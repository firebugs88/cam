import cv2
import threading
import queue
import time
import torch
import numpy as np
from ultralytics import YOLO
import json
import faiss
from torchvision import models, transforms
from PIL import Image
import os

class OptimizedYOLOTracker:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4, classes=None):
        """
        Inicializa el rastreador YOLO con un sistema de Re-identificación (Re-ID) para memoria persistente.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando el dispositivo: {self.device}")

        # --- Configuración del Modelo YOLO ---
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.confidence = confidence
        self.classes = classes

        # --- Configuración del Sistema de Re-ID ---
        self.reid_model = self._initialize_reid_model()
        self.embedding_dim = 1280  # Dimensión de salida de MobileNetV2
        self.similarity_threshold = 0.85  # Umbral de similitud (ajustar experimentalmente)
        self.db_index_path = "objects.index"
        self.db_names_path = "objects_names.json"
        self.faiss_index, self.name_map = self._load_or_create_db()
        
        # Diccionario para mantener los nombres identificados en la sesión actual {track_id: "nombre"}
        self.identified_tracks = {}
        self.last_seen_embeddings = {} # {track_id: embedding}

        # --- Configuración de Hilos y Colas ---
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)
        
        self.capture_thread = None
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

    def _initialize_reid_model(self):
        """Carga el modelo MobileNetV2 pre-entrenado y lo prepara para la extracción de características."""
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(self.device)
        # Usamos el modelo sin la capa de clasificación final
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten(1))
        feature_extractor.eval()
        return feature_extractor

    def _load_or_create_db(self):
        """Carga la base de datos de FAISS y el mapa de nombres, o los crea si no existen."""
        if os.path.exists(self.db_index_path) and os.path.exists(self.db_names_path):
            print("Cargando base de datos de objetos existente...")
            faiss_index = faiss.read_index(self.db_index_path)
            with open(self.db_names_path, 'r') as f:
                # Las claves de JSON se guardan como strings, las convertimos a int
                name_map = {int(k): v for k, v in json.load(f).items()}
            print(f"Base de datos cargada con {faiss_index.ntotal} objetos.")
        else:
            print("No se encontró base de datos. Creando una nueva.")
            faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            name_map = {}
        return faiss_index, name_map

    def _get_embedding(self, crop_img):
        """Genera un embedding (vector de características) para una imagen de objeto."""
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.reid_model(input_batch)
        
        # Normalizar el embedding para usarlo con similitud de coseno (producto punto)
        embedding_np = embedding.cpu().numpy()
        faiss.normalize_L2(embedding_np)
        return embedding_np

    def _capture_worker(self, cap):
        """Hilo para capturar frames de la cámara."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame. Saliendo...")
                self.stop_event.set()
                break
            try:
                self.frame_queue.put(frame, block=True, timeout=1)
            except queue.Full:
                pass

    def _process_worker(self):
        """Hilo para procesar los frames con el modelo YOLO."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            results = self.model.track(
                source=frame, 
                stream=False, 
                persist=True,
                verbose=False,
                conf=self.confidence,
                classes=self.classes,
                device=self.device
            )
            try:
                self.results_queue.put(results, block=True, timeout=1)
            except queue.Full:
                pass

    def _update_identities(self, frame, results):
        """Compara los objetos detectados con la base de datos para re-identificarlos."""
        if results[0].boxes.id is None:
            return

        current_track_ids = set(results[0].boxes.id.int().cpu().tolist())
        
        # Limpiar tracks que ya no están en pantalla
        disappeared_ids = set(self.identified_tracks.keys()) - current_track_ids
        for track_id in disappeared_ids:
            del self.identified_tracks[track_id]
            if track_id in self.last_seen_embeddings:
                del self.last_seen_embeddings[track_id]

        for box in results[0].boxes:
            track_id = int(box.id[0])
            
            # Si ya lo identificamos en esta sesión, continuamos
            if track_id in self.identified_tracks:
                continue

            # Extraer el crop del objeto
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = self._get_embedding(crop)
            self.last_seen_embeddings[track_id] = embedding

            # Buscar en la base de datos si hay suficientes objetos para comparar
            if self.faiss_index.ntotal > 0:
                # D es la distancia (L2), I es el índice del vecino más cercano
                D, I = self.faiss_index.search(embedding, 1)
                similarity = 1 - (D[0][0] / 4) # Convertir distancia L2 a una "similitud" aproximada
                
                if similarity > self.similarity_threshold:
                    db_id = I[0][0]
                    self.identified_tracks[track_id] = self.name_map[db_id]
                else:
                    # No es lo suficientemente similar, es un objeto no reconocido
                    class_name = self.model.names[int(box.cls[0])]
                    self.identified_tracks[track_id] = f"{class_name} ID:{track_id}"
            else:
                # La base de datos está vacía, solo mostrar ID
                class_name = self.model.names[int(box.cls[0])]
                self.identified_tracks[track_id] = f"{class_name} ID:{track_id}"


    def draw_results(self, frame, results):
        """Dibuja los resultados, usando los nombres personalizados del sistema de Re-ID."""
        if results[0].boxes.id is None:
            # Si no hay tracking, solo dibuja las detecciones
            return results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            track_id = int(box.id[0])
            label = self.identified_tracks.get(track_id, "Procesando...")
            
            color = (0, 255, 0) if "ID:" not in label else (255, 255, 0) # Verde para conocidos, Amarillo para desconocidos

            # Dibujar el bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar el texto (nombre o ID)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def _name_object(self):
        """Pausa el video y pide al usuario que nombre un objeto por su ID."""
        self.pause_event.set()
        print("\n--- ASIGNACIÓN DE NOMBRE ---")
        print("El video está en pausa. Revisa los IDs en pantalla.")
        
        try:
            target_id_str = input("Ingresa el ID del objeto que quieres nombrar (ej: 1, 2, ...): ")
            target_id = int(target_id_str)

            if target_id not in self.last_seen_embeddings:
                print(f"Error: El ID {target_id} no es válido o ya no está en pantalla.")
                self.pause_event.clear()
                return

            object_name = input(f"Ingresa el nombre para el objeto con ID {target_id}: ")
            if not object_name:
                print("El nombre no puede estar vacío.")
                self.pause_event.clear()
                return

            # Añadir el embedding a FAISS y el nombre al mapa
            embedding_to_add = self.last_seen_embeddings[target_id]
            new_db_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding_to_add)
            self.name_map[new_db_id] = object_name
            
            # Actualizar el nombre en la sesión actual
            self.identified_tracks[target_id] = object_name

            # Guardar los cambios en el disco
            faiss.write_index(self.faiss_index, self.db_index_path)
            with open(self.db_names_path, 'w') as f:
                json.dump(self.name_map, f, indent=4)
            
            print(f"¡Guardado! El objeto con ID {target_id} ahora es '{object_name}'.")

        except ValueError:
            print("Error: Entrada no válida. Debes ingresar un número para el ID.")
        except Exception as e:
            print(f"Ocurrió un error: {e}")
        
        print("--- Reanudando video ---")
        self.pause_event.clear()


    def run(self, cap):
        """Inicia el procesamiento de video en hilos."""
        window_name = "YOLOv8 con Memoria Persistente"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("\n--- Controles ---")
        print("Presiona 'n' para nombrar un objeto.")
        print("Presiona 'q' para salir.")
        print("Presiona 'f' para alternar pantalla completa.")

        self.capture_thread = threading.Thread(target=self._capture_worker, args=(cap,), daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        while not self.stop_event.is_set():
            try:
                results = self.results_queue.get(timeout=1)
                if results and results[0].orig_img is not None:
                    frame = results[0].orig_img.copy()
                    
                    # Lógica de Re-ID
                    self._update_identities(frame, results)
                    
                    # Dibujar resultados
                    frame_to_show = self.draw_results(frame, results)
                    cv2.imshow(window_name, frame_to_show)
                else:
                    continue
            except queue.Empty:
                # Si no hay resultados, mostramos el último frame capturado para que no se congele
                if not self.frame_queue.empty():
                     cv2.imshow(window_name, self.frame_queue.get())
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break
            elif key == ord('f'):
                is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                      cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN)
            elif key == ord('n'):
                self._name_object()

        self.stop_event.set()
        if self.capture_thread: self.capture_thread.join()
        if self.process_thread: self.process_thread.join()
        cv2.destroyAllWindows()
        cap.release()

def main():
    """Función principal para inicializar la aplicación de seguimiento."""
    # ================================================================
    # MODIFICACIÓN CLAVE: Reemplaza la URL con la de tu cámara IP o usa 0 para la webcam local.
    # Formato: "rtsp://usuario:contraseña@direccion_ip:puerto/ruta_del_stream"
    # ================================================================
    # camera_url = "rtsp://admin:123456@192.168.1.42:554/stream"  # Ejemplo de URL de cámara IP
    #camera_url = 0 # Usar la webcam por defecto
    camera_url = "rtsp://admin:123456@192.168.1.42:554/stream"
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Error: No se pudo abrir el stream de la cámara. Verifique la URL o el índice de la cámara.")
        return

    # Opcional: Ajustar resolución y FPS
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # Clases que quieres detectar (None para todas las clases de COCO)
    # Ejemplo: [0] para 'person', [2] para 'car', [16] para 'dog'
    target_classes = None 

    tracker = OptimizedYOLOTracker(
        model_path="yolov8n.pt", # Puedes usar 'yolov8s.pt' para más precisión a costa de velocidad
        confidence=0.5,
        classes=target_classes
    )
    tracker.run(cap)

if __name__ == "__main__":
    main()
