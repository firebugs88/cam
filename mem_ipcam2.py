import os
# SOLUCIÓN AL ERROR DE OpenMP - Debe ir ANTES de cualquier import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Configuración adicional para mejor rendimiento en AMD Ryzen
os.environ['OMP_NUM_THREADS'] = '6'  # Tu Ryzen 5 5600H tiene 6 cores
os.environ['MKL_NUM_THREADS'] = '6'

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
import traceback
import warnings
warnings.filterwarnings('ignore')  # Suprimir warnings no críticos

class OptimizedYOLOTracker:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4, classes=None):
        """
        Inicializa el rastreador YOLO optimizado para AMD Ryzen 5 5600H.
        """
        # Configuración optimizada para tu CPU AMD
        torch.set_num_threads(6)  # Optimizado para 6 cores físicos
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando el dispositivo: {self.device}")
        
        if self.device == 'cpu':
            print("Optimizando para AMD Ryzen 5 5600H...")
            # Habilitar optimizaciones para CPU AMD
            torch.backends.mkldnn.enabled = True
            torch.backends.cudnn.benchmark = False

        # --- Configuración del Modelo YOLO ---
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.confidence = confidence
        self.classes = classes

        # --- Configuración del Sistema de Re-ID ---
        self.reid_model = self._initialize_reid_model()
        self.embedding_dim = 1280
        self.similarity_threshold = 0.7
        self.db_index_path = "objects.index"
        self.db_names_path = "objects_names.json"
        self.faiss_index, self.name_map = self._load_or_create_db()
        
        # Diccionarios optimizados
        self.identified_tracks = {}
        self.last_seen_embeddings = {}
        self.embedding_cache = {}
        
        # --- Configuración de Hilos optimizada para Ryzen ---
        # Reducir el tamaño de las colas para menor uso de memoria
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)
        
        self.capture_thread = None
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Configuración de transformación optimizada
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Variables para control de FPS
        self.target_fps = 15  # Limitar FPS para mejor estabilidad
        self.frame_time = 1.0 / self.target_fps

    def _initialize_reid_model(self):
        """Carga el modelo MobileNetV2 optimizado para CPU AMD."""
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Optimizar para CPU si no hay CUDA
        if self.device == 'cpu':
            model = model.to('cpu')
            # Usar modo de evaluación para mejor rendimiento
            model.eval()
        else:
            model = model.to(self.device)
            
        feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1],
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1)
        )
        feature_extractor.eval()
        
        # Desactivar gradientes
        for param in feature_extractor.parameters():
            param.requires_grad = False
            
        # Optimización adicional para inferencia
        if self.device == 'cpu':
            feature_extractor = torch.jit.script(feature_extractor)
            
        return feature_extractor

    def _load_or_create_db(self):
        """Carga o crea la base de datos FAISS optimizada."""
        # Configurar FAISS para usar todos los cores disponibles
        faiss.omp_set_num_threads(6)
        
        if os.path.exists(self.db_index_path) and os.path.exists(self.db_names_path):
            print("Cargando base de datos de objetos existente...")
            try:
                faiss_index = faiss.read_index(self.db_index_path)
                with open(self.db_names_path, 'r') as f:
                    name_map = {int(k): v for k, v in json.load(f).items()}
                print(f"Base de datos cargada con {faiss_index.ntotal} objetos.")
            except Exception as e:
                print(f"Error cargando base de datos: {e}")
                print("Creando nueva base de datos...")
                faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                name_map = {}
        else:
            print("No se encontró base de datos. Creando una nueva.")
            faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            name_map = {}
        return faiss_index, name_map

    def _get_embedding(self, crop_img):
        """Genera un embedding con optimizaciones para CPU AMD."""
        try:
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                return None
            
            # Redimensionar imagen si es muy grande para mejorar rendimiento
            height, width = crop_img.shape[:2]
            if height > 300 or width > 300:
                scale = min(300/height, 300/width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                crop_img = cv2.resize(crop_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
            img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess(img_pil)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Usar autocast solo si hay GPU
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        embedding = self.reid_model(input_batch)
                else:
                    embedding = self.reid_model(input_batch)
            
            embedding_np = embedding.cpu().numpy().astype('float32')
            faiss.normalize_L2(embedding_np)
            
            return embedding_np
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return None

    def _capture_worker(self, cap):
        """Hilo optimizado para capturar frames."""
        last_frame_time = time.time()
        
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            
            # Control de FPS para reducir carga
            current_time = time.time()
            if current_time - last_frame_time < self.frame_time:
                time.sleep(0.001)
                continue
                
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame.")
                time.sleep(0.1)
                continue
            
            last_frame_time = current_time
            
            # Redimensionar frame si es muy grande
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_height = int(height * scale)
                new_width = int(width * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass

    def _process_worker(self):
        """Hilo para procesar frames con optimizaciones para CPU AMD."""
        # Configurar afinidad del thread para mejor rendimiento
        import psutil
        p = psutil.Process()
        # Usar cores físicos 0-5 para procesamiento
        p.cpu_affinity([0, 1, 2, 3, 4, 5])
        
        frame_skip = 0
        process_every_n_frames = 2  # Procesar cada 2 frames para mejor rendimiento
        
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
                
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Skip frames para mejorar rendimiento
            frame_skip += 1
            if frame_skip % process_every_n_frames != 0:
                continue

            # Configuración optimizada para tracking
            results = self.model.track(
                source=frame, 
                stream=False, 
                persist=True,
                verbose=False,
                conf=self.confidence,
                classes=self.classes,
                device=self.device,
                tracker="bytetrack.yaml",
                imgsz=640,  # Tamaño fijo para mejor rendimiento
                max_det=50,  # Limitar detecciones máximas
                retina_masks=False,  # Desactivar máscaras de alta resolución
                half=False  # No usar FP16 en CPU
            )
            
            if self.results_queue.full():
                try:
                    self.results_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            try:
                self.results_queue.put(results, block=False)
            except queue.Full:
                pass

    def _update_identities(self, frame, results):
        """Actualiza identidades con optimizaciones."""
        if results[0].boxes is None or results[0].boxes.id is None:
            return

        current_track_ids = set(results[0].boxes.id.int().cpu().tolist())
        
        # Limpiar tracks desaparecidos
        disappeared_ids = set(self.identified_tracks.keys()) - current_track_ids
        for track_id in disappeared_ids:
            self.embedding_cache.pop(track_id, None)
            self.identified_tracks.pop(track_id, None)
            self.last_seen_embeddings.pop(track_id, None)

        # Procesar solo un subset de objetos por frame para mejor rendimiento
        boxes_to_process = list(results[0].boxes)[:10]  # Máximo 10 objetos por frame
        
        for box in boxes_to_process:
            track_id = int(box.id[0])
            
            if track_id in self.identified_tracks:
                continue

            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            margin = 5
            y1 = max(0, y1 - margin)
            x1 = max(0, x1 - margin)
            y2 = min(frame.shape[0], y2 + margin)
            x2 = min(frame.shape[1], x2 + margin)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if track_id not in self.embedding_cache:
                embedding = self._get_embedding(crop)
                if embedding is None:
                    continue
                self.embedding_cache[track_id] = embedding
            else:
                embedding = self.embedding_cache[track_id]
                
            self.last_seen_embeddings[track_id] = embedding

            if self.faiss_index.ntotal > 0:
                D, I = self.faiss_index.search(embedding, 1)
                similarity = D[0][0]
                
                if similarity > self.similarity_threshold:
                    db_id = I[0][0]
                    if db_id in self.name_map:
                        self.identified_tracks[track_id] = self.name_map[db_id]
                else:
                    class_name = self.model.names[int(box.cls[0])]
                    self.identified_tracks[track_id] = f"{class_name} ID:{track_id}"
            else:
                class_name = self.model.names[int(box.cls[0])]
                self.identified_tracks[track_id] = f"{class_name} ID:{track_id}"

    def draw_results(self, frame, results):
        """Dibuja resultados con renderizado optimizado."""
        if results[0].boxes is None:
            return frame
            
        if results[0].boxes.id is None:
            return results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            track_id = int(box.id[0])
            label = self.identified_tracks.get(track_id, "Procesando...")
            
            if "ID:" not in label:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)

            # Dibujo simplificado para mejor rendimiento
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Texto con fondo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            text_y = y1 - baseline if y1 - text_height - baseline > 0 else y1 + text_height + baseline
            
            cv2.rectangle(
                frame, 
                (x1, text_y - text_height - baseline), 
                (x1 + text_width + 4, text_y + baseline), 
                color, 
                -1
            )
            cv2.putText(
                frame, 
                label, 
                (x1 + 2, text_y - baseline), 
                font, 
                font_scale, 
                (0, 0, 0), 
                thickness
            )

        return frame

    def _name_object(self):
        """Asigna nombres a objetos detectados."""
        self.pause_event.set()
        print("\n" + "="*50)
        print("ASIGNACIÓN DE NOMBRE")
        print("="*50)
        print("El video está en pausa. Objetos detectados:")
        
        available_ids = sorted(self.last_seen_embeddings.keys())
        if not available_ids:
            print("No hay objetos visibles en este momento.")
            self.pause_event.clear()
            return
            
        for track_id in available_ids:
            current_name = self.identified_tracks.get(track_id, f"ID:{track_id}")
            print(f"  - ID {track_id}: {current_name}")
        
        try:
            target_id_str = input("\nIngresa el ID del objeto a nombrar: ").strip()
            if not target_id_str:
                print("Operación cancelada.")
                return
                
            target_id = int(target_id_str)

            if target_id not in self.last_seen_embeddings:
                print(f"Error: El ID {target_id} no está disponible.")
                return

            object_name = input(f"Ingresa el nombre para el objeto ID {target_id}: ").strip()
            if not object_name:
                print("El nombre no puede estar vacío.")
                return

            embedding_to_add = self.last_seen_embeddings[target_id].copy()
            
            if embedding_to_add is None or embedding_to_add.shape != (1, self.embedding_dim):
                print(f"Error: Embedding inválido.")
                return
            
            new_db_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding_to_add)
            self.name_map[new_db_id] = object_name
            
            self.identified_tracks[target_id] = object_name
            
            faiss.write_index(self.faiss_index, self.db_index_path)
            json_safe_map = {str(k): v for k, v in self.name_map.items()}
            with open(self.db_names_path, 'w') as f:
                json.dump(json_safe_map, f, indent=4, ensure_ascii=False)
            
            print(f"\n✓ ¡Guardado exitosamente!")
            print(f"  El objeto ID {target_id} ahora se llama '{object_name}'")
            print(f"  Total de objetos en la base de datos: {self.faiss_index.ntotal}")

        except ValueError:
            print("Error: Debes ingresar un número válido para el ID.")
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            traceback.print_exc()
        finally:
            print("\nReanudando video en 2 segundos...")
            time.sleep(2)
            self.pause_event.clear()

    def run(self, cap):
        """Ejecuta el sistema de tracking optimizado."""
        window_name = "YOLOv8 con Memoria Persistente"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\n" + "="*50)
        print("SISTEMA DE TRACKING CON MEMORIA")
        print("="*50)
        print("\nControles:")
        print("  [N] - Nombrar un objeto")
        print("  [D] - Eliminar un objeto de la base de datos")
        print("  [L] - Listar objetos guardados")
        print("  [F] - Pantalla completa")
        print("  [Q] - Salir")
        print("="*50 + "\n")

        self.capture_thread = threading.Thread(
            target=self._capture_worker, 
            args=(cap,), 
            daemon=True
        )
        self.capture_thread.start()
        self.process_thread.start()

        fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        while not self.stop_event.is_set():
            try:
                results = self.results_queue.get(timeout=0.1)
                if results and results[0].orig_img is not None:
                    frame = results[0].orig_img.copy()
                    
                    self._update_identities(frame, results)
                    frame_to_show = self.draw_results(frame, results)
                    
                    # Calcular FPS
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - fps_time > 1.0:
                        current_fps = fps_counter / (current_time - fps_time)
                        fps_time = current_time
                        fps_counter = 0
                    
                    # Mostrar FPS y uso de CPU
                    cv2.putText(
                        frame_to_show, 
                        f"FPS: {current_fps:.1f} | Target: {self.target_fps}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
                    
                    cv2.imshow(window_name, frame_to_show)
            except queue.Empty:
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break
            elif key == ord('n'):
                self._name_object()
            elif key == ord('f'):
                is_fullscreen = cv2.getWindowProperty(
                    window_name, 
                    cv2.WND_PROP_FULLSCREEN
                ) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(
                    window_name, 
                    cv2.WND_PROP_FULLSCREEN, 
                    cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN
                )
            elif key == ord('l'):
                self._list_saved_objects()
            elif key == ord('d'):
                self._delete_object()

        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join()
        self.process_thread.join()
        cv2.destroyAllWindows()
        cap.release()

    def _list_saved_objects(self):
        """Lista objetos guardados."""
        print("\n" + "="*50)
        print("OBJETOS EN LA BASE DE DATOS")
        print("="*50)
        if not self.name_map:
            print("La base de datos está vacía.")
        else:
            for db_id, name in sorted(self.name_map.items()):
                print(f"  ID {db_id}: {name}")
        print("="*50)
        input("\nPresiona Enter para continuar...")

    def _delete_object(self):
        """Elimina un objeto de la base de datos."""
        self.pause_event.set()
        print("\n" + "="*50)
        print("ELIMINAR OBJETO DE LA BASE DE DATOS")
        print("="*50)
        
        if not self.name_map:
            print("La base de datos está vacía.")
            self.pause_event.clear()
            return
            
        for db_id, name in sorted(self.name_map.items()):
            print(f"  ID {db_id}: {name}")
            
        try:
            db_id_str = input("\nIngresa el ID del objeto a eliminar: ").strip()
            if not db_id_str:
                print("Operación cancelada.")
                return
                
            db_id = int(db_id_str)
            
            if db_id not in self.name_map:
                print(f"Error: No existe un objeto con ID {db_id}")
                return
                
            name_to_delete = self.name_map[db_id]
            confirm = input(f"¿Eliminar '{name_to_delete}'? (s/n): ").lower()
            
            if confirm == 's':
                print("Reconstruyendo base de datos...")
                
                new_index = faiss.IndexFlatIP(self.embedding_dim)
                new_name_map = {}
                
                for i in range(self.faiss_index.ntotal):
                    if i != db_id:
                        embedding = np.zeros((1, self.embedding_dim), dtype='float32')
                        self.faiss_index.reconstruct(i, embedding[0])
                        new_id = new_index.ntotal
                        new_index.add(embedding)
                        if i in self.name_map:
                            new_name_map[new_id] = self.name_map[i]
                
                self.faiss_index = new_index
                self.name_map = new_name_map
                
                faiss.write_index(self.faiss_index, self.db_index_path)
                json_safe_map = {str(k): v for k, v in self.name_map.items()}
                with open(self.db_names_path, 'w') as f:
                    json.dump(json_safe_map, f, indent=4, ensure_ascii=False)
                    
                print(f"✓ '{name_to_delete}' eliminado exitosamente.")
                
        except ValueError:
            print("Error: Debes ingresar un número válido.")
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            print("\nReanudando video...")
            time.sleep(1)
            self.pause_event.clear()


def main():
    """Función principal optimizada para AMD Ryzen 5 5600H."""
    # Verificar si psutil está instalado (necesario para optimizaciones)
    try:
        import psutil
    except ImportError:
        print("Instalando psutil para optimizaciones de CPU...")
        os.system("pip install psutil")
        import psutil
    
    # ================================================================
    # CONFIGURACIÓN DE LA CÁMARA
    # ================================================================
    camera_url = "rtsp://admin:123456@192.168.1.42:554/stream"  # Tu cámara IP
    # camera_url = 0  # Descomentar para usar webcam
    
    print("Iniciando conexión con la cámara...")
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el stream de la cámara.")
        print("Verifica:")
        print("  1. La URL es correcta")
        print("  2. La cámara está encendida y en la red")
        print("  3. Las credenciales son correctas")
        return

    # Optimizaciones específicas para streaming RTSP
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Limitar FPS desde la fuente
    
    # Usar backend optimizado si está disponible
    try:
        cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_FFMPEG)
    except:
        pass
    
    # Configurar códec H264 para mejor rendimiento
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Opcional: Reducir resolución para mejor rendimiento
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Clases a detectar (None = todas)
    target_classes = None 

    # Crear y ejecutar el tracker
    print("Inicializando sistema de tracking...")
    tracker = OptimizedYOLOTracker(
        model_path="yolov8n.pt",  # Usar yolov8n para mejor rendimiento en CPU
        confidence=0.5,
        classes=target_classes
    )
    
    try:
        tracker.run(cap)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"Error crítico: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Sistema cerrado correctamente.")


if __name__ == "__main__":
    main()