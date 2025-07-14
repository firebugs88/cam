from ultralytics import YOLO
import cv2
import time
import platform
from collections import defaultdict
import numpy as np
import threading
import queue
import torch
import gc

class OptimizedYOLOTracker:
    def __init__(self, model_path='yolov8s.pt'):
        # Configurar dispositivo y optimizaciones
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📦 Cargando modelo YOLOv8 en {self.device}...")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Optimizaciones para GPU
        if self.device == 'cuda':
            self.model.half()  # FP16 para GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        print("✅ Modelo cargado correctamente")
        
        # Configurar colas para multi-threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Control de threads
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Variables para tracking
        self.track_history = defaultdict(lambda: [])
        self.unique_objects = set()
        self.class_counts = defaultdict(set)
        self.show_trajectories = True
        
        # Variables para FPS
        self.fps_counter = 0
        self.start_time = time.time()
        self.frame_count = 0
        
        # Colores para las trayectorias
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
                      (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128)]
        
    def _capture_worker(self, cap):
        """Worker para captura de frames"""
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Descartar frames si hay backlog
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put(frame)
            
    def _process_worker(self):
        """Worker para procesamiento YOLO"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Procesamiento optimizado con tracking
                with torch.no_grad():
                    results = self.model.track(
                        frame, 
                        persist=True, 
                        verbose=False, 
                        conf=0.4,
                        device=self.device
                    )
                
                # Procesar resultados del tracking
                processed_data = None
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    processed_data = self._process_tracking_results(results[0])
                
                # Enviar resultado
                if not self.result_queue.full():
                    self.result_queue.put((frame, results, processed_data))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en procesamiento: {e}")
                
    def _process_tracking_results(self, result):
        """Procesar resultados de tracking"""
        boxes = result.boxes.xyxy
        track_ids = result.boxes.id
        classes = result.boxes.cls
        
        # Convertir a numpy arrays si son tensores de PyTorch
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        else:
            boxes = np.array(boxes)
        
        if hasattr(track_ids, 'cpu'):
            track_ids = track_ids.int().cpu().numpy().tolist()
        else:
            track_ids = np.array(track_ids, dtype=int).tolist()
        
        if hasattr(classes, 'cpu'):
            classes = classes.cpu().numpy().tolist()
        else:
            classes = np.array(classes).tolist()
        
        # Filtrar objetos por área mínima
        min_area = 1000
        for box, track_id, cls in zip(boxes, track_ids, classes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < min_area:
                continue
                
            # Añadir a objetos únicos
            self.unique_objects.add(track_id)
            
            # Añadir a conteo por clase
            class_name = self.model.names[int(cls)]
            self.class_counts[class_name].add(track_id)
            
            # Calcular centro del objeto
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Almacenar punto en la trayectoria
            track = self.track_history[track_id]
            track.append((center_x, center_y))
            
            # Limitar la longitud de la trayectoria
            if len(track) > 30:
                track.pop(0)
                
        return {'boxes': boxes, 'track_ids': track_ids, 'classes': classes}
        
    def get_result(self):
        """Obtener resultado procesado"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None, None, None
            
    def start_threads(self, cap):
        """Iniciar threads de captura y procesamiento"""
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_worker, args=(cap,))
        self.process_thread = threading.Thread(target=self._process_worker)
        
        self.capture_thread.start()
        self.process_thread.start()
        
    def stop_threads(self):
        """Detener threads"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
            
    def calculate_fps(self):
        """Calcular FPS"""
        self.fps_counter += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 1.0:
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.start_time = time.time()
            return fps
        return 0
        
    def reset_tracking(self):
        """Reiniciar tracking"""
        self.track_history.clear()
        self.unique_objects.clear()
        self.class_counts.clear()
        
    def toggle_trajectories(self):
        """Cambiar visualización de trayectorias"""
        self.show_trajectories = not self.show_trajectories
        return self.show_trajectories

def main():
    # --- 1. Inicializar Tracker Optimizado ---
    tracker = OptimizedYOLOTracker('yolov8s.pt')

    # --- 2. Configuración de la Cámara ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara.")
        return

    # Configurar resolución y optimizaciones de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir latencia
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Configurar la ventana
    window_name = 'YOLOv8 Tracking en Tiempo Real'
    if platform.system() == "Windows":
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("🚀 Iniciando detección y tracking en tiempo real...")
    print("✅ Cámara iniciada correctamente")
    print("⌨️  Presiona 'q' para salir")
    print("⌨️  Presiona 'r' para reiniciar el tracking")
    print("⌨️  Presiona 't' para mostrar/ocultar trayectorias")

    # --- 3. Iniciar Multi-threading ---
    tracker.start_threads(cap)

    try:
        while True:
            # Obtener resultado del procesamiento multi-thread
            frame, results, processed_data = tracker.get_result()
            
            if frame is not None and results is not None:
                # Dibujar las anotaciones de YOLO
                annotated_frame = results[0].plot()
                
                # --- Dibujar información adicional de tracking ---
                # Dibujar trayectorias si está activado
                if tracker.show_trajectories and results[0].boxes is not None:
                    for track_id, track in tracker.track_history.items():
                        if len(track) > 1:
                            # Seleccionar color basado en el ID
                            color = tracker.colors[track_id % len(tracker.colors)]
                            
                            # Dibujar la trayectoria
                            points = np.array(track, dtype=np.int32)
                            cv2.polylines(annotated_frame, [points], 
                                        isClosed=False, color=color, thickness=2)
                            
                            # Dibujar un círculo en la posición actual
                            if len(track) > 0:
                                cv2.circle(annotated_frame, track[-1], 5, color, -1)
                
                # Mostrar estadísticas de tracking
                y_offset = 30
                cv2.putText(annotated_frame, f'Objetos unicos totales: {len(tracker.unique_objects)}', 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar conteo por clase
                y_offset += 25
                for class_name, ids in tracker.class_counts.items():
                    count = len(ids)
                    if count > 0:
                        cv2.putText(annotated_frame, f'{class_name}: {count}', 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 25
                
                # Calcular y mostrar FPS
                fps = tracker.calculate_fps()
                if fps > 0:
                    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', 
                               (frame.shape[1] - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Mostrar estado de trayectorias
                traj_text = "Trayectorias: ON" if tracker.show_trajectories else "Trayectorias: OFF"
                cv2.putText(annotated_frame, traj_text, 
                           (frame.shape[1] - 200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Mostrar el frame
                cv2.imshow(window_name, annotated_frame)
                
                # Limpiar cache GPU cada 100 frames
                tracker.frame_count += 1
                if tracker.frame_count % 100 == 0 and tracker.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reiniciar tracking
                tracker.reset_tracking()
                tracker.model = YOLO('yolov8s.pt')  # Recargar modelo
                tracker.model.to(tracker.device)
                if tracker.device == 'cuda':
                    tracker.model.half()
                print("🔄 Tracking reiniciado")
            elif key == ord('t'):
                # Toggle trayectorias
                show_traj = tracker.toggle_trajectories()
                print(f"👁️  Trayectorias {'activadas' if show_traj else 'desactivadas'}")

    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 4. Liberar Recursos ---
        print("\n🔄 Cerrando aplicación...")
        
        # Detener threads
        tracker.stop_threads()
        
        # Mostrar estadísticas finales
        print(f"📊 Resumen final:")
        print(f"   - Total de objetos únicos detectados: {len(tracker.unique_objects)}")
        for class_name, ids in tracker.class_counts.items():
            print(f"   - {class_name}: {len(ids)} objetos únicos")
        
        # Limpiar memoria GPU
        if tracker.device == 'cuda':
            torch.cuda.empty_cache()
            del tracker.model
            gc.collect()
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados correctamente.")

if __name__ == "__main__":
    main()