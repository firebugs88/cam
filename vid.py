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
        
        # ROI (Region of Interest) configuration
        self.roi_enabled = False
        self.roi_regions = []  # Lista de ROIs: [(x1, y1, x2, y2), ...]
        self.roi_selection_mode = False
        self.roi_temp_points = []
        self.roi_adaptive = True  # ROI adaptativo basado en detecciones
        self.roi_expansion_factor = 1.2  # Factor de expansión para ROI adaptativo
        self.roi_min_size = (160, 160)  # Tamaño mínimo de ROI
        self.roi_memory_pool = []  # Pool de memoria GPU para ROIs
        
        # Configuración de memoria GPU para ROIs
        if self.device == 'cuda':
            self._init_gpu_memory_pool()
            
        # Variables para optimización de memoria
        self.frame_skip_counter = 0
        self.process_every_n_frames = 1  # Procesar cada N frames en ROI
        self.last_detections = []  # Cache de últimas detecciones
        
    def _init_gpu_memory_pool(self):
        """Inicializar pool de memoria GPU para ROIs"""
        try:
            # Pre-allocar memoria para diferentes tamaños de ROI
            common_sizes = [(320, 320), (416, 416), (512, 512), (640, 640)]
            for width, height in common_sizes:
                if self.device == 'cuda':
                    # Crear tensor vacío en GPU para cada tamaño
                    tensor = torch.empty((height, width, 3), dtype=torch.uint8, device=self.device)
                    self.roi_memory_pool.append({'size': (width, height), 'tensor': tensor})
            print("✅ Pool de memoria GPU inicializado")
        except Exception as e:
            print(f"⚠️ Error inicializando pool de memoria GPU: {e}")
            
    def _get_roi_from_pool(self, width, height):
        """Obtener tensor de memoria pre-allocada para ROI"""
        for pool_item in self.roi_memory_pool:
            pool_width, pool_height = pool_item['size']
            if pool_width >= width and pool_height >= height:
                return pool_item['tensor'][:height, :width, :]
        return None
        
    def add_roi_region(self, x1, y1, x2, y2):
        """Añadir nueva región ROI"""
        # Validar coordenadas
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Verificar tamaño mínimo
        width, height = x2 - x1, y2 - y1
        if width < self.roi_min_size[0] or height < self.roi_min_size[1]:
            return False
            
        self.roi_regions.append((x1, y1, x2, y2))
        return True
        
    def clear_roi_regions(self):
        """Limpiar todas las regiones ROI"""
        self.roi_regions.clear()
        
    def toggle_roi_mode(self):
        """Activar/desactivar modo ROI"""
        self.roi_enabled = not self.roi_enabled
        if not self.roi_enabled:
            self.roi_regions.clear()
        return self.roi_enabled
        
    def toggle_roi_selection(self):
        """Activar/desactivar modo de selección ROI"""
        self.roi_selection_mode = not self.roi_selection_mode
        self.roi_temp_points.clear()
        return self.roi_selection_mode
        
    def _update_adaptive_roi(self, detections):
        """Actualizar ROI adaptativo basado en detecciones"""
        if not self.roi_adaptive or not detections:
            return
            
        # Limpiar ROIs existentes
        self.roi_regions.clear()
        
        # Agrupar detecciones cercanas
        roi_candidates = []
        for detection in detections:
            x1, y1, x2, y2 = detection
            # Expandir la detección
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = (x2 - x1) * self.roi_expansion_factor, (y2 - y1) * self.roi_expansion_factor
            
            new_x1 = max(0, int(center_x - width / 2))
            new_y1 = max(0, int(center_y - height / 2))
            new_x2 = int(center_x + width / 2)
            new_y2 = int(center_y + height / 2)
            
            roi_candidates.append((new_x1, new_y1, new_x2, new_y2))
        
        # Fusionar ROIs solapadas
        merged_rois = self._merge_overlapping_rois(roi_candidates)
        self.roi_regions.extend(merged_rois)
        
    def _merge_overlapping_rois(self, rois):
        """Fusionar ROIs que se solapan"""
        if not rois:
            return []
            
        merged = []
        for roi in rois:
            x1, y1, x2, y2 = roi
            merged_with_existing = False
            
            for i, existing_roi in enumerate(merged):
                ex1, ey1, ex2, ey2 = existing_roi
                
                # Verificar solapamiento
                if (x1 < ex2 and x2 > ex1 and y1 < ey2 and y2 > ey1):
                    # Fusionar ROIs
                    merged[i] = (min(x1, ex1), min(y1, ey1), max(x2, ex2), max(y2, ey2))
                    merged_with_existing = True
                    break
                    
            if not merged_with_existing:
                merged.append(roi)
                
        return merged
        
    def _process_roi_regions(self, frame):
        """Procesar solo las regiones ROI del frame"""
        if not self.roi_enabled or not self.roi_regions:
            return frame, []
            
        roi_results = []
        processed_frame = frame.copy()
        
        for roi_idx, (x1, y1, x2, y2) in enumerate(self.roi_regions):
            # Extraer ROI del frame
            roi_frame = frame[y1:y2, x1:x2]
            
            if roi_frame.size == 0:
                continue
                
            # Usar memoria pre-allocada si está disponible
            roi_width, roi_height = roi_frame.shape[1], roi_frame.shape[0]
            gpu_tensor = self._get_roi_from_pool(roi_width, roi_height)
            
            if gpu_tensor is not None and self.device == 'cuda':
                # Copiar ROI a GPU tensor pre-allocado
                gpu_tensor[:roi_height, :roi_width] = torch.from_numpy(roi_frame).to(self.device)
                roi_input = gpu_tensor.cpu().numpy()
            else:
                roi_input = roi_frame
                
            # Procesar ROI con YOLO
            with torch.no_grad():
                roi_result = self.model.track(
                    roi_input,
                    persist=True,
                    verbose=False,
                    conf=0.3,  # Menor confianza para ROI
                    device=self.device
                )
                
            # Ajustar coordenadas al frame completo
            if roi_result[0].boxes is not None:
                adjusted_boxes = []
                for box in roi_result[0].boxes.xyxy:
                    if hasattr(box, 'cpu'):
                        box = box.cpu().numpy()
                    box[0] += x1  # x1
                    box[1] += y1  # y1
                    box[2] += x1  # x2
                    box[3] += y1  # y2
                    adjusted_boxes.append(box)
                    
                roi_results.append({
                    'roi_idx': roi_idx,
                    'boxes': adjusted_boxes,
                    'original_result': roi_result[0]
                })
                
        return processed_frame, roi_results

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
        """Worker para procesamiento YOLO con ROI optimizado"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Incrementar contador de frames
                self.frame_skip_counter += 1
                
                # Procesamiento con ROI si está habilitado
                if self.roi_enabled and self.roi_regions:
                    # Procesar solo regiones ROI
                    processed_frame, roi_results = self._process_roi_regions(frame)
                    
                    # Combinar resultados de todas las ROIs
                    all_boxes = []
                    for roi_result in roi_results:
                        all_boxes.extend(roi_result['boxes'])
                    
                    # Actualizar ROI adaptativo si está habilitado
                    if self.roi_adaptive and all_boxes:
                        self._update_adaptive_roi(all_boxes)
                    
                    # Crear resultado simulado para compatibilidad
                    results = roi_results
                    processed_data = {'boxes': all_boxes, 'roi_mode': True}
                    
                elif self.roi_enabled and not self.roi_regions and len(self.last_detections) > 0:
                    # Usar última detección para crear ROI adaptativo
                    self._update_adaptive_roi(self.last_detections)
                    # Procesar frame completo esta vez
                    with torch.no_grad():
                        results = self.model.track(
                            frame, 
                            persist=True, 
                            verbose=False, 
                            conf=0.4,
                            device=self.device
                        )
                    
                    processed_data = None
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        processed_data = self._process_tracking_results(results[0])
                        # Guardar detecciones para ROI adaptativo
                        if results[0].boxes is not None:
                            self.last_detections = results[0].boxes.xyxy.cpu().numpy().tolist()
                
                else:
                    # Procesamiento normal cuando ROI está deshabilitado
                    # Saltar frames para optimización cuando no hay ROI
                    if self.frame_skip_counter % self.process_every_n_frames != 0:
                        continue
                        
                    with torch.no_grad():
                        results = self.model.track(
                            frame, 
                            persist=True, 
                            verbose=False, 
                            conf=0.4,
                            device=self.device
                        )
                    
                    processed_data = None
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        processed_data = self._process_tracking_results(results[0])
                        # Guardar detecciones para posible ROI adaptativo
                        if results[0].boxes is not None:
                            self.last_detections = results[0].boxes.xyxy.cpu().numpy().tolist()
                
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
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para selección de ROI con mouse"""
        if not self.roi_selection_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Primer punto de la ROI
            self.roi_temp_points = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE and len(self.roi_temp_points) == 1:
            # Mostrar ROI temporal mientras se arrastra
            param['temp_roi'] = (self.roi_temp_points[0][0], self.roi_temp_points[0][1], x, y)
            
        elif event == cv2.EVENT_LBUTTONUP and len(self.roi_temp_points) == 1:
            # Segundo punto - finalizar ROI
            x1, y1 = self.roi_temp_points[0]
            if self.add_roi_region(x1, y1, x, y):
                print(f"✅ ROI agregada: ({x1}, {y1}) -> ({x}, {y})")
            else:
                print("❌ ROI demasiado pequeña")
            self.roi_temp_points.clear()
            param['temp_roi'] = None
            
    def draw_roi_overlay(self, frame, temp_roi=None):
        """Dibujar overlay de ROI en el frame"""
        overlay = frame.copy()
        
        # Dibujar ROIs existentes
        for i, (x1, y1, x2, y2) in enumerate(self.roi_regions):
            # Rectángulo de ROI
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Texto de ROI
            cv2.putText(overlay, f'ROI {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Área semi-transparente
            roi_area = overlay[y1:y2, x1:x2]
            roi_area[:] = cv2.addWeighted(roi_area, 0.7, 
                                        np.full_like(roi_area, (0, 255, 0)), 0.3, 0)
        
        # Dibujar ROI temporal si existe
        if temp_roi:
            x1, y1, x2, y2 = temp_roi
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, 'Nueva ROI', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return overlay

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

    # Configurar callback del mouse para ROI
    mouse_params = {'temp_roi': None}
    cv2.setMouseCallback(window_name, tracker.mouse_callback, mouse_params)

    print("🚀 Iniciando detección y tracking en tiempo real...")
    print("✅ Cámara iniciada correctamente")
    print("⌨️  Presiona 'q' para salir")
    print("⌨️  Presiona 'r' para reiniciar el tracking")
    print("⌨️  Presiona 't' para mostrar/ocultar trayectorias")
    print("⌨️  Presiona 'o' para activar/desactivar ROI")
    print("⌨️  Presiona 's' para seleccionar ROI con mouse")
    print("⌨️  Presiona 'c' para limpiar todas las ROIs")
    print("⌨️  Presiona 'a' para ROI adaptativo on/off")

    # --- 3. Iniciar Multi-threading ---
    tracker.start_threads(cap)

    try:
        while True:
            # Obtener resultado del procesamiento multi-thread
            frame, results, processed_data = tracker.get_result()
            
            if frame is not None and results is not None:
                # Manejar resultados ROI vs normales
                if processed_data and processed_data.get('roi_mode', False):
                    # Modo ROI - dibujar manualmente
                    annotated_frame = frame.copy()
                    
                    # Dibujar detecciones de ROI
                    for box in processed_data['boxes']:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, 'ROI Detection', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Modo normal - usar plot de YOLO
                    annotated_frame = results[0].plot()
                
                # --- Dibujar información adicional de tracking ---
                # Dibujar trayectorias si está activado
                if tracker.show_trajectories and not (processed_data and processed_data.get('roi_mode', False)):
                    if results[0].boxes is not None:
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
                
                # Mostrar estado de ROI
                roi_text = f"ROI: {'ON' if tracker.roi_enabled else 'OFF'} ({len(tracker.roi_regions)} regiones)"
                cv2.putText(annotated_frame, roi_text, 
                           (frame.shape[1] - 250, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Mostrar modo de selección ROI
                if tracker.roi_selection_mode:
                    cv2.putText(annotated_frame, "SELECCIONANDO ROI - Arrastra para crear", 
                               (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Dibujar overlay de ROI
                if tracker.roi_enabled or tracker.roi_selection_mode:
                    annotated_frame = tracker.draw_roi_overlay(annotated_frame, mouse_params.get('temp_roi'))
                
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
            elif key == ord('o'):
                # Toggle ROI mode
                roi_enabled = tracker.toggle_roi_mode()
                print(f"🎯 ROI {'activado' if roi_enabled else 'desactivado'}")
            elif key == ord('s'):
                # Toggle ROI selection mode
                roi_selection = tracker.toggle_roi_selection()
                print(f"🖱️  Selección ROI {'activada' if roi_selection else 'desactivada'}")
            elif key == ord('c'):
                # Clear all ROI regions
                tracker.clear_roi_regions()
                print("🗑️  Todas las ROIs eliminadas")
            elif key == ord('a'):
                # Toggle adaptive ROI
                tracker.roi_adaptive = not tracker.roi_adaptive
                print(f"🤖 ROI adaptativo {'activado' if tracker.roi_adaptive else 'desactivado'}")

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