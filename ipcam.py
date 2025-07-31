import cv2
import threading
import queue
import time
import torch
import numpy as np
from ultralytics import YOLO

class OptimizedYOLOTracker:
    def __init__(self, model_path="yolov8s.pt", confidence=0.4, classes=None):
        """
        Inicializa el rastreador YOLO con configuraciones optimizadas.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando el dispositivo: {self.device}")
        
        # Cargar el modelo YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.float16 = True  # Usar precisión FP16 para aceleración
        
        self.confidence = confidence
        self.classes = classes
        
        self.tracking_history = {}
        self.next_id = 0
        self.object_counts = {}
        
        self.processing_roi = False
        self.roi_areas = []
        self.adaptive_roi = False
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.results_queue = queue.Queue(maxsize=1)
        
        self.capture_thread = None # Se inicializará en run()
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        
        self.stop_event = threading.Event()

        # Configuración para la pre-asignación de memoria de GPU para ROIs
        self.roi_sizes = [(320, 320), (416, 416), (512, 512), (640, 640)]
        self.roi_tensors = {}
        if self.device == 'cuda':
            for size in self.roi_sizes:
                self.roi_tensors[size] = torch.zeros((1, 3, size[0], size[1]), dtype=torch.float16, device=self.device)

    def _capture_worker(self, cap):
        """
        Hilo para capturar frames de la cámara.
        """
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame. Saliendo...")
                self.stop_event.set()
                break
            
            # Poner el frame en la cola, descartando si ya está lleno
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    
    def _process_worker(self):
        """
        Hilo para procesar los frames con el modelo YOLO.
        """
        while not self.stop_event.is_set():
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

            # Poner los resultados en la cola, descartando si ya está lleno
            try:
                self.results_queue.put(results, block=False)
            except queue.Full:
                pass
            
    def process_frame(self, frame):
        """
        Función principal para procesar un solo frame (obsoleto con multihilo).
        """
        return self.model(frame)

    def draw_results(self, frame, results):
        """
        Dibuja los resultados de detección y rastreo en el frame.
        """
        # Dibujar los bounding boxes
        frame = results[0].plot()

        # Dibujar las trayectorias
        for track_id, history in self.tracking_history.items():
            for i in range(1, len(history)):
                cv2.line(frame, history[i - 1], history[i], (0, 255, 0), 2)
        
        # Dibujar las ROIs si están activas
        if self.processing_roi and self.roi_areas:
            for roi in self.roi_areas:
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Mostrar conteo de objetos
        y_offset = 30
        for class_name, count in self.object_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

        return frame

    def run(self, cap):
        """
        Inicia el procesamiento de video en hilos.
        """
        self.capture_thread = threading.Thread(target=self._capture_worker, args=(cap,), daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        while not self.stop_event.is_set():
            try:
                results = self.results_queue.get(timeout=1)
                frame = self.draw_results(results[0].orig_img.copy(), results)
                cv2.imshow("YOLOv8 Live Tracking", frame)
            except queue.Empty:
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break

        self.stop_event.set()
        self.capture_thread.join()
        self.process_thread.join()
        cv2.destroyAllWindows()
        cap.release()

def main():
    """
    Función principal para inicializar la aplicación de seguimiento.
    """
    # ================================================================
    # MODIFICACIÓN CLAVE: Reemplaza la URL de la cámara local con la de tu cámara IP.
    # El formato general es: "rtsp://usuario:contraseña@direccion_ip:puerto/ruta_del_stream"
    # Si la cámara no tiene usuario/contraseña, puedes omitirlos.
    # ================================================================
    camera_url = "rtsp://admin:123456@192.168.1.42:554/stream"  # Ejemplo de URL de cámara IP

    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Error: No se pudo abrir el stream de la cámara. Verifique la URL.")
        return

    # Ajustar la resolución y FPS de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Iniciar el rastreador
    tracker = OptimizedYOLOTracker()
    tracker.run(cap)

if __name__ == "__main__":
    main()