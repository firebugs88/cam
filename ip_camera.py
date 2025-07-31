import cv2
import numpy as np
import time
import threading
from urllib.parse import urlparse

class IPCameraStreamer:
    def __init__(self, camera_url, username=None, password=None):
        """
        Initialize IP Camera Streamer
        
        Args:
            camera_url (str): IP camera URL (e.g., 'http://192.168.1.100:8080/video')
            username (str): Username for authentication (optional)
            password (str): Password for authentication (optional)
        """
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.cap = None
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        
    def _build_url(self):
        """Build the complete URL with authentication if provided"""
        if self.username and self.password:
            parsed = urlparse(self.camera_url)
            return f"{parsed.scheme}://{self.username}:{self.password}@{parsed.netloc}{parsed.path}"
        return self.camera_url
    
    def connect(self):
        """Connect to the IP camera"""
        try:
            url = self._build_url()
            self.cap = cv2.VideoCapture(url)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test connection
            ret, frame = self.cap.read()
            if ret:
                print(f"Successfully connected to IP camera: {self.camera_url}")
                return True
            else:
                print("Failed to read frame from IP camera")
                return False
                
        except Exception as e:
            print(f"Error connecting to IP camera: {e}")
            return False
    
    def start_stream(self):
        """Start streaming from the IP camera"""
        if not self.connect():
            return False
            
        self.is_running = True
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self._capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        return True
    
    def _capture_frames(self):
        """Capture frames in a separate thread"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            else:
                print("Failed to capture frame")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the latest frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop the camera stream"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("IP camera stream stopped")

def main():
    # Configure your IP camera here
    # Common IP camera URL formats:
    # MJPEG: http://192.168.1.100:8080/video
    # RTSP: rtsp://192.168.1.100:554/stream
    # HTTP: http://192.168.1.100/mjpeg.cgi
    
    camera_url = input("Enter your IP camera URL: ").strip()
    
    # Optional authentication
    use_auth = input("Does your camera require authentication? (y/n): ").lower() == 'y'
    username = None
    password = None
    
    if use_auth:
        username = input("Username: ").strip()
        password = input("Password: ").strip()
    
    # Create camera streamer
    camera = IPCameraStreamer(camera_url, username, password)
    
    # Start streaming
    if camera.start_stream():
        print("\nCamera stream started successfully!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'f' to toggle fullscreen")
        
        cv2.namedWindow('IP Camera Stream', cv2.WINDOW_AUTOSIZE)
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = camera.get_frame()
                
                if frame is not None:
                    # Add timestamp and frame info
                    height, width = frame.shape[:2]
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add info overlay
                    cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Resolution: {width}x{height}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"FPS: {fps:.2f}")
                    
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('IP Camera Stream', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"ip_camera_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Frame saved as {filename}")
                    elif key == ord('f'):
                        # Toggle fullscreen
                        prop = cv2.getWindowProperty('IP Camera Stream', cv2.WND_PROP_FULLSCREEN)
                        if prop == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty('IP Camera Stream', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)
                        else:
                            cv2.setWindowProperty('IP Camera Stream', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    print("No frame available")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start camera stream")

if __name__ == "__main__":
    # Example usage with common camera URLs
    print("IP Camera Connection Tool")
    print("=" * 30)
    print("\nCommon IP camera URL formats:")
    print("- Android IP Webcam: http://192.168.1.XXX:8080/video")
    print("- DroidCam: http://192.168.1.XXX:4747/video")
    print("- RTSP stream: rtsp://192.168.1.XXX:554/stream")
    print("- Generic MJPEG: http://192.168.1.XXX/mjpeg.cgi")
    print()
    
    main()