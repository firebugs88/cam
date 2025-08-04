# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time YOLOv8 object detection and tracking system that processes both local camera and IP camera input with GPU acceleration. The application uses multi-threading for optimized performance and provides object trajectory visualization. Includes dedicated IP camera connection utilities with authentication support.

## Key Architecture

### Core Components

- **OptimizedYOLOTracker**: Main class that handles YOLO model loading, multi-threaded frame processing, and object tracking
- **IPCameraStreamer**: Dedicated class for IP camera connection, authentication, and streaming
- **Multi-threading Design**: Uses separate threads for frame capture (`_capture_worker`) and YOLO processing (`_process_worker`) with queue-based communication
- **GPU Optimization**: Supports CUDA acceleration with FP16 precision and memory management

### Key Files

- `vid.py`: Main application file containing the complete tracking system with local camera support
- `ipcam.py`: IP camera YOLO tracking system with multi-threading optimization
- `ip_camera.py`: Standalone IP camera connection and streaming utility
- `yolov8s.pt`: Pre-trained YOLOv8 model file

## Dependencies

The project uses these key libraries:
- `ultralytics` (YOLO implementation)
- `opencv-python` (cv2 for video processing)
- `torch` (PyTorch for GPU acceleration)
- `numpy` (numerical operations)
- `urllib.parse` (URL parsing for IP camera authentication)

## Common Commands

### Running the Application

**Local Camera (Webcam):**
```bash
python vid.py
```

**IP Camera with YOLO Tracking:**
```bash
python ipcam.py
```

**IP Camera Stream Only (No YOLO):**
```bash
python ip_camera.py
```

### Runtime Controls

**vid.py (Local Camera) Controls:**
- Press 'q' to quit
- Press 'r' to reset tracking
- Press 't' to toggle trajectory visualization
- Press 'o' to toggle ROI mode
- Press 's' to toggle ROI selection mode (mouse drag to create ROI)
- Press 'c' to clear all ROI regions
- Press 'a' to toggle adaptive ROI

**ipcam.py (IP Camera) Controls:**
- Press 'q' to quit

**ip_camera.py (Stream Only) Controls:**
- Press 'q' to quit
- Press 's' to save current frame
- Press 'f' to toggle fullscreen

## Development Notes

### GPU Configuration
- Automatically detects CUDA availability
- Uses FP16 precision for GPU acceleration
- Implements memory cleanup every 100 frames

### Performance Features
- Multi-threaded frame capture and processing
- Queue-based frame buffering with overflow protection
- Configurable confidence threshold (0.4 default)
- Minimum object area filtering (1000 pixels)
- ROI (Region of Interest) processing for computational optimization
- GPU memory pooling for efficient ROI processing
- Adaptive ROI adjustment based on object detection

### Tracking System
- Maintains trajectory history for each tracked object (max 30 points)
- Tracks unique objects across frames
- Provides per-class object counting
- Color-coded trajectory visualization

## Configuration

### Local Camera Settings (vid.py)
Camera settings are configured in `main()`:
- Resolution: 1280x720
- FPS: 30
- Buffer size: 1 (reduced latency)

### IP Camera Settings (ipcam.py)
- Resolution: 1280x720
- FPS: 30
- Default URL: `rtsp://admin:123456@192.168.1.42:554/stream` (modify in `main()`)

### IP Camera Connection (ip_camera.py)
- Supports various URL formats: MJPEG, RTSP, HTTP
- Interactive URL input with authentication support
- Buffer size: 1 (reduced latency)
- Automatic connection testing

Model confidence and tracking parameters can be adjusted in the `_process_worker` method.

### IP Camera URL Formats

Common IP camera URL formats supported:
- **Android IP Webcam**: `http://192.168.1.XXX:8080/video`
- **DroidCam**: `http://192.168.1.XXX:4747/video`
- **RTSP stream**: `rtsp://192.168.1.XXX:554/stream`
- **RTSP with auth**: `rtsp://username:password@192.168.1.XXX:554/stream`
- **Generic MJPEG**: `http://192.168.1.XXX/mjpeg.cgi`
- **HTTP with auth**: `http://username:password@192.168.1.XXX/video`

### ROI Optimization Features

The system includes advanced ROI processing to reduce computational load:

- **Manual ROI Selection**: Use mouse drag to define specific regions for processing
- **Adaptive ROI**: Automatically adjusts ROI regions based on object detections
- **GPU Memory Pooling**: Pre-allocated GPU memory for different ROI sizes (320x320, 416x416, 512x512, 640x640)
- **ROI Merging**: Automatically merges overlapping ROI regions for efficiency
- **Reduced Processing**: Only processes defined ROI regions instead of full frames

### ROI Configuration

Key ROI parameters in `OptimizedYOLOTracker`:
- `roi_expansion_factor`: 1.2 (how much to expand detected objects for ROI)
- `roi_min_size`: (160, 160) minimum ROI dimensions
- `process_every_n_frames`: Frame skipping for optimization
- ROI confidence threshold: 0.3 (lower than full frame for better detection in small regions)