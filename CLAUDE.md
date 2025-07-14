# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time YOLOv8 object detection and tracking system that processes camera input with GPU acceleration. The application uses multi-threading for optimized performance and provides object trajectory visualization.

## Key Architecture

### Core Components

- **OptimizedYOLOTracker**: Main class that handles YOLO model loading, multi-threaded frame processing, and object tracking
- **Multi-threading Design**: Uses separate threads for frame capture (`_capture_worker`) and YOLO processing (`_process_worker`) with queue-based communication
- **GPU Optimization**: Supports CUDA acceleration with FP16 precision and memory management

### Key Files

- `vid.py`: Main application file containing the complete tracking system
- `yolov8s.pt`: Pre-trained YOLOv8 model file

## Dependencies

The project uses these key libraries:
- `ultralytics` (YOLO implementation)
- `opencv-python` (cv2 for video processing)
- `torch` (PyTorch for GPU acceleration)
- `numpy` (numerical operations)

## Common Commands

### Running the Application
```bash
python vid.py
```

### Runtime Controls
- Press 'q' to quit
- Press 'r' to reset tracking
- Press 't' to toggle trajectory visualization

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

### Tracking System
- Maintains trajectory history for each tracked object (max 30 points)
- Tracks unique objects across frames
- Provides per-class object counting
- Color-coded trajectory visualization

## Configuration

Camera settings are configured in `main()`:
- Resolution: 1280x720
- FPS: 30
- Buffer size: 1 (reduced latency)

Model confidence and tracking parameters can be adjusted in the `_process_worker` method.