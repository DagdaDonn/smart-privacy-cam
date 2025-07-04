# Smart Privacy Cam

**Author:** Ethan O'Brien  
**Date:** 4th July 2025  
**License:** Open license, free to be redistributed

## Overview
Smart Privacy Cam is a privacy-focused, real-time vision system for Windows. It uses OpenCV and a webcam to detect faces and eyes, applies privacy and gamma correction features, and provides a modern PyQt5 GUI for control and feedback.

## Features

### Core Functionality
- **Real-time face and eye detection** using MediaPipe
- **Gamma correction** via PID controller based on face brightness
- **Automatic microphone mute/unmute** (with manual override)
- **Privacy modes**: face blur and anonymous mode (black bar over eyes)
- **Developer mode**: show bounding boxes for faces/eyes and rig model
- **Live video feedback** in modern PyQt5 GUI
- **Modular, extensible architecture**

### Fun Features
- **Mustache overlay**: Add fun mustache to detected faces
- **Glasses overlay**: Add stylish glasses frames
- **Hat overlay**: Add a top hat above the head
- **Mood detection**: Real-time facial expression analysis
- **Body tracking**: Full pose detection with rig model visualization

### GUI Controls
- **FPS Control**: 15, 30, or 60 FPS options
- **Resolution Control**: 640x480, 1280x720, or 1920x1080
- **Tracking Sensitivity**: Adjust landmark detection sensitivity (1-10)
- **Mood Sensitivity**: Fine-tune mood detection (1-10)
- **Privacy Mode Selection**: None, Face Blur, or Anonymous mode
- **Feature Toggles**: Mustache, Glasses, Hat, Developer Mode
- **Mute Override**: Manual microphone control
- **Real-time Status Indicators**: Mic status, system performance

### Data Analytics & Visualization
The application includes comprehensive data tracking and visualization:

#### Data Tracking
- **Mood Analysis**: Tracks detected moods with confidence levels
- **Light Correction**: Monitors gamma correction values over time
- **Face Detection**: Records face detection frequency and patterns
- **Microphone Status**: Tracks mute/unmute events
- **Privacy Mode Usage**: Monitors which privacy features are used
- **Performance Metrics**: FPS tracking and system performance
- **Brightness Levels**: Face brightness analysis

#### Generated Plots (saved in `data_plots/plots/`)
1. **Mood Analysis Dashboard** (`mood_analysis.png`)
   - Mood timeline and distribution
   - Confidence over time
   - Mood frequency analysis

2. **Light Correction Analysis** (`light_correction.png`)
   - Gamma correction over time
   - Gamma vs brightness correlation
   - Correction statistics

3. **Face Detection Analysis** (`face_detection.png`)
   - Face detection timeline
   - Detection rate analysis
   - Pattern recognition

4. **System Metrics Dashboard** (`system_metrics.png`)
   - Session information
   - Privacy mode usage
   - Performance metrics

5. **Performance Analysis** (`performance_analysis.png`)
   - FPS over time
   - Performance trends
   - System optimization insights

## Installation
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd smart-privacy-cam
   ```
2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```sh
python app.py
```

### GUI Navigation
- **Camera Tab**: FPS, resolution, and tracking controls
- **Privacy Tab**: Privacy mode selection and feature toggles
- **Settings Tab**: Sensitivity adjustments and system controls
- **Status Panel**: Real-time system status and indicators

### Requirements
- **OS**: Windows 10/11
- **Python**: 3.7-3.11 (64-bit required for MediaPipe)
- **Webcam**: USB webcam for video input
- **Dependencies**: See `requirements.txt`

### Key Dependencies
- `opencv-python` - Computer vision and video processing
- `mediapipe` - Face, pose, and landmark detection
- `PyQt5` - Modern GUI framework
- `pyautogui` - System automation (mic control)
- `matplotlib` - Data visualization
- `seaborn` - Enhanced plotting (optional)
- `numpy` - Numerical computing

## Project Structure
```
smart-privacy-cam/
├── app.py                 # Main application entry point
├── core/
│   ├── vision.py         # MediaPipe-based vision system
│   ├── control.py        # PID controller for gamma correction
│   ├── audio.py          # Microphone control system
│   └── privacy.py        # Privacy features and overlays
├── ui/
│   └── gui.py           # Modern PyQt5 GUI
├── data_plots/
│   ├── data_tracker.py  # Data collection and storage
│   ├── plot_generator.py # Visualization generation
│   ├── plots/           # Generated plot files
│   └── session_data.json # Session data storage
├── utils/               # Utility functions
└── tests/              # Test files
```

## Data Storage
Session data is automatically saved as JSON in `data_plots/session_data.json` for further analysis or integration with other tools. All plots are generated when the application closes and saved as high-resolution PNG files.

## License
Open license, free to be redistributed.

---
*Made with care for privacy, usability, and comprehensive analytics.* 
