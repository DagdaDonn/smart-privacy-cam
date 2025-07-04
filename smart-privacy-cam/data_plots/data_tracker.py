"""
Data Tracker Module
------------------
Collects and stores various metrics during the Smart Privacy Cam session
"""

from datetime import datetime
from collections import defaultdict, deque
import json
import os

class DataTracker:
    """
    DataTracker.__init__
    ------------------
    1. Initialize data storage for various metrics
    2. Set up time tracking and data structures
    3. Clear any existing data files on startup
    """
    def __init__(self):
        # Clear old data files on startup
        self._clear_old_data()
        
        # Data storage with timestamps
        self.mood_data = deque(maxlen=1000)  # Store last 1000 mood readings
        self.light_correction_data = deque(maxlen=1000)
        self.face_detection_data = deque(maxlen=1000)
        self.mic_status_data = deque(maxlen=1000)
        self.privacy_mode_data = deque(maxlen=1000)
        self.fps_data = deque(maxlen=1000)
        self.brightness_data = deque(maxlen=1000)
        
        # Statistics tracking
        self.session_start = datetime.now()
        self.total_frames = 0
        self.faces_detected = 0
        self.mood_counts = defaultdict(int)
        self.privacy_mode_usage = defaultdict(int)
        self.mic_mute_count = 0
        self.mic_unmute_count = 0

    """
    DataTracker._clear_old_data
    -------------------------
    1. Remove any existing data files and graphs
    """
    def _clear_old_data(self):
        files_to_remove = [
            'data_plots/mood_analysis.png',
            'data_plots/light_correction.png', 
            'data_plots/face_detection.png',
            'data_plots/system_metrics.png',
            'data_plots/performance_analysis.png',
            'data_plots/session_data.json'
        ]
        
        for file in files_to_remove:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")

    """
    DataTracker.add_mood_data
    -----------------------
    1. Add mood reading with timestamp and confidence
    """
    def add_mood_data(self, mood: str, confidence: float = 0.5):
        timestamp = datetime.now()
        self.mood_data.append({
            'timestamp': timestamp,
            'mood': mood,
            'confidence': confidence
        })
        self.mood_counts[mood] += 1

    """
    DataTracker.add_light_correction_data
    -----------------------------------
    1. Add gamma correction value with timestamp
    """
    def add_light_correction_data(self, gamma_value: float):
        timestamp = datetime.now()
        self.light_correction_data.append({
            'timestamp': timestamp,
            'gamma': gamma_value
        })

    """
    DataTracker.add_face_detection_data
    ---------------------------------
    1. Track face detection status and frequency
    """
    def add_face_detection_data(self, face_detected: bool):
        timestamp = datetime.now()
        self.face_detection_data.append({
            'timestamp': timestamp,
            'detected': face_detected
        })
        self.total_frames += 1
        if face_detected:
            self.faces_detected += 1

    """
    DataTracker.add_mic_status_data
    -----------------------------
    1. Track microphone mute/unmute status
    """
    def add_mic_status_data(self, is_muted: bool):
        timestamp = datetime.now()
        self.mic_status_data.append({
            'timestamp': timestamp,
            'muted': is_muted
        })
        
        # Track mute/unmute counts
        if is_muted:
            self.mic_mute_count += 1
        else:
            self.mic_unmute_count += 1

    """
    DataTracker.add_privacy_mode_data
    -------------------------------
    1. Track privacy mode usage
    """
    def add_privacy_mode_data(self, mode: str):
        timestamp = datetime.now()
        self.privacy_mode_data.append({
            'timestamp': timestamp,
            'mode': mode
        })
        self.privacy_mode_usage[mode] += 1

    """
    DataTracker.add_fps_data
    ----------------------
    1. Track FPS performance
    """
    def add_fps_data(self, fps: float):
        timestamp = datetime.now()
        self.fps_data.append({
            'timestamp': timestamp,
            'fps': fps
        })

    """
    DataTracker.add_brightness_data
    -----------------------------
    1. Track face brightness levels
    """
    def add_brightness_data(self, brightness: float):
        timestamp = datetime.now()
        self.brightness_data.append({
            'timestamp': timestamp,
            'brightness': brightness
        })

    """
    DataTracker.get_session_stats
    ----------------------------
    1. Get comprehensive session statistics
    """
    def get_session_stats(self):
        session_duration = datetime.now() - self.session_start
        detection_rate = (self.faces_detected / self.total_frames * 100) if self.total_frames > 0 else 0
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_seconds': session_duration.total_seconds(),
            'total_frames': self.total_frames,
            'faces_detected': self.faces_detected,
            'detection_rate': detection_rate,
            'mood_counts': dict(self.mood_counts),
            'privacy_mode_usage': dict(self.privacy_mode_usage),
            'mic_mute_count': self.mic_mute_count,
            'mic_unmute_count': self.mic_unmute_count,
            'average_fps': len(self.fps_data) / session_duration.total_seconds() if session_duration.total_seconds() > 0 else 0
        }

    """
    DataTracker.save_session_data
    ----------------------------
    1. Save all session data to JSON file
    """
    def save_session_data(self):
        session_data = self.get_session_stats()
        
        # Ensure data_plots directory exists
        os.makedirs('data_plots', exist_ok=True)
        
        with open('data_plots/session_data.json', 'w') as f:
            json.dump(session_data, f, indent=2) 