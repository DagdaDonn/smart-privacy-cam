"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

AppController (app.py)
----------------------
1. Initialize all core modules: VisionSystem, AudioController, PrivacyProcessor, GammaController
2. Initialize MainWindow GUI
3. Start video capture loop in a separate thread/timer
4. For each frame:
    a. Apply gamma correction
    b. Apply privacy/anonymous mode as selected
    c. Draw bounding boxes if developer mode is enabled
    d. Update GUI with processed frame
5. Wire up GUI controls to backend logic:
    a. Mute override toggle → AudioController
    b. Developer mode toggle → VisionSystem
    c. Privacy mode selection → PrivacyProcessor
6. Cleanly shut down on exit
"""
import os

 # Standard library imports first
import sys
import numpy as np
from datetime import datetime

# Import MediaPipe-dependent modules first
from core.vision import VisionSystem

# Then other imports
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from core.audio import AudioController
from core.control import GammaController
from core.privacy import PrivacyProcessor
from data_plots import DataTracker, PlotGenerator
from ui.gui import MainWindow
    
class AppController:
    """
    AppController.__init__
    ----------------------
    1. Initialize all core modules and GUI
    2. Set up video capture and timer for frame updates
    3. Connect GUI controls to backend logic
    """
    def __init__(self):
        # Core modules
        self.vision = VisionSystem()
        self.audio = AudioController()
        self.gamma = GammaController()
        self.privacy = PrivacyProcessor()
        self.data_tracker = DataTracker()  # Initialize data tracking
        self.plot_generator = PlotGenerator()  # Initialize plot generator
        # GUI
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.window.controller = self  # For cleanup on close
        
        # Connect GUI signals to backend logic
        self._connect_signals()
        
        # Video capture - try different camera indices
        self.cap = None
        for camera_index in [0, 1, 2]:  # Try different camera indices
            try:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    print(f'Successfully opened camera at index {camera_index}')
                    break
                else:
                    self.cap.release()
            except Exception as e:
                print(f'Failed to open camera at index {camera_index}: {e}')
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            print('Could not open any webcam. Please check if your camera is available and not in use by another application.')
            sys.exit(1)
        
        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # ~60 FPS for better performance
        
        self.window.show()

    """
    AppController._connect_signals
    -----------------------------
    1. Connect GUI controls to backend logic
    """
    def _connect_signals(self):
        self.window.mute_override_btn.clicked.connect(self._on_mute_override)
        self.window.dev_mode_checkbox.stateChanged.connect(self._on_dev_mode)
        self.window.privacy_mode_group.buttonClicked.connect(self._on_privacy_mode)
        self.window.mustache_checkbox.stateChanged.connect(self._on_mustache)
        self.window.glasses_checkbox.stateChanged.connect(self._on_glasses)
        self.window.hat_checkbox.stateChanged.connect(self._on_hat)
        self.window.fps_combo.currentTextChanged.connect(self._on_fps_change)
        self.window.resolution_combo.currentTextChanged.connect(self._on_resolution_change)
        self.window.gamma_slider.valueChanged.connect(self._on_gamma_change)
        self.window.brightness_slider.valueChanged.connect(self._on_brightness_change)
        self.window.tracking_sensitivity.valueChanged.connect(self._on_tracking_sensitivity)
        self.window.mood_sensitivity.valueChanged.connect(self._on_mood_sensitivity)

    """
    AppController._on_mute_override
    ------------------------------
    1. Set manual override in AudioController based on GUI toggle
    """
    def _on_mute_override(self):
        override = self.window.mute_override_btn.isChecked()
        self.audio.set_manual_override(override)

    """
    AppController._on_dev_mode
    -------------------------
    1. Set developer mode in VisionSystem based on GUI checkbox
    """
    def _on_dev_mode(self, state):
        enabled = state == Qt.CheckState.Checked
        self.vision.set_developer_mode(enabled)

    """
    AppController._on_privacy_mode
    -----------------------------
    1. Set privacy mode in AppController based on GUI radio selection
    """
    def _on_privacy_mode(self):
        mode_id = self.window.privacy_mode_group.checkedId()
        self.privacy_mode = mode_id  # 0: None, 1: Face Blur, 2: Anonymous

    """
    AppController._on_mustache
    -------------------------
    1. Set mustache mode in AppController based on GUI checkbox
    """
    def _on_mustache(self, state):
        self.mustache_enabled = state == Qt.CheckState.Checked

    """
    AppController._on_glasses
    ------------------------
    1. Set glasses mode in AppController based on GUI checkbox
    """
    def _on_glasses(self, state):
        self.glasses_enabled = state == Qt.CheckState.Checked

    """
    AppController._on_hat
    --------------------
    1. Set hat mode in AppController based on GUI checkbox
    """
    def _on_hat(self, state):
        self.hat_enabled = state == Qt.CheckState.Checked

    """
    AppController._on_fps_change
    ---------------------------
    1. Update FPS based on GUI combo box
    """
    def _on_fps_change(self, value):
        # Convert combo box text to FPS value
        if value == "Comic Book Mode":
            # Comic book mode: 1 frame per second (1000ms interval)
            interval = 1000
            self.timer.setInterval(interval)
            print(f"Comic Book Mode enabled (1 FPS, interval: {interval}ms)")
        else:
            try:
                fps = int(value)
                interval = int(1000 / fps)  # Convert FPS to milliseconds
                self.timer.setInterval(interval)
                print(f"FPS set to {fps} (interval: {interval}ms)")
            except ValueError:
                print(f"Invalid FPS value: {value}")

    """
    AppController._on_resolution_change
    ----------------------------------
    1. Update resolution based on GUI combo box
    """
    def _on_resolution_change(self, value):
        # Convert combo box text to resolution value
        try:
            resolution = value.split('x')
            width = int(resolution[0])
            height = int(resolution[1])
            
            # Set camera properties if camera is available
            if self.cap is not None and self.cap.isOpened():
                # Try to set new resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Give camera time to adjust
                import time
                time.sleep(0.1)
                
                # Verify the settings were applied
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Resolution set to {width}x{height} (actual: {actual_width:.0f}x{actual_height:.0f})")
                
                # If the camera fails after resolution change, reinitialize it
                if not self._test_camera_read():
                    print("Camera read failed after resolution change, reinitializing...")
                    self._reinitialize_camera(0)  # Use camera index 0 as fallback
            else:
                print("Camera not available for resolution change")
        except ValueError:
            print(f"Invalid resolution value: {value}")
        except Exception as e:
            print(f"Error changing resolution: {e}")
            # Try to reinitialize camera if there's an error
            self._reinitialize_camera(0)

    """
    AppController._on_tracking_sensitivity
    -------------------------------------
    1. Update tracking sensitivity in VisionSystem
    """
    def _on_tracking_sensitivity(self, value):
        # Convert slider value (1-10) to sensitivity parameter
        sensitivity = value / 10.0
        self.vision.set_tracking_sensitivity(sensitivity)

    """
    AppController._on_mood_sensitivity
    ---------------------------------
    1. Update mood detection sensitivity in VisionSystem
    """
    def _on_mood_sensitivity(self, value):
        # Convert slider value (1-10) to sensitivity parameter
        sensitivity = value / 10.0
        self.vision.set_mood_sensitivity(sensitivity)

    """
    AppController._on_gamma_change
    -----------------------------
    1. Update gamma correction value in GammaController
    """
    def _on_gamma_change(self, value):
        # Convert slider value (50-200) to gamma value (0.5-2.0)
        gamma = value / 100.0
        self.gamma.set_gamma(gamma)
        print(f"Gamma set to {gamma:.2f}")

    """
    AppController._on_brightness_change
    ---------------------------------
    1. Update brightness target in GammaController
    """
    def _on_brightness_change(self, value):
        # Convert slider value (50-200) to brightness target (50-200)
        brightness_target = value
        self.gamma.set_target_brightness(brightness_target)
        print(f"Brightness target set to {brightness_target}")

    """
    AppController._test_camera_read
    ------------------------------
    1. Test if camera can read a frame without errors
    """
    def _test_camera_read(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                return ret and frame is not None
            return False
        except Exception:
            return False

    """
    AppController._reinitialize_camera
    ---------------------------------
    1. Reinitialize camera with given index
    """
    def _reinitialize_camera(self, camera_index):
        try:
            # Release current camera
            if self.cap is not None:
                self.cap.release()
            
            # Try to open camera at specified index
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                print(f'Successfully reinitialized camera at index {camera_index}')
                return True
            else:
                print(f'Failed to reinitialize camera at index {camera_index}')
                return False
        except Exception as e:
            print(f'Error reinitializing camera: {e}')
            return False

    """
    AppController.update_frame
    -------------------------
    1. Capture frame from webcam
    2. Apply gamma correction
    3. Apply privacy/anonymous mode as selected
    4. Draw bounding boxes if developer mode is enabled
    5. Convert frame to QImage and update GUI
    """
    def update_frame(self):
        if self.cap is None:
            return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to read frame from camera")
                return
        except Exception as e:
            print(f"Error reading frame: {e}")
            # Try to reinitialize camera if there's a persistent error
            if not self._reinitialize_camera(0):
                print("Failed to reinitialize camera, stopping updates")
                self.timer.stop()
            return
        
        # Calculate FPS
        current_time = datetime.now()
        if hasattr(self, '_last_frame_time'):
            fps = 1.0 / (current_time - self._last_frame_time).total_seconds()
            self.data_tracker.add_fps_data(fps)
        self._last_frame_time = current_time
        
        # Get face bbox and eye landmarks
        face_bbox, eye_landmarks = self.vision.get_face_and_eyes(frame)
        # Get face landmarks for mustache
        face_landmarks = self.vision.get_face_landmarks(frame)
        # Get body bbox for gamma correction masking
        body_bbox = self.vision.get_body_bbox(frame)
        # Debug: Print body bounding box info
        if body_bbox:
            bx, by, bw, bh = body_bbox
            print(f"Body bbox: x={bx}, y={by}, w={bw}, h={bh}")
        else:
            print("No body bbox detected")
        # Get pose landmarks for rig model
        pose_landmarks = self.vision.get_pose_landmarks(frame)
        # Detect mood
        mood = self.vision.detect_mood(frame)
        
        # Track data for analytics
        self.data_tracker.add_face_detection_data(face_bbox is not None)
        self.data_tracker.add_mood_data(mood, 0.7)  # Default confidence
        
        # Update GUI mood display
        self.window.update_mood(mood)
        
        # Gamma correction: apply only to background (not face or body)
        if face_bbox:
            # Handle new ellipse-based bounding box format (center_x, center_y, width, height)
            center_x, center_y, width, height = face_bbox
            # Convert ellipse to rectangle for masking
            x = center_x - width // 2
            y = center_y - height // 2
            w = width
            h = height
            
            # Expand face bbox upward to include hair/forehead
            expand_up = int(h * 0.4)
            y_exp = max(0, y - expand_up)
            h_exp = h + expand_up
            face_roi = frame[y_exp:y+h, x:x+w].copy()
            brightness = self.vision.analyze_brightness(face_roi)
            gamma_val = self.gamma.update(brightness)
            
            # Track light correction and brightness data
            self.data_tracker.add_light_correction_data(gamma_val)
            self.data_tracker.add_brightness_data(brightness)
            
            # Create combined mask for face and body
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Add expanded face to mask with rounded edges
            # Create elliptical mask for face
            face_center = (x + w//2, y_exp + h_exp//2)
            face_axes = (w//2, h_exp//2)
            cv2.ellipse(mask, face_center, face_axes, 0, 0, 360, (255,), -1)
            
            # Add body to mask if detected with arch shape (semicircle top + rectangle bottom)
            if body_bbox:
                bx, by, bw, bh = body_bbox
                
                # Create arch-shaped mask for body
                # Top half: semicircle
                semicircle_center = (bx + bw//2, by + bh//4)  # Center of top half
                semicircle_radius = min(bw//2, bh//4)  # Radius for semicircle
                
                # Draw semicircle (top half of ellipse)
                cv2.ellipse(mask, semicircle_center, (semicircle_radius, semicircle_radius), 
                           0, 0, 180, (255,), -1)  # 0-180 degrees for top half
                
                # Bottom half: rectangle
                rect_y = by + bh//4  # Start from middle of body
                rect_height = bh * 3//4  # Bottom 3/4 of body
                cv2.rectangle(mask, (bx, rect_y), (bx + bw, by + bh), (255,), -1)
            
            # Apply Gaussian blur to the mask for smoother edges
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Normalize mask to 0-1 range for proper blending
            mask_float = mask.astype(np.float32) / 255.0
            
            # Apply gamma correction to the entire frame
            frame_gamma = self.apply_gamma(frame, gamma_val)
            
            # Create 3D mask for blending
            mask_3d = np.stack([mask_float] * 3, axis=2)
            
            # Apply manual brightness adjustment if set
            brightness_target = self.gamma.pid.setpoint
            if brightness_target != 120:  # If not at default
                # Apply brightness adjustment to background only
                frame_brightness = self.apply_brightness(frame, brightness_target)
                # Blend brightness-adjusted background with original person
                frame = (frame.astype(np.float32) * mask_3d + 
                        frame_brightness.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
            else:
                # Blend original and gamma-corrected frames using the mask
                # This prevents color inversion at edges
                frame = (frame.astype(np.float32) * mask_3d + 
                        frame_gamma.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
        else:
            # Apply gamma correction to entire frame when no face detected
            gamma_val = self.gamma.gamma  # Use current gamma value
            frame = self.apply_gamma(frame, gamma_val)
            
            # Apply manual brightness adjustment if set (to entire frame when no person detected)
            brightness_target = self.gamma.pid.setpoint
            if brightness_target != 120:  # If not at default
                frame = self.apply_brightness(frame, brightness_target)
        
        # Privacy/anonymous mode
        anonymous_active = False
        if hasattr(self, 'privacy_mode'):
            if self.privacy_mode == 1 and face_bbox:  # Face Blur
                frame = self.privacy.blur_faces(frame, [face_bbox])
                self.data_tracker.add_privacy_mode_data("Face Blur")
            elif self.privacy_mode == 2 and eye_landmarks:  # Anonymous
                frame = self.privacy.anonymous_mode(frame, eye_landmarks)
                anonymous_active = True
                self.data_tracker.add_privacy_mode_data("Anonymous")
            else:
                self.data_tracker.add_privacy_mode_data("None")
        
        # Add mustache if enabled
        if hasattr(self, 'mustache_enabled') and self.mustache_enabled and face_landmarks is not None and len(face_landmarks) > 0:
            frame = self.privacy.add_mustache(frame, face_landmarks)
        
        # Add glasses if enabled (after black bar for proper layering)
        if hasattr(self, 'glasses_enabled') and self.glasses_enabled and face_landmarks is not None and len(face_landmarks) > 0:
            frame = self.privacy.add_glasses(frame, face_landmarks, anonymous_active)
        
        # Add hat if enabled
        if hasattr(self, 'hat_enabled') and self.hat_enabled and face_landmarks is not None and len(face_landmarks) > 0:
            frame = self.privacy.add_hat(frame, face_landmarks)
        # Draw bounding boxes and rig model if developer mode
        if self.vision.developer_mode:
            if face_bbox:
                center_x, center_y, width, height = face_bbox
                # Draw ellipse for face bounding box
                cv2.ellipse(frame, (center_x, center_y), (width//2, height//2), 0, 0, 360, (255, 0, 0), 2)
                for eye in eye_landmarks:
                    for (ex, ey) in eye:
                        cv2.circle(frame, (ex, ey), 1, (0, 255, 0), -1)
            if body_bbox:
                x, y, w, h = body_bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Draw rig model
            if pose_landmarks:
                frame = self.vision.draw_rig_model(frame, pose_landmarks)
        # Mic mute indicator and status tracking
        is_muted = self.audio.is_muted()
        self.data_tracker.add_mic_status_data(is_muted)
        self.window.update_mic_status(is_muted)  # Update GUI indicator
        
        if is_muted:
            cv2.putText(frame, 'MIC MUTED', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        # Convert to QImage and update GUI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.window.video_label.setPixmap(QPixmap.fromImage(qt_image))

    """
    AppController.apply_gamma
    ------------------------
    1. Apply gamma correction to the input frame
    """
    def apply_gamma(self, image, gamma):
        inv_gamma = 1.0 / gamma if gamma > 0 else 1.0
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype('uint8')
        return cv2.LUT(image, table)

    """
    AppController.apply_brightness
    ------------------------------
    1. Apply manual brightness adjustment to the input frame
    """
    def apply_brightness(self, image, brightness_target):
        # Ensure brightness_target is within a reasonable range (e.g., 50-200)
        # This is a simplified example; a proper PID controller would be more robust
        # For now, we'll just apply a linear adjustment
        brightness_factor = brightness_target / 120.0 # Assuming 120 is the default
        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        return adjusted_image

    """
    AppController.cleanup
    --------------------
    1. Generate data analysis plots
    2. Release webcam and stop timer
    """
    def cleanup(self):
        # Generate data analysis plots
        try:
            self.plot_generator.generate_all_plots(self.data_tracker)
        except Exception as e:
            print(f"Error generating plots: {e}")
        
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()

    """
    AppController.run
    -----------------
    1. Start the Qt application event loop
    """
    def run(self):
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    controller = AppController()
    controller.run() 