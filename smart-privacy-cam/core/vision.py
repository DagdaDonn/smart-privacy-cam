"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

VisionSystem (vision.py)
------------------------
1. Initialize VisionSystem with developer_mode and camera_index
2. Use MediaPipe for face, eye, and pose detection
3. Provide method to get face bounding box, eye landmarks, and body landmarks
4. Provide method to analyze face brightness
5. Provide method to detect mood from facial expressions
6. Allow developer_mode to be toggled externally (e.g., from GUI)
"""

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.pose import Pose

class VisionSystem:
    """
    VisionSystem.__init__
    ---------------------
    1. Set developer_mode and camera_index attributes
    2. Initialize MediaPipe FaceMesh and Pose
    3. Open webcam stream using camera_index
    4. IF webcam cannot be opened THEN
        - Raise RuntimeError
    """
    def __init__(self, developer_mode=False, camera_index=0):
        self.developer_mode = developer_mode
        self.camera_index = camera_index
        self.face_mesh = FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.pose = Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open webcam')
        
        # Dynamic landmark tracking
        self.landmark_history = []  # Store recent landmarks
        self.max_history = 10  # Number of frames to average
        self.landmark_confidence = {}  # Track confidence for each landmark
        self.smoothing_factor = 0.7  # Exponential smoothing factor
        self.min_confidence_threshold = 0.5  # Minimum confidence for landmark tracking
        self.mood_sensitivity = 0.5  # Mood detection sensitivity

    """
    VisionSystem.set_developer_mode
    ------------------------------
    1. Set developer_mode attribute to given value (for GUI integration)
    """
    def set_developer_mode(self, mode: bool):
        self.developer_mode = mode

    """
    VisionSystem.get_face_and_eyes
    ------------------------------
    1. Process frame with MediaPipe FaceMesh
    2. Return face bounding box and eye landmarks using smoothed data
    3. If no face is detected, return (None, [])
    """
    def get_face_and_eyes(self, frame):
        # Get smoothed landmarks
        landmarks = self.get_face_landmarks(frame)
        if landmarks is None:
            return None, []
        
        # Calculate face bounding box from smoothed landmarks
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # Expand bounding box by 7% on each side
        width = x_max - x_min
        height = y_max - y_min
        expand_x = int(width * 0.07)
        expand_y = int(height * 0.07)
        
        # Ensure expanded coordinates stay within frame bounds
        h, w = frame.shape[:2]
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(w, x_max + expand_x)
        y_max = min(h, y_max + expand_y)
        
        # Create rounded bounding box (ellipse parameters)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Store ellipse parameters for drawing
        face_bbox = (center_x, center_y, width, height)
        
        # Extract eye regions from smoothed landmarks
        left_eye = landmarks[33:134]
        right_eye = landmarks[362:384]
        eye_landmarks = [left_eye, right_eye]
        
        return face_bbox, eye_landmarks

    """
    VisionSystem.get_face_landmarks
    ------------------------------
    1. Process frame with MediaPipe FaceMesh
    2. Update landmark history and return smoothed landmarks
    3. If no face is detected, return None
    """
    def get_face_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        h, w, _ = frame.shape
        multi_face_landmarks = getattr(results, "multi_face_landmarks", None)
        if multi_face_landmarks:
            for face_landmarks in multi_face_landmarks:
                points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
                # Update history with new landmarks
                self.update_landmark_history(points)
                # Return smoothed landmarks
                return self.get_smoothed_landmarks()
        return None

    """
    VisionSystem.get_body_landmarks
    ------------------------------
    1. Process frame with MediaPipe Pose
    2. Return body landmarks for masking
    """
    def get_body_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h, w, _ = frame.shape
        body_landmarks = []
        pose_landmarks = getattr(results, "pose_landmarks", None)
        if pose_landmarks:
            for landmark in pose_landmarks.landmark:
                if landmark.visibility > 0.5:  # Only use visible landmarks
                    body_landmarks.append((int(landmark.x * w), int(landmark.y * h)))
        return body_landmarks

    """
    VisionSystem.get_pose_landmarks
    ------------------------------
    1. Process frame with MediaPipe Pose
    2. Return pose landmarks for rig model visualization
    """
    def get_pose_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h, w, _ = frame.shape
        pose_landmarks = []
        pose_results = getattr(results, "pose_landmarks", None)
        if pose_results:
            for landmark in pose_results.landmark:
                if landmark.visibility > 0.5:  # Only use visible landmarks
                    pose_landmarks.append((int(landmark.x * w), int(landmark.y * h)))
        return pose_landmarks

    """
    VisionSystem.get_body_bbox
    --------------------------
    1. Get body landmarks and create bounding box
    2. Return body bounding box for gamma correction masking
    """
    def get_body_bbox(self, frame):
        body_landmarks = self.get_body_landmarks(frame)
        if body_landmarks:
            points = np.array(body_landmarks)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # If the body extends near the bottom of the frame, extend the bounding box
            # to include the full body area that would be cut off
            bottom_threshold = frame_height * 0.8  # If body is in bottom 20% of frame
            if y_max > bottom_threshold:
                # Extend the bounding box to the bottom of the frame
                y_max = frame_height
            
            # Ensure coordinates are within frame bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(frame_width, int(x_max))
            y_max = min(frame_height, int(y_max))
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        return None

    """
    VisionSystem.detect_mood
    ------------------------
    1. Analyze facial landmarks to detect mood with improved accuracy
    2. Use smoothed landmarks and confidence scores for stability
    3. Return mood string with confidence: 'Happy', 'Sad', 'Neutral', 'Surprised', 'Angry', 'Confused', 'Disgusted'
    """
    def detect_mood(self, frame):
        # Get smoothed landmarks
        landmarks = self.get_face_landmarks(frame)
        if landmarks is None:
            return "No Face Detected"
        
        # Enhanced facial feature analysis using smoothed landmarks
        # Eyebrows (more comprehensive)
        left_eyebrow = [landmarks[70], landmarks[63], landmarks[105], landmarks[66], landmarks[107], landmarks[55], landmarks[65]]
        right_eyebrow = [landmarks[336], landmarks[296], landmarks[334], landmarks[293], landmarks[300], landmarks[285], landmarks[295]]
        
        # Mouth features (more detailed)
        mouth_corners = [landmarks[61], landmarks[291]]  # Left and right corners
        upper_lip = [landmarks[13], landmarks[14], landmarks[15], landmarks[16], landmarks[17]]
        lower_lip = [landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22]]
        
        # Eye features
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        
        # Calculate features
        h, w = frame.shape[:2]
        eyebrow_height = np.mean([p[1] for p in left_eyebrow + right_eyebrow])
        eyebrow_angle = self._calculate_eyebrow_angle(left_eyebrow, right_eyebrow)
        
        mouth_width = np.linalg.norm(np.array(mouth_corners[1]) - np.array(mouth_corners[0]))
        mouth_height = np.linalg.norm(np.array(upper_lip[0]) - np.array(lower_lip[0]))
        mouth_openness = self._calculate_mouth_openness(upper_lip, lower_lip)
        
        eye_distance = np.linalg.norm(np.array(right_eye_corner) - np.array(left_eye_corner))
        
        # Normalize features relative to face size
        face_height = h
        normalized_eyebrow_height = eyebrow_height / face_height
        normalized_mouth_width = mouth_width / eye_distance
        normalized_mouth_openness = mouth_openness / eye_distance
        
        # Enhanced mood detection logic with better thresholds
        mood_scores = {}
        
        # Happy detection - balanced sensitivity
        happy_score = 0
        if normalized_mouth_width > 0.65:  # Wider mouth (increased threshold)
            happy_score += 2
        if normalized_mouth_openness > 0.08:  # Slightly open mouth (increased threshold)
            happy_score += 1
        if normalized_eyebrow_height > 0.32:  # Raised eyebrows (increased threshold)
            happy_score += 1
        mood_scores['Happy'] = happy_score
            
        # Sad detection - balanced sensitivity
        sad_score = 0
        if normalized_mouth_width < 0.42:  # Narrow mouth (adjusted threshold)
            sad_score += 2
        if normalized_eyebrow_height < 0.23:  # Lowered eyebrows (adjusted threshold)
            sad_score += 2
        if eyebrow_angle < -0.08:  # Drooping eyebrows (adjusted threshold)
            sad_score += 1
        mood_scores['Sad'] = sad_score
            
        # Angry detection - balanced sensitivity
        angry_score = 0
        if normalized_eyebrow_height < 0.18:  # Very lowered eyebrows (adjusted threshold)
            angry_score += 2
        if eyebrow_angle < -0.12:  # Furrowed brows (adjusted threshold)
            angry_score += 2
        if normalized_mouth_width < 0.38:  # Tight mouth (adjusted threshold)
            angry_score += 1
        mood_scores['Angry'] = angry_score
            
        # Surprised detection - balanced sensitivity
        surprised_score = 0
        if normalized_eyebrow_height > 0.42:  # Raised eyebrows (adjusted threshold)
            surprised_score += 2
        if normalized_mouth_openness > 0.18:  # Open mouth (adjusted threshold)
            surprised_score += 2
        mood_scores['Surprised'] = surprised_score
            
        # Confused detection - balanced sensitivity
        confused_score = 0
        if abs(eyebrow_angle) > 0.12:  # Asymmetric eyebrows (adjusted threshold)
            confused_score += 2
        if normalized_mouth_width < 0.48 and normalized_mouth_openness < 0.04:
            confused_score += 1
        mood_scores['Confused'] = confused_score
            
        # Disgusted detection - balanced sensitivity
        disgusted_score = 0
        if normalized_mouth_openness < 0.025 and normalized_mouth_width < 0.42:
            disgusted_score += 2
        if normalized_eyebrow_height < 0.23:
            disgusted_score += 1
        mood_scores['Disgusted'] = disgusted_score
            
        # Neutral (baseline) - balanced priority
        neutral_score = 1.5  # Balanced base neutral score
        if all(score < 2 for mood, score in mood_scores.items()):
            neutral_score += 1  # Bonus if no strong indicators
        mood_scores['Neutral'] = neutral_score
        
        # Find the mood with highest score
        detected_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
        confidence = mood_scores[detected_mood]
        
        # Return mood with confidence indicator
        if confidence >= 2.5:
            return detected_mood
        elif confidence >= 2:
            return f"{detected_mood}?"
        else:
            return "Neutral"

    def _calculate_eyebrow_angle(self, left_eyebrow, right_eyebrow):
        """Calculate eyebrow angle (negative = furrowed, positive = raised)"""
        left_center = np.mean(left_eyebrow, axis=0)
        right_center = np.mean(right_eyebrow, axis=0)
        return (left_center[1] + right_center[1]) / 2 - 0.5  # Normalized

    def _calculate_mouth_openness(self, upper_lip, lower_lip):
        """Calculate how open the mouth is"""
        upper_center = np.mean(upper_lip, axis=0)
        lower_center = np.mean(lower_lip, axis=0)
        return np.linalg.norm(upper_center - lower_center)

    def _calculate_eye_openness(self, left_eye, right_eye):
        """Calculate how open the eyes are"""
        # Simplified - could be enhanced with more eye landmarks
        return 0.5  # Placeholder

    """
    VisionSystem.analyze_brightness
    ------------------------------
    1. Convert face region of interest (ROI) to grayscale
    2. Compute mean pixel value (brightness) of grayscale ROI
    3. Return mean brightness
    """
    def analyze_brightness(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    """
    VisionSystem.draw_rig_model
    ---------------------------
    1. Draw stick figure overlay using pose landmarks
    2. Connect key body parts with lines
    3. Return frame with rig model drawn
    """
    def draw_rig_model(self, frame, pose_landmarks):
        if len(pose_landmarks) < 11:  # Need minimum landmarks for basic rig
            return frame
        
        # MediaPipe Pose landmark indices for key body parts
        # Head, shoulders, elbows, wrists, hips, knees, ankles
        connections = [
            # Head to shoulders
            (0, 11), (0, 12),  # Nose to shoulders
            # Shoulders
            (11, 12),  # Left to right shoulder
            # Arms
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            # Torso
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hips
            # Legs
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
        
        # Draw connections
        for connection in connections:
            if (connection[0] < len(pose_landmarks) and 
                connection[1] < len(pose_landmarks)):
                pt1 = pose_landmarks[connection[0]]
                pt2 = pose_landmarks[connection[1]]
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw key points
        for i, landmark in enumerate(pose_landmarks):
            if i < 33:  # Only draw main pose landmarks
                cv2.circle(frame, landmark, 3, (255, 0, 0), -1)
        
        return frame

    """
    VisionSystem.update_landmark_history
    ----------------------------------
    1. Add new landmarks to history
    2. Maintain rolling average of recent landmarks
    3. Calculate confidence scores for each landmark
    """
    def update_landmark_history(self, landmarks):
        if landmarks is None:
            return
        
        # Add to history
        self.landmark_history.append(landmarks)
        if len(self.landmark_history) > self.max_history:
            self.landmark_history.pop(0)
        
        # Update confidence scores
        if len(self.landmark_history) >= 3:
            self._update_landmark_confidence()

    """
    VisionSystem._update_landmark_confidence
    ---------------------------------------
    1. Calculate confidence for each landmark based on stability
    2. Higher confidence for landmarks that don't change much
    """
    def _update_landmark_confidence(self):
        if len(self.landmark_history) < 3:
            return
        
        # Calculate variance for each landmark across recent frames
        for i in range(len(self.landmark_history[0])):
            x_coords = [landmarks[i][0] for landmarks in self.landmark_history]
            y_coords = [landmarks[i][1] for landmarks in self.landmark_history]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            
            # Lower variance = higher confidence
            confidence = 1.0 / (1.0 + x_variance + y_variance)
            self.landmark_confidence[i] = confidence

    """
    VisionSystem.get_smoothed_landmarks
    ----------------------------------
    1. Return smoothed landmarks using exponential moving average
    2. Use confidence scores to weight the smoothing
    """
    def get_smoothed_landmarks(self):
        if not self.landmark_history:
            return None
        
        if len(self.landmark_history) == 1:
            return self.landmark_history[0]
        
        # Use exponential moving average with confidence weighting
        smoothed = np.array(self.landmark_history[-1], dtype=np.float32)
        
        for i in range(len(self.landmark_history) - 2, -1, -1):
            current = np.array(self.landmark_history[i], dtype=np.float32)
            alpha = self.smoothing_factor * (1.0 - i / len(self.landmark_history))
            smoothed = alpha * current + (1 - alpha) * smoothed
        
        return smoothed.astype(np.int32)

    """
    VisionSystem.set_tracking_sensitivity
    -----------------------------------
    1. Set tracking sensitivity for landmark detection
    """
    def set_tracking_sensitivity(self, sensitivity: float):
        # Adjust smoothing factor based on sensitivity (0.0 to 1.0)
        self.smoothing_factor = max(0.1, min(0.9, sensitivity))
        # Adjust confidence thresholds
        self.min_confidence_threshold = 0.5 - (sensitivity * 0.3)

    """
    VisionSystem.set_mood_sensitivity
    --------------------------------
    1. Set mood detection sensitivity
    """
    def set_mood_sensitivity(self, sensitivity: float):
        # Store mood sensitivity for use in mood detection
        self.mood_sensitivity = max(0.1, min(1.0, sensitivity))

    """
    VisionSystem.get_landmark_confidence
    -----------------------------------
    1. Return confidence score for a specific landmark
    2. Higher confidence means more stable landmark
    """
    def get_landmark_confidence(self, landmark_index):
        return self.landmark_confidence.get(landmark_index, 0.5)

if __name__ == '__main__':
     from mediapipe.python.solutions.face_mesh import FaceMesh
     print('VisionSystem and FaceMesh import successful!')