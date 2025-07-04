"""
Author: Ethan O'Brien
Date: 4th July 2025
License: Open license, free to be redistributed

PrivacyProcessor (privacy.py)
----------------------------
1. Initialize PrivacyProcessor
2. Provide method to blur faces in an image given face coordinates
3. Provide method to overlay a precise black bar over eye regions using eye landmarks
4. (Optionally) Provide toggles for privacy and anonymous modes
"""

import cv2
import numpy as np

class PrivacyProcessor:
    """
    PrivacyProcessor.__init__
    ------------------------
    1. Initialize any required attributes (none for now)
    """
    def __init__(self):
        pass

    """
    PrivacyProcessor.blur_faces
    --------------------------
    1. For each face coordinate in faces:
        a. Extract face region from image
        b. Apply Gaussian blur to face region
        c. Replace original face region with blurred version
    2. Return processed image
    """
    def blur_faces(self, image, faces):
        for face_bbox in faces:
            # Handle new ellipse-based bounding box format (center_x, center_y, width, height)
            if len(face_bbox) == 4:
                center_x, center_y, width, height = face_bbox
                # Convert ellipse to rectangle for blurring
                x = center_x - width // 2
                y = center_y - height // 2
                w = width
                h = height
            else:
                # Fallback to old format (x, y, w, h)
                x, y, w, h = face_bbox
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                face_roi = image[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                image[y:y+h, x:x+w] = blurred_face
        return image

    """
    PrivacyProcessor.anonymous_mode
    ------------------------------
    1. If both left and right eye landmarks are present:
        a. Create precise eye region masks for each eye
        b. Draw filled black rectangles over only the eye regions
    2. Return processed image
    """
    def anonymous_mode(self, image, eye_landmarks):
        try:
            # Draw one single black bar covering both eyes
            BAR_HEIGHT = 50  # pixels (increased from 14 by factor of 3)
            
            if eye_landmarks is None or len(eye_landmarks) < 2:
                return image
                
            # Ensure we have valid eye landmarks
            valid_eyes = []
            for eye in eye_landmarks:
                if eye is not None and len(eye) >= 2:
                    valid_eyes.append(eye)
            
            if len(valid_eyes) < 2:
                return image
            
            # Use specific eye landmarks for better positioning
            # Focus on the upper and lower eyelid landmarks for accurate positioning
            left_eye = valid_eyes[0]  # Left eye landmarks
            right_eye = valid_eyes[1]  # Right eye landmarks
            
            # Get the top and bottom of each eye (eyelid positions)
            left_eye_top = min(point[1] for point in left_eye if len(point) >= 2)
            left_eye_bottom = max(point[1] for point in left_eye if len(point) >= 2)
            right_eye_top = min(point[1] for point in right_eye if len(point) >= 2)
            right_eye_bottom = max(point[1] for point in right_eye if len(point) >= 2)
            
            # Get the leftmost and rightmost points for each eye separately
            left_eye_left = min(point[0] for point in left_eye if len(point) >= 2)
            left_eye_right = max(point[0] for point in left_eye if len(point) >= 2)
            right_eye_left = min(point[0] for point in right_eye if len(point) >= 2)
            right_eye_right = max(point[0] for point in right_eye if len(point) >= 2)
            
            # Calculate the overall eye region boundaries
            eye_top = min(left_eye_top, right_eye_top)
            eye_bottom = max(left_eye_bottom, right_eye_bottom)
            eye_left = min(left_eye_left, right_eye_left)
            eye_right = max(left_eye_right, right_eye_right)
            
            # Position the black bar to cover the UPPER half of the eye region
            # Start the bar at the top of the eyes and extend downward
            bar_y1 = max(0, eye_top + 40)  # Start further below the top of the eyes
            bar_y2 = min(image.shape[0], bar_y1 + BAR_HEIGHT)  # Extend downward by BAR_HEIGHT
            
            # Extend the bar horizontally to cover both eyes with some margin
            bar_x1 = max(0, eye_left - 10)  # Extend slightly beyond left eye
            bar_x2 = min(image.shape[1], eye_right + 10)  # Extend slightly beyond right eye
            
            # Draw the black bar
            cv2.rectangle(image, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 0), -1)
            
        except Exception as e:
            print(f"Error in anonymous mode: {e}")
            
        return image

    """
    PrivacyProcessor.add_mustache
    ---------------------------
    1. Add a fun mustache below the nose using face landmarks
    2. Draw a simple black mustache shape
    """
    def add_mustache(self, image, face_landmarks):
        try:
            if face_landmarks is None or len(face_landmarks) < 468:  # Need full face mesh
                return image
            
            # Get nose and mouth landmarks for mustache positioning
            # Use more robust landmark indices
            nose_tip = face_landmarks[4]  # Nose tip
            mouth_left = face_landmarks[61]  # Left mouth corner
            mouth_right = face_landmarks[291]  # Right mouth corner
            
            # Ensure landmarks are valid
            if not all(isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2 
                      for landmark in [nose_tip, mouth_left, mouth_right]):
                return image
            
            # Calculate mustache position and size
            mustache_y = int(nose_tip[1] + 15)  # Below nose
            mustache_width = int(np.linalg.norm(np.array(mouth_right) - np.array(mouth_left)) * 0.8)
            mustache_height = 8
            
            # Center the mustache horizontally
            mustache_x = int(nose_tip[0] - mustache_width // 2)
            
            # Ensure coordinates are within image bounds
            mustache_x = max(0, min(mustache_x, image.shape[1] - mustache_width))
            mustache_y = max(0, min(mustache_y, image.shape[0] - mustache_height))
            
            # Draw a simple curved mustache
            # Main bar
            cv2.rectangle(image, 
                         (mustache_x, mustache_y), 
                         (mustache_x + mustache_width, mustache_y + mustache_height), 
                         (0, 0, 0), -1)
            
            # Add curved ends (simple circles)
            left_end_x = max(0, mustache_x - 5)
            right_end_x = min(image.shape[1] - 1, mustache_x + mustache_width + 5)
            end_y = mustache_y + mustache_height // 2
            
            cv2.circle(image, (left_end_x, end_y), 6, (0, 0, 0), -1)
            cv2.circle(image, (right_end_x, end_y), 6, (0, 0, 0), -1)
            
        except Exception as e:
            print(f"Error adding mustache: {e}")
            return image
        
        return image

    """
    PrivacyProcessor.add_glasses
    --------------------------
    1. Add fun glasses using face landmarks
    2. Draw simple rectangular glasses frames
    """
    def add_glasses(self, image, face_landmarks, anonymous_mode_active=False):
        try:
            if face_landmarks is None or len(face_landmarks) < 468:  # Need full face mesh
                return image
            
            # Get eye landmarks for glasses positioning
            left_eye_center = face_landmarks[33]  # Left eye center
            right_eye_center = face_landmarks[263]  # Right eye center
            
            # Ensure landmarks are valid
            if not all(isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2 
                      for landmark in [left_eye_center, right_eye_center]):
                return image
            
            # Calculate glasses position and size
            eye_distance = np.linalg.norm(np.array(right_eye_center) - np.array(left_eye_center))
            lens_radius = int(eye_distance * 0.15)
            
            # Choose color based on anonymous mode
            if anonymous_mode_active:
                glasses_color = (255, 255, 255)  # White for anonymous mode
            else:
                glasses_color = (0, 0, 0)  # Black for normal mode
            
            # Draw left lens
            left_x, left_y = int(left_eye_center[0]), int(left_eye_center[1])
            cv2.circle(image, (left_x, left_y), lens_radius, glasses_color, 3)
            
            # Draw right lens
            right_x, right_y = int(right_eye_center[0]), int(right_eye_center[1])
            cv2.circle(image, (right_x, right_y), lens_radius, glasses_color, 3)
            
            # Draw bridge connecting the lenses
            bridge_start = (left_x + lens_radius, left_y)
            bridge_end = (right_x - lens_radius, right_y)
            cv2.line(image, bridge_start, bridge_end, glasses_color, 3)
            
            # Draw temple arms
            temple_length = int(lens_radius * 1.5)
            # Left temple
            cv2.line(image, (left_x - lens_radius, left_y), 
                    (left_x - lens_radius - temple_length, left_y - temple_length//2), glasses_color, 3)
            # Right temple
            cv2.line(image, (right_x + lens_radius, right_y), 
                    (right_x + lens_radius + temple_length, right_y - temple_length//2), glasses_color, 3)
            
        except Exception as e:
            print(f"Error adding glasses: {e}")
            return image
        
        return image

    """
    PrivacyProcessor.add_hat
    -----------------------
    1. Add a fun hat using face landmarks
    2. Draw a simple top hat above the head
    """
    def add_hat(self, image, face_landmarks):
        try:
            if face_landmarks is None or len(face_landmarks) < 468:  # Need full face mesh
                return image
            
            # Get forehead and head landmarks for hat positioning
            forehead_center = face_landmarks[10]  # Forehead center
            head_top = face_landmarks[10]  # Top of head
            
            # Ensure landmarks are valid
            if not all(isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2 
                      for landmark in [forehead_center, head_top]):
                return image
            
            # Calculate hat position and size
            hat_width = int(np.linalg.norm(np.array(face_landmarks[123]) - np.array(face_landmarks[352])) * 1.2)
            hat_height = int(hat_width * 0.8)
            
            # Position hat above the head
            hat_x = int(forehead_center[0] - hat_width // 2)
            hat_y = int(head_top[1] - hat_height - 20)  # 20 pixels above head
            
            # Ensure coordinates are within image bounds
            hat_x = max(0, min(hat_x, image.shape[1] - hat_width))
            hat_y = max(0, min(hat_y, image.shape[0] - hat_height))
            
            # Draw hat brim (bottom part)
            brim_height = int(hat_height * 0.2)
            cv2.rectangle(image, (hat_x, hat_y + hat_height - brim_height), 
                         (hat_x + hat_width, hat_y + hat_height), (0, 0, 0), -1)
            
            # Draw hat crown (top part)
            crown_width = int(hat_width * 0.8)
            crown_x = hat_x + (hat_width - crown_width) // 2
            cv2.rectangle(image, (crown_x, hat_y), 
                         (crown_x + crown_width, hat_y + hat_height - brim_height), (0, 0, 0), -1)
            
            # Add hat band
            band_y = hat_y + hat_height - brim_height - 5
            cv2.rectangle(image, (crown_x, band_y), 
                         (crown_x + crown_width, band_y + 10), (139, 69, 19), -1)  # Brown band
            
        except Exception as e:
            print(f"Error adding hat: {e}")
            return image
        
        return image 