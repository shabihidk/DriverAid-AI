"""
DriverAid - Vision Pipeline
Phase 3: MediaPipe Face Mesh Integration

Handles:
- Face detection and landmark extraction
- Eye Aspect Ratio (EAR) calculation
- Head pose estimation (pitch, yaw, roll)
- Eye region extraction for CNN inference
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, List


class VisionPipeline:
    """
    Real-time face analysis using MediaPipe Face Mesh.
    Optimized for performance on laptop CPUs.
    """
    
    # MediaPipe Face Mesh landmark indices
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    
    # Left eye landmarks (6 points for EAR calculation)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Right eye landmarks (6 points for EAR calculation)
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Face landmarks for head pose estimation
    FACE_LANDMARKS_FOR_POSE = [1, 33, 263, 61, 291, 199]  # Nose, eyes, mouth corners
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Only track driver's face
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # For visualization (optional)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera calibration (approximate values for standard webcam)
        self.focal_length = 1.0
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame and extract all vision features.
        
        Args:
            frame: BGR image from webcam (H, W, 3)
            
        Returns:
            Dictionary containing:
                - face_detected: bool
                - landmarks: List of (x, y, z) tuples (if detected)
                - ear_left: float (Eye Aspect Ratio - left eye)
                - ear_right: float (Eye Aspect Ratio - right eye)
                - ear_avg: float (Average EAR)
                - head_pose: Dict with pitch, yaw, roll angles
                - left_eye_region: np.ndarray (32x32 grayscale)
                - right_eye_region: np.ndarray (32x32 grayscale)
            or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None  # No face detected
        
        # Get first face (we only track one face)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        h, w, _ = frame.shape
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z  # Relative depth
            landmarks.append((x, y, z))
        
        # Calculate EAR for both eyes
        ear_left = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        ear_right = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        ear_avg = (ear_left + ear_right) / 2.0
        
        # Calculate head pose
        head_pose = self._calculate_head_pose(landmarks, frame.shape)
        
        # Extract eye regions for CNN
        left_eye_region = self._extract_eye_region(frame, landmarks, self.LEFT_EYE_INDICES)
        right_eye_region = self._extract_eye_region(frame, landmarks, self.RIGHT_EYE_INDICES)
        
        return {
            'face_detected': True,
            'landmarks': landmarks,
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear_avg': ear_avg,
            'head_pose': head_pose,
            'left_eye_region': left_eye_region,
            'right_eye_region': right_eye_region
        }
    
    def _calculate_ear(self, landmarks: List[Tuple], eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        
        Where p1-p6 are eye landmarks arranged as:
        p1, p4: horizontal corners
        p2, p3, p5, p6: vertical landmarks
        
        EAR decreases when eye closes.
        Typical values: Open eye ~0.25-0.30, Closed eye ~0.10-0.15
        
        Args:
            landmarks: List of all face landmarks
            eye_indices: Indices for specific eye (6 points)
            
        Returns:
            EAR value (float)
        """
        # Extract eye landmark coordinates
        eye_points = [landmarks[i] for i in eye_indices]
        
        # Calculate vertical distances
        vertical1 = self._euclidean_distance(eye_points[1], eye_points[5])
        vertical2 = self._euclidean_distance(eye_points[2], eye_points[4])
        
        # Calculate horizontal distance
        horizontal = self._euclidean_distance(eye_points[0], eye_points[3])
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def _euclidean_distance(self, p1: Tuple, p2: Tuple) -> float:
        """Calculate 2D Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_head_pose(self, landmarks: List[Tuple], frame_shape: Tuple) -> Dict:
        """
        Calculate head pose angles (pitch, yaw, roll) using solvePnP.
        
        Args:
            landmarks: List of facial landmarks
            frame_shape: (height, width, channels)
            
        Returns:
            Dictionary with pitch, yaw, roll in degrees
        """
        h, w, _ = frame_shape
        
        # 3D model points (generic face model in cm)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (-30.0, -30.0, -30.0),     # Left eye corner
            (30.0, -30.0, -30.0),      # Right eye corner
            (-20.0, 30.0, -20.0),      # Left mouth corner
            (20.0, 30.0, -20.0),       # Right mouth corner
            (0.0, 0.0, -50.0)          # Chin
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[1][:2],    # Nose tip
            landmarks[33][:2],   # Left eye corner
            landmarks[263][:2],  # Right eye corner
            landmarks[61][:2],   # Left mouth corner
            landmarks[291][:2],  # Right mouth corner
            landmarks[199][:2]   # Chin
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        return {
            'pitch': pitch,  # Up/down tilt
            'yaw': yaw,      # Left/right turn
            'roll': roll     # Head roll (tilt to side)
        }
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (in degrees)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        return np.degrees(x), np.degrees(y), np.degrees(z)
    
    def _extract_eye_region(self, frame: np.ndarray, landmarks: List[Tuple], 
                           eye_indices: List[int]) -> Optional[np.ndarray]:
        """
        Extract and preprocess eye region for CNN inference.
        
        Args:
            frame: Original BGR frame
            landmarks: All face landmarks
            eye_indices: Indices for specific eye
            
        Returns:
            32x32 grayscale eye region (ready for CNN), or None if extraction fails
        """
        try:
            # Get eye landmark coordinates
            eye_points = np.array([landmarks[i][:2] for i in eye_indices], dtype=np.int32)
            
            # Calculate bounding box with padding
            x_min = max(0, eye_points[:, 0].min() - 10)
            x_max = min(frame.shape[1], eye_points[:, 0].max() + 10)
            y_min = max(0, eye_points[:, 1].min() - 10)
            y_max = min(frame.shape[0], eye_points[:, 1].max() + 10)
            
            # Extract region
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                return None
            
            # Convert to grayscale
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to 32x32 (CNN input size)
            eye_resized = cv2.resize(eye_gray, (32, 32), interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            eye_normalized = eye_resized.astype('float32') / 255.0
            
            return eye_normalized
            
        except Exception as e:
            print(f"⚠️ Eye extraction failed: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, vision_data: Dict) -> np.ndarray:
        """
        Draw visualization on frame (for debugging/demo).
        
        Args:
            frame: Original BGR frame
            vision_data: Output from process_frame()
            
        Returns:
            Frame with visualizations drawn
        """
        if not vision_data or not vision_data['face_detected']:
            # No face detected - draw warning
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        landmarks = vision_data['landmarks']
        
        # Draw eye landmarks
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            x, y, _ = landmarks[idx]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw EAR values
        ear_text = f"EAR: {vision_data['ear_avg']:.3f}"
        cv2.putText(frame, ear_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw head pose
        pose = vision_data['head_pose']
        pose_text = f"Pitch: {pose['pitch']:.1f} Yaw: {pose['yaw']:.1f}"
        cv2.putText(frame, pose_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def cleanup(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


# Utility function for standalone testing
def test_vision_pipeline():
    """Test the vision pipeline with webcam."""
    cap = cv2.VideoCapture(0)
    vision = VisionPipeline()
    
    print("Testing Vision Pipeline...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vision_data = vision.process_frame(frame)
        
        # Draw visualizations
        frame = vision.draw_landmarks(frame, vision_data)
        
        # Display
        cv2.imshow('Vision Pipeline Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    vision.cleanup()


if __name__ == '__main__':
    test_vision_pipeline()
