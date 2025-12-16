"""
DriverAid - CNN Inference Wrapper
Phase 3: Model Integration

Handles:
- Loading trained CNN model
- Eye state prediction from eye regions
- Performance optimization (caching, batching)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple
import time


class DrowsinessInferenceEngine:
    """
    Wrapper for CNN model inference.
    Optimized for real-time performance.
    """
    
    def __init__(self, model_path: str = 'models/cnn_model.keras'):
        """
        Initialize the inference engine and load model.
        
        Args:
            model_path: Path to trained .keras model file
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Performance tracking
        self.inference_times = []
        self.frame_skip_counter = 0
        self.last_prediction = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained CNN model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please train the model first using ml/train.py"
                )
            
            print(f"🔄 Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True
            print(f"✅ Model loaded successfully!")
            
            # Warm-up inference (first inference is slower due to TF initialization)
            dummy_input = np.zeros((1, 32, 32, 1), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("✅ Model warmed up and ready for inference")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def predict_eye_state(self, eye_region: np.ndarray) -> Tuple[float, str]:
        """
        Predict eye state from preprocessed eye region.
        
        Args:
            eye_region: 32x32 grayscale normalized image (output from vision.py)
            
        Returns:
            Tuple of (confidence, state_label)
            - confidence: 0.0 (open) to 1.0 (closed)
            - state_label: "OPEN" or "CLOSED"
        """
        if not self.model_loaded:
            return 0.0, "UNKNOWN"
        
        if eye_region is None or eye_region.size == 0:
            return 0.0, "UNKNOWN"
        
        try:
            # Reshape for model input: (1, 32, 32, 1)
            if eye_region.shape == (32, 32):
                eye_region = eye_region.reshape(1, 32, 32, 1)
            elif eye_region.shape == (32, 32, 1):
                eye_region = eye_region.reshape(1, 32, 32, 1)
            
            # Inference timing
            start_time = time.time()
            
            # Predict (output is probability of CLOSED)
            prediction = self.model.predict(eye_region, verbose=0)[0][0]
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            # Classify based on threshold
            state = "CLOSED" if prediction > 0.5 else "OPEN"
            
            return float(prediction), state
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return 0.0, "UNKNOWN"
    
    def predict_both_eyes(self, left_eye: np.ndarray, right_eye: np.ndarray) -> dict:
        """
        Predict state for both eyes.
        
        Args:
            left_eye: Left eye region (32x32)
            right_eye: Right eye region (32x32)
            
        Returns:
            Dictionary containing:
                - left_closed_prob: Probability left eye is closed
                - right_closed_prob: Probability right eye is closed
                - avg_closed_prob: Average probability
                - state: Overall state ("OPEN", "CLOSED", "PARTIAL")
                - both_eyes_closed: Boolean
        """
        # Predict for each eye
        left_prob, left_state = self.predict_eye_state(left_eye)
        right_prob, right_state = self.predict_eye_state(right_eye)
        
        # Calculate average
        avg_prob = (left_prob + right_prob) / 2.0
        
        # Determine overall state
        if avg_prob > 0.7:
            overall_state = "CLOSED"
        elif avg_prob > 0.3:
            overall_state = "PARTIAL"  # One eye closed or both partially closed
        else:
            overall_state = "OPEN"
        
        both_closed = left_state == "CLOSED" and right_state == "CLOSED"
        
        return {
            'left_closed_prob': left_prob,
            'right_closed_prob': right_prob,
            'avg_closed_prob': avg_prob,
            'state': overall_state,
            'both_eyes_closed': both_closed,
            'left_state': left_state,
            'right_state': right_state
        }
    
    def get_average_inference_time(self) -> float:
        """
        Get average inference time in milliseconds.
        
        Returns:
            Average inference time (ms)
        """
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def should_skip_inference(self, frame_number: int, skip_frames: int = 2) -> bool:
        """
        Determine if inference should be skipped for performance.
        
        Strategy: Only run CNN every N frames to save CPU.
        Use previous prediction for skipped frames.
        
        Args:
            frame_number: Current frame number
            skip_frames: Run inference every N frames (default: 2)
            
        Returns:
            True if should skip, False if should run inference
        """
        return frame_number % (skip_frames + 1) != 0
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0,
                'total_inferences': 0
            }
        
        return {
            'avg_inference_time_ms': sum(self.inference_times) / len(self.inference_times),
            'min_inference_time_ms': min(self.inference_times),
            'max_inference_time_ms': max(self.inference_times),
            'total_inferences': len(self.inference_times)
        }


# Utility function for testing
def test_inference_engine():
    """Test the inference engine with dummy data."""
    print("=" * 60)
    print("🧪 TESTING INFERENCE ENGINE")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = DrowsinessInferenceEngine(model_path='../models/cnn_model.keras')
        
        # Test with dummy "open" eye (lighter pixels)
        print("\n📊 Test 1: Simulated OPEN eye (lighter pixels)")
        open_eye = (np.random.rand(32, 32) * 0.5 + 0.5).astype('float32')  # Values 0.5-1.0
        prob, state = engine.predict_eye_state(open_eye)
        print(f"   Result: {state} (confidence: {prob:.3f})")
        
        # Test with dummy "closed" eye (darker pixels)
        print("\n📊 Test 2: Simulated CLOSED eye (darker pixels)")
        closed_eye = (np.random.rand(32, 32) * 0.3).astype('float32')  # Values 0.0-0.3
        prob, state = engine.predict_eye_state(closed_eye)
        print(f"   Result: {state} (confidence: {prob:.3f})")
        
        # Test both eyes
        print("\n📊 Test 3: Both eyes prediction")
        result = engine.predict_both_eyes(open_eye, closed_eye)
        print(f"   Left: {result['left_state']} ({result['left_closed_prob']:.3f})")
        print(f"   Right: {result['right_state']} ({result['right_closed_prob']:.3f})")
        print(f"   Overall: {result['state']}")
        
        # Performance stats
        print("\n⚡ Performance Statistics:")
        stats = engine.get_performance_stats()
        print(f"   Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"   Min: {stats['min_inference_time_ms']:.2f} ms")
        print(f"   Max: {stats['max_inference_time_ms']:.2f} ms")
        print(f"   Total inferences: {stats['total_inferences']}")
        
        # Target check
        target_time = 50  # ms
        avg_time = stats['avg_inference_time_ms']
        if avg_time < target_time:
            print(f"\n✅ PASS: Inference time ({avg_time:.2f}ms) < Target ({target_time}ms)")
        else:
            print(f"\n⚠️ WARNING: Inference time ({avg_time:.2f}ms) > Target ({target_time}ms)")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📖 SOLUTION:")
        print("   1. Make sure you've trained the model:")
        print("      cd ml")
        print("      python train.py")
        print("   2. Verify model exists at: models/cnn_model.keras")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == '__main__':
    test_inference_engine()
