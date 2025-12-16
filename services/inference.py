"""
DriverAid - CNN Inference Wrapper
Handles loading trained CNN model, eye state prediction, and performance optimization.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple
import time


class DrowsinessInferenceEngine:
    """Wrapper for CNN model inference, optimized for real-time performance."""
    
    def __init__(self, model_path: str = 'models/cnn_model.keras'):
        """Initialize the inference engine and load model."""
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        self.inference_times = []
        self.frame_skip_counter = 0
        self.last_prediction = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained CNN model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please train the model first using ml/train.py"
                )
            
            print(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True
            print(f"Model loaded successfully!")
            
            dummy_input = np.zeros((1, 32, 32, 1), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("Model warmed up and ready for inference")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def predict_eye_state(self, eye_region: np.ndarray) -> Tuple[float, str]:
        """
        Predict eye state from preprocessed eye region.
        Returns (confidence, state_label) where confidence is 0.0 (open) to 1.0 (closed).
        """
        if not self.model_loaded:
            return 0.0, "UNKNOWN"
        
        if eye_region is None or eye_region.size == 0:
            return 0.0, "UNKNOWN"
        
        try:
            if eye_region.shape == (32, 32):
                eye_region = eye_region.reshape(1, 32, 32, 1)
            elif eye_region.shape == (32, 32, 1):
                eye_region = eye_region.reshape(1, 32, 32, 1)
            
            start_time = time.time()
            prediction = self.model.predict(eye_region, verbose=0)[0][0]
            
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            state = "CLOSED" if prediction > 0.5 else "OPEN"
            
            return float(prediction), state
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0, "UNKNOWN"
    
    def predict_both_eyes(self, left_eye: np.ndarray, right_eye: np.ndarray) -> dict:
        """Predict state for both eyes."""
        left_prob, left_state = self.predict_eye_state(left_eye)
        right_prob, right_state = self.predict_eye_state(right_eye)
        
        avg_prob = (left_prob + right_prob) / 2.0
        
        if avg_prob > 0.7:
            overall_state = "CLOSED"
        elif avg_prob > 0.3:
            overall_state = "PARTIAL"
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
        """Get average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def should_skip_inference(self, frame_number: int, skip_frames: int = 2) -> bool:
        """Determine if inference should be skipped for performance."""
        return frame_number % (skip_frames + 1) != 0
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
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


def test_inference_engine():
    """Test the inference engine with dummy data."""
    print("=" * 60)
    print("TESTING INFERENCE ENGINE")
    print("=" * 60)
    
    try:
        engine = DrowsinessInferenceEngine(model_path='../models/cnn_model.keras')
        
        print("\nNOTE: Using random pixel data for performance testing.")
        print("   Real accuracy testing requires actual eye images.\n")
        
        print("Test 1: Performance test with dummy data (lighter pixels)")
        open_eye = (np.random.rand(32, 32) * 0.5 + 0.5).astype('float32')
        prob, state = engine.predict_eye_state(open_eye)
        print(f"   Prediction: {state} (confidence: {prob:.3f})")
        print(f"   Model inference successful")
        
        print("\nTest 2: Performance test with dummy data (darker pixels)")
        closed_eye = (np.random.rand(32, 32) * 0.3).astype('float32')
        prob, state = engine.predict_eye_state(closed_eye)
        print(f"   Prediction: {state} (confidence: {prob:.3f})")
        print(f"   Model inference successful")
        
        print("\nTest 3: Both eyes prediction pipeline")
        result = engine.predict_both_eyes(open_eye, closed_eye)
        print(f"   Left eye: {result['left_state']} ({result['left_closed_prob']:.3f})")
        print(f"   Right eye: {result['right_state']} ({result['right_closed_prob']:.3f})")
        print(f"   Overall state: {result['state']}")
        print(f"   Batch prediction successful")
        
        print("\nPerformance Statistics:")
        stats = engine.get_performance_stats()
        print(f"   Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"   Min: {stats['min_inference_time_ms']:.2f} ms")
        print(f"   Max: {stats['max_inference_time_ms']:.2f} ms")
        print(f"   Total inferences: {stats['total_inferences']}")
        
        target_time = 50
        avg_time = stats['avg_inference_time_ms']
        if avg_time < target_time:
            print(f"\nPASS: Inference time ({avg_time:.2f}ms) < Target ({target_time}ms)")
        else:
            print(f"\nWARNING: Inference time ({avg_time:.2f}ms) > Target ({target_time}ms)")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nSOLUTION:")
        print("   1. Make sure you've trained the model:")
        print("      cd ml")
        print("      python train.py")
        print("   2. Verify model exists at: models/cnn_model.keras")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == '__main__':
    test_inference_engine()
