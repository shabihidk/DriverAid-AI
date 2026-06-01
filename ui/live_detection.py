"""
Live Detection Tab - Real-time drowsiness monitoring
"""

import streamlit as st
import cv2
import time
import platform
import sys
import os

# Safe cross-platform import for winsound (Windows only hardware audio)
if platform.system() == "Windows":
    import winsound
else:
    winsound = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from services.vision import VisionPipeline
from services.inference import DrowsinessInferenceEngine
from services.rules import ExpertSystem
from ui.components import render_alert_panel, get_metric_status

# IMPORT WEBRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

def trigger_alert_sound(alert_level):
    # [Your existing trigger_alert_sound code remains exactly the same]
    if not st.session_state.audio_enabled or not winsound:
        return
    current_time = time.time()
    if alert_level != st.session_state.last_alert_level:
        st.session_state.last_beep_time = 0
        st.session_state.last_alert_level = alert_level
    try:
        if alert_level == "MEDIUM":
            if current_time - st.session_state.last_beep_time > 3.0:
                winsound.Beep(800, 200)
                st.session_state.last_beep_time = current_time
        elif alert_level == "HIGH":
            if current_time - st.session_state.last_beep_time > 1.0:
                winsound.Beep(1000, 300)
                st.session_state.last_beep_time = current_time
        elif alert_level == "CRITICAL":
            if current_time - st.session_state.last_beep_time > 0.5:
                winsound.Beep(1500, 500)
                st.session_state.last_beep_time = current_time
    except Exception as e:
        pass


def initialize_system():
    # [Your existing initialize_system code remains exactly the same]
    try:
        with st.spinner("Initializing DriverAid system..."):
            st.session_state.vision = VisionPipeline()
            st.session_state.inference = DrowsinessInferenceEngine(model_path='models/cnn_model.keras')
            st.session_state.expert = ExpertSystem()
            st.session_state.system_initialized = True
        st.success("System initialized successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return False


def process_frame_with_system(frame):
    # [Your existing process_frame_with_system code remains exactly the same]
    st.session_state.frame_count += 1
    frame_num = st.session_state.frame_count
    
    vision_data = st.session_state.vision.process_frame(frame)
    
    cnn_prediction = None
    if vision_data and not st.session_state.inference.should_skip_inference(frame_num, skip_frames=2):
        left_eye = vision_data.get('left_eye_region')
        right_eye = vision_data.get('right_eye_region')
        if left_eye is not None and right_eye is not None:
            cnn_prediction = st.session_state.inference.predict_both_eyes(left_eye, right_eye)
    
    alert_data = st.session_state.expert.analyze(vision_data, cnn_prediction)
    
    if alert_data['alert_level'] != 'NONE':
        st.session_state.alert_count += 1
    
    annotated_frame = frame.copy()
    
    if vision_data and vision_data.get('face_detected'):
        landmarks = vision_data['landmarks']
        for idx in st.session_state.vision.LEFT_EYE_INDICES + st.session_state.vision.RIGHT_EYE_INDICES:
            x, y, _ = landmarks[idx]
            cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
        
        ear = vision_data['ear_avg']
        cv2.putText(annotated_frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        pose = vision_data['head_pose']
        cv2.putText(annotated_frame, f"Pitch: {pose['pitch']:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if cnn_prediction:
            state = cnn_prediction['state']
            prob = cnn_prediction['avg_closed_prob']
            cv2.putText(annotated_frame, f"Eyes: {state} ({prob:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    else:
        cv2.putText(annotated_frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    alert_color_bgr = {
        'NONE': (40, 167, 69),
        'LOW': (7, 193, 255),
        'MEDIUM': (20, 126, 253),
        'HIGH': (69, 53, 220),
        'CRITICAL': (0, 0, 139)
    }
    color = alert_color_bgr.get(alert_data['alert_level'], (128, 128, 128))
    cv2.putText(annotated_frame, f"Alert: {alert_data['alert_level']}", 
                (annotated_frame.shape[1] - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    metrics = {
        'ear': vision_data['ear_avg'] if vision_data else 0.0,
        'head_pitch': vision_data['head_pose']['pitch'] if vision_data else 0.0,
        'cnn_confidence': cnn_prediction['avg_closed_prob'] if cnn_prediction else 0.0,
        'inference_time': st.session_state.inference.get_average_inference_time()
    }
    
    return annotated_frame, alert_data, metrics

# --- NEW WEBRTC VIDEO TRANSFORMER CLASS ---
class DrowsinessVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Variables to store data to display in the Streamlit UI
        self.alert_data = None
        self.metrics = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # We must process the frame through your existing pipeline
        annotated_frame, alert_data, metrics = process_frame_with_system(img)
        
        # Save data so the main thread can render the UI text
        self.alert_data = alert_data
        self.metrics = metrics
        
        # Trigger sound (Note: winsound won't work on cloud, but keeping your logic)
        trigger_alert_sound(alert_data['alert_level'])
        
        return annotated_frame


def render_live_detection_tab():
    """Render the live detection tab."""
    if st.session_state.system_initialized:
        
        st.info("Click 'START' below to begin real-time drowsiness detection. Make sure to allow camera permissions in your browser.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Feed")
            
            # --- REPLACE cv2.VideoCapture WITH webrtc_streamer ---
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            webrtc_ctx = webrtc_streamer(
                key="drowsiness-detection",
                video_transformer_factory=DrowsinessVideoTransformer,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True
            )
        
        with col2:
            st.subheader("Alert Status")
            alert_placeholder = st.empty()
            
            if st.session_state.get('show_metrics', True):
                st.subheader("Metrics")
                metrics_placeholder = st.empty()
        
        # If the video stream is running, we poll the transformer for data to update the UI
        if webrtc_ctx.video_transformer:
            if webrtc_ctx.video_transformer.alert_data:
                with alert_placeholder.container():
                    render_alert_panel(webrtc_ctx.video_transformer.alert_data)
            
            if webrtc_ctx.video_transformer.metrics and st.session_state.get('show_metrics', True):
                metrics = webrtc_ctx.video_transformer.metrics
                with metrics_placeholder.container():
                    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                        st.markdown("""
                        **EAR (Eye Aspect Ratio):** Measures eye openness
                        - 🟢 > 0.25: Eyes wide open (alert)
                        - 🟡 0.20-0.25: Eyes partially closed (caution)
                        - 🔴 < 0.20: Eyes closing/closed (drowsy)
                        
                        **Head Pitch:** Vertical head angle
                        - 🟢 ±15°: Normal position
                        - 🟡 ±15-25°: Head tilting (mild concern)
                        - 🔴 > ±25°: Head nodding (drowsiness)
                        
                        **CNN Confidence:** AI model certainty
                        - Higher values = More confident prediction
                        
                        **Inference:** Processing speed per frame
                        - Target: < 50ms for real-time detection
                        """)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        ear_status, _ = get_metric_status("EAR", metrics['ear'])
                        st.metric("EAR (Eye Openness)", f"{metrics['ear']:.3f}", ear_status)
                    with metric_col2:
                        pitch_status, _ = get_metric_status("Head Pitch", metrics['head_pitch'])
                        st.metric("Head Pitch", f"{metrics['head_pitch']:.1f}°", pitch_status)
                    
                    st.metric("Inference Time", f"{metrics['inference_time']:.1f}ms")

    else:
        st.warning("Please initialize the system using the sidebar button")
        
        st.markdown("""
        ### System Requirements:
        - Webcam connected
        - Model trained (models/cnn_model.keras)
        - All dependencies installed
        
        ### Architecture:
        1. **Vision Pipeline** - Face detection, EAR calculation, head pose estimation
        2. **CNN Inference** - Eye state prediction (98.28% accuracy)
        3. **Expert Rules** - Drowsiness decision engine
        4. **Real-time Alerts** - Actionable warnings and recommendations
        """)