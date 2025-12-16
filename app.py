"""
DriverAid - Real-time Drowsiness Detection System
Main Application
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import winsound

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

from services.vision import VisionPipeline
from services.inference import DrowsinessInferenceEngine
from services.rules import ExpertSystem

st.set_page_config(
    page_title="DriverAid - Drowsiness Detection",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.vision = None
    st.session_state.inference = None
    st.session_state.expert = None
    st.session_state.frame_count = 0
    st.session_state.alert_count = 0
    st.session_state.last_alert_level = "NONE"
    st.session_state.last_beep_time = 0
    st.session_state.audio_enabled = True


def trigger_alert_sound(alert_level):
    """Play appropriate beep sound for alert level."""
    if not st.session_state.audio_enabled:
        return
    
    current_time = time.time()
    
    if alert_level != st.session_state.last_alert_level:
        st.session_state.last_beep_time = 0
        st.session_state.last_alert_level = alert_level
    
    try:
        if alert_level == "MEDIUM":
            if current_time - st.session_state.last_beep_time > 3.0:
                winsound.Beep(800, 150)
                st.session_state.last_beep_time = current_time
        elif alert_level == "HIGH":
            if current_time - st.session_state.last_beep_time > 1.0:
                winsound.Beep(1000, 150)
                st.session_state.last_beep_time = current_time
        elif alert_level == "CRITICAL":
            if current_time - st.session_state.last_beep_time > 0.5:
                winsound.Beep(1500, 300)
                st.session_state.last_beep_time = current_time
    except:
        pass


def initialize_system():
    """Initialize all system components."""
    try:
        with st.spinner("Initializing DriverAid system..."):
            st.session_state.vision = VisionPipeline(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.session_state.inference = DrowsinessInferenceEngine(
                model_path='models/cnn_model.keras'
            )
            st.session_state.expert = ExpertSystem()
            st.session_state.system_initialized = True
            return True
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        st.info("Please train the model first: `cd ml && python train.py`")
        return False
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return False


def get_alert_color(alert_level):
    """Get color hex code for alert level."""
    colors = {
        'NONE': '#28a745',
        'LOW': '#ffc107',
        'MEDIUM': '#fd7e14',
        'HIGH': '#dc3545',
        'CRITICAL': '#8B0000'
    }
    return colors.get(alert_level, '#6c757d')


def render_alert_panel(alert_data):
    """Render alert information panel."""
    alert_level = alert_data.get('alert_level', 'NONE')
    confidence = alert_data.get('confidence', 0.0)
    reason = alert_data.get('reason', 'No data')
    recommendations = alert_data.get('recommendations', [])
    
    alert_color = get_alert_color(alert_level)
    
    st.markdown(f"""
    <div style="background-color: {alert_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <h2 style="color: white; margin: 0;">ALERT: {alert_level}</h2>
        <p style="color: white; margin: 5px 0;">Confidence: {confidence:.1%}</p>
    </div>
    <div style="background-color: #ffffff; padding: 10px; color: #000000;">
        <p style="margin: 5px 0;"><strong>Reason:</strong> {reason}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if recommendations:
        st.markdown("**Recommendations:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")


def process_frame_with_system(frame):
    """Process frame through complete pipeline."""
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


st.title("DriverAid - Real-time Drowsiness Detection")
st.markdown("Vision + CNN + Expert Rules")

with st.sidebar:
    st.header("System Controls")
    
    if not st.session_state.system_initialized:
        if st.button("Initialize System", type="primary"):
            initialize_system()
    else:
        st.success("System Initialized")
        
        st.subheader("Settings")
        st.session_state.audio_enabled = st.checkbox("Audio Alerts", value=True)
        if st.session_state.audio_enabled:
            st.caption("MEDIUM: Slow beeps | HIGH: Fast beeps | CRITICAL: Loud beep")
        
        st.session_state.show_metrics = st.checkbox("Show Metrics", value=True)
        
        if st.button("Reset Statistics"):
            st.session_state.frame_count = 0
            st.session_state.alert_count = 0
            st.session_state.expert.reset()
            st.success("Statistics reset")
    
    st.markdown("---")
    
    st.subheader("System Info")
    if st.session_state.system_initialized:
        st.metric("Frames Processed", st.session_state.frame_count)
        st.metric("Total Alerts", st.session_state.alert_count)
        if st.session_state.frame_count > 0:
            alert_rate = (st.session_state.alert_count / st.session_state.frame_count) * 100
            st.metric("Alert Rate", f"{alert_rate:.1f}%")

if st.session_state.system_initialized:
    run_detection = st.checkbox("Start Detection", value=False)
    
    if run_detection:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Feed")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("Alert Status")
            alert_placeholder = st.empty()
            
            if st.session_state.get('show_metrics', True):
                st.subheader("Metrics")
                metrics_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check permissions and ensure no other application is using the camera.")
        else:
            while run_detection:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read frame from webcam. Camera may be in use by another application.")
                    break
                
                annotated_frame, alert_data, metrics = process_frame_with_system(frame)
                trigger_alert_sound(alert_data['alert_level'])
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                with alert_placeholder.container():
                    render_alert_panel(alert_data)
                
                if st.session_state.get('show_metrics', True):
                    with metrics_placeholder.container():
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("EAR", f"{metrics['ear']:.3f}")
                            st.metric("CNN Conf.", f"{metrics['cnn_confidence']:.3f}")
                        with metric_col2:
                            st.metric("Head Pitch", f"{metrics['head_pitch']:.1f}°")
                            st.metric("Inference", f"{metrics['inference_time']:.1f}ms")
                
                time.sleep(0.03)
            
            cap.release()
    else:
        st.info("Check the box above to start real-time drowsiness detection")
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

st.markdown("---")
st.caption("DriverAid v1.0 - MediaPipe + TensorFlow + Streamlit")
