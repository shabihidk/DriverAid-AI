"""
Live Detection Tab - Real-time drowsiness monitoring

IMPORTANT (Streamlit Cloud safety):
streamlit-webrtc runs the video callback (`recv`) in a SEPARATE worker thread.
`st.session_state` and any other `st.*` calls are NOT available in that thread and
will raise / warn ("missing ScriptRunContext"). Therefore the video processor below
holds its OWN pipeline objects and never touches st.session_state inside recv().
The main thread reads results back off the processor instance for the UI.
"""

import os
import sys

import av
import cv2
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from services.vision import VisionPipeline
from services.inference import DrowsinessInferenceEngine
from services.rules import ExpertSystem
from ui.components import render_alert_panel, get_metric_status

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'cnn_model.keras'
)


@st.cache_resource(show_spinner="Loading CNN model...")
def load_inference_engine():
    """Load the Keras model once per server process (cached across reruns/sessions)."""
    return DrowsinessInferenceEngine(model_path=MODEL_PATH)


def initialize_system():
    """Preflight check: make sure the model + pipeline load before showing the camera."""
    try:
        with st.spinner("Initializing DriverAid system..."):
            # Loading the engine here surfaces a clear error if the model is missing.
            _ = load_inference_engine()
            st.session_state.system_initialized = True
        st.success("System initialized successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return False


# --- WEBRTC VIDEO PROCESSOR (runs in a worker thread, NO st.* calls here) ---
class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Each processor owns its pipeline; the heavy CNN is shared via the cache.
        self.vision = VisionPipeline()
        self.inference = load_inference_engine()
        self.expert = ExpertSystem()

        self.frame_count = 0
        self.alert_count = 0

        # Read by the main thread to render the side panels.
        self.alert_data = None
        self.metrics = None

    def _process(self, frame):
        self.frame_count += 1
        frame_num = self.frame_count

        vision_data = self.vision.process_frame(frame)

        cnn_prediction = None
        if vision_data and not self.inference.should_skip_inference(frame_num, skip_frames=2):
            left_eye = vision_data.get('left_eye_region')
            right_eye = vision_data.get('right_eye_region')
            if left_eye is not None and right_eye is not None:
                cnn_prediction = self.inference.predict_both_eyes(left_eye, right_eye)

        alert_data = self.expert.analyze(vision_data, cnn_prediction)
        if alert_data['alert_level'] != 'NONE':
            self.alert_count += 1

        annotated_frame = frame.copy()

        if vision_data and vision_data.get('face_detected'):
            landmarks = vision_data['landmarks']
            for idx in self.vision.LEFT_EYE_INDICES + self.vision.RIGHT_EYE_INDICES:
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
            'inference_time': self.inference.get_average_inference_time()
        }

        return annotated_frame, alert_data, metrics

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_frame, alert_data, metrics = self._process(img)

        # Stash results for the main thread (plain attributes, thread-safe enough here).
        self.alert_data = alert_data
        self.metrics = metrics

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def render_live_detection_tab():
    """Render the live detection tab."""
    if not st.session_state.system_initialized:
        st.warning("Please initialize the system using the sidebar button")
        st.markdown("""
        ### System Requirements:
        - A webcam + a browser that allows camera access
        - Trained model committed at `models/cnn_model.keras`
        - All dependencies installed

        ### Architecture:
        1. **Vision Pipeline** - Face detection, EAR calculation, head pose estimation
        2. **CNN Inference** - Eye state prediction
        3. **Expert Rules** - Drowsiness decision engine
        4. **Real-time Alerts** - Actionable warnings and recommendations
        """)
        return

    st.info("Click 'START' below and allow camera access in your browser to begin detection.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        # STUN gets most networks connected. Restrictive networks (corporate/mobile)
        # may also need a TURN relay - see the deployment notes.
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=DrowsinessVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("Alert Status")
        alert_placeholder = st.empty()

        if st.session_state.get('show_metrics', True):
            st.subheader("Metrics")
            metrics_placeholder = st.empty()

    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.alert_data:
            with alert_placeholder.container():
                render_alert_panel(webrtc_ctx.video_processor.alert_data)

        if webrtc_ctx.video_processor.metrics and st.session_state.get('show_metrics', True):
            metrics = webrtc_ctx.video_processor.metrics
            with metrics_placeholder.container():
                with st.expander("What do these metrics mean?", expanded=False):
                    st.markdown("""
                    **EAR (Eye Aspect Ratio):** Measures eye openness
                    - > 0.25: Eyes wide open (alert)
                    - 0.20-0.25: Eyes partially closed (caution)
                    - < 0.20: Eyes closing/closed (drowsy)

                    **Head Pitch:** Vertical head angle
                    - Higher = head nodding forward (drowsiness)

                    **Inference:** CNN processing speed per frame
                    """)

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    ear_status, _ = get_metric_status("EAR", metrics['ear'])
                    st.metric("EAR (Eye Openness)", f"{metrics['ear']:.3f}", ear_status)
                with metric_col2:
                    pitch_status, _ = get_metric_status("Head Pitch", metrics['head_pitch'])
                    st.metric("Head Pitch", f"{metrics['head_pitch']:.1f}", pitch_status)

                st.metric("Inference Time", f"{metrics['inference_time']:.1f}ms")
