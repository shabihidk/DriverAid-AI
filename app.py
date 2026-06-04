"""
DriverAid - Real-time Drowsiness Detection System
Main Application Entry Point
"""

# ----------------------------------------------------------------------------
# Headless OpenCV guard (MUST run before anything imports cv2 / mediapipe).
#
# MediaPipe hard-depends on opencv-contrib-python (the GUI build), which needs
# libGL.so.1 + libgthread-2.0.so.0 - system libs that are NOT present on
# Streamlit Community Cloud. Rather than fight apt/packages.txt (which breaks
# every time Streamlit's Debian base image changes), we make cv2 headless at
# startup. opencv-python-headless ships the same cv2 API minus the GUI libs,
# and MediaPipe only uses standard cv2 calls at runtime.
# ----------------------------------------------------------------------------
import sys
import subprocess

_OPENCV_HEADLESS = "opencv-python-headless==4.10.0.84"

try:
    import cv2  # noqa: F401  (probe whether the current cv2 imports cleanly)
except (ImportError, OSError):
    # The GUI build failed to load (missing libGL). Replace it with headless.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "opencv-contrib-python", "opencv-python"],
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", _OPENCV_HEADLESS],
        check=True,
    )
    import importlib
    importlib.invalidate_caches()
    import cv2  # noqa: F401  (now resolves to the headless build)

import streamlit as st

# Configure page
st.set_page_config(
    page_title="DriverAid - Drowsiness Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
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

# Import UI modules
from ui.components import render_sidebar
from ui.live_detection import render_live_detection_tab
from ui.visualizations import render_visualizations_tab
from ui.documentation import render_documentation_tab

# Main title
st.title(" DriverAid - Drowsiness Detection System")
st.markdown("*Hybrid AI: Computer Vision + CNN + Expert Rules*")

# Render sidebar
render_sidebar()

# Main tabs
tab1, tab2, tab3 = st.tabs([" Live Detection", " Visualizations", " Documentation"])

with tab1:
    render_live_detection_tab()

with tab2:
    render_visualizations_tab()

with tab3:
    render_documentation_tab()

# Footer
st.markdown("---")
st.caption("DriverAid v1.0 - Hybrid AI System | MediaPipe + TensorFlow + Expert Rules")
