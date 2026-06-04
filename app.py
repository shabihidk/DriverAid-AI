"""
DriverAid - Real-time Drowsiness Detection System
Main Application Entry Point
"""

# ----------------------------------------------------------------------------
# Headless OpenCV guard (MUST run before anything imports cv2 / mediapipe).
#
# MediaPipe hard-depends on opencv-contrib-python (the GUI build), which needs
# libGL.so.1 + libgthread-2.0.so.0 - system libs that are NOT present on
# Streamlit Community Cloud. We cannot uninstall it at runtime because the venv
# site-packages is mounted READ-ONLY after the build phase.
#
# Instead, we install opencv-python-headless (same cv2 API, no GUI libs) into a
# WRITABLE temp dir and put it FIRST on sys.path so it shadows the read-only GUI
# build. MediaPipe then imports the headless cv2 too. No apt / packages.txt and
# no writes to the locked venv - works regardless of Streamlit's Debian image.
# ----------------------------------------------------------------------------
import os
import sys
import subprocess
import importlib

_OPENCV_HEADLESS = "opencv-python-headless==4.10.0.84"
_HEADLESS_DIR = "/tmp/driveraid_cv2"


def _ensure_headless_cv2():
    try:
        import cv2  # noqa: F401  (does the current cv2 import cleanly?)
        return
    except Exception:
        pass  # GUI build failed to load (missing libGL) - install headless below.

    os.makedirs(_HEADLESS_DIR, exist_ok=True)
    if not os.path.isdir(os.path.join(_HEADLESS_DIR, "cv2")):
        # --no-deps: numpy already exists in site-packages. Installing deps here
        # could drop numpy 2.x into /tmp and shadow the pinned numpy 1.26.4,
        # breaking TensorFlow/MediaPipe.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
             "--target", _HEADLESS_DIR, _OPENCV_HEADLESS],
            check=True,
        )

    if _HEADLESS_DIR not in sys.path:
        sys.path.insert(0, _HEADLESS_DIR)
    sys.modules.pop("cv2", None)  # drop the half-loaded GUI module
    importlib.invalidate_caches()
    import cv2  # noqa: F401  (now resolves to the headless build in /tmp)


_ensure_headless_cv2()

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
