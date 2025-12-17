"""
DriverAid - Real-time Drowsiness Detection System
Main Application Entry Point
"""

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
