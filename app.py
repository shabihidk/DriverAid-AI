"""
DriverAid - Real-time Drowsiness Detection System
Main Streamlit Application Entry Point

Phase 1: Webcam Test
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="DriverAid",
    page_icon="🚗",
    layout="wide"
)

st.title("DriverAid - Drowsiness Detection System")
st.markdown("**Phase 1: Webcam Verification Test**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.info("Testing webcam connectivity...")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    stframe = st.empty()

with col2:
    st.subheader("Status")
    status_placeholder = st.empty()
    
# Webcam Test
run_detection = st.checkbox("Start Webcam Test", value=False)

if run_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("ERROR: Cannot access webcam. Please check:")
        st.markdown("""
        - Webcam is connected
        - No other application is using it
        - Camera permissions are granted
        """)
    else:
        st.success("Webcam connected successfully!")
        
        # Frame counter for FPS calculation
        frame_count = 0
        
        while run_detection:
            ret, frame = cap.read()
            
            if not ret:
                status_placeholder.error("⚠️ Failed to read frame from webcam")
                break
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame counter overlay
            frame_count += 1
            cv2.putText(
                frame_rgb, 
                f"Frame: {frame_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display frame
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update status
            status_placeholder.success(f"""
            **Status:** Running  
            **Frames Captured:** {frame_count}  
            **Resolution:** {frame.shape[1]}x{frame.shape[0]}
            """)
        
        cap.release()
        st.info("Webcam test stopped.")

else:
    st.info("Check the box above to start the webcam test")

# Footer
st.markdown("---")
st.caption("DriverAid v0.1 - Phase 1: Setup & Skeleton")
