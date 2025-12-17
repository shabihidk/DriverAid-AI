"""
Documentation Tab - Project encyclopedia
"""

import streamlit as st


def render_documentation_tab():
    """Render the documentation/encyclopedia tab."""
    st.header("📖 Project Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Details", "Integration", "Usage Guide"])
    
    with tab1:
        st.markdown("""
        ## DriverAid - Drowsiness Detection System
        
        ### Problem Statement
        Driver drowsiness is a leading cause of road accidents worldwide. Traditional detection 
        methods rely on manual observation or invasive sensors. DriverAid provides a non-invasive, 
        real-time solution using computer vision and AI.
        
        ### Solution Approach
        DriverAid combines three complementary AI technologies:
        
        1. **Computer Vision (MediaPipe)**
           - Real-time face detection and landmark tracking
           - 468 facial landmarks for precise eye and head analysis
           - EAR (Eye Aspect Ratio) calculation
           - Head pose estimation (pitch, yaw, roll)
        
        2. **Deep Learning (CNN)**
           - Lightweight CNN for eye state classification
           - 98.28% accuracy on test set
           - <50ms inference time for real-time performance
           - Trained on 84,896 images
        
        3. **Expert System (Rule-Based AI)**
           - Multi-level drowsiness rules (LOW, MEDIUM, HIGH, CRITICAL)
           - Temporal smoothing to reduce false positives
           - Combines vision metrics with CNN predictions
           - Generates actionable recommendations
        
        ### Real-World Impact
        - **Accident Prevention:** Early warning before critical drowsiness
        - **Fleet Management:** Monitor driver alertness in commercial vehicles
        - **Personal Safety:** Individual drivers can self-monitor
        - **Insurance:** Telematics data for risk assessment
        """)
    
    with tab2:
        st.markdown("""
        ## Technical Architecture
        
        ### System Pipeline
        ```
        Camera Feed → Vision Pipeline → CNN Inference → Expert Rules → Alert System
        ```
        
        ### Component Details
        
        #### 1. Vision Pipeline (MediaPipe Face Mesh)
        - **Technology:** Google MediaPipe
        - **Landmarks:** 468 facial points
        - **Key Metrics:**
          - EAR (Eye Aspect Ratio): Measures eye openness
          - MAR (Mouth Aspect Ratio): Detects yawning
          - Head Pose: 3D orientation (pitch, yaw, roll)
        - **Performance:** 10-15 FPS on standard webcam
        
        #### 2. CNN Model
        - **Architecture:** Lightweight 2-layer CNN
        - **Input:** 32x32 grayscale eye images
        - **Output:** Binary classification (Open/Closed)
        - **Parameters:** ~51,000 (optimized for speed)
        - **Training:** Adam optimizer, Binary Cross-Entropy loss
        - **Regularization:** Dropout (0.25, 0.5) and Early Stopping
        
        #### 3. Expert Rules Engine
        
        **Rule 1: Eye Closure Detection**
        - **Severity:** HIGH (2-4s closed) → CRITICAL (>4s closed)
        - **Logic:** Monitors continuous eye closure duration
        - **Threshold:** 2 seconds = HIGH alert, 4 seconds = CRITICAL alert
        
        **Rule 2: Head Pose Monitoring**
        - **Severity:** MEDIUM
        - **Logic:** Detects head nodding or severe tilting
        - **Threshold:** Head pitch > 35° for 4+ seconds
        
        **Rule 3: Blink Rate Analysis**
        - **Severity:** LOW
        - **Logic:** Monitors blink frequency over 90-second window
        - **Threshold:** < 12 blinks per 90 seconds (early warning)
        
        **Temporal Smoothing:**
        - 30-frame rolling window
        - Prevents alert flickering
        - Requires sustained pattern for alert
        
        ### Alert System
        - **Visual:** Color-coded UI (Green → Yellow → Orange → Red → Dark Red)
        - **Audio:** Frequency-based beeps (800Hz → 1500Hz)
        - **Recommendations:** Context-aware suggestions
        """)
    
    with tab3:
        st.markdown("""
        ## ML-Traditional AI Integration
        
        ### Hybrid Architecture Benefits
        
        **Why Combine CNN + Rules?**
        - **CNN Strengths:** High accuracy on clean data, learns complex patterns
        - **CNN Weaknesses:** Black box, no temporal reasoning, requires lots of data
        - **Rules Strengths:** Explainable, temporal logic, domain knowledge
        - **Rules Weaknesses:** Hard to tune, can't learn from data
        
        ### Integration Flow
        
        1. **Vision Pipeline** extracts raw features (EAR, head pose)
        2. **CNN** classifies eye state from extracted eye regions
        3. **Expert System** receives:
           - Vision metrics (EAR, head pose, blink count)
           - CNN predictions (eye state, confidence)
        4. **Rule Engine** applies logic:
           ```python
           if EAR < 0.18 AND CNN_predicts_closed AND duration > 2s:
               ALERT = HIGH
           ```
        5. **Temporal Smoother** validates over 30-frame window
        6. **Alert Generator** produces final decision
        
        ### Why This Works
        - CNN provides **accurate momentary detection**
        - Rules provide **temporal context and reasoning**
        - Vision provides **complementary geometric features**
        - Integration provides **robust, explainable decisions**
        
        ### Feedback Mechanisms
        - Alert history influences smoothing thresholds
        - Multiple triggered rules increase confidence
        - System adapts to sustained vs. momentary patterns
        """)
    
    with tab4:
        st.markdown("""
        ## Usage Guide
        
        ### Setup Instructions
        
        1. **Install Dependencies**
           ```bash
           pip install -r requirements.txt
           ```
        
        2. **Train Model (if needed)**
           ```bash
           cd ml
           python train.py
           ```
        
        3. **Run Application**
           ```bash
           streamlit run app.py
           ```
        
        ### Using the System
        
        1. **Initialize:** Click "Initialize System" in sidebar
        2. **Configure:** Enable audio alerts and metrics display
        3. **Start Detection:** Check "Start Detection" box
        4. **Position:** Sit 50-100cm from camera, face forward
        5. **Monitor:** Watch for alert levels and recommendations
        
        ### Understanding Alerts
        
        - **🟢 NONE:** You're alert and attentive
        - **🟡 LOW:** Early warning - blink rate decreasing
        - **🟠 MEDIUM:** Caution - head tilting detected
        - **🔴 HIGH:** Danger - eyes closed too long
        - **🔴 CRITICAL:** IMMEDIATE - pull over now!
        
        ### Best Practices
        
        ✅ **DO:**
        - Use in well-lit environment
        - Position camera at eye level
        - Wear corrective lenses if needed
        - Take breaks when alerted
        
        ❌ **DON'T:**
        - Use while wearing sunglasses
        - Rely solely on system (stay alert!)
        - Ignore CRITICAL alerts
        - Use in poor lighting
        
        ### Troubleshooting
        
        **Issue:** "No face detected"
        - **Solution:** Adjust position, improve lighting
        
        **Issue:** False positives
        - **Solution:** System may be too sensitive, adjust thresholds in rules.py
        
        **Issue:** Slow performance
        - **Solution:** Close other applications, reduce camera resolution
        """)
