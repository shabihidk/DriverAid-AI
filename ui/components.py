"""
Reusable UI Components for DriverAid
"""

import streamlit as st


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


def get_metric_status(metric_name, value):
    """Get status indicator for metric value (matches expert system thresholds)."""
    if metric_name == "EAR":
        if value > 0.25:
            return "🟢 Alert", "#28a745"
        elif value > 0.20:
            return "🟡 Drowsy", "#ffc107"
        else:
            return "🔴 Critical", "#dc3545"
    elif metric_name == "Head Pitch":
        if abs(value) < 40:
            return "🟢 Normal", "#28a745"
        elif abs(value) < 48:
            return "🟡 Tilting", "#ffc107"
        else:
            return "🔴 Nodding", "#dc3545"
    return "", "#6c757d"


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


def render_sidebar():
    """Render the sidebar with system controls and info."""
    with st.sidebar:
        st.header("System Controls")
        
        if not st.session_state.system_initialized:
            if st.button("Initialize System", type="primary"):
                from ui.live_detection import initialize_system
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
