import streamlit as st
import cv2
import config
from fire_detection import FireDetector
import time
import asyncio

# Configure Streamlit UI page
st.set_page_config(page_title="FireGuard IoT Dashboard", page_icon="🔥", layout="wide")

st.title("🔥 FireGuard IoT - AI Hazard Monitoring Dashboard")
st.markdown("---")

# Sidebar configurations
st.sidebar.header("⚙️ System Configuration")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, config.CONFIDENCE_THRESHOLD)
frame_window = st.sidebar.number_input("Consecutive Frame Noise Filter", 1, 50, config.FRAME_WINDOW)

# Configure layout columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Stream & Detection")
    st_frame = st.empty()  # Video frame placeholder

with col2:
    st.subheader("📊 System Status")
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    st.subheader("📜 Event Log")
    event_log = st.empty()
    if 'events' not in st.session_state:
        st.session_state.events = []

# Instantiate detector
detector = FireDetector()

async def run_dashboard():
    # Capture live camera source (falls back to index 0 - webcam)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to connect to the Camera stream source!")
            break
            
        # Run AI detection on frame
        has_fire, ann_frame = detector.detect(frame)
        await detector.process_fire_logic(has_fire, frame)
        
        # Render image frame on dashboard
        st_frame.image(ann_frame, channels="BGR", use_column_width=True)
        
        # Update system status cards dynamically
        if detector.is_alerting:
            status_placeholder.error("🔥 STATUS: FIRE DETECTED!")
            alert_placeholder.warning("⚠️ Warning: Siren and Pump actuators are active!")
            
            # Log event timestamp
            new_event = f"🔥 {time.strftime('%H:%M:%S')} - Hazard Detected"
            if not st.session_state.events or st.session_state.events[0] != new_event:
                st.session_state.events.insert(0, new_event)
        else:
            status_placeholder.success("✅ STATUS: SAFE")
            alert_placeholder.info("ℹ️ Monitoring active - no threats detected.")

        # Print the 5 most recent logged events
        event_log.write("\n".join(st.session_state.events[:5]))
        
        # Non-blocking pause for Streamlit UI refresh
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(run_dashboard())
    except Exception as e:
        pass
