import streamlit as st
import cv2
import config
from fire_detection import FireDetector
import numpy as np
import time
import asyncio

# Thiết lập trang Streamlit
st.set_page_config(page_title="FireGuard IoT Dashboard", page_icon="🔥", layout="wide")

st.title("🔥 FireGuard IoT - Hệ Thống Giám Sát Báo Cháy AI")
st.markdown("---")

# Sidebar cấu hình
st.sidebar.header("⚙️ Cấu Hình Hệ Thống")
conf_threshold = st.sidebar.slider("Độ nhạy AI (Confidence)", 0.1, 1.0, config.CONFIDENCE_THRESHOLD)
frame_window = st.sidebar.number_input("Khung hình lọc nhiễu", 1, 50, config.FRAME_WINDOW)

# Khu vực hiển thị Video
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Stream & Detection")
    st_frame = st.empty() # Placeholder cho khung hình video

with col2:
    st.subheader("📊 Trạng Thái Hệ Thống")
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    st.subheader("📜 Lịch Sử Sự Kiện")
    event_log = st.empty()
    if 'events' not in st.session_state:
        st.session_state.events = []

# Khởi tạo detector
detector = FireDetector()

async def run_dashboard():
    # Sử dụng Webcam để demo (Nếu có ESP32-CAM thì thay bằng config.CAMERA_STREAM_URL)
    cap = cv2.VideoCapture(0) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể kết nối tới nguồn Camera!")
            break
            
        # AI Detection
        has_fire, ann_frame = detector.detect(frame)
        await detector.process_fire_logic(has_fire, frame)
        
        # Hiển thị Video
        st_frame.image(ann_frame, channels="BGR", use_column_width=True)
        
        # Cập nhật trạng thái
        if detector.is_alerting:
            status_placeholder.error("🔥 TRẠNG THÁI: PHÁT HIỆN HỎA HOẠN!")
            alert_placeholder.warning("⚠️ Đã kích hoạt Còi hú & Máy bơm!")
            
            # Lưu lịch sử
            new_event = f"🔥 {time.strftime('%H:%M:%S')} - Phát hiện cháy"
            if not st.session_state.events or st.session_state.events[0] != new_event:
                st.session_state.events.insert(0, new_event)
        else:
            status_placeholder.success("✅ TRẠNG THÁI: AN TOÀN")
            alert_placeholder.info("ℹ️ Hệ thống đang giám sát...")

        # Hiển thị lịch sử (5 sự kiện gần nhất)
        event_log.write("\n".join(st.session_state.events[:5]))
        
        # Sleep nhẹ để Streamlit render mượt
        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(run_dashboard())
    except Exception as e:
        pass
