🔥 FireGuard IoT - Hệ Thống Báo Cháy AI Thông Minh

Dự án IoT phát hiện lửa sớm sử dụng AI (YOLOv8), ESP32-CAM và điều khiển qua Web Dashboard (Streamlit) & Telegram.

🚀 Tính năng chính

Nhận diện lửa: Sử dụng mô hình YOLOv8 Nano, độ chính xác cao.

Giám sát thời gian thực: Xem video trực tiếp qua Web Dashboard.

Cảnh báo tức thì:

Hú còi/Bật bơm nước tự động (qua ESP32).

Gửi tin nhắn + Ảnh hiện trường qua Telegram.

Chống báo động giả: Logic xác nhận lửa liên tục trong 10 khung hình.

🛠️ Công nghệ sử dụng

AI: Python, Ultralytics YOLOv8, OpenCV.

Web App: Streamlit.

Phần cứng: ESP32-CAM, Relay, Máy bơm/Còi hú.

IoT: HTTP Request, Telegram Bot API.

⚙️ Cài đặt

Cài đặt thư viện:

pip install ultralytics streamlit opencv-python requests


Chạy hệ thống:

Kết nối ESP32 với WiFi.

Chạy Web Dashboard:

streamlit run dashboard.py


📸 Hình ảnh dự án

(Bạn có thể chèn ảnh chụp màn hình Dashboard hoặc ảnh phần cứng vào đây)

📞 Liên hệ

Dự án được thực hiện bởi: [CAO TIEN DUNG- DUNG HARRY]