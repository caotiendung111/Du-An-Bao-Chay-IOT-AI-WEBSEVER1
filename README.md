🔥 FireGuard IoT - Hệ Thống Báo Cháy AI Thông Minh

Một giải pháp an toàn, thông minh và phản ứng tức thì cho ngôi nhà của bạn.

📖 Giới thiệu

FireGuard IoT là dự án tích hợp sức mạnh của Trí tuệ nhân tạo (AI) và Internet vạn vật (IoT) để phát hiện sớm nguy cơ hỏa hoạn. Hệ thống không chỉ hú còi tại chỗ mà còn gửi hình ảnh hiện trường ngay lập tức đến điện thoại của chủ nhà, giúp bạn xử lý tình huống dù đang ở bất cứ đâu.

🚀 Tính năng nổi bật

👁️ Mắt thần AI: Sử dụng mô hình YOLOv8 Nano được huấn luyện chuyên sâu, nhận diện ngọn lửa chính xác kể cả lửa nhỏ (bật lửa, nến).

📹 Giám sát trực quan: Xem video trực tiếp (Live Stream) từ hiện trường qua giao diện Web Dashboard hiện đại.

⚡ Phản ứng tức thì:

Kích hoạt Còi hú & Máy bơm tự động qua ESP32.

Gửi Cảnh báo khẩn cấp + Ảnh chụp hiện trường qua Telegram.

🛡️ Chống báo động giả: Thuật toán thông minh lọc nhiễu (chỉ báo động khi phát hiện lửa liên tục trong 10 khung hình).

🛠️ Công nghệ sử dụng

Lĩnh vực

Công nghệ / Phần cứng

🧠 AI Core

Python, Ultralytics YOLOv8, OpenCV

💻 Web App

Streamlit (Giao diện điều khiển)

🤖 Phần cứng

ESP32-CAM, Module Relay, Còi hú, Máy bơm mini

🌐 IoT & Net

HTTP Request (REST API), Telegram Bot API

⚙️ Hướng dẫn cài đặt

1. Chuẩn bị môi trường

Cài đặt các thư viện Python cần thiết:

pip install ultralytics streamlit opencv-python requests


2. Thiết lập phần cứng

Nạp code cho ESP32-CAM (sử dụng Arduino IDE).

Đấu nối Relay và Còi báo động theo sơ đồ chân (GPIO 12, 13...).

3. Chạy hệ thống

Kết nối máy tính và ESP32 vào cùng một mạng WiFi.

Khởi chạy Web Dashboard:

streamlit run dashboard.py


📸 Hình ảnh dự án


📞 Liên hệ & Tác giả

Dự án được phát triển và duy trì bởi Cao Tiến Dũng (Dung Harry). Rất mong nhận được sự đóng góp ý kiến từ cộng đồng!

👨‍💻 Developer: Cao Tiến Dũng (Dung Harry)
🚀 Motto: "Code bằng đam mê, debug bằng cà phê ☕"

🤝 Kết nối với mình:

Facebook: Dung Harry (Harry Yiu Oi)

GitHub: caotiendung111

⭐ Nếu thấy dự án này hữu ích, hãy ủng hộ mình bằng cách bấm Star cho Repository này nhé! Cảm ơn bạn rất nhiều! ❤️