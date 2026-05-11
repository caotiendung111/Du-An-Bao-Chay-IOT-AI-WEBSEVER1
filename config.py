# Cấu hình hệ thống FireGuard IoT

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# ESP32 Configuration
ESP32_IP = "192.168.1.100"  # Địa chỉ IP của ESP32-CAM khi kết nối WiFi
CAMERA_STREAM_URL = f"http://{ESP32_IP}:81/stream"  # Luồng stream từ ESP32-CAM

# AI Configuration
MODEL_PATH = "best.pt"  # Đường dẫn tới file model YOLOv8
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy nhận diện lửa
FRAME_WINDOW = 10  # Số khung hình liên tiếp phát hiện lửa để kích hoạt báo động

# Hardware Control Pins (Thông qua API trên ESP32)
RELAY_ALARM_PIN = 12
RELAY_PUMP_PIN = 13
