# FireGuard IoT System Configuration

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# ESP32 Configuration
ESP32_IP = "192.168.1.100"  # ESP32-CAM local IP address on Wi-Fi network
CAMERA_STREAM_URL = f"http://{ESP32_IP}:81/stream"  # Video streaming endpoint

# AI Inference Configuration
MODEL_PATH = "best.pt"  # YOLOv8 pre-trained model weights path
CONFIDENCE_THRESHOLD = 0.5  # Object detection confidence threshold
FRAME_WINDOW = 10  # Number of consecutive frames detecting fire before triggering the alarm (noise filtering)

# Hardware Control Pins (Controlled via ESP32 HTTP API)
RELAY_ALARM_PIN = 12
RELAY_PUMP_PIN = 13
