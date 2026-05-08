# 🔥 FireGuard IoT - Hệ Thống Báo Cháy AI Thông Minh

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00A4EF?style=flat-square&logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ESP32](https://img.shields.io/badge/ESP32-E7352C?style=flat-square&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](#)

**Giải pháp an toàn thông minh tích hợp AI & IoT cho phát hiện và ngăn chặn hỏa hoạn**

[Demo](#-cách-sử-dụng) • [Tài Liệu](#-hướng-dẫn-cài-đặt) • [Liên Hệ](#-liên-hệ--tác-giả)

</div>

---

## 📖 Giới Thiệu

**FireGuard IoT** là một hệ thống báo cháy tích hợp sức mạnh của **Trí tuệ nhân tạo (AI)** và **Internet vạn vật (IoT)** để phát hiện sớm nguy cơ hỏa hoạn. 

Hệ thống không chỉ nhận diện lửa mà còn:
- 🎯 **Phản ứng tức thì** kích hoạt các thiết bị an toàn
- 📱 **Cảnh báo khẩn cấp** gửi ngay đến người dùng
- 🛡️ **Chống báo động giả** với thuật toán lọc nhiễu thông minh
- 🔴 **Mở rộng dễ dàng** cho các ứng dụng khác

---

## 🚀 Tính Năng Nổi Bật

| Tính Năng | Mô Tả | Công Nghệ |
|-----------|-------|----------|
| 👁️ **Mắt thần AI** | Nhận diện lửa chính xác kể cả lửa nhỏ (bật lửa, nến) | YOLOv8 Nano + OpenCV |
| 📹 **Giám sát trực quan** | Xem video trực tiếp (Live Stream) từ hiện trường | Streamlit Web Dashboard |
| ⚡ **Phản ứng tức thì** | Kích hoạt Còi hú & Máy bơm tự động | ESP32 + Relay Module |
| 📲 **Cảnh báo khẩn cấp** | Gửi cảnh báo + ảnh chụp qua Telegram | Telegram Bot API |
| 🛡️ **Chống báo động giả** | Lọc nhiễu thông minh (báo động khi phát hiện 10 khung liên tiếp) | Custom Algorithm |

---

## 🛠️ Công Nghệ Sử Dụng

### 🧠 AI Core
```
Python 3.8+
├── Ultralytics YOLOv8 (Model nhận diện lửa)
├── OpenCV (Xử lý hình ảnh video)
└── NumPy/Pandas (Xử lý dữ liệu)
```

### 💻 Web Application
```
Streamlit
├── Real-time Video Streaming
├── Dashboard điều khiển
├── Lịch sử sự kiện
└── Thiết lập cảnh báo
```

### 🤖 Phần Cứng IoT
```
ESP32-CAM
├── Camera OV2640 (1600x1200)
├── Module Relay 4 Channel
├── Còi hú DC 12V
├── Máy bơm mini 12V
└── Cảm biến nhiệt độ (tùy chọn)
```

### 🌐 Kết Nối & API
```
HTTP REST API
├── ESP32 → Server (gửi frame video)
├── Server → Telegram Bot (cảnh báo)
└── Server → Web Dashboard (live stream)
```

---

## ⚙️ Hướng Dẫn Cài Đặt

### 1️⃣ Chuẩn Bị Môi Trường

```bash
# Clone repository
git clone https://github.com/caotiendung111/Du-An-Bao-Chay-IOT-AI-WEBSEVER1.git
cd Du-An-Bao-Chay-IOT-AI-WEBSEVER1

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện Python
pip install -r requirements.txt
```

**Requirements:**
```bash
pip install ultralytics streamlit opencv-python requests pillow python-telegram-bot
```

### 2️⃣ Thiết Lập Phần Cứng

#### GPIO Pinout (ESP32-CAM)
| Thiết Bị | GPIO Pin | Mô Tả |
|----------|----------|-------|
| Relay Còi Hú | GPIO 12 | Kích hoạt còi báo động |
| Relay Máy Bơm | GPIO 13 | Kích hoạt hệ thống tưới nước |
| LED Chỉ Báo | GPIO 14 | LED trạng thái |
| Cảm Biến Nhiệt | GPIO 35 | Cảm biến (ADC) |

#### Nạp Code cho ESP32-CAM
1. Mở Arduino IDE
2. Chọn Board: `ESP32 Wrover Module`
3. Nạp code từ thư mục `esp32_code/`
4. Cấu hình WiFi trong code

### 3️⃣ Cấu Hình Telegram Bot

1. Tạo Telegram Bot với [@BotFather](https://t.me/botfather)
2. Lấy `TELEGRAM_BOT_TOKEN` và `CHAT_ID`
3. Thêm vào `config.py`:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
ESP32_IP = "192.168.1.100"  # IP của ESP32
```

### 4️⃣ Chạy Hệ Thống

```bash
# Kết nối máy tính và ESP32 vào cùng mạng WiFi
# Khởi chạy Web Dashboard
streamlit run dashboard.py

# Truy cập: http://localhost:8501
```

---

## 📁 Cấu Trúc Dự Án

```
Du-An-Bao-Chay-IOT-AI-WEBSEVER1/
├── dashboard.py                 # Streamlit Web App
├── fire_detection.py           # Module AI nhận diện lửa
├── telegram_alert.py           # Module gửi cảnh báo Telegram
├── config.py                   # Thiết lập cấu hình
├── esp32_code/                 # Code cho ESP32-CAM
│   ├── camera_streaming.ino
│   └── relay_control.ino
├── models/                     # Pre-trained YOLOv8 models
│   └── fire_detection_v8n.pt
├── requirements.txt            # Dependencies
├── README.md                   # Tài liệu này
└── LICENSE
```

---

## 🔧 Cách Sử Dụng

### Bắt Đầu Hệ Thống

```bash
# Terminal 1: Chạy Server nhận diện lửa
python fire_detection.py

# Terminal 2: Chạy Web Dashboard
streamlit run dashboard.py
```

### Giao Diện Dashboard
- **Live Stream**: Xem video trực tiếp từ ESP32-CAM
- **Trạng Thái Hệ Thống**: Hiển thị tình trạng kết nối
- **Lịch Sự Kiện**: Danh sách các lần phát hiện lửa
- **Cài Đặt**: Điều chỉnh độ nhạy AI, thời gian chờ, v.v.

---

## 🎓 Kỹ Thuật Chính

### 1. Nhận Diện Lửa với YOLOv8
```python
from ultralytics import YOLO
model = YOLO('fire_detection_v8n.pt')
results = model(frame)
```

### 2. Lọc Nhiễu - Thuật Toán Logic
- Yêu cầu phát hiện lửa liên tiếp ≥ 10 khung hình
- Giảm báo động giả từ ánh sáng, phản chiếu, v.v.

### 3. Gửi Cảnh Báo Telegram
```python
bot.send_photo(chat_id, photo, caption="🔥 Phát hiện lửa!")
bot.send_message(chat_id, "Thời gian: " + timestamp)
```

### 4. Điều Khiển ESP32 qua HTTP
```python
requests.get(f'http://{ESP32_IP}/relay/on?pin=12')
```

---

## 📊 Hiệu Suất

| Chỉ Số | Giá Trị |
|--------|--------|
| **FPS (Frame/sec)** | 15-20 FPS |
| **Độ Chính Xác AI** | >92% (tập test) |
| **Thời Gian Phản Ứng** | <2 giây |
| **Tiêu Thụ Điện** | ~5-8W |
| **Độ Phân Giải Video** | 1280x960 @ 15FPS |

---

## ⚠️ Lưu Ý Quan Trọng

- 🔐 **Bảo Mật**: Cấu hình WiFi an toàn, không share mã nguồn công khai
- ⚡ **Điện Áp**: Đảm bảo cấp điện 12V ổn định cho relay & các thiết bị
- 🌡️ **Nhiệt Độ**: Đặt ESP32-CAM ở nơi thoáng mát, tránh quá nóng
- 🔌 **Kết Nối**: Kiểm tra kết nối WiFi trước khi vận hành

---

## 🤝 Đóng Góp

Chúng tôi rất hoan nghênh các đóng góp từ cộng đồng!

1. Fork dự án
2. Tạo branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📝 License

Dự án này được cấp phép dưới **MIT License** - xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 📸 Hình Ảnh & Demo

| Mô Tả | Hình Ảnh |
|-------|---------|
| Dashboard Streamlit | ![Dashboard](docs/dashboard.png) |
| Phát Hiện Lửa | ![Fire Detection](docs/fire_detection.png) |
| Sơ Đồ Kết Nối | ![Wiring](docs/wiring_diagram.png) |

*Thêm hình ảnh vào thư mục `docs/`*

---

## 🐛 Gỡ Lỗi & Troubleshooting

### ❌ Vấn Đề: ESP32 không kết nối được
```
✅ Giải pháp:
- Kiểm tra WiFi SSID & Password trong code
- Chắc chắn ESP32 & PC cùng mạng
- Reset ESP32 (nhấn nút RESET)
```

### ❌ Vấn Đề: Model YOLOv8 không load được
```
✅ Giải pháp:
- Cài đặt lại: pip install --upgrade ultralytics
- Kiểm tra file model: models/fire_detection_v8n.pt
- Tải model từ Hugging Face nếu cần
```

### ❌ Vấn Đề: Cảnh báo Telegram không gửi
```
✅ Giải pháp:
- Kiểm tra TELEGRAM_BOT_TOKEN & CHAT_ID
- Đảm bảo có kết nối internet
- Kiểm tra firewall cho phép Telegram
```

---

## 📚 Tài Liệu Tham Khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Streamlit Official Docs](https://docs.streamlit.io/)
- [ESP32-CAM Guide](https://randomnerdtutorials.com/esp32-cam-video-streaming-web-server/)
- [Telegram Bot API](https://core.telegram.org/bots/api)

---

## 📞 Liên Hệ & Tác Giả

### 👨‍💻 Nhà Phát Triển
**Cao Tiến Dũng** (Dung Harry)

### 🚀 Motto
> "Code bằng đam mê, debug bằng cà phê ☕"

### 🤝 Kết Nối Với Mình

| Nền Tảng | Thông Tin |
|----------|----------|
| 📘 Facebook | [Dung Harry (Harry Yiu Oi)](https://facebook.com/dungh) |
| 🐙 GitHub | [@caotiendung111](https://github.com/caotiendung111) |
| 💼 LinkedIn | [Cao Tiến Dũng](https://linkedin.com/in/caotiendung) |
| 📧 Email | caotiendung111@email.com |

---

## ⭐ Hỗ Trợ Dự Án

Nếu thấy dự án này hữu ích, hãy ủng hộ mình bằng cách:

- ⭐ **Bấm Star** cho Repository này
- 🍴 **Fork** để phát triển thêm
- 💬 **Chia sẻ** với cộng đồng
- 💖 **Donate** để hỗ trợ phát triển tiếp

---

<div align="center">

**Made with ❤️ by Cao Tiến Dũng**

![GitHub followers](https://img.shields.io/github/followers/caotiendung111?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/caotiendung111/Du-An-Bao-Chay-IOT-AI-WEBSEVER1?style=social)

*Cảm ơn bạn rất nhiều! 🙏*

</div>