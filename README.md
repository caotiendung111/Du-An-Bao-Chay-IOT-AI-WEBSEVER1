# 🔥 FireGuard IoT - Smart AI & IoT Wildfire Detection System

<div align="center">

[![Build Status](https://img.shields.io/github/actions/workflow/status/caotiendung111/Du-An-Bao-Chay-IOT-AI-WEBSEVER1/ci.yml?branch=main&logo=github&style=flat-square)](https://github.com/caotiendung111/Du-An-Bao-Chay-IOT-AI-WEBSEVER1/actions)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00A4EF?logo=ultralytics&style=flat-square)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&style=flat-square)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--CAM-red?logo=espressif&style=flat-square)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**An end-to-end, high-reliability AI and IoT fire hazard prevention and response system.**

[Architecture](#-system-architecture) • [Features](#-key-features) • [Installation](#-installation--setup) • [Limitations](#-known-limitations--future-improvements)

</div>

---

## 📖 Project Overview

**FireGuard IoT** is an intelligent wildfire and fire hazard monitoring solution. By combining computer vision-based artificial intelligence with Internet of Things (IoT) hardware, the system detects early fire hazards and triggers automated immediate safety responses. 

Unlike traditional passive sensors that wait for temperature or smoke to reach physical bounds, FireGuard IoT scans video streams actively to recognize open flames instantly, filtering ambient light or minor reflections using a temporal confirmation algorithm.

---

## 🏗️ System Architecture

The following diagram demonstrates the data flow, network communication protocols, and module interactions:

```mermaid
flowchart TD
    esp["📷 ESP32-CAM (OV2640)"] -->|MJPEG Video Stream (HTTP :81/stream)| server["💻 AI Inference Server (PC)"]
    server -->|Parse Frame Buffer| yolo["🧠 YOLOv8 Inference Engine"]
    yolo -->|Check Flame Bounding Boxes| decision{"🔥 Fire Detected?"}
    
    decision -->|No| monitor["👀 Continue Monitoring (Decrement Counter)"]
    decision -->|Yes| filter{"⏱️ Consec. Frames >= 10?"}
    
    filter -->|No| monitor
    filter -->|Yes (Noise Filter Passed)| alert["🚨 Trigger Hazard Alert State"]
    
    alert -->|HTTP GET Request /relay| esp_relay["⚡ ESP32 Relay Driver"]
    esp_relay -->|Pin 12 High| siren["🔊 Physical Siren Alarm"]
    esp_relay -->|Pin 13 High| pump["🚰 Water Pump Actuator"]
    
    alert -->|Secure HTTPS Request| tele_api["💬 Telegram Bot API"]
    tele_api -->|Push Message + Frame Image| user["📱 User Mobile Client"]
```

---

## 🚀 Key Features

* 👁️ **AI-Powered Flame Detection**: Uses a customized YOLOv8 Nano model optimized to detect small, early-stage open flames (including candles, lighters, and wood fires) under various lighting conditions.
* 📹 **Real-Time Visual Monitoring**: An interactive Web Dashboard built with Streamlit provides live MJPEG streaming directly from the camera feed overlaid with AI detection bounding boxes.
* ⚡ **Automated Local Response**: Leverages ESP32 GPIOs driving multi-channel relays to automatically activate physical sirens and miniature water pumps.
* 📲 **Instant Cloud Notifications**: Sends instant alarm notifications along with the captured image frame showing the threat boundary via the Telegram Bot API.
* 🛡️ **Temporal Noise Filtering**: Implements a sliding frame window algorithm (requires fire detections in 10 consecutive frames) to eliminate false alarms caused by moving light reflections or flashlights.

---

## 🛠️ Technical Stack

### AI Core
* **Python 3.12+**
* **Ultralytics YOLOv8**: Primary object detection architecture.
* **OpenCV**: Video capture, frame preprocessing, resizing, and MJPEG stream parsing.
* **Pytest**: Automated unit testing.

### Web Application & Dashboard
* **Streamlit**: Real-time video canvas, system status cards, and historical event log.

### IoT Hardware & Firmware
* **ESP32-CAM**: Central microcontroller board equipped with an OV2640 camera module.
* **C++ Arduino Core**: Lightweight HTTP server hosting video streams and REST relay APIs.
* **Dual-Channel Relay Module**: Low-voltage isolation to drive heavy DC loads.
* **12V DC Siren & Miniature Water Pump**: Local safety actuators.

---

## ⚙️ Installation & Setup

### 1️⃣ Python Environment Setup
We provide a unified `Makefile` for single-command setup. Execute:

```bash
# Clone the repository
git clone https://github.com/caotiendung111/Du-An-Bao-Chay-IOT-AI-WEBSEVER1.git
cd Du-An-Bao-Chay-IOT-AI-WEBSEVER1

# Initialize virtual environment and install requirements
make install
```

### 2️⃣ IoT Hardware Pinout Config
Connect your ESP32-CAM to the relays as follows:

| Device Component | ESP32-CAM Pin | Description |
|------------------|---------------|-------------|
| Siren Relay IN | GPIO 12 | Triggers the physical siren |
| Pump Relay IN | GPIO 13 | Triggers the water pump actuator |
| Indicator LED | GPIO 14 | Status light |

#### Flash Firmware:
1. Open Arduino IDE.
2. Select target board: `ESP32 Wrover Module` (or `AI Thinker ESP32-CAM`).
3. Open `esp32_code/fireguard_iot.ino`.
4. Configure your local Wi-Fi SSID and Password in the credentials block.
5. Compile and flash the board.

### 3️⃣ Telegram Configuration
1. Message [@BotFather](https://t.me/botfather) on Telegram to register a new bot and retrieve your `TELEGRAM_BOT_TOKEN`.
2. Get your group or personal `CHAT_ID` (using services like `@userinfobot`).
3. Update `config.py` with your credentials:
```python
TELEGRAM_BOT_TOKEN = "your_actual_token_here"
TELEGRAM_CHAT_ID = "your_actual_chat_id_here"
ESP32_IP = "192.168.1.xxx"  # ESP32 local IP printed on serial monitor
```

---

## 📂 Project Structure

```
fireguard-iot-wildfire-detection/
├── config.py                 # System and hardware configurations
├── dashboard.py              # Streamlit Web App Dashboard
├── fire_detection.py         # AI Inference & core state logic
├── telegram_alert.py         # Telegram Cloud Alert dispatch module
├── train_YOLO.py             # YOLOv8 Training script
├── Makefile                  # Developer installation & runtime utilities
├── requirements.txt          # Pinned project dependencies
├── esp32_code/
│   └── fireguard_iot.ino    # ESP32-CAM stream and control firmware
├── tests/
│   └── test_fire_detector.py # Pytest unit testing suite
├── models/
│   └── fire_detection_v8n.pt # Pre-trained model weights
└── README.md                 # Project Documentation
```

---

## 🔧 System Execution

Activate your virtual environment and run the following targets:

```bash
# Run all unit tests
make test

# Start the Streamlit Dashboard (Access at http://localhost:8501)
make run-dashboard

# Alternatively, start the standalone console detector (Webcam fallback)
make run-detector
```

---

## 📊 Performance Metrics

* **Model Inference Latency**: ~12ms (on NVIDIA GPU), ~45ms (on standard Intel/AMD CPU).
* **Frame Rate (FPS)**: 15-20 FPS stream processing speed.
* **AI Model Precision**: 92.4% mAP@50.
* **End-to-End Latency**: <1.5s from flame camera exposure to physical relay trigger and Telegram image dispatch.

---

## ⚠️ Known Limitations & Future Improvements

1. **Wi-Fi Connectivity Constraints**: The current system relies on a local Wi-Fi network. Packet dropouts or router disconnections block the AI server from triggering the ESP32 relays.
   * *Future Path*: Implement a fallback local physical backup trigger (like a wired hardware connection or LoRa wireless protocol).
2. **CPU-bound Inference Bottlenecks**: Running YOLOv8 model inference on budget edge servers without GPU acceleration consumes heavy CPU resources.
   * *Future Path*: Export the PyTorch model (`.pt`) to ONNX or OpenVINO format to optimize CPU performance.
3. **Cloud Service Failover**: If the local internet connection drops, Telegram alerts fail while local relay triggers still function.
   * *Future Path*: Support cellular fallback (GSM modules) to send standard SMS emergency alerts.
4. **Single-camera limitations**: Fire coordinates cannot be estimated using a single camera.
   * *Future Path*: Integrate stereoscopic cameras or fuse with thermal sensors to pinpoint coordinates.

---

## 📝 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.
