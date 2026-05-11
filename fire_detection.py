import cv2
from ultralytics import YOLO
import config
import requests
import asyncio
from telegram_alert import TelegramAlert

class FireDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = YOLO(model_path)
        self.alert_system = TelegramAlert()
        self.fire_counter = 0
        self.is_alerting = False

    def detect(self, frame):
        """Nhận diện lửa trong một khung hình"""
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        
        has_fire = False
        annotated_frame = frame.copy()
        
        # Duyệt qua các kết quả nhận diện
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                has_fire = True
            
            # Vẽ khung nhận diện lên ảnh
            annotated_frame = result.plot()
            
        return has_fire, annotated_frame

    async def process_fire_logic(self, has_fire, frame):
        """Xử lý logic lọc nhiễu và kích hoạt báo động"""
        if has_fire:
            self.fire_counter += 1
        else:
            self.fire_counter = max(0, self.fire_counter - 1)

        # Nếu phát hiện lửa liên tiếp đủ số khung hình quy định
        if self.fire_counter >= config.FRAME_WINDOW and not self.is_alerting:
            self.is_alerting = True
            print("🔥 PHÁT HIỆN CHÁY! Đang kích hoạt báo động...")
            
            # 1. Kích hoạt thiết bị ngoại vi qua ESP32
            self.trigger_hardware(on=True)
            
            # 2. Gửi thông báo Telegram
            await self.alert_system.send_alert_photo(frame)
            await self.alert_system.send_alert_message("⚡ Hệ thống đã tự động kích hoạt máy bơm và còi hú!")
            
        # Nếu lửa đã tắt hoàn toàn (sau một khoảng thời gian)
        elif self.fire_counter == 0 and self.is_alerting:
            self.is_alerting = False
            print("✅ Đã dập tắt lửa. Đang tắt hệ thống báo động...")
            self.trigger_hardware(on=False)

    def trigger_hardware(self, on=True):
        """Gửi lệnh HTTP tới ESP32 để điều khiển Relay"""
        state = "on" if on else "off"
        try:
            # Gửi lệnh còi hú
            requests.get(f"http://{config.ESP32_IP}/relay?pin={config.RELAY_ALARM_PIN}&state={state}", timeout=2)
            # Gửi lệnh máy bơm
            requests.get(f"http://{config.ESP32_IP}/relay?pin={config.RELAY_PUMP_PIN}&state={state}", timeout=2)
        except Exception as e:
            print(f"Không thể kết nối tới ESP32: {e}")

# Module test
async def main():
    detector = FireDetector()
    cap = cv2.VideoCapture(0) # Test với Webcam
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        has_fire, ann_frame = detector.detect(frame)
        await detector.process_fire_logic(has_fire, frame)
        
        cv2.imshow("FireGuard AI Detection", ann_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
