import cv2
from ultralytics import YOLO
import config
import requests
import asyncio
from telegram_alert import TelegramAlert

class FireDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        """Initializes the YOLOv8 model, alert system, and status counters."""
        self.model = YOLO(model_path)
        self.alert_system = TelegramAlert()
        self.fire_counter = 0
        self.is_alerting = False

    def detect(self, frame):
        """Performs YOLOv8 object detection on a single frame."""
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        
        has_fire = False
        annotated_frame = frame.copy()
        
        # Parse detection coordinates and plot bounding boxes
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                has_fire = True
            
            # Draw prediction bounding box on the image frame
            annotated_frame = result.plot()
            
        return has_fire, annotated_frame

    async def process_fire_logic(self, has_fire, frame):
        """Applies a consecutive-frame noise filter and triggers alerts/actuators."""
        if has_fire:
            self.fire_counter += 1
        else:
            self.fire_counter = max(0, self.fire_counter - 1)

        # Alarm trigger if fire is detected for >= FRAME_WINDOW consecutive frames
        if self.fire_counter >= config.FRAME_WINDOW and not self.is_alerting:
            self.is_alerting = True
            print("🔥 HAZARD DETECTED! Triggering alarm and actuators...")
            
            # 1. Trigger hardware relays on ESP32
            self.trigger_hardware(on=True)
            
            # 2. Dispatch Telegram alert notification with snapshot
            await self.alert_system.send_alert_photo(frame)
            await self.alert_system.send_alert_message("⚡ Automated response system activated: Siren and Pump are ON!")
            
        # Alarm reset when fire signal goes to 0
        elif self.fire_counter == 0 and self.is_alerting:
            self.is_alerting = False
            print("✅ Hazard cleared. Turning off alarm and actuators...")
            self.trigger_hardware(on=False)

    def trigger_hardware(self, on=True):
        """Sends HTTP requests to the ESP32 API to control Relay pins."""
        state = "on" if on else "off"
        try:
            # Trigger Alarm/Siren relay
            requests.get(f"http://{config.ESP32_IP}/relay?pin={config.RELAY_ALARM_PIN}&state={state}", timeout=2)
            # Trigger Water Pump relay
            requests.get(f"http://{config.ESP32_IP}/relay?pin={config.RELAY_PUMP_PIN}&state={state}", timeout=2)
        except Exception as e:
            print(f"ESP32 hardware control connection failed: {e}")

# Standalone execution loop (for local testing/webcam usage)
async def main():
    detector = FireDetector()
    cap = cv2.VideoCapture(0)  # Use index 0 for system default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        has_fire, ann_frame = detector.detect(frame)
        await detector.process_fire_logic(has_fire, frame)
        
        cv2.imshow("FireGuard AI Detection", ann_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
