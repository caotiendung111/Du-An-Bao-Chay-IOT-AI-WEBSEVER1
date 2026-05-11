import telegram
import asyncio
import cv2
import config
from datetime import datetime

class TelegramAlert:
    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.bot = telegram.Bot(token=self.token)

    async def send_alert_message(self, message):
        """Gửi tin nhắn văn bản tới Telegram"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except Exception as e:
            print(f"Lỗi gửi tin nhắn Telegram: {e}")
            return False

    async def send_alert_photo(self, frame, caption="🔥 CẢNH BÁO: PHÁT HIỆN HỎA HOẠN!"):
        """Gửi ảnh chụp màn hình lúc phát hiện cháy tới Telegram"""
        try:
            # Chuyển đổi frame sang định dạng jpg trong bộ nhớ
            _, buffer = cv2.imencode('.jpg', frame)
            photo_bytes = buffer.tobytes()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_caption = f"{caption}\n⏰ Thời gian: {timestamp}"
            
            await self.bot.send_photo(chat_id=self.chat_id, photo=photo_bytes, caption=full_caption)
            return True
        except Exception as e:
            print(f"Lỗi gửi ảnh Telegram: {e}")
            return False

# Ví dụ test nhanh:
# if __name__ == "__main__":
#     alert = TelegramAlert()
#     asyncio.run(alert.send_alert_message("🔔 FireGuard IoT System is Starting..."))
