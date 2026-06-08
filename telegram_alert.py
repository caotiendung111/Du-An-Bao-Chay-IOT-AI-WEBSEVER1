import telegram
import asyncio
import cv2
import config
from datetime import datetime

class TelegramAlert:
    def __init__(self):
        """Initializes the Telegram Bot using token and chat credentials from configuration."""
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.bot = telegram.Bot(token=self.token)

    async def send_alert_message(self, message):
        """Sends a plain text notification to the target Telegram chat."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except Exception as e:
            print(f"Telegram message dispatch error: {e}")
            return False

    async def send_alert_photo(self, frame, caption="🔥 ALERT: FIRE DETECTED!"):
        """Compresses the video frame to JPEG and dispatches it with a caption to Telegram."""
        try:
            # Compress frame buffer to JPEG in memory
            _, buffer = cv2.imencode('.jpg', frame)
            photo_bytes = buffer.tobytes()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_caption = f"{caption}\n⏰ Time: {timestamp}"
            
            await self.bot.send_photo(chat_id=self.chat_id, photo=photo_bytes, caption=full_caption)
            return True
        except Exception as e:
            print(f"Telegram photo dispatch error: {e}")
            return False
