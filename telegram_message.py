import os
import requests

def send_telegram_message(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    r = requests.post(url, data=payload)
    if r.status_code == 200:
        print("📲 Message envoyé sur Telegram.")
    else:
        print(f"❌ Erreur Telegram {r.status_code} : {r.text}")

if __name__ == "__main__":
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # ou mets directement ton token pour tester
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # ou "@NomDuCanal"
    MESSAGE = "Hello depuis XGenius ⚽📊"
    
    send_telegram_message(BOT_TOKEN, CHAT_ID, MESSAGE)
