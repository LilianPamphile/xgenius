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
