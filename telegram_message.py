import requests

def send_telegram_message(bot_token: str, chat_id: str, message: str, parse_mode: str = None):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code == 200:
        print("📲 Message envoyé sur Telegram.")
    else:
        print(f"❌ Erreur Telegram {r.status_code} : {r.text}")
