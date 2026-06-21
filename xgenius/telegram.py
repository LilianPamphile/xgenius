# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List
import requests

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

MAX_TELEGRAM_LEN = 3900


def split_message(text: str, limit: int = MAX_TELEGRAM_LEN) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in text.split("\n"):
        add_len = len(paragraph) + 1
        if current and current_len + add_len > limit:
            chunks.append("\n".join(current).strip())
            current = [paragraph]
            current_len = add_len
        else:
            current.append(paragraph)
            current_len += add_len
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def send_telegram_message(message: str, dry_run: bool = False) -> None:
    if dry_run:
        print("\n--- TELEGRAM DRY RUN ---")
        print(message)
        print("--- END DRY RUN ---\n")
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in split_message(message):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        r = requests.post(url, data=payload, timeout=30)
        if r.status_code >= 300:
            raise RuntimeError(f"Erreur Telegram {r.status_code}: {r.text}")
        print("Message Telegram envoyé.")
