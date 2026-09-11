"""
Telegram notification helper.

Requires env vars:
    TELEGRAM_BOT_TOKEN  — token from @BotFather
    TELEGRAM_CHAT_ID    — your personal chat id (get via /getUpdates)
"""
import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Log of every message_id we get back, so a bad message can be found and
# deleted later (Telegram's Bot API has no endpoint to list past messages —
# this is the only record that will ever exist).
SENT_LOG_FILE = Path(__file__).parent / "logs" / "telegram_sent.jsonl"


def _log_sent(message_id: int, chat_id: str, text: str):
    SENT_LOG_FILE.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "message_id": message_id,
        "preview": text[:120],
    }
    with SENT_LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping notification")
        return False

    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        message_id = resp.json().get("result", {}).get("message_id")
        if message_id is not None:
            _log_sent(message_id, chat_id, text)
            print(f"[telegram] Sent message_id={message_id}")
        return True
    except requests.RequestException as e:
        print(f"[telegram] Error sending message: {e}")
        return False
