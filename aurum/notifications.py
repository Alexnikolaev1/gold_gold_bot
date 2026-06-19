import logging

import requests

from aurum.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_telegram_alert(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Telegram alert failed: %s", exc)
        return False


def format_signal_alert(
    signal: str,
    price: float,
    confidence: float,
    tp1: float,
    tp2: float,
    sl: float,
    oz: float,
) -> str:
    emoji = "🟢" if signal == "BUY" else "🔴"
    return (
        f"{emoji} <b>AURUM SIGNAL — {signal}</b>\n"
        f"Цена: <b>${price:.2f}</b>\n"
        f"Confidence: <b>{confidence * 100:.1f}%</b>\n"
        f"SL: ${sl:.1f} | TP1: ${tp1:.1f} | TP2: ${tp2:.1f}\n"
        f"Размер: {oz:.1f} oz"
    )
