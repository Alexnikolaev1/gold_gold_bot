import logging

import requests

from aurum.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from aurum.telegram_bot import load_chat_id, send_message

logger = logging.getLogger(__name__)


def send_telegram_alert(message: str) -> bool:
    chat_id = load_chat_id() or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    return send_message(chat_id, message)


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
