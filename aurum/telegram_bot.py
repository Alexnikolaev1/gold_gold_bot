"""Telegram bot — handles /start, /status, /help via long polling."""

import json
import logging
import time
from pathlib import Path

import requests

from aurum.config import DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from aurum.data import fetch_realtime_data, is_comex_session_active
from aurum.execution.state import StateStore
from aurum.features import calculate_indicators
from aurum.ml import models_exist
from aurum.signals import process_signals

logger = logging.getLogger(__name__)

CHAT_ID_PATH = DATA_DIR / "telegram_chat_id.txt"
API = "https://api.telegram.org/bot{token}/{method}"


def _api(method: str, **kwargs) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        return {}
    url = API.format(token=TELEGRAM_BOT_TOKEN, method=method)
    resp = requests.post(url, json=kwargs, timeout=35)
    resp.raise_for_status()
    return resp.json()


def save_chat_id(chat_id: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_ID_PATH.write_text(str(chat_id), encoding="utf-8")


def load_chat_id() -> str:
    if CHAT_ID_PATH.exists():
        return CHAT_ID_PATH.read_text(encoding="utf-8").strip()
    return TELEGRAM_CHAT_ID


def send_message(chat_id: int | str, text: str, parse_mode: str = "HTML") -> bool:
    try:
        result = _api("sendMessage", chat_id=chat_id, text=text, parse_mode=parse_mode)
        return result.get("ok", False)
    except Exception as exc:
        logger.warning("sendMessage failed: %s", exc)
        return False


def _welcome_text() -> str:
    session_ok, session_msg = is_comex_session_active()
    return (
        "⚡ <b>AURUM Gold Bot</b> активен!\n\n"
        "Команды:\n"
        "/status — текущий сигнал и цена\n"
        "/balance — баланс и открытая позиция\n"
        "/help — справка\n\n"
        f"Сессия: {session_msg}\n"
        "Алерты о сделках приходят автоматически."
    )


def _status_text() -> str:
    if not models_exist():
        return "⚠️ ML-модели не обучены. Запустите терминал один раз."
    df = fetch_realtime_data()
    if df.empty or len(df) < 200:
        return "⚠️ Нет рыночных данных. Подождите открытия сессии."
    feat = calculate_indicators(df)
    result = process_signals(feat)
    price = float(feat.iloc[-1]["Close"])
    return (
        f"📊 <b>AURUM Status</b>\n"
        f"Цена: <b>${price:.2f}</b>\n"
        f"Сигнал: <b>{result.signal}</b>\n"
        f"Confidence: <b>{result.confidence * 100:.1f}%</b>\n"
        f"P(Buy): {result.conf_buy * 100:.1f}% | P(Sell): {result.conf_sell * 100:.1f}%\n"
        f"Правила: {result.rules_buy}/{result.rules_sell}"
    )


def _balance_text() -> str:
    state = StateStore().load()
    lines = [
        f"💰 Баланс: <b>${state.balance:,.2f}</b>",
        f"Daily PnL: <b>${state.daily_pnl:,.2f}</b>",
        f"Режим: <b>{state.mode}</b>",
    ]
    if state.open_trade:
        t = state.open_trade
        lines.append(
            f"\n📈 Открыта: <b>{t.side}</b> @ ${t.entry_price:.2f}\n"
            f"SL=${t.sl} | TP1=${t.tp1} | TP2=${t.tp2}"
        )
    else:
        lines.append("\nНет открытых позиций.")
    return "\n".join(lines)


def handle_command(text: str, chat_id: int) -> str:
    cmd = text.strip().split()[0].lower().split("@")[0]
    if cmd in ("/start", "/start@aurum_gold_bot"):
        save_chat_id(chat_id)
        return _welcome_text()
    if cmd == "/status":
        return _status_text()
    if cmd == "/balance":
        return _balance_text()
    if cmd == "/help":
        return (
            "<b>Команды AURUM Bot</b>\n"
            "/start — регистрация + приветствие\n"
            "/status — сигнал и цена золота\n"
            "/balance — баланс и позиция\n"
            "/help — эта справка\n\n"
            "Веб-терминал: открой URL Railway в браузере (не /start)."
        )
    return "Неизвестная команда. Напиши /help"


def process_update(update: dict) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    text = msg.get("text", "")
    chat_id = msg["chat"]["id"]
    if not text.startswith("/"):
        return
    reply = handle_command(text, chat_id)
    send_message(chat_id, reply)


def run_polling(offset: int = 0) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment")

    logger.info("Telegram bot polling started")

    while True:
        try:
            data = _api("getUpdates", offset=offset, timeout=30)
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                process_update(update)
        except requests.exceptions.Timeout:
            continue
        except Exception as exc:
            logger.exception("Polling error: %s", exc)
            time.sleep(5)
