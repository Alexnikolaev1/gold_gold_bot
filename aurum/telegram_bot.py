"""Telegram bot — /start, /status, /balance via long polling."""

import logging
import os
import sys
import time

import requests

from aurum.config import DATA_DIR, TELEGRAM_CHAT_ID
from aurum.data import fetch_realtime_data, is_comex_session_active
from aurum.execution.state import StateStore
from aurum.features import calculate_indicators
from aurum.ml import models_exist
from aurum.signals import process_signals

logger = logging.getLogger(__name__)

CHAT_ID_PATH = DATA_DIR / "telegram_chat_id.txt"
OFFSET_PATH = DATA_DIR / "telegram_offset.txt"


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _log(msg: str) -> None:
    """Railway shows stdout — дублируем важные события."""
    logger.info(msg)
    print(f"[AURUM TG] {msg}", flush=True)


def _api(method: str, **params) -> dict:
    token = _token()
    if not token:
        return {"ok": False}
    url = f"https://api.telegram.org/bot{token}/{method}"
    resp = requests.get(url, params=params, timeout=35)
    resp.raise_for_status()
    return resp.json()


def _api_post(method: str, **payload) -> dict:
    token = _token()
    url = f"https://api.telegram.org/bot{token}/{method}"
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def save_chat_id(chat_id: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_ID_PATH.write_text(str(chat_id), encoding="utf-8")
    _log(f"Chat ID saved: {chat_id}")


def load_chat_id() -> str:
    if CHAT_ID_PATH.exists():
        return CHAT_ID_PATH.read_text(encoding="utf-8").strip()
    return TELEGRAM_CHAT_ID


def _load_offset() -> int:
    if OFFSET_PATH.exists():
        try:
            return int(OFFSET_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return 0


def _save_offset(offset: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(str(offset), encoding="utf-8")


def send_message(chat_id: int | str, text: str, parse_mode: str = "HTML") -> bool:
    try:
        result = _api_post("sendMessage", chat_id=chat_id, text=text, parse_mode=parse_mode)
        ok = result.get("ok", False)
        if not ok:
            _log(f"sendMessage failed: {result}")
        return ok
    except Exception as exc:
        _log(f"sendMessage error: {exc}")
        return False


def _welcome_text() -> str:
    _, session_msg = is_comex_session_active()
    return (
        "⚡ <b>AURUM Gold Bot</b> активен!\n\n"
        "Команды:\n"
        "/status — сигнал и цена\n"
        "/balance — баланс и позиция\n"
        "/help — справка\n\n"
        f"Сессия: {session_msg}"
    )


def _status_text() -> str:
    if not models_exist():
        return "⚠️ ML-модели не обучены. Открой веб-терминал один раз."
    df = fetch_realtime_data()
    if df.empty or len(df) < 200:
        return "⚠️ Нет данных по золоту. Подождите открытия сессии."
    feat = calculate_indicators(df)
    result = process_signals(feat)
    price = float(feat.iloc[-1]["Close"])
    return (
        f"📊 <b>AURUM Status</b>\n"
        f"Цена: <b>${price:.2f}</b>\n"
        f"Сигнал: <b>{result.signal}</b>\n"
        f"Confidence: <b>{result.confidence * 100:.1f}%</b>\n"
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
            f"\n📈 {t.side} @ ${t.entry_price:.2f}\nSL=${t.sl} TP1=${t.tp1} TP2=${t.tp2}"
        )
    else:
        lines.append("\nНет открытых позиций.")
    return "\n".join(lines)


def handle_command(text: str, chat_id: int) -> str:
    cmd = text.strip().split()[0].lower().split("@")[0]
    if cmd == "/start":
        save_chat_id(chat_id)
        return _welcome_text()
    if cmd == "/status":
        return _status_text()
    if cmd == "/balance":
        return _balance_text()
    if cmd == "/help":
        return (
            "<b>AURUM Bot</b>\n"
            "/start — регистрация\n"
            "/status — сигнал\n"
            "/balance — баланс\n"
            "/help — справка"
        )
    return "Неизвестная команда. /help"


def process_update(update: dict) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    text = msg.get("text", "")
    chat_id = msg["chat"]["id"]
    if not text.startswith("/"):
        return
    _log(f"Command from {chat_id}: {text}")
    reply = handle_command(text, chat_id)
    if send_message(chat_id, reply):
        _log(f"Replied to {chat_id}")
    else:
        _log(f"Failed to reply to {chat_id}")


def _bootstrap() -> None:
    """Verify token, remove webhook (blocks polling), log bot name."""
    token = _token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    me = _api("getMe")
    if not me.get("ok"):
        raise RuntimeError(f"Invalid bot token: {me}")
    username = me["result"].get("username", "?")
    _log(f"Bot verified: @{username}")

    wh = _api("deleteWebhook", drop_pending_updates=True)
    _log(f"Webhook cleared: {wh.get('ok')}")


def run_polling() -> None:
    try:
        _bootstrap()
    except Exception as exc:
        _log(f"Bootstrap failed: {exc}")
        return

    offset = _load_offset()
    _log(f"Polling started (offset={offset})")

    while True:
        try:
            data = _api("getUpdates", offset=offset, timeout=30)
            if not data.get("ok"):
                _log(f"getUpdates error: {data}")
                time.sleep(5)
                continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                _save_offset(offset)
                process_update(update)
        except requests.exceptions.ReadTimeout:
            continue
        except requests.exceptions.ConnectionError as exc:
            _log(f"Connection error: {exc}")
            time.sleep(10)
        except Exception as exc:
            _log(f"Polling error: {exc}")
            time.sleep(5)
