"""Telegram bot — /start, /status, /balance via long polling + background entry alerts."""

import logging
import os
import threading
import time

import requests

from aurum.config import DATA_DIR, DEFAULT_DEPOSIT, DEFAULT_RISK_PCT, TELEGRAM_ALERTS_ENABLED, TELEGRAM_CHAT_ID, TOTAL_RULES
from aurum.data import fetch_realtime_data, is_comex_session_active
from aurum.execution.state import StateStore
from aurum.features import calculate_indicators
from aurum.ml import models_exist
from aurum.risk import calculate_trade_levels, position_size_oz
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


def _proxies() -> dict[str, str] | None:
    proxy = os.getenv("TELEGRAM_PROXY", "").strip() or os.getenv("HTTPS_PROXY", "").strip()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _request_kwargs(read_timeout: float) -> dict:
    kwargs: dict = {"timeout": (5, read_timeout)}
    proxies = _proxies()
    if proxies:
        kwargs["proxies"] = proxies
    return kwargs


def _network_hint(exc: Exception) -> str:
    msg = str(exc).lower()
    if "getaddrinfo failed" in msg or "name resolution" in msg or "failed to resolve" in msg:
        return (
            "api.telegram.org недоступен (DNS/блокировка). "
            "Локально: включите VPN или задайте TELEGRAM_PROXY=socks5://127.0.0.1:1080"
        )
    if _proxies():
        return "Проверьте TELEGRAM_PROXY / HTTPS_PROXY — прокси не отвечает."
    return "Проверьте интернет и доступ к api.telegram.org."


def _api(method: str, *, read_timeout: float = 35, **params) -> dict:
    token = _token()
    if not token:
        return {"ok": False}
    url = f"https://api.telegram.org/bot{token}/{method}"
    resp = requests.get(url, params=params, **_request_kwargs(read_timeout))
    resp.raise_for_status()
    return resp.json()


def _api_post(method: str, **payload) -> dict:
    token = _token()
    url = f"https://api.telegram.org/bot{token}/{method}"
    resp = requests.post(url, json=payload, **_request_kwargs(15))
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
    alerts = "включены" if TELEGRAM_ALERTS_ENABLED else "выключены"
    return (
        "⚡ <b>AURUM Gold Bot</b> активен!\n\n"
        "Команды:\n"
        "/status — сигнал, SL/TP\n"
        "/balance — баланс и позиция\n"
        "/help — справка\n\n"
        f"🔔 Алерты на вход: <b>{alerts}</b> (24/7, без браузера)\n"
        f"Сессия: {session_msg}"
    )


def _status_text() -> str:
    if not models_exist():
        return (
            "⚠️ ML-модели не обучены.\n"
            "Перезапустите сервер — бот обучит их автоматически (2–3 мин).\n"
            "Или откройте веб-терминал один раз."
        )
    df = fetch_realtime_data()
    if df.empty or len(df) < 200:
        return "⚠️ Нет данных по золоту. Подождите открытия сессии."
    feat = calculate_indicators(df)
    result = process_signals(feat)
    price = float(feat.iloc[-1]["Close"])
    lines = [
        f"📊 <b>AURUM Status</b>",
        f"Цена: <b>${price:.2f}</b>",
        f"Сигнал: <b>{result.signal}</b>",
        f"Confidence: <b>{result.confidence * 100:.1f}%</b>",
        f"Правила: {result.rules_buy}/{result.rules_sell}",
    ]
    if result.signal in ("BUY", "SELL"):
        levels = calculate_trade_levels(feat, result.signal)
        rules = result.rules_buy if result.signal == "BUY" else result.rules_sell
        oz = position_size_oz(DEFAULT_DEPOSIT, DEFAULT_RISK_PCT, levels.risk_per_oz)
        lines.extend(
            [
                f"SL: <b>${levels.sl:.1f}</b> | TP1: <b>${levels.tp1:.1f}</b> | TP2: <b>${levels.tp2:.1f}</b>",
                f"R:R → TP1: <b>{levels.risk_reward_tp1:.2f}</b> | Консенсус: {rules}/{TOTAL_RULES}",
                f"Размер (~${DEFAULT_DEPOSIT:.0f}, {DEFAULT_RISK_PCT}%): <b>{oz:.2f} oz</b>",
            ]
        )
        from aurum.signal_alerts import evaluate_entry_opportunity

        opp = evaluate_entry_opportunity()
        lines.append(
            "\n✅ <b>Условия входа выполнены</b> — алерт отправлен"
            if opp and opp.side == result.signal
            else "\n⏳ Вход: ждём confidence + консенсус + R:R"
        )
    return "\n".join(lines)


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

    proxy = _proxies()
    if proxy:
        _log(f"Using proxy: {proxy['https']}")
    _log("Connecting to api.telegram.org (getMe)...")

    try:
        me = _api("getMe", read_timeout=10)
    except Exception as exc:
        raise RuntimeError(f"{_network_hint(exc)} ({exc})") from exc

    if not me.get("ok"):
        raise RuntimeError(f"Invalid bot token: {me}")
    username = me["result"].get("username", "?")
    _log(f"Bot verified: @{username}")

    wh = _api("deleteWebhook", drop_pending_updates=True)
    _log(f"Webhook cleared: {wh.get('ok')}")


def run_polling() -> bool:
    try:
        _bootstrap()
    except Exception as exc:
        _log(f"Bootstrap failed: {exc}")
        return False

    stop_alerts = threading.Event()
    if TELEGRAM_ALERTS_ENABLED:
        from aurum.signal_alerts import run_alert_loop

        threading.Thread(
            target=run_alert_loop,
            args=(stop_alerts, _log),
            name="aurum-signal-alerts",
            daemon=True,
        ).start()

    offset = _load_offset()
    _log(f"Polling started (offset={offset})")

    while True:
        try:
            data = _api("getUpdates", offset=offset, timeout=30, read_timeout=40)
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
