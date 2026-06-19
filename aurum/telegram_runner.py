"""Start Telegram bot in a background thread (single Railway service)."""

import logging
import os
import threading

_lock = threading.Lock()
_started = False
_thread: threading.Thread | None = None

logger = logging.getLogger(__name__)


def start_telegram_bot_background() -> bool:
    """Launch polling once per process. Safe to call from Streamlit."""
    global _started, _thread
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")
        return False

    with _lock:
        if _started and _thread and _thread.is_alive():
            return True

        from aurum.logging_setup import setup_logging
        from aurum.telegram_bot import run_polling

        setup_logging("aurum.telegram")
        _thread = threading.Thread(
            target=run_polling,
            name="aurum-telegram",
            daemon=True,
        )
        _thread.start()
        _started = True
        logger.info("Telegram bot background thread started")
        print("[AURUM] Telegram bot thread started", flush=True)
        return True


def telegram_bot_alive() -> bool:
    return _thread is not None and _thread.is_alive()
