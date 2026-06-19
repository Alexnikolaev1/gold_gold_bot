#!/usr/bin/env python3
"""Telegram bot worker — responds to /start, /status, /balance."""

from aurum.logging_setup import setup_logging
from aurum.telegram_bot import run_polling

if __name__ == "__main__":
    setup_logging("aurum.telegram")
    run_polling()
