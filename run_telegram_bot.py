#!/usr/bin/env python3
"""Telegram bot worker — commands + 24/7 entry alerts (no browser needed)."""

import os
import sys

print("[AURUM TG] run_telegram_bot.py starting...", flush=True)
token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not token:
    print("[AURUM TG] FATAL: TELEGRAM_BOT_TOKEN not set!", flush=True)
    sys.exit(1)
print(f"[AURUM TG] Token present (len={len(token)})", flush=True)

from aurum.logging_setup import setup_logging
from aurum.telegram_bot import run_polling

if __name__ == "__main__":
    setup_logging("aurum.telegram")
    print("[AURUM TG] Entering polling loop...", flush=True)
    if not run_polling():
        print("[AURUM TG] FATAL: bot stopped (see errors above)", flush=True)
        sys.exit(1)
