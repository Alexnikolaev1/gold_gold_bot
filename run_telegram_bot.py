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
from aurum.ml import ensure_models
from aurum.telegram_bot import run_polling

if __name__ == "__main__":
    setup_logging("aurum.telegram")
    print("[AURUM TG] Checking ML models...", flush=True)
    meta = ensure_models()
    if meta:
        print(
            f"[AURUM TG] ML ready: {meta.samples} bars | "
            f"buy={meta.buy_precision:.0%} sell={meta.sell_precision:.0%}",
            flush=True,
        )
    else:
        print("[AURUM TG] WARNING: ML training failed — /status and alerts disabled", flush=True)
    print("[AURUM TG] Entering polling loop...", flush=True)
    try:
        if not run_polling():
            print("[AURUM TG] FATAL: bot stopped (see errors above)", flush=True)
            sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
