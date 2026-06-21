#!/usr/bin/env python3
"""
Railway entry point — запускает Telegram-бота и Streamlit в одном контейнере.

Telegram работает как отдельный процесс (не фоновый поток Streamlit).
"""

import os
import signal
import subprocess
import sys
import time


def _mask_token(token: str) -> str:
    if len(token) < 12:
        return "(too short)"
    return f"{token[:8]}...{token[-4:]}"


def main() -> None:
    port = os.getenv("PORT", "8501")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    print("=" * 50, flush=True)
    print("[AURUM] Starting AURUM server", flush=True)
    print(f"[AURUM] PORT={port}", flush=True)
    print(f"[AURUM] TELEGRAM_BOT_TOKEN={'set ' + _mask_token(token) if token else 'NOT SET'}", flush=True)
    print("=" * 50, flush=True)

    print("[AURUM] Ensuring ML models (required for /status and alerts)...", flush=True)
    try:
        from aurum.ml import ensure_models

        meta = ensure_models()
        if meta:
            print(
                f"[AURUM] ML ready: {meta.samples} bars | "
                f"buy={meta.buy_precision:.0%} sell={meta.sell_precision:.0%}",
                flush=True,
            )
        else:
            print("[AURUM] WARNING: ML training failed — open terminal once or check Yahoo data", flush=True)
    except Exception as exc:
        print(f"[AURUM] WARNING: ML setup error: {exc}", flush=True)

    telegram_proc = None
    if token:
        telegram_proc = subprocess.Popen(
            [sys.executable, "run_telegram_bot.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"[AURUM] Telegram subprocess PID={telegram_proc.pid}", flush=True)
        time.sleep(5)
        if telegram_proc.poll() is not None:
            print(
                f"[AURUM] ERROR: Telegram bot exited with code {telegram_proc.returncode}. "
                "Локально часто блокируют api.telegram.org — VPN или TELEGRAM_PROXY.",
                flush=True,
            )
        else:
            print("[AURUM] Telegram bot running", flush=True)
    else:
        print("[AURUM] WARNING: No TELEGRAM_BOT_TOKEN — /start will not work", flush=True)

    def shutdown(signum, _frame):
        print(f"[AURUM] Signal {signum}, stopping...", flush=True)
        if telegram_proc and telegram_proc.poll() is None:
            telegram_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "main.py",
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    print(f"[AURUM] Starting Streamlit on port {port}", flush=True)
    streamlit_proc = subprocess.Popen(streamlit_cmd)
    streamlit_proc.wait()

    if telegram_proc and telegram_proc.poll() is None:
        telegram_proc.terminate()
    sys.exit(streamlit_proc.returncode or 0)


if __name__ == "__main__":
    main()
