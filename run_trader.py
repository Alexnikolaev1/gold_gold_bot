#!/usr/bin/env python3
"""
AURUM Auto-Trader Worker — 24/7 execution for FxPro (cTrader).

  python run_trader.py                     # paper
  TRADING_MODE=demo python run_trader.py   # FxPro demo
  TRADING_MODE=live python run_trader.py   # LIVE (real money)
"""

import os
import signal
import sys
import time

from aurum.config import DEFAULT_DEPOSIT, DEFAULT_RISK_PCT, TRADER_LOOP_SEC, TRADING_MODE
from aurum.data import fetch_training_data
from aurum.execution.engine import ExecutionEngine
from aurum.logging_setup import setup_logging
from aurum.ml import models_exist, train_models
from aurum.preflight import run_preflight
from aurum.telegram_runner import start_telegram_bot_background

logger = setup_logging("aurum.trader")
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received (%s)", signum)
    _shutdown = True


def ensure_models() -> None:
    if models_exist():
        return
    logger.info("Training ML models...")
    df = fetch_training_data()
    if df.empty:
        raise RuntimeError("Cannot fetch training data")
    train_models(df)
    logger.info("Models trained")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    mode = TRADING_MODE
    if mode == "live":
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING — REAL MONEY AT RISK")
        logger.warning("=" * 60)

    report = run_preflight()
    for check in report.checks:
        level = logger.info if check.ok else logger.warning
        level("Preflight [%s]: %s — %s", check.name, "OK" if check.ok else "FAIL", check.detail)
    if not report.critical_ok:
        logger.error("Critical preflight checks failed. Aborting.")
        sys.exit(1)

    ensure_models()

    start_telegram_bot_background()

    deposit = float(os.getenv("DEPOSIT", DEFAULT_DEPOSIT))
    risk_pct = float(os.getenv("RISK_PCT", DEFAULT_RISK_PCT))
    engine = ExecutionEngine(mode=mode, deposit=deposit, risk_pct=risk_pct)

    if mode in ("demo", "live"):
        if not engine.connect():
            logger.error("FxPro/cTrader connection failed. Check OAuth tokens.")
            sys.exit(1)
        logger.info("Connected to FxPro cTrader (%s) | balance=$%.2f", mode, engine.state.balance)
    else:
        engine.connect()
        logger.info("Paper mode | balance=$%.2f", engine.state.balance)

    logger.info("Trader loop started (interval=%ds)", TRADER_LOOP_SEC)
    consecutive_errors = 0

    while not _shutdown:
        try:
            summary = engine.run_cycle()
            logger.info("Cycle: %s", summary)
            consecutive_errors = 0
        except Exception:
            consecutive_errors += 1
            logger.exception("Cycle error (%d)", consecutive_errors)
            if consecutive_errors >= 5:
                logger.error("Too many consecutive errors, pausing 5 min")
                time.sleep(300)
                consecutive_errors = 0
        time.sleep(TRADER_LOOP_SEC)

    engine.disconnect()
    logger.info("Trader stopped cleanly")


if __name__ == "__main__":
    main()
