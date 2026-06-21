"""Background entry alerts — same filters as ExecutionEngine, no browser required."""

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from aurum.config import (
    DATA_DIR,
    DEFAULT_DEPOSIT,
    DEFAULT_RISK_PCT,
    MIN_BARS_FOR_FEATURES,
    MIN_RR_TP1,
    SIGNAL_ALERT_INTERVAL_SEC,
    TELEGRAM_ALERTS_ENABLED,
    TOTAL_RULES,
    TRADE_COOLDOWN_BARS,
)
from aurum.data import fetch_realtime_data, is_comex_session_active
from aurum.execution.state import StateStore
from aurum.features import calculate_indicators
from aurum.ml import models_exist
from aurum.notifications import format_entry_alert, send_telegram_alert
from aurum.risk import calculate_trade_levels, position_size_oz
from aurum.signals import process_signals

logger = logging.getLogger(__name__)

ALERT_STATE_PATH = DATA_DIR / "signal_alert_state.json"


@dataclass
class EntryOpportunity:
    side: str
    price: float
    confidence: float
    tp1: float
    tp2: float
    sl: float
    oz: float
    rr_tp1: float
    bar_time: str
    rules_active: int


@dataclass
class AlertState:
    last_alert_key: str = ""
    last_alert_at: str = ""


def _load_alert_state() -> AlertState:
    if not ALERT_STATE_PATH.exists():
        return AlertState()
    try:
        raw = json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        return AlertState(**raw)
    except (json.JSONDecodeError, TypeError):
        return AlertState()


def _save_alert_state(state: AlertState) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_PATH.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _alert_key(bar_time: str, side: str) -> str:
    return f"{bar_time}:{side}"


def _cooldown_active(df_feat, bar_time: str, last_closed_bar: str) -> bool:
    if not last_closed_bar:
        return False
    times = [str(t) for t in df_feat.index]
    try:
        closed_idx = times.index(last_closed_bar)
        current_idx = times.index(bar_time)
        return (current_idx - closed_idx) < TRADE_COOLDOWN_BARS
    except ValueError:
        return False


def evaluate_entry_opportunity(
    deposit: float | None = None,
    risk_pct: float | None = None,
) -> EntryOpportunity | None:
    """Return entry setup when all AURUM entry filters pass (mirrors ExecutionEngine)."""
    session_ok, _ = is_comex_session_active()
    if not session_ok:
        return None
    if not models_exist():
        return None

    df = fetch_realtime_data()
    if df.empty or len(df) < MIN_BARS_FOR_FEATURES:
        return None

    df_feat = calculate_indicators(df)
    bar_time = str(df_feat.index[-1])
    price = float(df_feat.iloc[-1]["Close"])

    exec_state = StateStore().load()
    if exec_state.open_trade:
        return None
    if _cooldown_active(df_feat, bar_time, exec_state.last_closed_bar):
        return None

    result = process_signals(df_feat)
    if result.signal not in ("BUY", "SELL"):
        return None

    levels = calculate_trade_levels(df_feat, result.signal)
    if levels.risk_reward_tp1 < MIN_RR_TP1:
        return None

    dep = deposit if deposit is not None else DEFAULT_DEPOSIT
    risk = risk_pct if risk_pct is not None else DEFAULT_RISK_PCT
    oz = position_size_oz(dep, risk, levels.risk_per_oz)
    if oz <= 0:
        return None

    rules_active = result.rules_buy if result.signal == "BUY" else result.rules_sell
    return EntryOpportunity(
        side=result.signal,
        price=price,
        confidence=result.confidence,
        tp1=levels.tp1,
        tp2=levels.tp2,
        sl=levels.sl,
        oz=oz,
        rr_tp1=levels.risk_reward_tp1,
        bar_time=bar_time,
        rules_active=rules_active,
    )


def send_entry_alert(opp: EntryOpportunity, *, force: bool = False) -> bool:
    """Send Telegram entry alert once per bar+side."""
    key = _alert_key(opp.bar_time, opp.side)
    alert_state = _load_alert_state()
    if not force and alert_state.last_alert_key == key:
        return False

    message = format_entry_alert(
        signal=opp.side,
        price=opp.price,
        confidence=opp.confidence,
        tp1=opp.tp1,
        tp2=opp.tp2,
        sl=opp.sl,
        oz=opp.oz,
        rr_tp1=opp.rr_tp1,
        rules_active=opp.rules_active,
        total_rules=TOTAL_RULES,
        bar_time=opp.bar_time,
    )
    if not send_telegram_alert(message):
        return False

    alert_state.last_alert_key = key
    alert_state.last_alert_at = datetime.now(timezone.utc).isoformat()
    _save_alert_state(alert_state)
    logger.info("Entry alert sent: %s @ %.2f bar=%s", opp.side, opp.price, opp.bar_time)
    return True


def scan_and_alert() -> bool:
    """One alert scan cycle. Returns True if alert was sent."""
    from aurum.telegram_bot import load_chat_id

    if not models_exist():
        return False
    if not load_chat_id():
        return False
    opp = evaluate_entry_opportunity()
    if opp is None:
        return False
    return send_entry_alert(opp)


def run_alert_loop(stop_event: threading.Event, log_fn=None) -> None:
    """Background loop — runs inside Telegram bot process."""
    _log = log_fn or logger.info
    if not TELEGRAM_ALERTS_ENABLED:
        _log("Signal alerts disabled (TELEGRAM_ALERTS_ENABLED=false)")
        return

    _log(f"Signal alert worker started (interval={SIGNAL_ALERT_INTERVAL_SEC}s)")
    warned_no_models = False
    warned_no_chat = False
    while not stop_event.is_set():
        try:
            from aurum.telegram_bot import load_chat_id

            if not models_exist():
                if not warned_no_models:
                    _log("Alerts paused: ML models not ready")
                    warned_no_models = True
            elif not load_chat_id():
                if not warned_no_chat:
                    _log("Alerts paused: send /start to register chat")
                    warned_no_chat = True
            else:
                warned_no_models = False
                warned_no_chat = False
                if scan_and_alert():
                    _log("Entry alert sent")
        except Exception as exc:
            logger.exception("Alert scan failed: %s", exc)
            if log_fn:
                log_fn(f"Alert scan error: {exc}")
        stop_event.wait(SIGNAL_ALERT_INTERVAL_SEC)
