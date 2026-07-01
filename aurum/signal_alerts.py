"""Background entry alerts — gold ML + Hash Hedge multi-asset scanner."""

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from aurum.config import (
    DATA_DIR,
    DEFAULT_DEPOSIT,
    DEFAULT_RISK_PCT,
    HASHHEDGE_ALERTS_ENABLED,
    HASHHEDGE_SCAN_BATCH,
    MIN_BARS_FOR_FEATURES,
    MIN_FEAT_BARS_ALERT,
    SIGNAL_ALERT_INTERVAL_SEC,
    TELEGRAM_ALERTS_ENABLED,
    TOTAL_RULES,
    TRADE_COOLDOWN_BARS,
)
from aurum.data import fetch_asset_analysis_data, is_asset_session_active
from aurum.entry_filters import passes_strong_entry
from aurum.thresholds import get_entry_thresholds
from aurum.execution.state import StateStore
from aurum.features import calculate_indicators
from aurum.hashhedge import hashhedge_pair_label, load_hashhedge_symbols, uses_ml_model
from aurum.ml import models_exist
from aurum.notifications import format_entry_alert, send_telegram_alert
from aurum.risk import calculate_trade_levels, position_size_oz
from aurum.signals import process_signals

logger = logging.getLogger(__name__)

ALERT_STATE_PATH = DATA_DIR / "signal_alert_state.json"


@dataclass
class EntryOpportunity:
    symbol: str
    pair: str
    side: str
    price: float
    confidence: float
    tp1: float
    tp2: float
    sl: float
    size: float
    rr_tp1: float
    bar_time: str
    rules_active: int
    use_ml: bool = False


@dataclass
class AlertState:
    last_alerts: dict[str, str] = field(default_factory=dict)
    scan_offset: int = 0
    # legacy single-key format
    last_alert_key: str = ""
    last_alert_at: str = ""


def _load_alert_state() -> AlertState:
    if not ALERT_STATE_PATH.exists():
        return AlertState()
    try:
        raw = json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        if "last_alerts" in raw:
            return AlertState(
                last_alerts=raw.get("last_alerts") or {},
                scan_offset=int(raw.get("scan_offset", 0)),
                last_alert_at=raw.get("last_alert_at", ""),
            )
        # migrate legacy
        legacy = raw.get("last_alert_key", "")
        last_alerts = {"XAU": legacy} if legacy else {}
        return AlertState(last_alerts=last_alerts, last_alert_at=raw.get("last_alert_at", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return AlertState()


def _save_alert_state(state: AlertState) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_alerts": state.last_alerts,
        "scan_offset": state.scan_offset,
        "last_alert_at": state.last_alert_at,
    }
    ALERT_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _alert_key(symbol: str, bar_time: str, side: str) -> str:
    return f"{symbol}:{bar_time}:{side}"


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
    symbol: str = "XAU",
    deposit: float | None = None,
    risk_pct: float | None = None,
) -> EntryOpportunity | None:
    """Return entry setup when all filters pass for a Hash Hedge symbol."""
    sym = symbol.upper()
    session_ok, _ = is_asset_session_active(sym)
    if not session_ok:
        return None

    use_ml = uses_ml_model(sym)
    if use_ml and not models_exist():
        return None

    df = fetch_asset_analysis_data(sym)
    if df.empty or len(df) < MIN_BARS_FOR_FEATURES:
        return None

    df_feat = calculate_indicators(df)
    if df_feat.empty or len(df_feat) < MIN_FEAT_BARS_ALERT:
        return None

    bar_time = str(df_feat.index[-1])
    price = float(df_feat.iloc[-1]["Close"])

    if sym == "XAU":
        exec_state = StateStore().load()
        if exec_state.open_trade:
            return None
        if _cooldown_active(df_feat, bar_time, exec_state.last_closed_bar):
            return None

    result = process_signals(df_feat, use_ml=use_ml, include_macro=use_ml)
    if not passes_strong_entry(result, use_ml=use_ml):
        return None

    levels = calculate_trade_levels(df_feat, result.signal)
    min_rr = get_entry_thresholds(use_ml=use_ml).min_rr
    if levels.risk_reward_tp1 < min_rr:
        return None

    dep = deposit if deposit is not None else DEFAULT_DEPOSIT
    risk = risk_pct if risk_pct is not None else DEFAULT_RISK_PCT
    size = position_size_oz(dep, risk, levels.risk_per_oz)
    if size <= 0:
        return None

    rules_active = result.rules_buy if result.signal == "BUY" else result.rules_sell
    return EntryOpportunity(
        symbol=sym,
        pair=hashhedge_pair_label(sym),
        side=result.signal,
        price=price,
        confidence=result.confidence,
        tp1=levels.tp1,
        tp2=levels.tp2,
        sl=levels.sl,
        size=size,
        rr_tp1=levels.risk_reward_tp1,
        bar_time=bar_time,
        rules_active=rules_active,
        use_ml=use_ml,
    )


def send_entry_alert(opp: EntryOpportunity, *, force: bool = False) -> bool:
    """Send Telegram entry alert once per symbol+bar+side."""
    key = _alert_key(opp.symbol, opp.bar_time, opp.side)
    alert_state = _load_alert_state()
    if not force and alert_state.last_alerts.get(opp.symbol) == key:
        return False

    size_label = "oz" if opp.symbol == "XAU" else "units"
    message = format_entry_alert(
        signal=opp.side,
        price=opp.price,
        confidence=opp.confidence,
        tp1=opp.tp1,
        tp2=opp.tp2,
        sl=opp.sl,
        oz=opp.size,
        rr_tp1=opp.rr_tp1,
        rules_active=opp.rules_active,
        total_rules=TOTAL_RULES,
        bar_time=opp.bar_time,
        symbol=opp.symbol,
        pair=opp.pair,
        size_label=size_label,
        platform="Hash Hedge",
    )
    if not send_telegram_alert(message):
        return False

    alert_state.last_alerts[opp.symbol] = key
    alert_state.last_alert_at = datetime.now(timezone.utc).isoformat()
    _save_alert_state(alert_state)
    logger.info("Entry alert sent: %s %s @ %.4f bar=%s", opp.symbol, opp.side, opp.price, opp.bar_time)
    return True


def scan_hashhedge_batch() -> list[str]:
    """Scan next batch of Hash Hedge symbols; return symbols that fired alerts."""
    from aurum.telegram_bot import load_chat_id

    if not load_chat_id():
        return []
    if not HASHHEDGE_ALERTS_ENABLED:
        return []

    symbols = load_hashhedge_symbols()
    if not symbols:
        return []

    state = _load_alert_state()
    batch = max(1, min(HASHHEDGE_SCAN_BATCH, len(symbols)))
    start = state.scan_offset % len(symbols)
    chunk = list(symbols[start : start + batch])
    if len(chunk) < batch:
        chunk.extend(symbols[: batch - len(chunk)])

    sent: list[str] = []
    for sym in chunk:
        try:
            opp = evaluate_entry_opportunity(sym)
            if opp and send_entry_alert(opp):
                sent.append(sym)
        except Exception as exc:
            logger.warning("Hash Hedge scan failed for %s: %s", sym, exc)

    state.scan_offset = (start + batch) % len(symbols)
    _save_alert_state(state)
    return sent


def scan_and_alert() -> bool:
    """One alert scan cycle across Hash Hedge universe."""
    sent = scan_hashhedge_batch()
    return len(sent) > 0


def run_alert_loop(stop_event: threading.Event, log_fn=None) -> None:
    """Background loop — runs inside Telegram bot process."""
    _log = log_fn or logger.info
    if not TELEGRAM_ALERTS_ENABLED:
        _log("Signal alerts disabled (TELEGRAM_ALERTS_ENABLED=false)")
        return

    symbols = load_hashhedge_symbols()
    _log(
        f"Signal alert worker started (interval={SIGNAL_ALERT_INTERVAL_SEC}s, "
        f"Hash Hedge: {len(symbols)} coins, batch={HASHHEDGE_SCAN_BATCH})"
    )
    warned_no_models = False
    warned_no_chat = False
    warned_idle = False
    while not stop_event.is_set():
        try:
            from aurum.telegram_bot import load_chat_id

            if not load_chat_id():
                if not warned_no_chat:
                    _log("Alerts paused: send /start to register chat")
                    warned_no_chat = True
            else:
                warned_no_chat = False
                if HASHHEDGE_ALERTS_ENABLED and uses_ml_model("XAU") and not models_exist():
                    if not warned_no_models:
                        _log("Gold ML not ready — crypto/metals still scanned with technical mode")
                        warned_no_models = True
                else:
                    warned_no_models = False

                sent = scan_hashhedge_batch()
                if sent:
                    _log(f"Entry alerts sent: {', '.join(sent)}")
                    warned_idle = False
                elif not warned_idle:
                    state = _load_alert_state()
                    pos = state.scan_offset
                    _log(
                        f"Hash Hedge scan OK — no max-confidence entry in batch "
                        f"(offset {pos}/{len(symbols)})"
                    )
                    warned_idle = True
        except Exception as exc:
            logger.exception("Alert scan failed: %s", exc)
            if log_fn:
                log_fn(f"Alert scan error: {exc}")
        stop_event.wait(SIGNAL_ALERT_INTERVAL_SEC)
