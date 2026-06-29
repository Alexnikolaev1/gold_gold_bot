import logging
import uuid
from datetime import datetime, timezone

from aurum.broker.ctrader.client import CTraderBroker
from aurum.broker.paper import PaperBroker
from aurum.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_RISK_PCT,
    MAX_DAILY_LOSS_PCT,
    MIN_RR_TP1,
    TP1_CLOSE_FRACTION,
    TRADE_COOLDOWN_BARS,
    TRADING_MODE,
)
from aurum.data import fetch_analysis_data, is_comex_session_active
from aurum.execution.state import ExecutionState, ManagedTrade, StateStore
from aurum.features import calculate_indicators
from aurum.risk import calculate_trade_levels, position_size_oz
from aurum.signals import process_signals
from aurum.thresholds import get_entry_thresholds, passes_strong_entry
from aurum.hashhedge import hashhedge_pair_label
from aurum.signal_alerts import EntryOpportunity, send_entry_alert

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Signal → order pipeline with TP1 partial + TP2 full exit."""

    def __init__(self, mode: str | None = None, deposit: float = 50_000, risk_pct: float = DEFAULT_RISK_PCT):
        self.mode = (mode or TRADING_MODE).lower()
        self.deposit = deposit
        self.risk_pct = risk_pct
        self.store = StateStore()
        self.state = self.store.load()
        self.broker = self._create_broker()

    def _create_broker(self):
        if self.mode in ("demo", "live"):
            return CTraderBroker(host=self.mode if self.mode == "live" else "demo")
        return PaperBroker(initial_balance=self.state.balance or self.deposit)

    def connect(self) -> bool:
        ok = self.broker.connect()
        if ok:
            bal = self.broker.get_balance()
            if bal > 0:
                self.state.balance = bal
            self.sync_with_broker()
        self.state.mode = self.mode
        self.store.save(self.state)
        return ok

    def disconnect(self) -> None:
        self.broker.disconnect()

    def sync_with_broker(self) -> None:
        """Reconcile local state with broker positions."""
        if self.mode == "paper":
            return
        positions = self.broker.get_positions()
        if not positions:
            if self.state.open_trade:
                self.store.log(self.state, "Broker: position closed externally")
                self.state.open_trade = None
            return
        if self.state.open_trade:
            bp = positions[0]
            self.state.open_trade.position_id = bp.position_id
            self.state.open_trade.entry_price = bp.entry_price
            return
        bp = positions[0]
        self.state.open_trade = ManagedTrade(
            trade_id=str(uuid.uuid4())[:8],
            side=bp.side,
            entry_price=bp.entry_price,
            volume_oz=self.broker.oz_from_volume(bp.volume) if hasattr(self.broker, "oz_from_volume") else 1.0,
            sl=bp.sl,
            tp1=bp.tp,
            tp2=bp.tp,
            opened_at=datetime.now(timezone.utc).isoformat(),
            position_id=bp.position_id,
            broker_volume=bp.volume,
        )
        self.store.log(self.state, f"Recovered open {bp.side} position from broker")

    def emergency_close_all(self) -> str:
        if self.mode != "paper" and not self.broker.is_connected():
            self.broker.connect()
        for pos in self.broker.get_positions():
            self.broker.close_position(pos.position_id)
        self.state.open_trade = None
        self.store.log(self.state, "EMERGENCY: all positions closed")
        self.store.save(self.state)
        return "closed"

    def run_cycle(self) -> dict:
        self._reset_daily_if_needed()
        session_ok, session_msg = is_comex_session_active()
        summary = {"action": "none", "session": session_msg, "mode": self.mode}

        if self.mode != "paper" and not self.broker.is_connected():
            if not self.broker.connect():
                self.store.log(self.state, "Broker reconnect failed")
                self.store.save(self.state)
                return summary

        if not session_ok:
            self.store.log(self.state, f"Session inactive: {session_msg}")
            self.store.save(self.state)
            return summary

        df = fetch_analysis_data()
        if df.empty or len(df) < 200:
            self.store.log(self.state, "Insufficient market data")
            self.store.save(self.state)
            return summary

        df_feat = calculate_indicators(df)
        if df_feat.empty:
            self.store.log(self.state, "Indicators unavailable (sparse market data)")
            self.store.save(self.state)
            return summary

        bar_time = str(df_feat.index[-1])
        current_price = float(df_feat.iloc[-1]["Close"])

        if self.state.open_trade:
            if self.mode != "paper":
                self.sync_with_broker()
            summary["action"] = self._manage_open_trade(current_price)
            self.store.save(self.state)
            return summary

        if self._daily_loss_limit_hit():
            self.store.log(self.state, "Daily loss limit — trading paused")
            self.store.save(self.state)
            return summary

        if self._cooldown_active(df_feat, bar_time):
            self.store.save(self.state)
            return summary

        if bar_time == self.state.last_signal_bar:
            self.store.save(self.state)
            return summary

        signal_result = process_signals(df_feat, use_ml=True, include_macro=True)
        self.state.last_signal_bar = bar_time

        if signal_result.signal not in ("BUY", "SELL"):
            self.store.log(self.state, f"HOLD | conf={signal_result.confidence:.1%}")
            self.store.save(self.state)
            return summary

        if not passes_strong_entry(signal_result, use_ml=True):
            self.store.log(
                self.state,
                f"Filtered {signal_result.signal} | conf={signal_result.confidence:.1%} (not max confidence)",
            )
            self.store.save(self.state)
            return summary

        levels = calculate_trade_levels(df_feat, signal_result.signal)
        min_rr = get_entry_thresholds(use_ml=True).min_rr
        if levels.risk_reward_tp1 < min_rr:
            self.store.log(self.state, f"Rejected: R:R {levels.risk_reward_tp1:.2f} < {min_rr}")
            self.store.save(self.state)
            return summary

        oz = position_size_oz(self.state.balance, self.risk_pct, levels.risk_per_oz)
        if oz <= 0:
            self.store.save(self.state)
            return summary

        volume = self.broker.volume_from_oz(oz) if hasattr(self.broker, "volume_from_oz") else max(1, int(oz * 100))
        if volume <= 0:
            self.store.save(self.state)
            return summary

        if not self.broker.is_connected() and not self.broker.connect():
            self.store.log(self.state, "Broker connection failed")
            self.store.save(self.state)
            return summary

        order = self.broker.open_market(
            signal_result.signal, volume, levels.sl, levels.tp2, entry_hint=current_price
        )
        if not order.success:
            self.store.log(self.state, f"Order failed: {order.message}")
            self.store.save(self.state)
            return summary

        trade = ManagedTrade(
            trade_id=str(uuid.uuid4())[:8],
            side=signal_result.signal,
            entry_price=order.entry_price or current_price,
            volume_oz=oz,
            sl=levels.sl,
            tp1=levels.tp1,
            tp2=levels.tp2,
            opened_at=datetime.now(timezone.utc).isoformat(),
            position_id=order.position_id,
            broker_volume=volume,
            confidence=signal_result.confidence,
        )
        self.state.open_trade = trade
        self.state.last_trade_at = trade.opened_at
        self.store.log(
            self.state,
            f"OPEN {trade.side} @ {trade.entry_price:.2f} | "
            f"SL={levels.sl} TP1={levels.tp1} TP2={levels.tp2} | {oz:.2f}oz | R:R={levels.risk_reward_tp1:.2f}",
        )
        rules_active = signal_result.rules_buy if trade.side == "BUY" else signal_result.rules_sell
        send_entry_alert(
            EntryOpportunity(
                symbol="XAU",
                pair=hashhedge_pair_label("XAU"),
                side=trade.side,
                price=trade.entry_price,
                confidence=trade.confidence,
                tp1=levels.tp1,
                tp2=levels.tp2,
                sl=levels.sl,
                size=oz,
                rr_tp1=levels.risk_reward_tp1,
                bar_time=bar_time,
                rules_active=rules_active,
                use_ml=True,
            )
        )
        summary["action"] = f"opened_{trade.side.lower()}"
        self.store.save(self.state)
        return summary

    def _manage_open_trade(self, price: float) -> str:
        trade = self.state.open_trade
        if not trade:
            return "none"
        if trade.side == "BUY":
            if price <= trade.sl:
                return self._close_trade(price, "SL")
            if not trade.tp1_hit and price >= trade.tp1:
                return self._partial_tp1(price)
            if price >= trade.tp2:
                return self._close_trade(price, "TP2")
        else:
            if price >= trade.sl:
                return self._close_trade(price, "SL")
            if not trade.tp1_hit and price <= trade.tp1:
                return self._partial_tp1(price)
            if price <= trade.tp2:
                return self._close_trade(price, "TP2")
        return "holding"

    def _partial_tp1(self, price: float) -> str:
        trade = self.state.open_trade
        if not trade or trade.tp1_hit:
            return "holding"
        close_vol = max(1, int(trade.broker_volume * TP1_CLOSE_FRACTION))
        if trade.position_id:
            self.broker.close_position(trade.position_id, volume=close_vol)
            trade.broker_volume -= close_vol
            trade.volume_oz *= 1 - TP1_CLOSE_FRACTION
        trade.tp1_hit = True
        trade.sl = trade.entry_price
        if trade.position_id:
            self.broker.amend_sl_tp(trade.position_id, sl=trade.entry_price, tp=trade.tp2)
        pnl = self._calc_pnl(trade, price, TP1_CLOSE_FRACTION)
        self._apply_pnl(pnl)
        self.store.log(self.state, f"TP1 @ {price:.2f} | partial {TP1_CLOSE_FRACTION:.0%} | SL→breakeven")
        return "tp1_partial"

    def _close_trade(self, price: float, reason: str) -> str:
        trade = self.state.open_trade
        if not trade:
            return "none"
        if trade.position_id:
            self.broker.close_position(trade.position_id)
        fraction = 1.0 if not trade.tp1_hit else (1 - TP1_CLOSE_FRACTION)
        pnl = self._calc_pnl(trade, price, fraction)
        self._apply_pnl(pnl)
        self.state.trade_history.append(
            {
                "trade_id": trade.trade_id,
                "side": trade.side,
                "entry": trade.entry_price,
                "exit": price,
                "pnl": round(pnl, 2),
                "reason": reason,
                "confidence": round(trade.confidence, 3),
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.state.trade_history = self.state.trade_history[-100:]
        self.state.last_closed_bar = self.state.last_signal_bar
        self.store.log(self.state, f"CLOSE {reason} @ {price:.2f} | PnL=${pnl:.2f}")
        self.state.open_trade = None
        return f"closed_{reason.lower()}"

    def _apply_pnl(self, pnl: float) -> None:
        self.state.daily_pnl += pnl
        if isinstance(self.broker, PaperBroker):
            self.broker.apply_pnl(pnl)
            self.state.balance += pnl
        else:
            bal = self.broker.get_balance()
            if bal > 0:
                self.state.balance = bal

    def _calc_pnl(self, trade: ManagedTrade, exit_price: float, fraction: float) -> float:
        direction = 1 if trade.side == "BUY" else -1
        return direction * (exit_price - trade.entry_price) * trade.volume_oz * fraction

    def _daily_loss_limit_hit(self) -> bool:
        limit = self.state.balance * (MAX_DAILY_LOSS_PCT / 100)
        return self.state.daily_pnl <= -limit

    def _cooldown_active(self, df_feat, bar_time: str) -> bool:
        if not self.state.last_closed_bar:
            return False
        times = [str(t) for t in df_feat.index]
        try:
            closed_idx = times.index(self.state.last_closed_bar)
            current_idx = times.index(bar_time)
            return (current_idx - closed_idx) < TRADE_COOLDOWN_BARS
        except ValueError:
            return False

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.daily_reset_date != today:
            self.state.daily_reset_date = today
            self.state.daily_pnl = 0.0
