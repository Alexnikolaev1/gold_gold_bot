import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aurum.broker.base import BrokerPosition, OrderResult
from aurum.config import EXECUTION_STATE_PATH

logger = logging.getLogger(__name__)


@dataclass
class ManagedTrade:
    trade_id: str
    side: str
    entry_price: float
    volume_oz: float
    sl: float
    tp1: float
    tp2: float
    opened_at: str
    tp1_hit: bool = False
    status: str = "open"
    position_id: int | None = None
    broker_volume: int = 0
    confidence: float = 0.0


@dataclass
class ExecutionState:
    mode: str = "paper"
    balance: float = 50_000.0
    daily_pnl: float = 0.0
    daily_reset_date: str = ""
    last_signal_bar: str = ""
    last_trade_at: str = ""
    last_closed_bar: str = ""
    open_trade: ManagedTrade | None = None
    trade_history: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class StateStore:
    def __init__(self, path: Path = EXECUTION_STATE_PATH):
        self.path = path

    def load(self) -> ExecutionState:
        if not self.path.exists():
            return ExecutionState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        open_trade = raw.get("open_trade")
        if open_trade:
            raw["open_trade"] = ManagedTrade(**open_trade)
        return ExecutionState(**raw)

    def save(self, state: ExecutionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def log(self, state: ExecutionState, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        state.logs.append(entry)
        state.logs = state.logs[-200:]
        logger.info(message)
