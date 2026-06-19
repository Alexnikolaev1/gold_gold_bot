from aurum.broker.base import BrokerPosition, OrderResult


class PaperBroker:
    """Local paper broker — simulates FxPro execution without real orders."""

    LOT_SIZE_CENTS = 10000  # 100 oz per lot (cTrader convention)

    def __init__(self, initial_balance: float = 50_000.0):
        self.balance = initial_balance
        self._positions: dict[int, BrokerPosition] = {}
        self._next_id = 1
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_balance(self) -> float:
        return self.balance

    def get_positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    def volume_from_oz(self, oz: float) -> int:
        lots = oz / (self.LOT_SIZE_CENTS / 100.0)
        return max(1, int(round(lots * 100)))

    def oz_from_volume(self, volume: int) -> float:
        return (volume / 100.0) * (self.LOT_SIZE_CENTS / 100.0)

    def open_market(self, side: str, volume: int, sl: float, tp: float, entry_hint: float = 0.0) -> OrderResult:
        if entry_hint <= 0:
            return OrderResult(False, None, None, "No entry price")
        pid = self._next_id
        self._next_id += 1
        pos = BrokerPosition(
            position_id=pid,
            symbol="XAUUSD",
            side=side.upper(),
            volume=volume,
            entry_price=entry_hint,
            sl=sl,
            tp=tp,
        )
        self._positions[pid] = pos
        return OrderResult(True, pid, pid, "Paper market order filled", entry_hint)

    def close_position(self, position_id: int, volume: int | None = None) -> OrderResult:
        pos = self._positions.get(position_id)
        if not pos:
            return OrderResult(False, None, None, "Position not found")
        if volume is None or volume >= pos.volume:
            del self._positions[position_id]
            return OrderResult(True, position_id, None, "Paper position closed")
        pos.volume -= volume
        return OrderResult(True, position_id, None, "Paper partial close")

    def amend_sl_tp(self, position_id: int, sl: float | None, tp: float | None) -> bool:
        pos = self._positions.get(position_id)
        if not pos:
            return False
        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp
        return True

    def apply_pnl(self, pnl: float) -> None:
        self.balance += pnl
