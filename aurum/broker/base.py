from dataclasses import dataclass
from typing import Protocol


@dataclass
class BrokerPosition:
    position_id: int
    symbol: str
    side: str
    volume: int
    entry_price: float
    sl: float
    tp: float
    unrealized_pnl: float = 0.0


@dataclass
class OrderResult:
    success: bool
    position_id: int | None
    order_id: int | None
    message: str
    entry_price: float = 0.0


class Broker(Protocol):
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def get_balance(self) -> float: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def open_market(
        self,
        side: str,
        volume: int,
        sl: float,
        tp: float,
    ) -> OrderResult: ...
    def close_position(self, position_id: int, volume: int | None = None) -> OrderResult: ...
    def amend_sl_tp(self, position_id: int, sl: float | None, tp: float | None) -> bool: ...
