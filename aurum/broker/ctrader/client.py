import logging
import threading
import time
import uuid
from dataclasses import dataclass

from aurum.broker.base import BrokerPosition, OrderResult
from aurum.broker.ctrader.auth import CTraderAuth
from aurum.broker.ctrader.orders import oz_to_protocol_volume, price_distance_to_relative, protocol_volume_to_oz
from aurum.config import (
    CTRADER_ACCOUNT_ID,
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_HOST,
    CTRADER_SYMBOL,
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    symbol_id: int
    name: str
    digits: int
    lot_size: int
    min_volume: int
    step_volume: int


class CTraderBroker:
    """Synchronous FxPro/cTrader broker wrapper (Twisted reactor in background thread)."""

    def __init__(
        self,
        host: str = CTRADER_HOST,
        account_id: int = CTRADER_ACCOUNT_ID,
        symbol_name: str = CTRADER_SYMBOL,
        auth: CTraderAuth | None = None,
    ):
        self.host = host.lower()
        self.account_id = account_id
        self.symbol_name = symbol_name.upper()
        self.auth = auth or CTraderAuth()
        self._client = None
        self._reactor_thread: threading.Thread | None = None
        self._connected = False
        self._account_authorized = False
        self._symbol: SymbolInfo | None = None
        self._balance = 0.0
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, object] = {}
        self._lock = threading.Lock()
        self._accounts: list = []

    def connect(self) -> bool:
        try:
            from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAccountAuthReq,
                ProtoOAAccountAuthRes,
                ProtoOAAmendPositionSLTPReq,
                ProtoOAAmendPositionSLTPRes,
                ProtoOAApplicationAuthReq,
                ProtoOAApplicationAuthRes,
                ProtoOAClosePositionReq,
                ProtoOAClosePositionRes,
                ProtoOAErrorRes,
                ProtoOAExecutionEvent,
                ProtoOAGetAccountListByAccessTokenReq,
                ProtoOAGetAccountListByAccessTokenRes,
                ProtoOANewOrderReq,
                ProtoOAOrderType,
                ProtoOAReconcileReq,
                ProtoOAReconcileRes,
                ProtoOASymbolsListReq,
                ProtoOASymbolsListRes,
                ProtoOATradeSide,
                ProtoOATraderReq,
                ProtoOATraderRes,
            )
            from twisted.internet import reactor
        except ImportError as exc:
            logger.error("ctrader-open-api not installed: %s", exc)
            return False

        self._imports = {
            "Client": Client,
            "EndPoints": EndPoints,
            "Protobuf": Protobuf,
            "TcpProtocol": TcpProtocol,
            "ProtoOAApplicationAuthReq": ProtoOAApplicationAuthReq,
            "ProtoOAApplicationAuthRes": ProtoOAApplicationAuthRes,
            "ProtoOAGetAccountListByAccessTokenReq": ProtoOAGetAccountListByAccessTokenReq,
            "ProtoOAGetAccountListByAccessTokenRes": ProtoOAGetAccountListByAccessTokenRes,
            "ProtoOAAccountAuthReq": ProtoOAAccountAuthReq,
            "ProtoOAAccountAuthRes": ProtoOAAccountAuthRes,
            "ProtoOASymbolsListReq": ProtoOASymbolsListReq,
            "ProtoOASymbolsListRes": ProtoOASymbolsListRes,
            "ProtoOATraderReq": ProtoOATraderReq,
            "ProtoOATraderRes": ProtoOATraderRes,
            "ProtoOANewOrderReq": ProtoOANewOrderReq,
            "ProtoOAOrderType": ProtoOAOrderType,
            "ProtoOATradeSide": ProtoOATradeSide,
            "ProtoOAReconcileReq": ProtoOAReconcileReq,
            "ProtoOAReconcileRes": ProtoOAReconcileRes,
            "ProtoOAClosePositionReq": ProtoOAClosePositionReq,
            "ProtoOAClosePositionRes": ProtoOAClosePositionRes,
            "ProtoOAAmendPositionSLTPReq": ProtoOAAmendPositionSLTPReq,
            "ProtoOAAmendPositionSLTPRes": ProtoOAAmendPositionSLTPRes,
            "ProtoOAExecutionEvent": ProtoOAExecutionEvent,
            "ProtoOAErrorRes": ProtoOAErrorRes,
            "ProtoHeartbeatEvent": ProtoHeartbeatEvent,
            "reactor": reactor,
        }

        EndPoints = self._imports["EndPoints"]
        host_addr = EndPoints.PROTOBUF_LIVE_HOST if self.host == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self._client = Client(host_addr, EndPoints.PROTOBUF_PORT, self._imports["TcpProtocol"])
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        if not reactor.running:
            self._reactor_thread = threading.Thread(
                target=lambda: reactor.run(installSignalHandlers=False),
                daemon=True,
                name="ctrader-reactor",
            )
            self._reactor_thread.start()
            time.sleep(0.5)

        with self._lock:
            self._pending["app_auth"] = threading.Event()
            self._pending["account_auth"] = threading.Event()

        self._client.startService()
        if not self._wait_flag("app_auth", timeout=20):
            return False
        if not self._wait_flag("account_auth", timeout=20):
            return False
        self._load_trader_info()
        self._load_symbol()
        return self._connected and self._account_authorized and self._symbol is not None

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.stopService()
            except Exception:
                pass
        self._connected = False
        self._account_authorized = False

    def is_connected(self) -> bool:
        return self._connected and self._account_authorized

    def get_balance(self) -> float:
        return self._balance

    def get_symbol(self) -> SymbolInfo | None:
        return self._symbol

    def get_accounts(self) -> list[dict]:
        return [
            {"id": int(a.ctidTraderAccountId), "broker": getattr(a, "brokerName", ""), "live": bool(a.isLive)}
            for a in self._accounts
        ]

    def get_last_error(self) -> str | None:
        err = self._responses.get("error") or self._responses.get("last")
        if err and hasattr(err, "errorCode"):
            return str(getattr(err, "description", err.errorCode))
        return None

    def get_positions(self) -> list[BrokerPosition]:
        resp = self._send(self._make_reconcile_req())
        if resp is None:
            return []
        positions = []
        sym = self._symbol
        for pos in resp.position:
            side = "BUY" if pos.tradeSide == 1 else "SELL"
            vol = int(pos.volume)
            entry = float(pos.price)
            sl = float(pos.stopLoss) if pos.stopLoss else 0.0
            tp = float(pos.takeProfit) if pos.takeProfit else 0.0
            positions.append(
                BrokerPosition(
                    position_id=int(pos.positionId),
                    symbol=self.symbol_name,
                    side=side,
                    volume=vol,
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                )
            )
        return positions

    def open_market(self, side: str, volume: int, sl: float, tp: float, entry_hint: float = 0.0) -> OrderResult:
        if not self._symbol:
            return OrderResult(False, None, None, "Symbol not loaded")
        sym = self._symbol
        entry = entry_hint or self._last_price or 0.0
        if entry <= 0:
            return OrderResult(False, None, None, "No reference price for SL/TP")

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        req = self._imports["ProtoOANewOrderReq"]()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = sym.symbol_id
        req.orderType = self._imports["ProtoOAOrderType"].MARKET
        req.tradeSide = (
            self._imports["ProtoOATradeSide"].BUY if side.upper() == "BUY" else self._imports["ProtoOATradeSide"].SELL
        )
        req.volume = volume
        req.relativeStopLoss = price_distance_to_relative(sl_dist)
        req.relativeTakeProfit = price_distance_to_relative(tp_dist)

        resp = self._send(req, timeout=30)
        if resp is None:
            return OrderResult(False, None, None, "Order timeout")

        if hasattr(resp, "errorCode"):
            return OrderResult(False, None, None, str(getattr(resp, "description", resp.errorCode)))

        positions = self.get_positions()
        if not positions:
            return OrderResult(True, None, None, "Order sent, position pending confirmation")

        pos = positions[-1]
        return OrderResult(True, pos.position_id, None, "Market order filled", pos.entry_price)

    def close_position(self, position_id: int, volume: int | None = None) -> OrderResult:
        req = self._imports["ProtoOAClosePositionReq"]()
        req.ctidTraderAccountId = self.account_id
        req.positionId = int(position_id)
        req.volume = volume if volume is not None else 0
        resp = self._send(req, timeout=30)
        if resp is None:
            return OrderResult(False, None, None, "Close timeout")
        return OrderResult(True, position_id, None, "Position close requested")

    def amend_sl_tp(self, position_id: int, sl: float | None, tp: float | None) -> bool:
        req = self._imports["ProtoOAAmendPositionSLTPReq"]()
        req.ctidTraderAccountId = self.account_id
        req.positionId = int(position_id)
        if sl is not None:
            req.stopLoss = float(sl)
        if tp is not None:
            req.takeProfit = float(tp)
        resp = self._send(req, timeout=20)
        return resp is not None

    def volume_from_oz(self, oz: float) -> int:
        sym = self._symbol
        if not sym:
            return 0
        return oz_to_protocol_volume(oz, sym.lot_size, sym.step_volume, sym.min_volume)

    def oz_from_volume(self, volume: int) -> float:
        sym = self._symbol
        if not sym:
            return 0.0
        return protocol_volume_to_oz(volume, sym.lot_size)

    # --- internal ---

    _last_price: float = 0.0

    def _on_connected(self, _client) -> None:
        logger.info("cTrader TCP connected")
        req = self._imports["ProtoOAApplicationAuthReq"]()
        req.clientId = CTRADER_CLIENT_ID
        req.clientSecret = CTRADER_CLIENT_SECRET
        self._client.send(req)

    def _on_disconnected(self, _client, reason) -> None:
        logger.warning("cTrader disconnected: %s", reason)
        self._connected = False
        self._account_authorized = False

    def _on_message(self, _client, message) -> None:
        Protobuf = self._imports["Protobuf"]
        payload_type = message.payloadType
        skip = {
            self._imports["ProtoHeartbeatEvent"]().payloadType,
        }
        if payload_type in skip:
            return

        extracted = Protobuf.extract(message)
        client_msg_id = getattr(message, "clientMsgId", None) or ""
        app_res = self._imports["ProtoOAApplicationAuthRes"]().payloadType
        acct_list_res = self._imports["ProtoOAGetAccountListByAccessTokenRes"]().payloadType
        acct_auth_res = self._imports["ProtoOAAccountAuthRes"]().payloadType
        error_res = self._imports["ProtoOAErrorRes"]().payloadType

        if payload_type == app_res:
            self._connected = True
            self._set_flag("app_auth")
            token = self.auth.ensure_valid_token()
            req = self._imports["ProtoOAGetAccountListByAccessTokenReq"]()
            req.accessToken = token
            self._client.send(req)
        elif payload_type == acct_list_res:
            self._accounts = list(extracted.ctidTraderAccount)
            if self.account_id <= 0 and self._accounts:
                self.account_id = int(self._accounts[0].ctidTraderAccountId)
            req = self._imports["ProtoOAAccountAuthReq"]()
            req.ctidTraderAccountId = self.account_id
            req.accessToken = self.auth.ensure_valid_token()
            self._client.send(req)
        elif payload_type == acct_auth_res:
            self._account_authorized = True
            self._set_flag("account_auth")
        elif payload_type == error_res:
            logger.error("cTrader error: %s", extracted)
            self._resolve_pending(client_msg_id, extracted)
        else:
            self._resolve_pending(client_msg_id, extracted)

    def _send(self, request, timeout: float = 20):
        msg_id = str(uuid.uuid4())
        event = threading.Event()
        with self._lock:
            self._pending[msg_id] = event
        try:
            self._client.send(request, clientMsgId=msg_id)
        except Exception as exc:
            logger.error("Send failed: %s", exc)
            return None
        if not event.wait(timeout):
            return None
        with self._lock:
            return self._responses.pop(msg_id, self._responses.get("last"))

    def _resolve_pending(self, client_msg_id: str | None, value) -> None:
        with self._lock:
            self._responses["last"] = value
            if client_msg_id and client_msg_id in self._pending:
                self._responses[client_msg_id] = value
                self._pending[client_msg_id].set()
            elif len(self._pending) == 1:
                only_id = next(iter(self._pending))
                if only_id not in ("app_auth", "account_auth"):
                    self._responses[only_id] = value
                    self._pending[only_id].set()

    def _set_flag(self, key: str) -> None:
        with self._lock:
            event = self._pending.get(key)
            if event:
                event.set()
            else:
                self._pending[key] = threading.Event()
                self._pending[key].set()

    def _wait_flag(self, key: str, timeout: float) -> bool:
        with self._lock:
            if key not in self._pending:
                self._pending[key] = threading.Event()
            event = self._pending[key]
        return event.wait(timeout)

    def _make_reconcile_req(self):
        req = self._imports["ProtoOAReconcileReq"]()
        req.ctidTraderAccountId = self.account_id
        return req

    def _load_trader_info(self) -> None:
        req = self._imports["ProtoOATraderReq"]()
        req.ctidTraderAccountId = self.account_id
        resp = self._send(req)
        if resp and hasattr(resp, "trader"):
            self._balance = float(resp.trader.balance) / 100.0

    def _load_symbol(self) -> None:
        req = self._imports["ProtoOASymbolsListReq"]()
        req.ctidTraderAccountId = self.account_id
        req.includeArchivedSymbols = False
        resp = self._send(req, timeout=30)
        if not resp:
            return
        target = self.symbol_name.replace("/", "")
        for sym in resp.symbol:
            name = sym.symbolName.upper().replace("/", "")
            if name == target or target in name:
                self._symbol = SymbolInfo(
                    symbol_id=int(sym.symbolId),
                    name=sym.symbolName,
                    digits=int(sym.digits),
                    lot_size=int(sym.lotSize or 10000),
                    min_volume=int(sym.minVolume or 1),
                    step_volume=int(sym.stepVolume or 1),
                )
                logger.info("Symbol loaded: %s (id=%s)", sym.symbolName, sym.symbolId)
                break
