"""Execution layer (spec section 4.5): plumbing, not a decision point.
Receives a risk-approved trade verbatim and submits it. Two brokers:
SimBroker (no gateway needed; fills at the proposed price and is loudly
labeled simulated) and IBKRBroker (ib_async -> IB Gateway/TWS paper
account, port 7497 = TWS paper default). Same interface for paper now
and live later (spec section 4.1)."""
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: str  # "filled" | "submitted" | "rejected" | "timeout"
    filled_qty: int
    avg_fill_price: float | None
    mode: str  # "simulated" | "ibkr_paper"
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SimBroker:
    """Instant full fill at the proposed price. NOT a market simulation —
    a stand-in so the pipeline runs end-to-end without an IB Gateway."""

    def submit(self, trade: dict) -> OrderResult:
        return OrderResult(
            order_id=f"sim-{uuid.uuid4().hex[:10]}",
            status="filled",
            filled_qty=int(trade["quantity"]),
            avg_fill_price=float(trade["price"]),
            mode="simulated",
            detail="simulated fill at proposed price",
        )


class IBKRBroker:
    """Submits market orders to IB Gateway/TWS (paper account first —
    port 7497 is the TWS paper default; IB Gateway paper is 4002)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, timeout_s: float = 30.0, ib=None):
        self.host, self.port, self.client_id = host, port, client_id
        self.timeout_s = timeout_s
        self._ib = ib

    def _connect(self):
        if self._ib is None:
            from ib_async import IB
            self._ib = IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        return self._ib

    def submit(self, trade: dict) -> OrderResult:
        from ib_async import MarketOrder, Stock

        ib = self._connect()
        contract = Stock(trade["ticker"], "SMART", "USD")
        order = MarketOrder(trade["side"], int(trade["quantity"]))
        ib_trade = ib.placeOrder(contract, order)

        deadline = time.monotonic() + self.timeout_s
        while not ib_trade.isDone() and time.monotonic() < deadline:
            ib.sleep(0.25)

        status = ib_trade.orderStatus.status
        filled = int(ib_trade.orderStatus.filled or 0)
        avg = ib_trade.orderStatus.avgFillPrice
        if status == "Filled":
            result_status = "filled"
        elif status in ("Cancelled", "Inactive", "ApiCancelled"):
            result_status = "rejected"
        elif not ib_trade.isDone():
            result_status = "timeout"
        else:
            result_status = "submitted"
        return OrderResult(
            order_id=str(ib_trade.order.orderId),
            status=result_status,
            filled_qty=filled,
            avg_fill_price=float(avg) if avg else None,
            mode="ibkr_paper",
            detail=f"IB status {status}",
        )
