from types import SimpleNamespace

from tenx.execution import IBKRBroker, OrderResult, SimBroker


def make_trade(qty=40, price=200.0, side="BUY"):
    return {"ticker": "NVDA", "side": side, "quantity": qty, "price": price,
            "notional_usd": 8_000.0, "signal_as_of": "2026-08-27",
            "hypothetical": True}


def test_sim_broker_fills_at_proposed_price():
    result = SimBroker().submit(make_trade())
    assert isinstance(result, OrderResult)
    assert result.status == "filled"
    assert result.filled_qty == 40
    assert result.avg_fill_price == 200.0
    assert result.mode == "simulated"
    assert result.order_id


def test_sim_broker_order_ids_are_unique():
    b = SimBroker()
    assert b.submit(make_trade()).order_id != b.submit(make_trade()).order_id


class FakeIB:
    """Mimics the slice of ib_async.IB that IBKRBroker touches."""

    def __init__(self, final_status="Filled", filled=40, avg_price=201.5,
                 done=True):
        self._final_status = final_status
        self._filled = filled
        self._avg_price = avg_price
        self._done = done
        self.placed = []

    def isConnected(self):
        return True

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return SimpleNamespace(
            order=SimpleNamespace(orderId=7),
            orderStatus=SimpleNamespace(
                status=self._final_status,
                filled=self._filled,
                avgFillPrice=self._avg_price,
            ),
            isDone=lambda: self._done,
        )

    def sleep(self, seconds):
        pass


def test_ibkr_broker_maps_trade_to_order():
    fake = FakeIB()
    result = IBKRBroker(ib=fake).submit(make_trade(qty=40, side="BUY"))
    assert result.status == "filled"
    assert result.filled_qty == 40
    assert result.avg_fill_price == 201.5
    assert result.mode == "ibkr_paper"
    contract, order = fake.placed[0]
    assert contract.symbol == "NVDA"
    assert order.action == "BUY"
    assert order.totalQuantity == 40


def test_ibkr_broker_reports_unfilled_as_timeout():
    fake = FakeIB(final_status="Submitted", filled=0, avg_price=None, done=False)
    result = IBKRBroker(ib=fake, timeout_s=0.0).submit(make_trade())
    assert result.status == "timeout"
    assert result.mode == "ibkr_paper"
