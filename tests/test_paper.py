import pytest

from tenx.paper import build_paper_trade
from tenx.signals.base import Signal


def make_signal(action="BUY", last_close=200.0):
    return Signal(
        ticker="NVDA", as_of="2026-08-27", action=action,
        confidence=0.5, features={"last_close": last_close}, rationale="test",
    )


def test_buy_signal_builds_trade():
    trade = build_paper_trade(make_signal("BUY", last_close=200.0))
    assert trade == {
        "ticker": "NVDA",
        "side": "BUY",
        "quantity": 50,
        "price": 200.0,
        "notional_usd": 10_000.0,
        "signal_as_of": "2026-08-27",
        "hypothetical": True,
    }


def test_sell_signal_builds_sell_side():
    assert build_paper_trade(make_signal("SELL"))["side"] == "SELL"


def test_hold_returns_none():
    assert build_paper_trade(make_signal("HOLD")) is None


def test_price_above_notional_raises():
    with pytest.raises(ValueError):
        build_paper_trade(make_signal("BUY", last_close=20_000.0))
