import pytest

from tenx.paper import build_paper_trade


def test_buy_builds_trade():
    trade = build_paper_trade("BUY", "NVDA", 200.0, "2026-08-27")
    assert trade == {
        "ticker": "NVDA",
        "side": "BUY",
        "quantity": 50,
        "price": 200.0,
        "notional_usd": 10_000.0,
        "signal_as_of": "2026-08-27",
        "hypothetical": True,
    }


def test_sell_builds_sell_side():
    assert build_paper_trade("SELL", "NVDA", 200.0, "2026-08-27")["side"] == "SELL"


def test_hold_returns_none():
    assert build_paper_trade("HOLD", "NVDA", 200.0, "2026-08-27") is None


def test_price_above_notional_raises():
    with pytest.raises(ValueError):
        build_paper_trade("BUY", "NVDA", 20_000.0, "2026-08-27")
