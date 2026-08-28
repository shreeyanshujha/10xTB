import numpy as np
import pandas as pd
import pytest

from tenx.signals.base import Signal
from tenx.signals.placeholder import sma_crossover_signal


def make_bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
         "volume": np.full(len(c), 1_000_000)},
        index=idx,
    )


def test_rising_prices_produce_buy():
    df = make_bars(np.linspace(100, 200, 60))
    sig = sma_crossover_signal(df, "NVDA")
    assert isinstance(sig, Signal)
    assert sig.action == "BUY"
    assert sig.ticker == "NVDA"
    assert sig.as_of == df.index[-1].date().isoformat()
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.features["sma_fast"] > sig.features["sma_slow"]
    assert sig.rationale


def test_falling_prices_produce_sell():
    df = make_bars(np.linspace(200, 100, 60))
    sig = sma_crossover_signal(df, "NVDA")
    assert sig.action == "SELL"
    assert sig.features["sma_fast"] < sig.features["sma_slow"]


def test_too_little_history_raises():
    df = make_bars(np.linspace(100, 110, 10))
    with pytest.raises(ValueError):
        sma_crossover_signal(df, "NVDA")


def test_signal_serializes_to_dict():
    df = make_bars(np.linspace(100, 200, 60))
    d = sma_crossover_signal(df, "NVDA").to_dict()
    assert d["action"] == "BUY"
    assert isinstance(d["features"], dict)
