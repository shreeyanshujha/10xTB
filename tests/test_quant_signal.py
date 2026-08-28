import numpy as np
import pandas as pd
import pytest

from tenx.signals.base import Signal
from tenx.signals.quant_model import DEFAULT_MODEL, quant_signal


def make_bars(n=900, drift=0.0005, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n))), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


def test_quant_signal_produces_valid_signal():
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600)
    assert isinstance(sig, Signal)
    assert sig.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.as_of == df.index[-1].date().isoformat()
    assert sig.features["model"] == DEFAULT_MODEL
    assert 0.0 <= sig.features["p_up"] <= 1.0
    assert sig.features["last_close"] == float(df["close"].iloc[-1])
    assert "walk-forward" in sig.rationale


def test_action_mapping_follows_thresholds():
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600)
    p = sig.features["p_up"]
    expected = "BUY" if p >= 0.55 else "SELL" if p <= 0.45 else "HOLD"
    assert sig.action == expected
    expected_conf = p if p >= 0.55 else (1 - p) if p <= 0.45 else max(p, 1 - p)
    assert sig.confidence == pytest.approx(expected_conf)


def test_training_respects_embargo():
    """Train window must end `horizon` bars before the prediction bar."""
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600, horizon=5)
    assert sig.features["train_end"] <= df.index[-6].date().isoformat()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        quant_signal(make_bars(n=200), "NVDA", train_size=600)


def test_deterministic():
    df = make_bars()
    s1 = quant_signal(df, "NVDA", train_size=600)
    s2 = quant_signal(df, "NVDA", train_size=600)
    assert s1.features["p_up"] == s2.features["p_up"]
