import numpy as np
import pandas as pd

from tenx.journal import Journal
from tenx.pipeline import run_pipeline
from tenx.signals.base import Signal


def fake_fetch_rising(ticker, start_date, end_date=None):
    n = 900
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


def test_pipeline_end_to_end_offline(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising)

    assert result["ticker"] == "NVDA"
    assert result["signal"]["action"] in {"BUY", "SELL", "HOLD"}
    assert result["signal"]["features"]["model"]  # quant signal metadata present
    assert 0.0 <= result["signal"]["features"]["p_up"] <= 1.0

    records = Journal(jpath).read_all()
    stages = [r["stage"] for r in records]
    assert stages == ["data_pull", "signal", "paper_trade"]
    assert len({r["run_id"] for r in records}) == 1
    assert records[0]["payload"]["rows"] == 900
    assert records[1]["payload"]["rationale"]
    # trade record is consistent with the signal's action
    if result["signal"]["action"] == "HOLD":
        assert records[2]["payload"]["trade"] is None
    else:
        assert records[2]["payload"]["trade"]["side"] == result["signal"]["action"]
        assert records[2]["payload"]["trade"]["hypothetical"] is True


def test_pipeline_journals_no_trade_on_hold(tmp_path):
    def hold_signal(df, ticker):
        return Signal(
            ticker=ticker, as_of=df.index[-1].date().isoformat(), action="HOLD",
            confidence=0.5, features={"last_close": float(df["close"].iloc[-1])},
            rationale="handcrafted HOLD for the no-trade branch",
        )

    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising,
                          signal_fn=hold_signal)
    assert result["trade"] is None
    last = Journal(jpath).read_all()[-1]
    assert last["stage"] == "paper_trade"
    assert last["payload"]["trade"] is None
    assert "reason" in last["payload"]
