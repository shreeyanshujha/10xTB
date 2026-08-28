import numpy as np
import pandas as pd

from tenx.journal import Journal
from tenx.pipeline import run_pipeline


def fake_fetch_rising(ticker, start_date, end_date=None):
    idx = pd.date_range("2026-03-01", periods=60, freq="B")
    c = pd.Series(np.linspace(100, 200, 60), index=idx)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c,
         "volume": np.full(60, 1_000_000)},
        index=idx,
    )


def test_pipeline_end_to_end_offline(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising)

    assert result["ticker"] == "NVDA"
    assert result["signal"]["action"] == "BUY"
    assert result["trade"]["side"] == "BUY"
    assert result["trade"]["hypothetical"] is True

    records = Journal(jpath).read_all()
    stages = [r["stage"] for r in records]
    assert stages == ["data_pull", "signal", "paper_trade"]
    assert len({r["run_id"] for r in records}) == 1
    assert records[0]["payload"]["rows"] == 60
    assert records[1]["payload"]["rationale"]
    assert records[2]["payload"]["trade"]["quantity"] >= 1


def test_pipeline_journals_no_trade_on_hold(tmp_path):
    def fetch_flat(ticker, start_date, end_date=None):
        idx = pd.date_range("2026-03-01", periods=60, freq="B")
        c = pd.Series(np.full(60, 100.0), index=idx)
        return pd.DataFrame(
            {"open": c, "high": c, "low": c, "close": c,
             "volume": np.full(60, 1_000_000)},
            index=idx,
        )

    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fetch_flat)
    assert result["trade"] is None
    last = Journal(jpath).read_all()[-1]
    assert last["stage"] == "paper_trade"
    assert last["payload"]["trade"] is None
    assert "reason" in last["payload"]
