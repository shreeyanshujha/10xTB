import numpy as np
import pandas as pd

from tenx.agents.decision import Decision
from tenx.journal import Journal
from tenx.pipeline import run_pipeline


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


def make_fake_decide(action, conviction=0.7):
    def fake_decide(signal, research):
        return Decision(
            ticker=signal.ticker, action=action, conviction=conviction,
            rationale="fake decision", model="fake",
            trail={"system": "s", "user": "u", "response_text": "{}",
                   "response_id": "req_fake", "stop_reason": "end_turn",
                   "usage": {}},
        )
    return fake_decide


def test_pipeline_end_to_end_offline(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    # decision says BUY regardless of what the quant signal said — the
    # decision, not the raw signal, must drive the trade
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising,
                          decide_fn=make_fake_decide("BUY"))

    assert result["ticker"] == "NVDA"
    assert result["signal"]["action"] in {"BUY", "SELL", "HOLD"}
    assert result["research"]["status"] == "stub"
    assert result["decision"]["action"] == "BUY"
    assert result["trade"]["side"] == "BUY"
    assert result["trade"]["hypothetical"] is True

    records = Journal(jpath).read_all()
    stages = [r["stage"] for r in records]
    assert stages == ["data_pull", "signal", "research_context", "decision",
                      "paper_trade"]
    assert len({r["run_id"] for r in records}) == 1
    assert records[0]["payload"]["rows"] == 900
    assert records[1]["payload"]["rationale"]
    assert records[2]["payload"]["status"] == "stub"
    # decision stage journals the FULL reasoning trail, not just the outcome
    assert records[3]["payload"]["trail"]["system"]
    assert records[3]["payload"]["trail"]["response_text"]
    assert records[4]["payload"]["trade"]["side"] == "BUY"


def test_pipeline_journals_no_trade_on_hold_decision(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising,
                          decide_fn=make_fake_decide("HOLD"))
    assert result["trade"] is None
    last = Journal(jpath).read_all()[-1]
    assert last["stage"] == "paper_trade"
    assert last["payload"]["trade"] is None
    assert "reason" in last["payload"]
