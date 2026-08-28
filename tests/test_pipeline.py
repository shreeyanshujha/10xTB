import numpy as np
import pandas as pd

from tenx.agents.decision import Decision
from tenx.journal import Journal
from tenx.pipeline import run_pipeline
from tenx.risk import RiskLimits


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


def run(tmp_path, **kw):
    kw.setdefault("journal_path", tmp_path / "journal.jsonl")
    kw.setdefault("portfolio_path", tmp_path / "pf.json")
    kw.setdefault("fetch", fake_fetch_rising)
    kw.setdefault("risk_limits",
                  RiskLimits(kill_switch_path=str(tmp_path / "KILL_SWITCH")))
    return run_pipeline("NVDA", **kw)


def test_pipeline_approved_trade_executes_and_updates_portfolio(tmp_path):
    result = run(tmp_path, decide_fn=make_fake_decide("BUY"))

    assert result["decision"]["action"] == "BUY"
    assert result["risk"]["approved"] is True
    assert result["order"]["mode"] == "simulated"
    assert result["order"]["status"] == "filled"
    filled = result["order"]["filled_qty"]
    assert result["portfolio"]["positions"]["NVDA"]["qty"] == filled
    assert (tmp_path / "pf.json").exists()

    records = Journal(tmp_path / "journal.jsonl").read_all()
    stages = [r["stage"] for r in records]
    assert stages == ["data_pull", "signal", "research_context", "decision",
                      "proposed_trade", "risk_check", "execution"]
    assert len({r["run_id"] for r in records}) == 1
    # risk stage journals every check with detail
    risk_payload = records[5]["payload"]
    assert len(risk_payload["checks"]) == 5
    assert all(c["detail"] for c in risk_payload["checks"])
    # execution journals the order and the portfolio after it
    exec_payload = records[6]["payload"]
    assert exec_payload["order"]["mode"] == "simulated"
    assert exec_payload["portfolio_after"]["positions"]["NVDA"]["qty"] == filled


def test_pipeline_hold_decision_skips_risk_and_execution(tmp_path):
    result = run(tmp_path, decide_fn=make_fake_decide("HOLD"))
    assert result["trade"] is None
    assert result["risk"] is None
    assert result["order"] is None
    records = Journal(tmp_path / "journal.jsonl").read_all()
    assert [r["stage"] for r in records][-1] == "proposed_trade"
    assert records[-1]["payload"]["trade"] is None
    assert "reason" in records[-1]["payload"]


def test_pipeline_blocked_trade_never_reaches_broker(tmp_path):
    class ExplodingBroker:
        def submit(self, trade):
            raise AssertionError("broker must not be called on a blocked trade")

    result = run(
        tmp_path,
        decide_fn=make_fake_decide("BUY"),
        broker=ExplodingBroker(),
        risk_limits=RiskLimits(max_position_notional=1.0,
                               kill_switch_path=str(tmp_path / "KILL_SWITCH")),
    )
    assert result["risk"]["approved"] is False
    assert result["order"] is None
    assert result["portfolio"]["positions"] == {}
    records = Journal(tmp_path / "journal.jsonl").read_all()
    assert [r["stage"] for r in records][-1] == "risk_check"


def test_pipeline_kill_switch_blocks_trade(tmp_path):
    (tmp_path / "KILL_SWITCH").touch()
    result = run(tmp_path, decide_fn=make_fake_decide("BUY"))
    assert result["risk"]["approved"] is False
    assert result["order"] is None
    kill = next(c for c in result["risk"]["checks"] if c["name"] == "kill_switch")
    assert kill["passed"] is False
