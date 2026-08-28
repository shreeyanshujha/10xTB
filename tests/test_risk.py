import pytest

from tenx.portfolio import Portfolio
from tenx.risk import RiskLimits, RiskVerdict, check_trade

CHECK_NAMES = ["kill_switch", "price_sanity", "position_cap",
               "gross_exposure_cap", "drawdown_breaker"]


def make_trade(qty=40, price=200.0, side="BUY", ticker="NVDA"):
    return {"ticker": ticker, "side": side, "quantity": qty, "price": price,
            "notional_usd": 10_000.0, "signal_as_of": "2026-08-27",
            "hypothetical": True}


def limits(tmp_path, **kw):
    kw.setdefault("kill_switch_path", str(tmp_path / "KILL_SWITCH"))
    return RiskLimits(**kw)


def test_clean_trade_is_approved_and_all_checks_reported(tmp_path):
    v = check_trade(make_trade(), Portfolio(tmp_path / "pf.json"),
                    {"NVDA": 200.0}, limits(tmp_path))
    assert isinstance(v, RiskVerdict)
    assert v.approved is True
    assert [c["name"] for c in v.checks] == CHECK_NAMES
    assert all(c["passed"] for c in v.checks)
    assert all(c["detail"] for c in v.checks)


def test_kill_switch_blocks_everything(tmp_path):
    (tmp_path / "KILL_SWITCH").touch()
    v = check_trade(make_trade(), Portfolio(tmp_path / "pf.json"),
                    {"NVDA": 200.0}, limits(tmp_path))
    assert v.approved is False
    kill = next(c for c in v.checks if c["name"] == "kill_switch")
    assert kill["passed"] is False
    # later checks still run and are reported — no silent short-circuit
    assert len(v.checks) == len(CHECK_NAMES)


def test_position_cap_blocks_oversized_position(tmp_path):
    pf = Portfolio(tmp_path / "pf.json")
    pf.apply_fill("NVDA", "BUY", 45, 200.0)  # $9,000 already held
    v = check_trade(make_trade(qty=10, price=200.0), pf,
                    {"NVDA": 200.0}, limits(tmp_path))  # would be $11,000
    assert v.approved is False
    assert next(c for c in v.checks if c["name"] == "position_cap")["passed"] is False


def test_gross_exposure_cap_counts_other_tickers(tmp_path):
    pf = Portfolio(tmp_path / "pf.json")
    pf.apply_fill("AMD", "BUY", 100, 290.0)  # $29,000 gross elsewhere
    v = check_trade(make_trade(qty=10, price=200.0), pf,
                    {"NVDA": 200.0, "AMD": 290.0}, limits(tmp_path))
    assert v.approved is False
    assert next(c for c in v.checks if c["name"] == "gross_exposure_cap")["passed"] is False


def test_drawdown_breaker_blocks_after_20pct_loss(tmp_path):
    pf = Portfolio(tmp_path / "pf.json")
    pf.update_equity_peak(100_000.0)
    pf.cash = 79_000.0  # equity 79k < 80% of peak
    v = check_trade(make_trade(), pf, {"NVDA": 200.0}, limits(tmp_path))
    assert v.approved is False
    assert next(c for c in v.checks if c["name"] == "drawdown_breaker")["passed"] is False


def test_nonpositive_price_blocked(tmp_path):
    v = check_trade(make_trade(price=0.0), Portfolio(tmp_path / "pf.json"),
                    {}, limits(tmp_path))
    assert v.approved is False
    assert next(c for c in v.checks if c["name"] == "price_sanity")["passed"] is False


def test_verdict_serializes(tmp_path):
    v = check_trade(make_trade(), Portfolio(tmp_path / "pf.json"),
                    {"NVDA": 200.0}, limits(tmp_path))
    d = v.to_dict()
    assert d["approved"] is True and len(d["checks"]) == len(CHECK_NAMES)
