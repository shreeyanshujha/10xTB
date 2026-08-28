"""Deterministic risk layer (spec section 4.5) — plain code, hard limits,
sits between the Decision Agent and execution. No LLM output is an input
here and nothing can waive a failed check: the agent's rationale is not
even passed in. Every check is reported pass/block (spec section 4.6).
Rejection is total — this layer never resizes a trade."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tenx.portfolio import Portfolio


@dataclass(frozen=True)
class RiskLimits:
    max_position_notional: float = 10_000.0
    max_gross_exposure: float = 30_000.0
    max_drawdown_pct: float = 0.20
    kill_switch_path: str = "data/KILL_SWITCH"


@dataclass(frozen=True)
class RiskVerdict:
    approved: bool
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_trade(
    trade: dict,
    portfolio: Portfolio,
    prices: dict,
    limits: RiskLimits = RiskLimits(),
) -> RiskVerdict:
    ticker = trade["ticker"]
    qty = int(trade["quantity"])
    price = float(trade["price"])
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    kill = Path(limits.kill_switch_path).exists()
    add("kill_switch", not kill,
        f"kill switch file {limits.kill_switch_path} "
        + ("EXISTS — all trading halted" if kill else "absent"))

    add("price_sanity", price > 0, f"price {price}")

    proposed_notional = portfolio.position_notional(ticker, price) + abs(qty) * price
    add("position_cap", proposed_notional <= limits.max_position_notional,
        f"{ticker} notional after trade {proposed_notional:.2f} "
        f"vs cap {limits.max_position_notional:.2f}")

    proposed_gross = portfolio.gross_exposure(prices) + abs(qty) * price
    add("gross_exposure_cap", proposed_gross <= limits.max_gross_exposure,
        f"gross exposure after trade {proposed_gross:.2f} "
        f"vs cap {limits.max_gross_exposure:.2f}")

    equity = portfolio.equity(prices)
    floor = portfolio.equity_peak * (1 - limits.max_drawdown_pct)
    add("drawdown_breaker", equity >= floor,
        f"equity {equity:.2f} vs floor {floor:.2f} "
        f"(peak {portfolio.equity_peak:.2f}, max dd {limits.max_drawdown_pct:.0%})")

    return RiskVerdict(approved=all(c["passed"] for c in checks), checks=checks)
