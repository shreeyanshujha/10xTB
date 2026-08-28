"""Phase 0 walking-skeleton orchestrator: data pull -> placeholder signal ->
hypothetical paper trade, journaling every stage with a shared run_id so any
trade (or non-trade) traces back to the data and rationale that produced it."""
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from tenx.agents.decision import decide
from tenx.agents.research_stub import stub_research_context
from tenx.execution import SimBroker
from tenx.journal import Journal
from tenx.paper import build_paper_trade
from tenx.portfolio import Portfolio
from tenx.risk import RiskLimits, check_trade
from tenx.signals.quant_model import quant_signal


def run_pipeline(
    ticker: str = "NVDA",
    journal_path: str | Path = "data/journal.jsonl",
    fetch: Callable | None = None,
    signal_fn: Callable = quant_signal,
    research_fn: Callable = stub_research_context,
    decide_fn: Callable = decide,
    broker=None,
    risk_limits: RiskLimits = RiskLimits(),
    portfolio_path: str | Path = "data/portfolio.json",
    lookback_days: int = 1600,
) -> dict:
    if fetch is None:
        from tenx.data.openbb_provider import get_daily_bars
        fetch = get_daily_bars

    journal = Journal(journal_path)
    run_id = uuid.uuid4().hex[:12]
    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    df = fetch(ticker, start_date=start_date)
    journal.log(run_id, "data_pull", {
        "ticker": ticker,
        "start_date": start_date,
        "rows": len(df),
        "first_bar": df.index[0].date().isoformat(),
        "last_bar": df.index[-1].date().isoformat(),
        "last_close": float(df["close"].iloc[-1]),
    })

    signal = signal_fn(df, ticker)
    journal.log(run_id, "signal", signal.to_dict())

    research = research_fn(ticker)
    journal.log(run_id, "research_context", research)

    decision = decide_fn(signal, research)
    journal.log(run_id, "decision", decision.to_dict())

    proposed = build_paper_trade(
        decision.action, ticker,
        price=signal.features["last_close"], as_of=signal.as_of,
    )
    portfolio = Portfolio(portfolio_path)
    prices = {ticker: float(signal.features["last_close"])}
    risk_verdict = None
    order = None

    if proposed is None:
        journal.log(run_id, "proposed_trade", {
            "trade": None,
            "reason": f"decision was {decision.action}; no trade proposed",
        })
    else:
        journal.log(run_id, "proposed_trade", {"trade": proposed})
        risk_verdict = check_trade(proposed, portfolio, prices, risk_limits)
        journal.log(run_id, "risk_check", risk_verdict.to_dict())
        if risk_verdict.approved:
            if broker is None:
                broker = SimBroker()
            order = broker.submit(proposed)
            if order.status == "filled":
                portfolio.apply_fill(ticker, proposed["side"],
                                     order.filled_qty, order.avg_fill_price)
                portfolio.update_equity_peak(portfolio.equity(prices))
                portfolio.save()
            journal.log(run_id, "execution", {
                "order": order.to_dict(),
                "portfolio_after": portfolio.to_dict(),
            })

    return {
        "run_id": run_id,
        "ticker": ticker,
        "signal": signal.to_dict(),
        "research": research,
        "decision": decision.to_dict(),
        "trade": proposed,
        "risk": risk_verdict.to_dict() if risk_verdict else None,
        "order": order.to_dict() if order else None,
        "portfolio": portfolio.to_dict(),
    }


if __name__ == "__main__":
    import os

    broker = None
    if os.environ.get("TENX_BROKER") == "ibkr":
        from tenx.execution import IBKRBroker
        broker = IBKRBroker(
            host=os.environ.get("TENX_IB_HOST", "127.0.0.1"),
            port=int(os.environ.get("TENX_IB_PORT", "7497")),
            client_id=int(os.environ.get("TENX_IB_CLIENT_ID", "1")),
        )
    result = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "NVDA",
                          broker=broker)
    print(result)
