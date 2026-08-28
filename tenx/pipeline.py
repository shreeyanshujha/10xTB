"""Phase 0 walking-skeleton orchestrator: data pull -> placeholder signal ->
hypothetical paper trade, journaling every stage with a shared run_id so any
trade (or non-trade) traces back to the data and rationale that produced it."""
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from tenx.journal import Journal
from tenx.paper import build_paper_trade
from tenx.signals.placeholder import sma_crossover_signal


def run_pipeline(
    ticker: str = "NVDA",
    journal_path: str | Path = "data/journal.jsonl",
    fetch: Callable | None = None,
    lookback_days: int = 120,
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

    signal = sma_crossover_signal(df, ticker)
    journal.log(run_id, "signal", signal.to_dict())

    trade = build_paper_trade(signal)
    if trade is None:
        journal.log(run_id, "paper_trade", {
            "trade": None,
            "reason": f"signal action was {signal.action}; no trade taken",
        })
    else:
        journal.log(run_id, "paper_trade", {"trade": trade})

    return {
        "run_id": run_id,
        "ticker": ticker,
        "signal": signal.to_dict(),
        "trade": trade,
    }


if __name__ == "__main__":
    result = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "NVDA")
    print(result)
