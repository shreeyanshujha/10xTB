"""Hypothetical paper trades for the Phase 0 skeleton. No broker involved —
real IBKR paper execution (and the deterministic risk layer in front of it)
arrives in Phase 3 (spec: docs/BUILD_PLAN.md section 6)."""
from tenx.signals.base import Signal


def build_paper_trade(signal: Signal, notional_usd: float = 10_000.0) -> dict | None:
    if signal.action == "HOLD":
        return None
    price = float(signal.features["last_close"])
    quantity = int(notional_usd // price)
    if quantity < 1:
        raise ValueError(
            f"price {price} exceeds notional {notional_usd}; no whole share affordable"
        )
    return {
        "ticker": signal.ticker,
        "side": signal.action,
        "quantity": quantity,
        "price": price,
        "notional_usd": notional_usd,
        "signal_as_of": signal.as_of,
        "hypothetical": True,
    }
