"""Hypothetical paper trades for the pre-Phase-3 pipeline. The side comes
from the Decision Agent's recommendation; no broker is involved — real IBKR
paper execution and the deterministic risk layer arrive in Phase 3."""


def build_paper_trade(
    side: str, ticker: str, price: float, as_of: str,
    notional_usd: float = 10_000.0,
) -> dict | None:
    if side == "HOLD":
        return None
    price = float(price)
    quantity = int(notional_usd // price)
    if quantity < 1:
        raise ValueError(
            f"price {price} exceeds notional {notional_usd}; no whole share affordable"
        )
    return {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "price": price,
        "notional_usd": notional_usd,
        "signal_as_of": as_of,
        "hypothetical": True,
    }
