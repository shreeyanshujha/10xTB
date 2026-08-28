"""JSON-file-backed paper portfolio state: positions, cash, equity peak.
Read by the deterministic risk layer (tenx/risk.py) and updated by the
execution layer on fills. Not an agent, not a decision point."""
import json
from pathlib import Path


class Portfolio:
    def __init__(self, path: Path | str, starting_cash: float = 100_000.0):
        self.path = Path(path)
        if self.path.exists():
            state = json.loads(self.path.read_text())
            self.cash = float(state["cash"])
            self.positions = state["positions"]
            self.equity_peak = float(state["equity_peak"])
        else:
            self.cash = float(starting_cash)
            self.positions = {}
            self.equity_peak = float(starting_cash)

    def apply_fill(self, ticker: str, side: str, qty: int, price: float) -> None:
        qty, price = int(qty), float(price)
        pos = self.positions.setdefault(ticker, {"qty": 0, "avg_price": 0.0})
        signed = qty if side == "BUY" else -qty
        self.cash -= signed * price
        new_qty = pos["qty"] + signed
        if new_qty != 0 and (pos["qty"] == 0 or (pos["qty"] > 0) != (new_qty > 0)):
            pos["avg_price"] = price  # opened or flipped: basis resets
        elif abs(new_qty) > abs(pos["qty"]):
            total = pos["avg_price"] * abs(pos["qty"]) + price * qty
            pos["avg_price"] = total / abs(new_qty)  # added to position
        pos["qty"] = new_qty

    def position_qty(self, ticker: str) -> int:
        return self.positions.get(ticker, {}).get("qty", 0)

    def position_notional(self, ticker: str, price: float) -> float:
        return abs(self.position_qty(ticker)) * float(price)

    def _price_for(self, ticker: str, prices: dict) -> float:
        if ticker in prices:
            return float(prices[ticker])
        return float(self.positions[ticker]["avg_price"])

    def gross_exposure(self, prices: dict) -> float:
        return sum(
            abs(p["qty"]) * self._price_for(t, prices)
            for t, p in self.positions.items() if p["qty"] != 0
        )

    def equity(self, prices: dict) -> float:
        return self.cash + sum(
            p["qty"] * self._price_for(t, prices)
            for t, p in self.positions.items() if p["qty"] != 0
        )

    def update_equity_peak(self, equity: float) -> float:
        self.equity_peak = max(self.equity_peak, float(equity))
        return self.equity_peak

    def to_dict(self) -> dict:
        return {"cash": self.cash, "positions": self.positions,
                "equity_peak": self.equity_peak}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2))
