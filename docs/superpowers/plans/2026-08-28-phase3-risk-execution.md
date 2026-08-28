# Phase 3 — Deterministic Risk Layer + IBKR Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a deterministic risk layer (hard caps, drawdown circuit breaker, kill switch — plain code, no LLM anywhere near it) between the Decision Agent and execution, and wire real order submission behind a broker interface: IBKR paper via `ib_async`, with a clearly-labeled simulated fallback.

**Architecture:** Three new modules. `tenx/portfolio.py`: JSON-file-backed position/cash/equity-peak state the risk checks read and fills update. `tenx/risk.py`: `RiskLimits` (frozen config), `check_trade()` returning a `RiskVerdict` that lists EVERY check with pass/block and detail (spec §4.6 — the risk layer journals every check, not just outcomes). `tenx/execution.py`: `OrderResult`, `SimBroker` (always available, fills at the proposed price, labeled `mode: "simulated"`), and `IBKRBroker` (`ib_async` → IB Gateway/TWS paper account; the `ib` object is injectable so order-mapping logic is unit-testable without a gateway). The pipeline chain becomes data → signal → research → decision → proposed_trade → risk_check → execution, with portfolio state updated on fill and the full causal chain journaled under one run_id.

**Why `ib_async` and not the session's IBKR MCP connector:** the MCP connector is bound to an interactive claude.ai session and cannot serve the autonomous n8n-scheduled pipeline on BlackICE. `ib_async==2.1.0` (pinned, verified installing 2026-08-28) talks directly to IB Gateway/TWS — same interface for paper now and live later, exactly as spec §4.1 requires.

**Tech Stack:** existing stack + `ib_async==2.1.0`. New pytest marker `broker` (live-gateway tests, deselected by default — no IB Gateway runs on this dev machine).

**Spec:** `docs/BUILD_PLAN.md` — Phase 3 row of §6, §4.5 (deterministic guardrail layer), §4.6 (logging), §9 (guardrails).

## Global Constraints

- The risk layer is plain code with hard limits. No LLM output can alter, waive, or resize anything in it; the Decision Agent's text is not even an input to `check_trade` (spec §4.5, §9). Rejection is total — the risk layer never "adjusts" a trade.
- Every check is journaled pass or block with a reason, every order with its full causal chain back to signal + rationale (spec §4.6).
- Kill switch = existence of the file `data/KILL_SWITCH`. Anything (user, n8n, a script) can `touch` it; while it exists, every trade is blocked. No code path deletes it automatically.
- Execution is plumbing, not a decision point: a broker receives an approved trade verbatim.
- Simulated fills must be unmistakably labeled (`mode: "simulated"`) in both `OrderResult` and the journal — a sim fill must never be confusable with an IBKR paper fill.
- Default limits (config, overridable): max position notional $10,000/ticker; max gross exposure $30,000; max drawdown 20% from equity peak; starting paper cash $100,000.

---

### Task 1: Portfolio state

**Files:**
- Create: `tenx/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Produces: `Portfolio(path: Path | str, starting_cash: float = 100_000.0)` — loads JSON state from `path` if it exists, else initializes `{cash: starting_cash, positions: {}, equity_peak: starting_cash}`. Methods: `apply_fill(ticker, side, qty, price)` (BUY decreases cash / increases signed qty, SELL the reverse; avg_price tracked for reporting), `position_qty(ticker) -> int`, `position_notional(ticker, price) -> float` (abs), `gross_exposure(prices: dict[str, float]) -> float` (sum of abs(qty)·price; falls back to stored avg_price when a price is missing), `equity(prices) -> float` (cash + signed market value), `update_equity_peak(equity) -> float` (ratchets up only), `save()`, `to_dict()`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_portfolio.py
from tenx.portfolio import Portfolio


def test_new_portfolio_defaults(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    assert p.cash == 100_000.0
    assert p.position_qty("NVDA") == 0
    assert p.equity({}) == 100_000.0
    assert p.to_dict()["equity_peak"] == 100_000.0


def test_buy_and_sell_accounting(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "BUY", 40, 200.0)
    assert p.cash == 100_000.0 - 8_000.0
    assert p.position_qty("NVDA") == 40
    assert p.position_notional("NVDA", 210.0) == 40 * 210.0
    p.apply_fill("NVDA", "SELL", 15, 210.0)
    assert p.cash == 92_000.0 + 15 * 210.0
    assert p.position_qty("NVDA") == 25


def test_short_position_counts_in_gross_exposure(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "SELL", 10, 200.0)
    assert p.position_qty("NVDA") == -10
    assert p.gross_exposure({"NVDA": 200.0}) == 2_000.0
    # equity: short position subtracts market value, cash was credited
    assert p.equity({"NVDA": 200.0}) == 100_000.0


def test_equity_peak_only_ratchets_up(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    assert p.update_equity_peak(110_000.0) == 110_000.0
    assert p.update_equity_peak(90_000.0) == 110_000.0


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "pf.json"
    p = Portfolio(path)
    p.apply_fill("NVDA", "BUY", 40, 200.0)
    p.update_equity_peak(101_000.0)
    p.save()
    p2 = Portfolio(path)
    assert p2.cash == p.cash
    assert p2.position_qty("NVDA") == 40
    assert p2.to_dict()["equity_peak"] == 101_000.0


def test_gross_exposure_falls_back_to_avg_price(tmp_path):
    p = Portfolio(tmp_path / "pf.json")
    p.apply_fill("NVDA", "BUY", 10, 200.0)
    assert p.gross_exposure({}) == 2_000.0  # no live price -> avg_price
```

- [ ] **Step 2: Verify fail (ModuleNotFoundError), implement**

```python
# tenx/portfolio.py
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
```

- [ ] **Step 3: Verify 6 pass; commit** — `git add -A && git commit -m "feat: JSON-backed paper portfolio state"`

---

### Task 2: Deterministic risk layer

**Files:**
- Create: `tenx/risk.py`
- Test: `tests/test_risk.py`

**Interfaces:**
- Consumes: proposed trade dict (Phase 2 `build_paper_trade` shape), `Portfolio` (Task 1).
- Produces: `RiskLimits` frozen dataclass (`max_position_notional=10_000.0`, `max_gross_exposure=30_000.0`, `max_drawdown_pct=0.20`, `kill_switch_path="data/KILL_SWITCH"`); `RiskVerdict` frozen dataclass (`approved: bool`, `checks: list[dict]` each `{"name", "passed", "detail"}`, `to_dict()`); `check_trade(trade: dict, portfolio: Portfolio, prices: dict[str, float], limits: RiskLimits = RiskLimits()) -> RiskVerdict`. ALL checks always run and are reported (no short-circuit hiding later checks); `approved = all passed`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_risk.py
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
```

- [ ] **Step 2: Verify fail, implement**

```python
# tenx/risk.py
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
```

- [ ] **Step 3: Verify 7 pass; commit** — `git add -A && git commit -m "feat: deterministic risk layer — caps, drawdown breaker, kill switch"`

---

### Task 3: Execution layer (SimBroker + IBKRBroker)

**Files:**
- Create: `tenx/execution.py`
- Modify: `pyproject.toml` (add `ib_async==2.1.0` dep + `broker` marker; addopts `-m 'not network and not llm and not broker'`)
- Test: `tests/test_execution.py`

**Interfaces:**
- Produces: `OrderResult` frozen dataclass (`order_id: str`, `status: str` — `"filled" | "submitted" | "rejected" | "timeout"`, `filled_qty: int`, `avg_fill_price: float | None`, `mode: str` — `"simulated" | "ibkr_paper"`, `detail: str`, `to_dict()`); `SimBroker()` with `submit(trade: dict) -> OrderResult` (fills fully at `trade["price"]`); `IBKRBroker(host="127.0.0.1", port=7497, client_id=1, timeout_s=30.0, ib=None)` with the same `submit` signature (`ib` injectable; lazily imports `ib_async` and connects on first use).

- [ ] **Step 1: pyproject: add dep + `broker` marker, reinstall `-e .`, refreeze lock**

- [ ] **Step 2: Failing tests**

```python
# tests/test_execution.py
from types import SimpleNamespace

from tenx.execution import IBKRBroker, OrderResult, SimBroker


def make_trade(qty=40, price=200.0, side="BUY"):
    return {"ticker": "NVDA", "side": side, "quantity": qty, "price": price,
            "notional_usd": 8_000.0, "signal_as_of": "2026-08-27",
            "hypothetical": True}


def test_sim_broker_fills_at_proposed_price():
    result = SimBroker().submit(make_trade())
    assert isinstance(result, OrderResult)
    assert result.status == "filled"
    assert result.filled_qty == 40
    assert result.avg_fill_price == 200.0
    assert result.mode == "simulated"
    assert result.order_id


def test_sim_broker_order_ids_are_unique():
    b = SimBroker()
    assert b.submit(make_trade()).order_id != b.submit(make_trade()).order_id


class FakeIB:
    """Mimics the slice of ib_async.IB that IBKRBroker touches."""

    def __init__(self, final_status="Filled", filled=40, avg_price=201.5):
        self._status = SimpleNamespace(status=final_status)
        self._fill = SimpleNamespace(filled=filled, avgFillPrice=avg_price)
        self.placed = []

    def isConnected(self):
        return True

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return SimpleNamespace(
            order=SimpleNamespace(orderId=7),
            orderStatus=SimpleNamespace(
                status=self._status.status,
                filled=self._fill.filled,
                avgFillPrice=self._fill.avgFillPrice,
            ),
            isDone=lambda: True,
        )

    def sleep(self, seconds):
        pass


def test_ibkr_broker_maps_trade_to_order():
    fake = FakeIB()
    result = IBKRBroker(ib=fake).submit(make_trade(qty=40, side="BUY"))
    assert result.status == "filled"
    assert result.filled_qty == 40
    assert result.avg_fill_price == 201.5
    assert result.mode == "ibkr_paper"
    contract, order = fake.placed[0]
    assert contract.symbol == "NVDA"
    assert order.action == "BUY"
    assert order.totalQuantity == 40


def test_ibkr_broker_reports_unfilled_as_timeout():
    fake = FakeIB(final_status="Submitted", filled=0, avg_price=None)
    fake_trade_not_done = fake.placeOrder  # keep behavior; isDone False below

    class FakeIBPending(FakeIB):
        def placeOrder(self, contract, order):
            t = super().placeOrder(contract, order)
            t.isDone = lambda: False
            return t

    result = IBKRBroker(ib=FakeIBPending(final_status="Submitted", filled=0,
                                         avg_price=None),
                        timeout_s=0.0).submit(make_trade())
    assert result.status == "timeout"
    assert result.mode == "ibkr_paper"
```

- [ ] **Step 3: Verify fail, implement**

```python
# tenx/execution.py
"""Execution layer (spec section 4.5): plumbing, not a decision point.
Receives a risk-approved trade verbatim and submits it. Two brokers:
SimBroker (no gateway needed; fills at the proposed price and is loudly
labeled simulated) and IBKRBroker (ib_async -> IB Gateway/TWS paper
account, port 7497 = TWS paper default). Same interface for paper now
and live later (spec section 4.1)."""
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: str  # "filled" | "submitted" | "rejected" | "timeout"
    filled_qty: int
    avg_fill_price: float | None
    mode: str  # "simulated" | "ibkr_paper"
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SimBroker:
    """Instant full fill at the proposed price. NOT a market simulation —
    a stand-in so the pipeline runs end-to-end without an IB Gateway."""

    def submit(self, trade: dict) -> OrderResult:
        return OrderResult(
            order_id=f"sim-{uuid.uuid4().hex[:10]}",
            status="filled",
            filled_qty=int(trade["quantity"]),
            avg_fill_price=float(trade["price"]),
            mode="simulated",
            detail="simulated fill at proposed price",
        )


class IBKRBroker:
    """Submits market orders to IB Gateway/TWS (paper account first —
    port 7497 is the TWS paper default; IB Gateway paper is 4002)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, timeout_s: float = 30.0, ib=None):
        self.host, self.port, self.client_id = host, port, client_id
        self.timeout_s = timeout_s
        self._ib = ib

    def _connect(self):
        if self._ib is None:
            from ib_async import IB
            self._ib = IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        return self._ib

    def submit(self, trade: dict) -> OrderResult:
        from ib_async import MarketOrder, Stock

        ib = self._connect()
        contract = Stock(trade["ticker"], "SMART", "USD")
        order = MarketOrder(trade["side"], int(trade["quantity"]))
        ib_trade = ib.placeOrder(contract, order)

        deadline = time.monotonic() + self.timeout_s
        while not ib_trade.isDone() and time.monotonic() < deadline:
            ib.sleep(0.25)

        status = ib_trade.orderStatus.status
        filled = int(ib_trade.orderStatus.filled or 0)
        avg = ib_trade.orderStatus.avgFillPrice
        if status == "Filled":
            result_status = "filled"
        elif status in ("Cancelled", "Inactive", "ApiCancelled"):
            result_status = "rejected"
        elif not ib_trade.isDone():
            result_status = "timeout"
        else:
            result_status = "submitted"
        return OrderResult(
            order_id=str(ib_trade.order.orderId),
            status=result_status,
            filled_qty=filled,
            avg_fill_price=float(avg) if avg else None,
            mode="ibkr_paper",
            detail=f"IB status {status}",
        )
```

Note for the timeout test: `IBKRBroker` must not import `ib_async` when `ib` is injected AND the loop must consult `isDone()` before status mapping — the fake with `isDone -> False` and `timeout_s=0.0` exercises the timeout branch. `Stock`/`MarketOrder` ARE imported in `submit` even with a fake `ib` — that import is cheap and the pinned dependency is installed, so the fake-injected tests still pass.

- [ ] **Step 4: Verify 4 pass; commit** — `git add -A && git commit -m "feat: execution layer — SimBroker + ib_async IBKRBroker"`

---

### Task 4: Pipeline wiring — risk gate + execution + portfolio update

**Files:**
- Modify: `tenx/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- `run_pipeline(..., broker=None, risk_limits: RiskLimits = RiskLimits(), portfolio_path="data/portfolio.json")` — `broker=None` → `SimBroker()`. Stage order journaled: `data_pull`, `signal`, `research_context`, `decision`, `proposed_trade`, then if a trade was proposed `risk_check` (payload = `verdict.to_dict()`), then if approved `execution` (payload = `{"order": order.to_dict(), "portfolio_after": portfolio.to_dict()}`). Portfolio updated+saved only on `status == "filled"`. Returns dict with added keys `"risk"` (verdict dict or None), `"order"` (order dict or None), `"portfolio"` (state dict).
- CLI: `TENX_BROKER=ibkr` env var selects `IBKRBroker` (`TENX_IB_HOST`/`TENX_IB_PORT`/`TENX_IB_CLIENT_ID` optional overrides); default stays `SimBroker`.

- [ ] **Step 1: Update `tests/test_pipeline.py`** — keep the fake fetch/decide helpers; assertions become:
  - BUY-decision path: stages `[data_pull, signal, research_context, decision, proposed_trade, risk_check, execution]`; `result["risk"]["approved"] is True`; `result["order"]["mode"] == "simulated"` and `status == "filled"`; `result["portfolio"]["positions"]["NVDA"]["qty"] == result["order"]["filled_qty"]`; portfolio JSON file exists under `tmp_path`.
  - HOLD-decision path: stages end at `proposed_trade` with `{"trade": None, "reason": ...}`; `result["risk"] is None and result["order"] is None`.
  - Blocked path: pass `risk_limits=RiskLimits(max_position_notional=1.0, kill_switch_path=str(tmp_path/"KS"))` → stages end at `risk_check`; `result["risk"]["approved"] is False`; `result["order"] is None`; portfolio unchanged (no NVDA position).
  - All three use `portfolio_path=tmp_path / "pf.json"` so state never leaks between tests.

- [ ] **Step 2: Verify fail, implement pipeline changes**

```python
# pipeline.py additions (signature + after the decision stage)
from tenx.execution import SimBroker
from tenx.portfolio import Portfolio
from tenx.risk import RiskLimits, check_trade

    proposed = build_paper_trade(
        decision.action, ticker,
        price=signal.features["last_close"], as_of=signal.as_of,
    )
    portfolio = Portfolio(portfolio_path)
    prices = {ticker: float(signal.features["last_close"])}
    risk_verdict = order = None

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

    return {"run_id": run_id, "ticker": ticker, "signal": signal.to_dict(),
            "research": research, "decision": decision.to_dict(),
            "trade": proposed,
            "risk": risk_verdict.to_dict() if risk_verdict else None,
            "order": order.to_dict() if order else None,
            "portfolio": portfolio.to_dict()}
```

`__main__` block grows the broker factory:

```python
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
```

- [ ] **Step 3: Full suite green; commit** — `git add -A && git commit -m "feat: risk-gated execution pipeline with portfolio state; Phase 3 complete"`

---

## Self-Review Notes

- §4.5 coverage: risk layer is pure code, runs between decision and execution, rejection is total, kill switch + caps + drawdown breaker present; execution is plumbing behind a stable broker interface (paper now, live later — §4.1). §4.6: every check journaled with detail; every order journaled with portfolio-after and the run_id causal chain. §9: nothing in the agent's output reaches `check_trade` except the proposed trade's numeric fields.
- Live-gateway verification (`IBKRBroker` against a real paper account) cannot run on this machine — no IB Gateway/TWS installed. The broker's order-mapping logic is fake-tested; first real-gateway run is a documented follow-up for the BlackICE deployment.
- Type consistency: `OrderResult.to_dict()`/`RiskVerdict.to_dict()`/`Portfolio.to_dict()` keys match the pipeline's journal payloads and return dict; `build_paper_trade`'s Phase 2 shape is unchanged as the proposed-trade shape.
