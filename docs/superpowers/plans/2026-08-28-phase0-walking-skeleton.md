# Phase 0 — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the full pipe runs end-to-end for one stock (NVDA): OpenBB data pull → deliberately-dumb signal → hypothetical paper trade, with every stage journaled.

**Architecture:** A single Python package (`tenx`) with one module per future layer: `data/` (OpenBB wrapper — the only module allowed to import openbb), `signals/` (placeholder SMA crossover), `paper.py` (hypothetical trade builder — replaced by real IBKR execution in Phase 3), `journal.py` (append-only JSONL reasoning trail), and `pipeline.py` (orchestrator with an injectable data-fetch so the test suite runs offline). No agents, no risk layer, no IBKR yet — those are Phases 2–4.

**Tech Stack:** Python 3.12.14 (uv-managed; OpenBB does not yet support the system Python 3.14), OpenBB Platform `4.7.2` with `openbb-yfinance 1.6.3` (verified working 2026-08-28), pandas 3.0.5, pytest.

**Spec:** `docs/BUILD_PLAN.md` (the full multi-phase build plan; this plan implements only its Phase 0 row).

## Global Constraints

- Only `tenx/data/openbb_provider.py` may import from `openbb` — every other layer reads the standardized DataFrame it returns (spec §4.2).
- Every stage logs its full reasoning/inputs to the journal, not just outcomes (spec §4.6).
- The placeholder signal is intentionally dumb — do NOT make it smart; that is Phase 1's job (spec §6, Phase 0 row).
- Pin exact dependency versions: `openbb==4.7.2`; freeze full tree to `requirements.lock`. Full fork/vendor of OpenBB is a BlackICE-deployment concern, deferred past Phase 0 but before relying on it in production (spec §4.2 notes OpenBB the company wound down Aug 2026).
- Test suite must pass offline; network-dependent tests are marked `network` and deselected by default.
- Package name is `tenx` (directory `10xTB` is not a valid Python identifier).

---

### Task 1: Repo scaffold, environment, pinned dependencies

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `docs/BUILD_PLAN.md` (paste the spec verbatim), `tenx/__init__.py`, `tenx/data/__init__.py`, `tenx/signals/__init__.py`, `tests/__init__.py`, `requirements.lock`
- Pre-existing: `.venv/` (uv venv on Python 3.12.14 with openbb 4.7.2 already installed)

**Interfaces:**
- Produces: an importable `tenx` package; `pytest` runnable via `.venv/bin/python -m pytest`; git repo with initial commit.

- [ ] **Step 1: git init and .gitignore**

```bash
cd /home/shreeyanshu/Projects/active/10xTB && git init -b main
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
/data/
*.egg-info/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "tenx"
version = "0.0.1"
description = "Agentic trading platform — Phase 0 walking skeleton"
requires-python = ">=3.12,<3.13"
dependencies = [
    "openbb==4.7.2",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools]
packages = ["tenx", "tenx.data", "tenx.signals"]

[tool.pytest.ini_options]
addopts = "-m 'not network'"
markers = [
    "network: tests that hit live data providers (deselected by default)",
]
```

- [ ] **Step 3: Create package skeleton and save the spec**

Create empty `tenx/__init__.py`, `tenx/data/__init__.py`, `tenx/signals/__init__.py`, `tests/__init__.py`. Save the full build-plan document (provided in the kickoff message) verbatim to `docs/BUILD_PLAN.md`.

- [ ] **Step 4: Install dev deps + editable package, freeze lock**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python .venv/bin/python pytest -e .
uv pip freeze --python .venv/bin/python > requirements.lock
```

- [ ] **Step 5: Verify pytest runs (0 tests, exit cleanly)**

Run: `.venv/bin/python -m pytest --collect-only -q`
Expected: "no tests ran" / empty collection, no import errors.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: Phase 0 scaffold — pinned OpenBB 4.7.2, py3.12 venv, spec vendored"
```

---

### Task 2: OpenBB data provider

**Files:**
- Create: `tenx/data/openbb_provider.py`
- Test: `tests/test_provider.py`

**Interfaces:**
- Produces: `get_daily_bars(ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame` — returns a DataFrame indexed by date with exactly columns `["open", "high", "low", "close", "volume"]`, sorted ascending by date. Raises `ValueError` if the result is empty.

- [ ] **Step 1: Write the failing (network-marked) test**

```python
# tests/test_provider.py
import pytest

from tenx.data.openbb_provider import get_daily_bars

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@pytest.mark.network
def test_get_daily_bars_nvda_live():
    df = get_daily_bars("NVDA", start_date="2026-07-01")
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) > 10
    assert df.index.is_monotonic_increasing
    assert (df["close"] > 0).all()


@pytest.mark.network
def test_get_daily_bars_bad_ticker_raises():
    with pytest.raises(ValueError):
        get_daily_bars("ZZZZNOTATICKER", start_date="2026-07-01")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_provider.py -m network -v`
Expected: FAIL/ERROR with `ModuleNotFoundError` or `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Implement the provider**

```python
# tenx/data/openbb_provider.py
"""The only module in the codebase allowed to import from openbb.

Every other layer consumes the standardized DataFrame returned here
(spec: docs/BUILD_PLAN.md section 4.2).
"""
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_daily_bars(
    ticker: str, start_date: str, end_date: str | None = None
) -> pd.DataFrame:
    from openbb import obb  # deferred: openbb import is slow (~seconds)

    try:
        result = obb.equity.price.historical(
            symbol=ticker,
            provider="yfinance",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        raise ValueError(f"no daily bars returned for {ticker!r}: {exc}") from exc

    df = result.to_dataframe()
    if df.empty:
        raise ValueError(f"no daily bars returned for {ticker!r}")
    df = df[REQUIRED_COLUMNS].sort_index()
    return df
```

- [ ] **Step 4: Run network tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_provider.py -m network -v`
Expected: 2 PASS. Also run `.venv/bin/python -m pytest -q` and confirm the default suite deselects them.

- [ ] **Step 5: Commit**

```bash
git add tenx/data/openbb_provider.py tests/test_provider.py
git commit -m "feat: OpenBB daily-bars provider (yfinance), network-marked tests"
```

---

### Task 3: Signal dataclass + placeholder SMA-crossover signal

**Files:**
- Create: `tenx/signals/base.py`, `tenx/signals/placeholder.py`
- Test: `tests/test_placeholder_signal.py`

**Interfaces:**
- Consumes: the bars DataFrame shape from Task 2 (`open/high/low/close/volume`, date index).
- Produces: `Signal` frozen dataclass with fields `ticker: str`, `as_of: str` (ISO date of last bar), `action: str` (`"BUY" | "SELL" | "HOLD"`), `confidence: float`, `features: dict`, `rationale: str`, and method `to_dict() -> dict`. Function `sma_crossover_signal(df: pd.DataFrame, ticker: str, fast: int = 5, slow: int = 20) -> Signal`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_placeholder_signal.py
import numpy as np
import pandas as pd
import pytest

from tenx.signals.base import Signal
from tenx.signals.placeholder import sma_crossover_signal


def make_bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
         "volume": np.full(len(c), 1_000_000)},
        index=idx,
    )


def test_rising_prices_produce_buy():
    df = make_bars(np.linspace(100, 200, 60))
    sig = sma_crossover_signal(df, "NVDA")
    assert isinstance(sig, Signal)
    assert sig.action == "BUY"
    assert sig.ticker == "NVDA"
    assert sig.as_of == df.index[-1].date().isoformat()
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.features["sma_fast"] > sig.features["sma_slow"]
    assert sig.rationale


def test_falling_prices_produce_sell():
    df = make_bars(np.linspace(200, 100, 60))
    sig = sma_crossover_signal(df, "NVDA")
    assert sig.action == "SELL"
    assert sig.features["sma_fast"] < sig.features["sma_slow"]


def test_too_little_history_raises():
    df = make_bars(np.linspace(100, 110, 10))
    with pytest.raises(ValueError):
        sma_crossover_signal(df, "NVDA")


def test_signal_serializes_to_dict():
    df = make_bars(np.linspace(100, 200, 60))
    d = sma_crossover_signal(df, "NVDA").to_dict()
    assert d["action"] == "BUY"
    assert isinstance(d["features"], dict)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_placeholder_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.signals.base'`.

- [ ] **Step 3: Implement**

```python
# tenx/signals/base.py
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Signal:
    ticker: str
    as_of: str  # ISO date of the last bar the signal was computed from
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    features: dict = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
```

```python
# tenx/signals/placeholder.py
"""Phase 0 placeholder signal. Deliberately dumb — a 5/20 SMA crossover.

Exists only to prove the pipeline runs end-to-end. Replaced in Phase 1
by a walk-forward-validated model (spec: docs/BUILD_PLAN.md section 6).
"""
import pandas as pd

from tenx.signals.base import Signal


def sma_crossover_signal(
    df: pd.DataFrame, ticker: str, fast: int = 5, slow: int = 20
) -> Signal:
    if len(df) < slow:
        raise ValueError(
            f"need at least {slow} bars for {ticker}, got {len(df)}"
        )
    sma_fast = float(df["close"].rolling(fast).mean().iloc[-1])
    sma_slow = float(df["close"].rolling(slow).mean().iloc[-1])
    action = "BUY" if sma_fast > sma_slow else "SELL" if sma_fast < sma_slow else "HOLD"
    return Signal(
        ticker=ticker,
        as_of=df.index[-1].date().isoformat(),
        action=action,
        confidence=0.5,  # placeholder: no calibrated confidence until Phase 1
        features={
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "fast_window": fast,
            "slow_window": slow,
            "last_close": float(df["close"].iloc[-1]),
        },
        rationale=(
            f"Placeholder 5/20 SMA crossover: fast={sma_fast:.2f} "
            f"{'>' if sma_fast > sma_slow else '<=' } slow={sma_slow:.2f} -> {action}"
        ),
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_placeholder_signal.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tenx/signals/ tests/test_placeholder_signal.py
git commit -m "feat: Signal dataclass + placeholder SMA-crossover signal"
```

---

### Task 4: JSONL reasoning journal

**Files:**
- Create: `tenx/journal.py`
- Test: `tests/test_journal.py`

**Interfaces:**
- Produces: `Journal(path: Path | str)` with `log(run_id: str, stage: str, payload: dict) -> dict` (returns the full record it wrote, including UTC `ts`) and `read_all() -> list[dict]`. Records are one-JSON-object-per-line appends; the file and parent dirs are created on first write.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_journal.py
import json

from tenx.journal import Journal


def test_log_appends_jsonl_records(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    rec1 = j.log("run-1", "data_pull", {"ticker": "NVDA", "rows": 40})
    rec2 = j.log("run-1", "signal", {"action": "BUY"})

    lines = (tmp_path / "journal.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["stage"] == "data_pull"
    assert parsed[0]["run_id"] == "run-1"
    assert parsed[0]["payload"]["ticker"] == "NVDA"
    assert parsed[1]["payload"]["action"] == "BUY"
    assert rec1["ts"] <= rec2["ts"]
    assert rec1["ts"].endswith("+00:00")


def test_read_all_round_trips(tmp_path):
    j = Journal(tmp_path / "sub" / "journal.jsonl")  # parent dir auto-created
    j.log("run-1", "a", {})
    j.log("run-2", "b", {"x": 1})
    records = j.read_all()
    assert [r["run_id"] for r in records] == ["run-1", "run-2"]


def test_read_all_missing_file_is_empty(tmp_path):
    assert Journal(tmp_path / "nope.jsonl").read_all() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_journal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.journal'`.

- [ ] **Step 3: Implement**

```python
# tenx/journal.py
"""Append-only JSONL reasoning trail. Every layer logs full reasoning here,
not just outcomes (spec: docs/BUILD_PLAN.md section 4.6)."""
import json
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def log(self, run_id: str, stage: str, payload: dict) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "stage": stage,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_journal.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tenx/journal.py tests/test_journal.py
git commit -m "feat: append-only JSONL reasoning journal"
```

---

### Task 5: Hypothetical paper-trade builder

**Files:**
- Create: `tenx/paper.py`
- Test: `tests/test_paper.py`

**Interfaces:**
- Consumes: `Signal` from Task 3.
- Produces: `build_paper_trade(signal: Signal, notional_usd: float = 10_000.0) -> dict | None`. For BUY/SELL returns `{"ticker", "side", "quantity", "price", "notional_usd", "signal_as_of", "hypothetical": True}` where `price` is `signal.features["last_close"]` and `quantity = int(notional_usd // price)` (≥ 1 enforced via ValueError if price > notional). For HOLD returns `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_paper.py
import pytest

from tenx.paper import build_paper_trade
from tenx.signals.base import Signal


def make_signal(action="BUY", last_close=200.0):
    return Signal(
        ticker="NVDA", as_of="2026-08-27", action=action,
        confidence=0.5, features={"last_close": last_close}, rationale="test",
    )


def test_buy_signal_builds_trade():
    trade = build_paper_trade(make_signal("BUY", last_close=200.0))
    assert trade == {
        "ticker": "NVDA",
        "side": "BUY",
        "quantity": 50,
        "price": 200.0,
        "notional_usd": 10_000.0,
        "signal_as_of": "2026-08-27",
        "hypothetical": True,
    }


def test_sell_signal_builds_sell_side():
    assert build_paper_trade(make_signal("SELL"))["side"] == "SELL"


def test_hold_returns_none():
    assert build_paper_trade(make_signal("HOLD")) is None


def test_price_above_notional_raises():
    with pytest.raises(ValueError):
        build_paper_trade(make_signal("BUY", last_close=20_000.0))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_paper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.paper'`.

- [ ] **Step 3: Implement**

```python
# tenx/paper.py
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_paper.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tenx/paper.py tests/test_paper.py
git commit -m "feat: hypothetical paper-trade builder"
```

---

### Task 6: Pipeline orchestrator (offline-testable) + live end-to-end run

**Files:**
- Create: `tenx/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `get_daily_bars` (Task 2), `sma_crossover_signal`/`Signal` (Task 3), `Journal` (Task 4), `build_paper_trade` (Task 5).
- Produces: `run_pipeline(ticker: str = "NVDA", journal_path: str | Path = "data/journal.jsonl", fetch: Callable | None = None, lookback_days: int = 120) -> dict` returning `{"run_id", "ticker", "signal", "trade"}` (`trade` may be `None`). Journals stages `data_pull`, `signal`, `paper_trade` (payload `{"trade": None, "reason": ...}` on no-trade). Runnable as `python -m tenx.pipeline [TICKER]`.

- [ ] **Step 1: Write the failing tests (offline via injected fetch)**

```python
# tests/test_pipeline.py
import numpy as np
import pandas as pd

from tenx.journal import Journal
from tenx.pipeline import run_pipeline


def fake_fetch_rising(ticker, start_date, end_date=None):
    idx = pd.date_range("2026-03-01", periods=60, freq="B")
    c = pd.Series(np.linspace(100, 200, 60), index=idx)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c,
         "volume": np.full(60, 1_000_000)},
        index=idx,
    )


def test_pipeline_end_to_end_offline(tmp_path):
    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fake_fetch_rising)

    assert result["ticker"] == "NVDA"
    assert result["signal"]["action"] == "BUY"
    assert result["trade"]["side"] == "BUY"
    assert result["trade"]["hypothetical"] is True

    records = Journal(jpath).read_all()
    stages = [r["stage"] for r in records]
    assert stages == ["data_pull", "signal", "paper_trade"]
    assert len({r["run_id"] for r in records}) == 1
    assert records[0]["payload"]["rows"] == 60
    assert records[1]["payload"]["rationale"]
    assert records[2]["payload"]["trade"]["quantity"] >= 1


def test_pipeline_journals_no_trade_on_hold(tmp_path):
    def fetch_flat(ticker, start_date, end_date=None):
        idx = pd.date_range("2026-03-01", periods=60, freq="B")
        c = pd.Series(np.full(60, 100.0), index=idx)
        return pd.DataFrame(
            {"open": c, "high": c, "low": c, "close": c,
             "volume": np.full(60, 1_000_000)},
            index=idx,
        )

    jpath = tmp_path / "journal.jsonl"
    result = run_pipeline("NVDA", journal_path=jpath, fetch=fetch_flat)
    assert result["trade"] is None
    last = Journal(jpath).read_all()[-1]
    assert last["stage"] == "paper_trade"
    assert last["payload"]["trade"] is None
    assert "reason" in last["payload"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.pipeline'`.

- [ ] **Step 3: Implement**

```python
# tenx/pipeline.py
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
```

- [ ] **Step 4: Run offline tests to verify pass, then full suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v` → 2 PASS.
Run: `.venv/bin/python -m pytest -q` → all offline tests pass, network tests deselected.

- [ ] **Step 5: Live end-to-end run (the actual Phase 0 exit criterion)**

Run: `.venv/bin/python -m tenx.pipeline NVDA`
Expected: prints a result dict with a real signal computed from live NVDA data; `data/journal.jsonl` contains exactly 3 new records (`data_pull`, `signal`, `paper_trade`) sharing one run_id. Inspect the journal to confirm.

- [ ] **Step 6: Commit**

```bash
git add tenx/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end pipeline orchestrator; Phase 0 skeleton complete"
```

---

## Self-Review Notes

- Spec coverage: implements exactly the Phase 0 row of `docs/BUILD_PLAN.md` §6 (one stock, OpenBB pull, dumb signal, logged hypothetical trade). Deliberately excludes quant models, agents, risk layer, IBKR — later phases.
- The `data/` output dir is gitignored so journal runs don't pollute history; the journal format itself is tested.
- Type consistency checked: `Signal.to_dict()`, `features["last_close"]`, and journal record shape are used identically across Tasks 3–6.
