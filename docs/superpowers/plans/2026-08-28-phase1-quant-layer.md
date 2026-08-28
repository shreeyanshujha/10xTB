# Phase 1 — Real Quant Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder SMA signal with a real, walk-forward-validated tabular model producing a confidence-scored `Signal` for NVDA — still one stock, still no agents.

**Architecture:** New `tenx/quant/` package: `features.py` (causal OHLCV feature engineering + forward-return labels), `walkforward.py` (rolling train/test splits with a label embargo, fold-by-fold evaluation), `models.py` (three candidate model factories: logistic baseline, HistGradientBoosting, XGBoost), and `evaluate.py` (CLI that runs the full walk-forward comparison on real NVDA history and writes a JSON report). A new `tenx/signals/quant_model.py` turns the winning model into a `Signal` (train on trailing window with embargo, predict today, map probability to BUY/SELL/HOLD). `pipeline.py` swaps to the quant signal. The placeholder stays in the tree as the Phase 0 artifact.

**Tech Stack:** Existing Phase 0 stack plus `xgboost==3.4.1`, `scikit-learn==1.9.0` (verified installing on the py3.12 venv, 2026-08-28). NVDA history: 3,684 daily bars from 2012-01-03 (verified live).

**Spec:** `docs/BUILD_PLAN.md` — Phase 1 row of §6, quant-layer rules in §4.3, guardrails §9.

## Global Constraints

- Walk-forward validation ONLY — no single static train/test split anywhere (spec §4.3, §9).
- No LLM anywhere in this layer (spec §9); models are sklearn-API tabular models.
- Output remains the Phase 0 `Signal` dataclass — the Decision Agent (Phase 2) consumes it unchanged.
- All feature computation must be causal: features at bar *t* use data ≤ *t*; labels at *t* use only bars *t+1 … t+horizon*. A lookahead test enforces this.
- Training windows must respect a label embargo: a sample at index *t* has a label that peeks `horizon` bars forward, so training must end `horizon` bars before any test/prediction bar.
- Deterministic seeds everywhere (`random_state=42`); evaluation must be reproducible.
- Model choice is decided EMPIRICALLY by the evaluation run (spec §8) — the plan defines candidates and the selection rule, not the winner.
- Full reasoning trail (spec §4.6): the signal journals model name, probability, train window, and walk-forward provenance; the validation run writes a persistent report.

## Design Decisions (made now, recorded so they aren't silently drifted from)

- **Prediction target:** binary — is `close[t+5] > close[t]` (5 trading days ≈ the spec's multi-hour-to-multi-day holding period, on daily bars).
- **Features (all causal, from OHLCV only — fundamentals join in a later phase):** returns over 1/5/10/20 bars; close/SMA5, close/SMA20, SMA5/SMA20 ratios; 5- and 20-bar return volatility; RSI(14); 20-bar volume z-score; (high−low)/close daily range and its 5-bar mean; distance of close from 20-bar high and low.
- **Walk-forward scheme:** rolling window — train 756 bars (~3 years), embargo 5 bars, test 63 bars (~1 quarter), step 63. On 3,684 bars this yields ~45 out-of-sample quarterly folds spanning 2015→2026 (multiple regimes: 2018 correction, 2020 crash, 2022 bear, 2023–24 AI run).
- **Selection rule (pre-committed):** primary metric = mean out-of-sample accuracy edge over the fold's base rate (majority-class frequency); tie-break by mean AUC. The winner becomes `DEFAULT_MODEL`.
- **Probability → action:** p ≥ 0.55 BUY, p ≤ 0.45 SELL, else HOLD; `confidence = p` for BUY, `1 − p` for SELL, else `max(p, 1−p)`. Note: NVDA's 5-day up-move base rate is well above 50%, so SELL will be rare — expected for a long-biased name, and measured honestly via the edge-over-base-rate metric.
- **Phase 1 validates predictive power** (accuracy edge, AUC), not trading P&L — portfolio-level comparison vs. buy-and-hold is the Phase 6 bar (spec §7) and is not smuggled in early.

---

### Task 1: Dependencies + causal features and labels

**Files:**
- Modify: `pyproject.toml` (add pinned deps, add `tenx.quant` package)
- Create: `tenx/quant/__init__.py`, `tenx/quant/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `build_features(df: pd.DataFrame) -> pd.DataFrame` (same index as `df`, columns = FEATURE_COLUMNS, NaN in warmup rows); `make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series` (1.0 if close[t+horizon] > close[t], NaN for final `horizon` rows); `build_dataset(df, horizon=5) -> tuple[pd.DataFrame, pd.Series]` (aligned X, y with all NaN rows dropped); constant `FEATURE_COLUMNS: list[str]`.

- [ ] **Step 1: Pin deps in pyproject and freeze**

In `pyproject.toml` dependencies add `"xgboost==3.4.1", "scikit-learn==1.9.0"`; in `[tool.setuptools] packages` add `"tenx.quant"`. Then:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python .venv/bin/python -e .
uv pip freeze --python .venv/bin/python > requirements.lock
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_features.py
import numpy as np
import pandas as pd

from tenx.quant.features import FEATURE_COLUMNS, build_dataset, build_features, make_labels


def make_bars(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close * 1.01,
         "low": close * 0.99, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


def test_features_have_expected_columns_and_index():
    df = make_bars()
    X = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.equals(df.index)


def test_features_are_causal_no_lookahead():
    """Changing future bars must not change features at bar t."""
    df = make_bars()
    t = 150
    X_full = build_features(df)
    df_mutated = df.copy()
    df_mutated.iloc[t + 1:, df.columns.get_loc("close")] = 9999.0
    df_mutated.iloc[t + 1:, df.columns.get_loc("volume")] = 1.0
    X_mut = build_features(df_mutated)
    pd.testing.assert_series_equal(X_full.iloc[t], X_mut.iloc[t])


def test_labels_look_exactly_horizon_forward():
    df = make_bars()
    y = make_labels(df, horizon=5)
    t = 100
    expected = 1.0 if df["close"].iloc[t + 5] > df["close"].iloc[t] else 0.0
    assert y.iloc[t] == expected
    assert y.iloc[-5:].isna().all()  # final horizon rows have no label


def test_build_dataset_aligns_and_drops_nans():
    df = make_bars()
    X, y = build_dataset(df, horizon=5)
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert set(y.unique()) <= {0.0, 1.0}
    # warmup (longest lookback 20 + RSI 14 style) and tail horizon rows are gone
    assert len(X) < len(df)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.quant'`.

- [ ] **Step 4: Implement**

```python
# tenx/quant/features.py
"""Causal OHLCV feature engineering for the quant layer (spec section 4.3).

Every feature at bar t uses only data at or before t. Labels at bar t use
only bars t+1..t+horizon. tests/test_features.py enforces both properties.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "close_sma5", "close_sma20", "sma5_sma20",
    "vol_5", "vol_20",
    "rsi_14",
    "volume_z20",
    "range_pct", "range_pct_5",
    "dist_high20", "dist_low20",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    ret_1 = close.pct_change()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_14 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    range_pct = (high - low) / close

    X = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": close.pct_change(5),
            "ret_10": close.pct_change(10),
            "ret_20": close.pct_change(20),
            "close_sma5": close / sma5 - 1,
            "close_sma20": close / sma20 - 1,
            "sma5_sma20": sma5 / sma20 - 1,
            "vol_5": ret_1.rolling(5).std(),
            "vol_20": ret_1.rolling(20).std(),
            "rsi_14": rsi_14,
            "volume_z20": (volume - volume.rolling(20).mean())
            / volume.rolling(20).std(),
            "range_pct": range_pct,
            "range_pct_5": range_pct.rolling(5).mean(),
            "dist_high20": close / high20 - 1,
            "dist_low20": close / low20 - 1,
        },
        index=df.index,
    )
    return X[FEATURE_COLUMNS]


def make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    close = df["close"]
    future = close.shift(-horizon)
    y = (future > close).astype(float)
    y[future.isna()] = np.nan
    return y


def build_dataset(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    X = build_features(df)
    y = make_labels(df, horizon=horizon)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]
```

Create empty `tenx/quant/__init__.py`.

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.lock tenx/quant/ tests/test_features.py
git commit -m "feat: causal OHLCV features + forward-return labels for quant layer"
```

---

### Task 2: Model factories

**Files:**
- Create: `tenx/quant/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `MODEL_FACTORIES: dict[str, Callable[[], object]]` with keys `"logistic"`, `"hist_gb"`, `"xgboost"`; each factory returns an UNFITTED sklearn-API estimator with `fit(X, y)` and `predict_proba(X)`, seeded with `random_state=42`. `make_model(name: str)` helper that raises `ValueError` on unknown names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import numpy as np
import pytest

from tenx.quant.models import MODEL_FACTORIES, make_model


def toy_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(float)
    return X, y


@pytest.mark.parametrize("name", ["logistic", "hist_gb", "xgboost"])
def test_factory_fits_and_predicts_proba(name):
    X, y = toy_data()
    model = make_model(name)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # learnable signal -> better than chance in-sample
    assert ((proba[:, 1] > 0.5) == (y == 1)).mean() > 0.6


@pytest.mark.parametrize("name", ["logistic", "hist_gb", "xgboost"])
def test_factory_is_deterministic(name):
    X, y = toy_data()
    p1 = make_model(name).fit(X, y).predict_proba(X)
    p2 = make_model(name).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        make_model("transformer_llm")


def test_registry_has_exactly_three_candidates():
    assert set(MODEL_FACTORIES) == {"logistic", "hist_gb", "xgboost"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.quant.models'`.

- [ ] **Step 3: Implement**

```python
# tenx/quant/models.py
"""Candidate models for the quant layer. Tabular models only — no LLMs
(spec section 4.3, guardrail section 9). Winner chosen empirically by
tenx/quant/evaluate.py, never hardcoded ahead of the evidence."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


def _logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )


def _hist_gb():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=40, random_state=SEED,
    )


def _xgboost():
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, random_state=SEED,
        eval_metric="logloss",
    )


MODEL_FACTORIES = {"logistic": _logistic, "hist_gb": _hist_gb, "xgboost": _xgboost}


def make_model(name: str):
    if name not in MODEL_FACTORIES:
        raise ValueError(f"unknown model {name!r}; choices: {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name]()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add tenx/quant/models.py tests/test_models.py
git commit -m "feat: seeded candidate model factories (logistic, hist_gb, xgboost)"
```

---

### Task 3: Walk-forward engine with label embargo

**Files:**
- Create: `tenx/quant/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `make_model` (Task 2).
- Produces: `walkforward_splits(n: int, train_size: int, test_size: int, embargo: int, step: int) -> list[tuple[range, range]]` — ordered (train_range, test_range) pairs where `train.stop + embargo == test.start`; `run_walkforward(X: pd.DataFrame, y: pd.Series, model_name: str, train_size: int = 756, test_size: int = 63, embargo: int = 5, step: int = 63) -> dict` — report `{"model", "params": {...}, "n_folds", "folds": [{"train_start","train_end","test_start","test_end","n_test","base_rate","accuracy","edge","auc"}...], "mean_accuracy", "mean_base_rate", "mean_edge", "mean_auc"}` where dates are ISO strings from the index, `edge = accuracy - base_rate`, `base_rate = max(p_up, 1 - p_up)` on the test fold, and `auc` is `None` for single-class test folds.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_walkforward.py
import numpy as np
import pandas as pd

from tenx.quant.walkforward import run_walkforward, walkforward_splits


def test_splits_are_ordered_with_embargo():
    splits = walkforward_splits(n=1000, train_size=500, test_size=100, embargo=5, step=100)
    assert len(splits) > 0
    for train, test in splits:
        assert len(train) == 500
        assert len(test) <= 100
        assert train.stop + 5 == test.start  # embargo gap: train labels can't peek into test
        assert test.stop <= 1000
    # consecutive folds step forward monotonically
    starts = [test.start for _, test in splits]
    assert starts == sorted(starts)
    assert all(b - a == 100 for a, b in zip(starts, starts[1:]))


def test_splits_never_produced_when_history_too_short():
    assert walkforward_splits(n=100, train_size=500, test_size=100, embargo=5, step=100) == []


def make_learnable_dataset(n=1500, seed=1):
    """Feature 0 genuinely predicts the label; models should find it."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, 5)),
                     columns=[f"f{i}" for i in range(5)], index=idx)
    y = pd.Series(((X["f0"] + rng.normal(0, 0.7, n)) > 0).astype(float), index=idx)
    return X, y


def test_run_walkforward_report_shape_and_learnability():
    X, y = make_learnable_dataset()
    report = run_walkforward(X, y, "logistic", train_size=500, test_size=100,
                             embargo=5, step=100)
    assert report["model"] == "logistic"
    assert report["n_folds"] == len(report["folds"]) > 3
    for fold in report["folds"]:
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert 0.5 <= fold["base_rate"] <= 1.0
        assert fold["test_start"] > fold["train_end"]  # ISO dates compare lexicographically
    # a genuinely learnable signal must show positive mean edge out-of-sample
    assert report["mean_edge"] > 0.05
    assert report["mean_auc"] > 0.6


def test_run_walkforward_is_deterministic():
    X, y = make_learnable_dataset()
    r1 = run_walkforward(X, y, "hist_gb", train_size=500, test_size=100, embargo=5, step=200)
    r2 = run_walkforward(X, y, "hist_gb", train_size=500, test_size=100, embargo=5, step=200)
    assert r1 == r2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.quant.walkforward'`.

- [ ] **Step 3: Implement**

```python
# tenx/quant/walkforward.py
"""Walk-forward validation: train on a rolling window, test on the next
quarter, roll forward, retrain (spec section 4.3 — the ONLY acceptable
validation scheme; a single static split is a guardrail violation, section 9).

The embargo leaves `horizon` bars between train end and test start: a sample
at bar t carries a label computed from bar t+horizon, so without the gap the
last training labels would peek into the test window."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tenx.quant.models import make_model


def walkforward_splits(
    n: int, train_size: int, test_size: int, embargo: int, step: int
) -> list[tuple[range, range]]:
    splits = []
    test_start = train_size + embargo
    while test_start < n:
        test_stop = min(test_start + test_size, n)
        splits.append(
            (range(test_start - embargo - train_size, test_start - embargo),
             range(test_start, test_stop))
        )
        test_start += step
    return splits


def run_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    train_size: int = 756,
    test_size: int = 63,
    embargo: int = 5,
    step: int = 63,
) -> dict:
    params = {"train_size": train_size, "test_size": test_size,
              "embargo": embargo, "step": step}
    folds = []
    for train, test in walkforward_splits(len(X), train_size, test_size, embargo, step):
        tr, te = list(train), list(test)
        model = make_model(model_name)
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        y_te = y.iloc[te].to_numpy()
        pred = (proba > 0.5).astype(float)
        accuracy = float((pred == y_te).mean())
        p_up = float(y_te.mean())
        base_rate = float(max(p_up, 1 - p_up))
        auc = (float(roc_auc_score(y_te, proba))
               if len(np.unique(y_te)) > 1 else None)
        folds.append({
            "train_start": X.index[tr[0]].date().isoformat(),
            "train_end": X.index[tr[-1]].date().isoformat(),
            "test_start": X.index[te[0]].date().isoformat(),
            "test_end": X.index[te[-1]].date().isoformat(),
            "n_test": len(te),
            "base_rate": base_rate,
            "accuracy": accuracy,
            "edge": accuracy - base_rate,
            "auc": auc,
        })
    aucs = [f["auc"] for f in folds if f["auc"] is not None]
    return {
        "model": model_name,
        "params": params,
        "n_folds": len(folds),
        "folds": folds,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])) if folds else None,
        "mean_base_rate": float(np.mean([f["base_rate"] for f in folds])) if folds else None,
        "mean_edge": float(np.mean([f["edge"] for f in folds])) if folds else None,
        "mean_auc": float(np.mean(aucs)) if aucs else None,
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_walkforward.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tenx/quant/walkforward.py tests/test_walkforward.py
git commit -m "feat: walk-forward engine with label embargo"
```

---

### Task 4: Evaluation CLI — run the real comparison on NVDA

**Files:**
- Create: `tenx/quant/evaluate.py`
- Test: `tests/test_evaluate.py` (offline via injected fetch)

**Interfaces:**
- Consumes: `get_daily_bars`, `build_dataset`, `run_walkforward`, `MODEL_FACTORIES`.
- Produces: `evaluate(ticker: str = "NVDA", start_date: str = "2012-01-01", fetch=None, out_path: str | Path = "data/walkforward_report.json", **wf_kwargs) -> dict` — report `{"ticker", "generated_at", "start_date", "n_bars", "horizon", "models": {name: <run_walkforward report>}, "winner", "selection_rule"}` where winner = highest `mean_edge`, tie-break `mean_auc`. Writes JSON to `out_path`. Runnable as `python -m tenx.quant.evaluate NVDA`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import json

import numpy as np
import pandas as pd

from tenx.quant.evaluate import evaluate


def fake_fetch(ticker, start_date, end_date=None):
    n = 1400
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


def test_evaluate_compares_all_models_and_picks_winner(tmp_path):
    out = tmp_path / "report.json"
    report = evaluate("TEST", fetch=fake_fetch, out_path=out,
                      train_size=500, test_size=100, embargo=5, step=200)
    assert set(report["models"]) == {"logistic", "hist_gb", "xgboost"}
    assert report["winner"] in report["models"]
    edges = {n: m["mean_edge"] for n, m in report["models"].items()}
    assert edges[report["winner"]] == max(edges.values())
    assert report["ticker"] == "TEST"
    assert report["horizon"] == 5
    # persisted report round-trips
    assert json.loads(out.read_text())["winner"] == report["winner"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.quant.evaluate'`.

- [ ] **Step 3: Implement**

```python
# tenx/quant/evaluate.py
"""Walk-forward model comparison on real history. This run IS the empirical
model-choice evidence the spec calls for (sections 4.3 and 8) — its JSON
report is the reasoning trail for why DEFAULT_MODEL is what it is."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tenx.quant.features import build_dataset
from tenx.quant.models import MODEL_FACTORIES
from tenx.quant.walkforward import run_walkforward

SELECTION_RULE = "highest mean_edge (accuracy minus per-fold base rate); tie-break mean_auc"


def evaluate(
    ticker: str = "NVDA",
    start_date: str = "2012-01-01",
    fetch=None,
    out_path: str | Path = "data/walkforward_report.json",
    horizon: int = 5,
    **wf_kwargs,
) -> dict:
    if fetch is None:
        from tenx.data.openbb_provider import get_daily_bars
        fetch = get_daily_bars

    df = fetch(ticker, start_date=start_date)
    X, y = build_dataset(df, horizon=horizon)

    models = {}
    for name in sorted(MODEL_FACTORIES):
        models[name] = run_walkforward(X, y, name, embargo=horizon, **wf_kwargs)

    winner = max(
        models,
        key=lambda n: (models[n]["mean_edge"], models[n]["mean_auc"] or 0.0),
    )
    report = {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "n_bars": len(df),
        "horizon": horizon,
        "selection_rule": SELECTION_RULE,
        "models": models,
        "winner": winner,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = evaluate(sys.argv[1] if len(sys.argv) > 1 else "NVDA")
    for name, m in report["models"].items():
        print(f"{name:10s} folds={m['n_folds']} acc={m['mean_accuracy']:.3f} "
              f"base={m['mean_base_rate']:.3f} edge={m['mean_edge']:+.3f} "
              f"auc={m['mean_auc']:.3f}")
    print(f"winner: {report['winner']}  ({report['selection_rule']})")
```

- [ ] **Step 4: Run offline test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 1 PASS (allow ~1–2 min; it trains 3 models across folds).

- [ ] **Step 5: Run the REAL evaluation on NVDA (the Phase 1 evidence)**

Run: `.venv/bin/python -m tenx.quant.evaluate NVDA`
Expected: prints per-model walk-forward stats over ~45 quarterly folds and a winner; writes `data/walkforward_report.json`. Record the numbers — they go into `docs/phase1-validation.md` (Task 5) and decide `DEFAULT_MODEL`.

- [ ] **Step 6: Commit**

```bash
git add tenx/quant/evaluate.py tests/test_evaluate.py
git commit -m "feat: walk-forward evaluation CLI comparing candidate models"
```

---

### Task 5: Quant signal + pipeline swap + validation writeup

**Files:**
- Create: `tenx/signals/quant_model.py`, `docs/phase1-validation.md`
- Modify: `tenx/pipeline.py` (swap placeholder → quant signal)
- Test: `tests/test_quant_signal.py`; modify `tests/test_pipeline.py` (fake data long enough to train)

**Interfaces:**
- Consumes: `build_features`/`build_dataset` (Task 1), `make_model` (Task 2), `Signal` (Phase 0).
- Produces: `quant_signal(df: pd.DataFrame, ticker: str, model_name: str = DEFAULT_MODEL, horizon: int = 5, train_size: int = 756, buy_threshold: float = 0.55, sell_threshold: float = 0.45) -> Signal`. `DEFAULT_MODEL` set from the Task 4 real-run winner. Trains on the trailing `train_size` labeled bars ending `horizon` bars before the last bar (embargo), predicts on the last bar's features. `Signal.features` includes `p_up`, `model`, `train_size`, `train_start`, `train_end`, `horizon`, `last_close`. Raises `ValueError` if history < `train_size + horizon + 30` bars. `run_pipeline` grows an optional `signal_fn` parameter (default `quant_signal`) and journals unchanged stages.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quant_signal.py
import numpy as np
import pandas as pd
import pytest

from tenx.signals.base import Signal
from tenx.signals.quant_model import DEFAULT_MODEL, quant_signal


def make_bars(n=900, drift=0.0005, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n))), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


def test_quant_signal_produces_valid_signal():
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600)
    assert isinstance(sig, Signal)
    assert sig.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.as_of == df.index[-1].date().isoformat()
    assert sig.features["model"] == DEFAULT_MODEL
    assert 0.0 <= sig.features["p_up"] <= 1.0
    assert sig.features["last_close"] == float(df["close"].iloc[-1])
    assert "walk-forward" in sig.rationale


def test_action_mapping_follows_thresholds():
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600)
    p = sig.features["p_up"]
    expected = "BUY" if p >= 0.55 else "SELL" if p <= 0.45 else "HOLD"
    assert sig.action == expected
    expected_conf = p if p >= 0.55 else (1 - p) if p <= 0.45 else max(p, 1 - p)
    assert sig.confidence == pytest.approx(expected_conf)


def test_training_respects_embargo():
    """Train window must end `horizon` bars before the prediction bar."""
    df = make_bars()
    sig = quant_signal(df, "NVDA", train_size=600, horizon=5)
    assert sig.features["train_end"] <= df.index[-6].date().isoformat()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        quant_signal(make_bars(n=200), "NVDA", train_size=600)


def test_deterministic():
    df = make_bars()
    s1 = quant_signal(df, "NVDA", train_size=600)
    s2 = quant_signal(df, "NVDA", train_size=600)
    assert s1.features["p_up"] == s2.features["p_up"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_quant_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenx.signals.quant_model'`.

- [ ] **Step 3: Implement the quant signal**

```python
# tenx/signals/quant_model.py
"""Phase 1 quant signal: trains DEFAULT_MODEL on a trailing window of causal
features (with a label embargo so training never peeks past the prediction
bar) and emits a probability-scored Signal. DEFAULT_MODEL was chosen by the
walk-forward comparison in data/walkforward_report.json — see
docs/phase1-validation.md for the numbers."""
import pandas as pd

from tenx.quant.features import build_dataset, build_features
from tenx.quant.models import make_model
from tenx.signals.base import Signal

DEFAULT_MODEL = "WINNER_FROM_TASK4"  # set from the real NVDA evaluation run


def quant_signal(
    df: pd.DataFrame,
    ticker: str,
    model_name: str = DEFAULT_MODEL,
    horizon: int = 5,
    train_size: int = 756,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> Signal:
    if len(df) < train_size + horizon + 30:
        raise ValueError(
            f"need at least {train_size + horizon + 30} bars for {ticker}, got {len(df)}"
        )
    X_all, y_all = build_dataset(df, horizon=horizon)
    # build_dataset drops the last `horizon` bars (no label yet) — that IS the
    # embargo: the newest training label is computed entirely before "today".
    X_train, y_train = X_all.iloc[-train_size:], y_all.iloc[-train_size:]

    model = make_model(model_name)
    model.fit(X_train, y_train)

    x_today = build_features(df).iloc[[-1]]
    if x_today.isna().any().any():
        raise ValueError(f"latest bar for {ticker} has NaN features")
    p_up = float(model.predict_proba(x_today)[0, 1])

    if p_up >= buy_threshold:
        action, confidence = "BUY", p_up
    elif p_up <= sell_threshold:
        action, confidence = "SELL", 1 - p_up
    else:
        action, confidence = "HOLD", max(p_up, 1 - p_up)

    return Signal(
        ticker=ticker,
        as_of=df.index[-1].date().isoformat(),
        action=action,
        confidence=confidence,
        features={
            "model": model_name,
            "p_up": p_up,
            "horizon": horizon,
            "train_size": len(X_train),
            "train_start": X_train.index[0].date().isoformat(),
            "train_end": X_train.index[-1].date().isoformat(),
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "last_close": float(df["close"].iloc[-1]),
        },
        rationale=(
            f"{model_name} (walk-forward validated, see data/walkforward_report.json) "
            f"trained on {len(X_train)} bars "
            f"[{X_train.index[0].date()}..{X_train.index[-1].date()}] with "
            f"{horizon}-bar label embargo: P(close up in {horizon}d)={p_up:.3f} "
            f"-> {action}"
        ),
    )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/test_quant_signal.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Swap the pipeline to the quant signal**

In `tenx/pipeline.py`: replace the `sma_crossover_signal` import with `from tenx.signals.quant_model import quant_signal`; add parameter `signal_fn: Callable = quant_signal`; call `signal = signal_fn(df, ticker)`; change default `lookback_days` from `120` to `1600` (≈ 1,100 business days: covers 756 training bars + warmup + embargo). Update `tests/test_pipeline.py`: fakes must produce 900 bars (reuse the `make_bars` generator shape from `tests/test_quant_signal.py`, rising drift for the BUY case), and the BUY-case assertion becomes action-agnostic: assert `result["signal"]["action"] in {"BUY","SELL","HOLD"}` and the journal chain — the HOLD/no-trade case instead injects `signal_fn=lambda df, t: <handcrafted HOLD Signal>` to keep that branch deterministic.

- [ ] **Step 6: Full suite, then live run**

Run: `.venv/bin/python -m pytest -q` → all pass.
Run: `.venv/bin/python -m tenx.pipeline NVDA` → journal gains a run whose `signal` stage shows `model`, `p_up`, train window, and a rationale referencing walk-forward validation.

- [ ] **Step 7: Write `docs/phase1-validation.md`**

Contents: the walk-forward scheme (windows, embargo, fold count, date span), the per-model table from the real Task 4 run (mean accuracy, base rate, edge, AUC), the winner and the pre-committed selection rule, honest caveats (single ticker; edge measured against base rate, not P&L; long-biased base rate makes SELL rare; Phase 6 is the real bar), and how to reproduce (`.venv/bin/python -m tenx.quant.evaluate NVDA`).

- [ ] **Step 8: Commit**

```bash
git add tenx/signals/quant_model.py tenx/pipeline.py tests/ docs/phase1-validation.md
git commit -m "feat: walk-forward-validated quant signal wired into pipeline; Phase 1 complete"
```

---

## Self-Review Notes

- Spec coverage: Phase 1 row of §6 (real model, walk-forward, one stock, no agents) — covered by Tasks 1–5. §4.3 output contract (structured, confidence-scored signal) — `quant_signal` keeps the Phase 0 `Signal` shape the Phase 2 Decision Agent will consume. §9 guardrails: no static split (Task 3), no LLM (Task 2), placeholder replaced not deleted.
- `DEFAULT_MODEL = "WINNER_FROM_TASK4"` is a deliberate placeholder token — Task 5 Step 3 must substitute the actual Task 4 winner before tests run (tests import it; the token would fail `make_model` validation loudly, not silently).
- Type consistency: `run_walkforward` fold dicts feed `evaluate` unchanged; `quant_signal` features keys match what Task 5's tests and the validation doc reference.
