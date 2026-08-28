# Phase 1 Validation — Walk-Forward Model Comparison (NVDA)

**Run date:** 2026-08-28 · **Reproduce:** `.venv/bin/python -m tenx.quant.evaluate NVDA` · **Raw report:** `data/walkforward_report.json`

## Setup

- **Data:** 3,684 NVDA daily bars, 2012-01-03 → 2026-08-27, via the OpenBB provider (yfinance).
- **Target:** binary — is `close[t+5] > close[t]` (5 trading days ≈ the spec's multi-hour-to-multi-day holding period).
- **Features:** 15 causal OHLCV features (returns, SMA ratios, volatility, RSI-14, volume z-score, range, distance from 20-bar high/low). Lookahead is impossible by construction and enforced by `tests/test_features.py::test_features_are_causal_no_lookahead`.
- **Scheme:** rolling walk-forward — train 756 bars (~3y), 5-bar label embargo, test 63 bars (~1 quarter), step 63. **46 out-of-sample quarterly folds** spanning 2015→2026, covering the 2018 correction, 2020 crash, 2022 bear, and the 2023–24 AI run.
- **Selection rule (pre-committed before the run):** highest mean edge = accuracy − per-fold base rate; tie-break mean AUC.

## Results

| Model | Folds | Mean accuracy | Mean base rate | **Mean edge** | Mean AUC |
|---|---|---|---|---|---|
| logistic (winner) | 46 | 0.562 | 0.618 | **−0.055** | 0.523 |
| xgboost | 46 | 0.534 | 0.618 | −0.084 | 0.504 |
| hist_gb | 46 | 0.521 | 0.618 | −0.097 | 0.511 |

## Honest read — do not sugarcoat this

**No model beats the base rate.** NVDA closed higher 5 days later in ~62% of bars over this period; every candidate predicts direction *less* accurately than the constant "always up" guess. The logistic model's AUC of 0.523 indicates only the faintest ranking signal; the boosted trees are indistinguishable from noise out-of-sample (AUC ≈ 0.50–0.51) — they overfit the training windows.

What this does and does not mean:

- The **validation machinery works and is trustworthy** — it just delivered bad news instead of a flattering backtest, which is precisely the failure mode this project was designed to avoid (spec §4.3, §9). An in-sample or single-split evaluation of the same models looks far better; walk-forward exposed it.
- `DEFAULT_MODEL = "logistic"` is the best of the candidates per the pre-committed rule, but a signal built on it currently carries **no demonstrated directional edge over baseline**. Its probability output still functions as the structured, confidence-scored input the Phase 2 Decision Agent consumes — with this documented weakness attached.
- This is **one ticker** during a strongly trending regime; a high base rate is intrinsically hard to beat on direction. It says nothing yet about the other basket names.
- Phase 1's deliverable (real model, walk-forward validated, honest report) is met. **The bar for real money is Phase 6's**, and nothing here suggests we're near it.

## Implications carried forward

1. Daily-bar OHLCV technicals alone are insufficient for a 5-day directional edge on NVDA. The planned fundamentals (Phase 1+ data) and qualitative research context (Phase 4) are not decoration — on this evidence they're where any edge has to come from.
2. Candidate future work (deliberately NOT iterated on now, to avoid threshold-shopping the same data until something "works"): alternative targets (excess return vs. market, volatility-scaled returns), longer horizons, cross-sectional features. Each retest of the same history consumes statistical validity — batch these deliberately, don't drip-tune.
3. The BUY/SELL thresholds (0.55/0.45) around a ~0.62 base rate mean the signal will lean BUY on NVDA. Expected for a long-biased name; the risk layer (Phase 3), not the signal, is what bounds exposure.
