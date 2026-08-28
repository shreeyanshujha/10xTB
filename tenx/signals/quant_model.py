"""Phase 1 quant signal: trains DEFAULT_MODEL on a trailing window of causal
features (with a label embargo so training never peeks past the prediction
bar) and emits a probability-scored Signal. DEFAULT_MODEL was chosen by the
walk-forward comparison in data/walkforward_report.json — see
docs/phase1-validation.md for the numbers."""
import pandas as pd

from tenx.quant.features import build_dataset, build_features
from tenx.quant.models import make_model
from tenx.signals.base import Signal

# Winner of the walk-forward comparison (data/walkforward_report.json,
# 2026-08-28): best of three candidates by mean edge, though NOTE the edge
# was negative vs base rate — see docs/phase1-validation.md before trusting.
DEFAULT_MODEL = "logistic"


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
