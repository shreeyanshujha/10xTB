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
            f"{'>' if sma_fast > sma_slow else '<='} slow={sma_slow:.2f} -> {action}"
        ),
    )
