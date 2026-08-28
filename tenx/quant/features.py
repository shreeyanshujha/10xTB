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
