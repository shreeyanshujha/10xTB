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
