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
