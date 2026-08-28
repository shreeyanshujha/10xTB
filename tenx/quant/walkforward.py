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
