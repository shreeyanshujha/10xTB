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
