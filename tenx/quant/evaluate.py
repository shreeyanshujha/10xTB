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

    wf_kwargs.setdefault("embargo", horizon)
    models = {}
    for name in sorted(MODEL_FACTORIES):
        models[name] = run_walkforward(X, y, name, **wf_kwargs)

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
