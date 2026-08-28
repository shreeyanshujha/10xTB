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
