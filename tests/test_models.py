import numpy as np
import pytest

from tenx.quant.models import MODEL_FACTORIES, make_model


def toy_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(float)
    return X, y


@pytest.mark.parametrize("name", ["logistic", "hist_gb", "xgboost"])
def test_factory_fits_and_predicts_proba(name):
    X, y = toy_data()
    model = make_model(name)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # learnable signal -> better than chance in-sample
    assert ((proba[:, 1] > 0.5) == (y == 1)).mean() > 0.6


@pytest.mark.parametrize("name", ["logistic", "hist_gb", "xgboost"])
def test_factory_is_deterministic(name):
    X, y = toy_data()
    p1 = make_model(name).fit(X, y).predict_proba(X)
    p2 = make_model(name).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        make_model("transformer_llm")


def test_registry_has_exactly_three_candidates():
    assert set(MODEL_FACTORIES) == {"logistic", "hist_gb", "xgboost"}
