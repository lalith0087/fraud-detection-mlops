from fastapi.testclient import TestClient

from src import api
from src.data import FEATURES, make_transactions
from src.drift import build_reference, drift_report
import numpy as np
import pytest
import src.train as train_mod
from src.train import best_cost_threshold, total_cost, train


def test_data_is_imbalanced():
    df = make_transactions(10_000)
    assert 0.003 < df["is_fraud"].mean() < 0.03


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Keep tests away from the repo's models/ dir and MLflow registry."""
    monkeypatch.setattr(train_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(api, "_bundle", None)
    return tmp_path


def serve(tmp_path):
    (tmp_path / "model.joblib").write_bytes((tmp_path / "candidate.joblib").read_bytes())


def test_train_and_predict(isolated):
    metrics = train()
    serve(isolated)
    assert metrics["pr_auc"] > 0.4
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    risky = dict(amount=900, hour=2, merchant_risk=0.9, distance_from_home=300, txn_last_24h=8, is_foreign=1)
    safe = dict(amount=20, hour=13, merchant_risk=0.1, distance_from_home=3, txn_last_24h=1, is_foreign=0)
    assert client.post("/predict", json={"features": risky}).json()["fraud_probability"] > \
        client.post("/predict", json={"features": safe}).json()["fraud_probability"]


def test_cost_threshold_moves_with_costs():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    proba = np.clip(0.3 * y + rng.normal(0.3, 0.15, 5000), 0, 1)
    strict = best_cost_threshold(y, proba, cost_fn=1, cost_fp=50)    # false alarms expensive
    lenient = best_cost_threshold(y, proba, cost_fn=50, cost_fp=1)   # misses expensive
    assert lenient < strict
    assert total_cost(y, proba, lenient, 50, 1) <= total_cost(y, proba, 0.5, 50, 1)


def test_drift_detects_shift():
    X = make_transactions(20_000)[FEATURES]
    ref = build_reference(X.iloc[:10_000])
    same = drift_report(ref, X.iloc[10_000:])
    shifted = drift_report(ref, X.iloc[10_000:] + X.std())
    assert same["overall"] == "stable"
    assert shifted["overall"] == "significant"


def test_drift_endpoint(isolated):
    train()
    serve(isolated)
    api._recent.clear()
    client = TestClient(api.app)
    assert client.get("/drift").json()["status"] == "insufficient_data"
    row = dict(amount=20, hour=13, merchant_risk=0.1, distance_from_home=3, txn_last_24h=1, is_foreign=0)
    for _ in range(api.MIN_DRIFT_SAMPLES):
        client.post("/predict", json={"features": row})
    r = client.get("/drift").json()
    assert r["overall"] == "significant"  # 100 identical rows look nothing like the training mix


def test_registry_promotion_gate(isolated):
    from src import registry
    train()
    out = registry.promote()
    assert out["promoted"] == 1 and (isolated / "model.joblib").exists()
    train(); registry.promote()                      # equal PR-AUC, same data: allowed
    assert [v["version"] for v in registry.list_versions()] == [1, 2]
    train()                                          # v3, then pretend it scored worse
    c = registry._client()
    c.log_metric(c.get_model_version(train_mod.MODEL_NAME, "3").run_id, "pr_auc", 0.0)
    with pytest.raises(SystemExit):
        registry.promote()
    assert registry.promote(force=True)["promoted"] == 3
