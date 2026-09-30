from fastapi.testclient import TestClient

from src import api
from src.data import make_transactions
from src.train import train


def test_data_is_imbalanced():
    df = make_transactions(10_000)
    assert 0.003 < df["is_fraud"].mean() < 0.03


def test_train_and_predict():
    metrics = train()
    assert metrics["pr_auc"] > 0.5
    api._model = None
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    risky = dict(amount=900, hour=2, merchant_risk=0.9, distance_from_home=300, txn_last_24h=8, is_foreign=1)
    safe = dict(amount=20, hour=13, merchant_risk=0.1, distance_from_home=3, txn_last_24h=1, is_foreign=0)
    assert client.post("/predict", json=risky).json()["fraud_probability"] > \
        client.post("/predict", json=safe).json()["fraud_probability"]
