"""FastAPI service: POST /predict scores a transaction given its feature dict."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Fraud Detection API")
_bundle = None


class Transaction(BaseModel):
    features: dict[str, float]


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run: python -m src.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": get_bundle()["features"]}


@app.post("/predict")
def predict(txn: Transaction, threshold: float | None = None):
    b = get_bundle()
    missing = [f for f in b["features"] if f not in txn.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    proba = float(b["model"].predict_proba(pd.DataFrame([txn.features])[b["features"]])[0, 1])
    t = b["threshold"] if threshold is None else threshold
    return {"fraud_probability": round(proba, 4), "is_fraud": proba >= t, "threshold": t}
