"""FastAPI service: POST /predict scores a transaction given its feature dict."""
from collections import deque
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.drift import drift_report

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Fraud Detection API")
_bundle = None
MIN_DRIFT_SAMPLES = 100
_recent = deque(maxlen=1000)  # most recent scored transactions


class Transaction(BaseModel):
    features: dict[str, float]


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run: python -m src.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/")
def root():
    return {"service": "fraud-detection-api", "docs": "/docs", "health": "/health", "drift": "/drift", "features": "/features", "example": "/example"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": get_bundle()["features"]}


@app.get("/example")
def example():
    """Ready-to-send /predict request bodies: one real fraud and one legitimate transaction."""
    b = get_bundle()
    if "examples" not in b:
        raise HTTPException(409, "Model has no examples. Retrain: python -m src.train")
    return {name: {"features": feats} for name, feats in b["examples"].items() if feats}


@app.post("/predict")
def predict(txn: Transaction, threshold: float | None = None):
    b = get_bundle()
    missing = [f for f in b["features"] if f not in txn.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    _recent.append({f: txn.features[f] for f in b["features"]})
    proba = float(b["model"].predict_proba(pd.DataFrame([txn.features])[b["features"]])[0, 1])
    t = b["threshold"] if threshold is None else threshold
    return {"fraud_probability": round(proba, 4), "is_fraud": proba >= t, "threshold": t}


@app.get("/drift")
def drift():
    """PSI of recently scored transactions vs the training distribution."""
    b = get_bundle()
    if "reference" not in b:
        raise HTTPException(409, "Model has no drift reference. Retrain: python -m src.train")
    if len(_recent) < MIN_DRIFT_SAMPLES:
        return {"status": "insufficient_data", "n_samples": len(_recent), "needed": MIN_DRIFT_SAMPLES}
    return drift_report(b["reference"], pd.DataFrame(list(_recent)))
