"""FastAPI service: POST /predict scores a transaction."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Fraud Detection API")
_model = None


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1)
    distance_from_home: float = Field(ge=0)
    txn_last_24h: int = Field(ge=0)
    is_foreign: int = Field(ge=0, le=1)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run: python -m src.train")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction, threshold: float = 0.5):
    proba = float(get_model().predict_proba(pd.DataFrame([txn.model_dump()])[FEATURES])[0, 1])
    return {"fraud_probability": round(proba, 4), "is_fraud": proba >= threshold}
