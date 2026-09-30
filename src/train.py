"""Train a fraud model, log to MLflow, write models/model.joblib + models/metrics.json.

Usage: python -m src.train [--data synthetic|real]
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def best_threshold(y, proba) -> float:
    """Threshold maximising F1 on the validation split."""
    best, best_f1 = 0.5, -1.0
    for t in [i / 20 for i in range(1, 20)]:
        f1 = precision_recall_fscore_support(y, proba >= t, average="binary", zero_division=0)[2]
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def train(source: str = "synthetic") -> dict:
    df, features = load(source)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[features], df["is_fraud"], test_size=0.2, stratify=df["is_fraud"], random_state=42
    )
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    params = {"class_weight": "balanced", "random_state": 42}
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name=source):
        model = HistGradientBoostingClassifier(**params).fit(X_fit, y_fit)
        threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
        proba = model.predict_proba(X_te)[:, 1]
        p, r, f1, _ = precision_recall_fscore_support(y_te, proba >= threshold, average="binary", zero_division=0)
        metrics = {
            "pr_auc": round(average_precision_score(y_te, proba), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        }
        mlflow.log_params({**params, "data": source, "threshold": threshold, "n_rows": len(df)})
        mlflow.log_metrics(metrics)
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "features": features, "threshold": threshold}, MODEL_DIR / "model.joblib")
        metrics.update(threshold=threshold, data=source)
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    print(json.dumps(train(ap.parse_args().data), indent=2))
