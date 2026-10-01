"""Train a fraud model, log to MLflow, write models/candidate.joblib + models/metrics.json,
and register a new model version (see src/registry.py to promote it to production).

Usage: python -m src.train [--data synthetic|real] [--cost-fn 100] [--cost-fp 5]
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load
from src.drift import build_reference

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_NAME = "fraud-detector"


def best_threshold(y, proba) -> float:
    """Threshold maximising F1 on the validation split."""
    best, best_f1 = 0.5, -1.0
    for t in [i / 20 for i in range(1, 20)]:
        f1 = precision_recall_fscore_support(y, proba >= t, average="binary", zero_division=0)[2]
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def total_cost(y, proba, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Missed frauds cost `cost_fn` each, false alarms cost `cost_fp` each."""
    pred = np.asarray(proba) >= threshold
    y = np.asarray(y).astype(bool)
    return float(cost_fn * (y & ~pred).sum() + cost_fp * (~y & pred).sum())


def best_cost_threshold(y, proba, cost_fn: float, cost_fp: float) -> float:
    """Threshold minimising total business cost on the validation split."""
    grid = np.linspace(0.01, 0.99, 99)
    return float(min(grid, key=lambda t: total_cost(y, proba, t, cost_fn, cost_fp)))


def train(source: str = "synthetic", cost_fn: float = 100.0, cost_fp: float = 5.0) -> dict:
    df, features = load(source)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[features], df["is_fraud"], test_size=0.2, stratify=df["is_fraud"], random_state=42
    )
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    params = {"class_weight": "balanced", "random_state": 42}
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name=source):
        model = HistGradientBoostingClassifier(**params).fit(X_fit, y_fit)
        val_proba = model.predict_proba(X_val)[:, 1]
        f1_threshold = best_threshold(y_val, val_proba)
        threshold = best_cost_threshold(y_val, val_proba, cost_fn, cost_fp)
        proba = model.predict_proba(X_te)[:, 1]
        p, r, f1, _ = precision_recall_fscore_support(y_te, proba >= threshold, average="binary", zero_division=0)
        metrics = {
            "pr_auc": round(average_precision_score(y_te, proba), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "cost_per_1k_txns": round(1000 * total_cost(y_te, proba, threshold, cost_fn, cost_fp) / len(y_te), 2),
            "cost_per_1k_at_f1_threshold": round(1000 * total_cost(y_te, proba, f1_threshold, cost_fn, cost_fp) / len(y_te), 2),
            "cost_per_1k_at_0.5": round(1000 * total_cost(y_te, proba, 0.5, cost_fn, cost_fp) / len(y_te), 2),
        }
        mlflow.log_params({**params, "data": source, "threshold": threshold, "cost_fn": cost_fn, "cost_fp": cost_fp, "n_rows": len(df)})
        mlflow.log_metrics(metrics)
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "features": features, "threshold": threshold,
                     "reference": build_reference(X_fit)}, MODEL_DIR / "candidate.joblib")
        metrics.update(threshold=threshold, data=source)
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
        mlflow.log_artifact(str(MODEL_DIR / "candidate.joblib"), artifact_path="bundle")
        run = mlflow.active_run()
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    version = client.create_model_version(MODEL_NAME, source=f"{run.info.artifact_uri}/bundle", run_id=run.info.run_id)
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
    metrics["registered_version"] = int(version.version)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--cost-fn", type=float, default=100.0, help="cost of a missed fraud")
    ap.add_argument("--cost-fp", type=float, default=5.0, help="cost of a false alarm")
    a = ap.parse_args()
    print(json.dumps(train(a.data, a.cost_fn, a.cost_fp), indent=2))
