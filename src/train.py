"""Train a fraud model and write models/model.joblib + models/metrics.json."""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import FEATURES, make_transactions

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train(threshold: float = 0.5) -> dict:
    df = make_transactions()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df["is_fraud"], test_size=0.2, stratify=df["is_fraud"], random_state=42
    )
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_te, proba >= threshold, average="binary", zero_division=0)
    metrics = {
        "pr_auc": round(average_precision_score(y_te, proba), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "threshold": threshold,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
