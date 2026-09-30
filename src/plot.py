"""Render docs/results.png (PR curve + confusion matrix) from the saved model on the held-out split."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import train_test_split

from src.data import load

ROOT = Path(__file__).resolve().parent.parent


def main(source: str = "real"):
    b = joblib.load(ROOT / "models" / "model.joblib")
    df, features = load(source)
    _, X_te, _, y_te = train_test_split(df[features], df["is_fraud"], test_size=0.2, stratify=df["is_fraud"], random_state=42)
    proba = b["model"].predict_proba(X_te[b["features"]])[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax[0], name="Model")
    ax[0].axhline(y_te.mean(), ls="--", c="gray", label=f"Baseline ({y_te.mean():.4f})")
    ax[0].set_title("Precision-Recall curve"); ax[0].legend(loc="upper right")
    ConfusionMatrixDisplay.from_predictions(y_te, proba >= b["threshold"], ax=ax[1], cmap="Blues", values_format="d",
                                            display_labels=["Legit", "Fraud"], colorbar=False)
    ax[1].set_title(f"Confusion matrix (threshold={b['threshold']})")
    fig.suptitle(f"Fraud detection - {source} data, held-out 20%")
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "results.png", dpi=150)


if __name__ == "__main__":
    main()
