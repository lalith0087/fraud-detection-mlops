"""Data-drift monitoring with the Population Stability Index (PSI).

PSI < 0.1 stable, 0.1-0.25 moderate drift, > 0.25 significant drift.
"""
import numpy as np
import pandas as pd

BINS = 10
EPS = 1e-4


def build_reference(X: pd.DataFrame) -> dict:
    """Per-feature quantile bin edges and training-set bin proportions."""
    ref = {}
    for col in X.columns:
        edges = np.unique(np.quantile(X[col], np.linspace(0, 1, BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        props = np.histogram(X[col], bins=edges)[0] / len(X)
        ref[col] = {"edges": edges.tolist(), "props": props.tolist()}
    return ref


def psi(reference: dict, values) -> float:
    edges = np.array(reference["edges"])
    expected = np.clip(np.array(reference["props"]), EPS, None)
    actual = np.clip(np.histogram(values, bins=edges)[0] / len(values), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(score: float) -> str:
    return "stable" if score < 0.1 else "moderate" if score < 0.25 else "significant"


def drift_report(reference: dict, X: pd.DataFrame) -> dict:
    scores = {c: round(psi(reference[c], X[c].to_numpy()), 4) for c in reference}
    worst = max(scores, key=scores.get)
    return {
        "n_samples": len(X),
        "max_psi": scores[worst],
        "overall": status(scores[worst]),
        "features": {c: {"psi": s, "status": status(s)} for c, s in sorted(scores.items(), key=lambda kv: -kv[1])},
    }


if __name__ == "__main__":
    # Demo: compare held-out data against the trained reference, then a shifted copy.
    import argparse

    import joblib
    from sklearn.model_selection import train_test_split

    from src.data import load
    from src.train import MODEL_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    src = ap.parse_args().data
    b = joblib.load(MODEL_DIR / "model.joblib")
    df, feats = load(src)
    _, X_te = train_test_split(df[feats], test_size=0.2, random_state=42)
    for name, X in [("held-out (no drift)", X_te), ("shifted +1 std (drift)", X_te + X_te.std())]:
        r = drift_report(b["reference"], X)
        top = ", ".join(f"{c}={v['psi']}" for c, v in list(r["features"].items())[:3])
        print(f"{name}: overall={r['overall']} max_psi={r['max_psi']}  top: {top}")
