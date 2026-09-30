"""Synthetic, highly imbalanced card-transaction data (~1% fraud)."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "merchant_risk", "distance_from_home", "txn_last_24h", "is_foreign"]


def make_transactions(n: int = 50_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    # only ~55% of fraud looks anomalous; the rest blends in with normal traffic
    f = (y == 1) & (rng.random(n) < 0.55)
    df = pd.DataFrame({
        "amount": np.where(f, rng.lognormal(5.0, 1.0, n), rng.lognormal(3.5, 0.9, n)),
        "hour": np.where(f, rng.choice([0, 1, 2, 3, 4, 23], n), rng.integers(6, 23, n)),
        "merchant_risk": np.clip(np.where(f, rng.normal(0.65, 0.2, n), rng.normal(0.3, 0.2, n)), 0, 1),
        "distance_from_home": np.where(f, rng.exponential(120, n), rng.exponential(15, n)),
        "txn_last_24h": np.where(f, rng.poisson(6, n), rng.poisson(2, n)),
        "is_foreign": np.where(f, rng.random(n) < 0.4, rng.random(n) < 0.05).astype(int),
        "is_fraud": y,
    })
    return df


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_real() -> pd.DataFrame:
    """ULB credit-card fraud dataset (284,807 txns, 0.17% fraud) via OpenML; cached in data/."""
    cache = DATA_DIR / "creditcard.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    df = df.rename(columns={"Class": "is_fraud"}).astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache)
    return df


def load(source: str = "synthetic") -> tuple[pd.DataFrame, list[str]]:
    df = load_real() if source == "real" else make_transactions()
    return df, [c for c in df.columns if c != "is_fraud"]
