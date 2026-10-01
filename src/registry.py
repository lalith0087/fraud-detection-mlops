"""MLflow model registry workflow.

Every `python -m src.train` registers a new version of `fraud-detector` and tags it `candidate`.
Promotion is gated: a candidate only becomes `production` if it is not worse than the current
production model on PR-AUC (same data source). Promoting exports the bundle to models/model.joblib,
which the API serves.

    python -m src.registry list
    python -m src.registry promote [--version N] [--force]
"""
import argparse
import shutil

import mlflow
from mlflow import MlflowClient

import src.train as T
from src.train import MODEL_NAME


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(T.TRACKING_URI)  # read at call time so tests can redirect it
    return MlflowClient()


def _info(client: MlflowClient, mv) -> dict:
    mv = client.get_model_version(MODEL_NAME, str(mv.version))  # search results omit aliases
    run = client.get_run(mv.run_id)
    return {"version": int(mv.version), "run_id": mv.run_id, "data": run.data.params.get("data"),
            "pr_auc": run.data.metrics.get("pr_auc"), "cost": run.data.metrics.get("cost_per_1k_txns"),
            "aliases": sorted(mv.aliases)}


def list_versions() -> list[dict]:
    client = _client()
    return sorted((_info(client, mv) for mv in client.search_model_versions(f"name='{MODEL_NAME}'")),
                  key=lambda d: d["version"])


def promote(version: int | None = None, force: bool = False) -> dict:
    client = _client()
    if version is None:
        version = int(client.get_model_version_by_alias(MODEL_NAME, "candidate").version)
    cand = _info(client, client.get_model_version(MODEL_NAME, str(version)))
    try:
        prod = _info(client, client.get_model_version_by_alias(MODEL_NAME, "production"))
    except mlflow.exceptions.MlflowException:
        prod = None
    if prod and not force:
        if prod["data"] != cand["data"]:
            raise SystemExit(f"Refusing: v{version} trained on '{cand['data']}' but production uses "
                             f"'{prod['data']}'. Use --force to override.")
        if cand["pr_auc"] < prod["pr_auc"]:
            raise SystemExit(f"Refusing: v{version} PR-AUC {cand['pr_auc']} < production v{prod['version']} "
                             f"PR-AUC {prod['pr_auc']}. Use --force to override.")
    client.set_registered_model_alias(MODEL_NAME, "production", version)
    path = mlflow.artifacts.download_artifacts(run_id=cand["run_id"], artifact_path="bundle/candidate.joblib")
    T.MODEL_DIR.mkdir(exist_ok=True)
    shutil.copy(path, T.MODEL_DIR / "model.joblib")
    return {"promoted": version, "replaced": prod["version"] if prod else None, **{k: cand[k] for k in ("data", "pr_auc")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("promote")
    p.add_argument("--version", type=int)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        for v in list_versions():
            print(v)
    else:
        print(promote(a.version, a.force))
