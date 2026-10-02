"""Drift-triggered retraining.

Reads the live drift report and, only if overall drift is `significant`, trains a new candidate
and tries to promote it through the registry gate. Run it on a schedule (cron / CI):

    python -m src.retrain --api-url https://<service>/drift --data real
    python -m src.retrain --report drift.json            # use a saved report instead

Note: this retrains on the configured data source. In production you would retrain on fresh
*labelled* transactions (e.g. confirmed fraud cases joined back to the logged traffic), because the
logged traffic itself has no labels.
"""
import argparse
import json
import urllib.request

from src import registry
from src.train import train


def fetch_report(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)


def maybe_retrain(report: dict, source: str = "synthetic", cost_fn: float = 100.0, cost_fp: float = 5.0) -> dict:
    if report.get("overall") != "significant":
        why = report.get("status") or f"drift is {report.get('overall', 'unknown')}"
        return {"retrained": False, "reason": why}
    metrics = train(source, cost_fn, cost_fp)
    version = metrics["registered_version"]
    try:
        out = registry.promote(version)
        return {"retrained": True, "version": version, "promoted": True, "pr_auc": metrics["pr_auc"],
                "replaced": out["replaced"], "trigger": report["overall"], "max_psi": report["max_psi"]}
    except SystemExit as e:  # gate refused the candidate
        return {"retrained": True, "version": version, "promoted": False, "reason": str(e),
                "trigger": report["overall"], "max_psi": report["max_psi"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--api-url", help="URL of the service's /drift endpoint")
    src.add_argument("--report", help="path to a saved drift report JSON")
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--cost-fn", type=float, default=100.0)
    ap.add_argument("--cost-fp", type=float, default=5.0)
    a = ap.parse_args()
    report = fetch_report(a.api_url) if a.api_url else json.load(open(a.report))
    print(json.dumps(maybe_retrain(report, a.data, a.cost_fn, a.cost_fp), indent=2))
