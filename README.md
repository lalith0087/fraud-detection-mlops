# Fraud Detection MLOps

## Live demo
| | Link |
|---|---|
| Interactive API docs | https://fraud-detection-api-o6vb.onrender.com/docs |
| Sample fraud + legitimate transactions | https://fraud-detection-api-o6vb.onrender.com/example |
| Health check | https://fraud-detection-api-o6vb.onrender.com/health |
| Drift report | https://fraud-detection-api-o6vb.onrender.com/drift |
| Source code | https://github.com/lalith0087/fraud-detection-mlops |

Try it: open `/example`, copy the `fraud` object, then in `/docs` use **POST /predict** > Try it out, paste it in the request body and Execute.
Hosted on Render's free tier, so the first request after idle can take ~50 s to wake the service.

[![CI](https://github.com/lalith0087/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/lalith0087/fraud-detection-mlops/actions/workflows/ci.yml)

End-to-end fraud detection: data generation → training → REST API → Docker → CI.

## Overview
- **Data**: the real ULB credit-card fraud dataset (fetched from OpenML, cached in `data/`) or a synthetic generator (`src/data.py`).
- **Tracking**: every run logs params, metrics and artifacts to MLflow (SQLite backend).
- **Model**: `HistGradientBoostingClassifier` with balanced class weights, evaluated on PR-AUC (the right metric for imbalanced data).
- **Serving**: FastAPI `POST /predict` with validated input and a tunable threshold.
- **Ops**: Dockerfile trains the model at build time; GitHub Actions runs tests and builds the image.

## Results (held-out 20%)
| Data | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Real (ULB credit-card, 284,807 txns, 0.17% fraud) | 0.735 | 0.962 | 0.480 | 0.878 | 0.621 |
| Synthetic (1% fraud, overlapping classes) | 0.586 | 0.791 | 0.457 | 0.567 | 0.506 |

![Results](docs/results.png)

On the real data at the tuned threshold: 86 of 98 frauds caught, with 93 false alarms out of 56,864 legitimate transactions.

## Cost-based threshold tuning
The decision threshold is chosen to minimise business cost, not F1: a missed fraud costs `--cost-fn` (default 100) and a false alarm costs `--cost-fp` (default 5). On the real data (threshold 0.62):

| Threshold strategy | Cost per 1,000 transactions |
|---|---|
| Default 0.5 | 30.63 |
| Best F1 | 34.15 |
| **Best cost (used)** | **29.23** |

Optimising F1 alone is actually the most expensive choice here, because it trades away recall for precision that costs more than it saves. Change the costs to match your business: `python -m src.train --data real --cost-fn 200 --cost-fp 2`.

## Drift monitoring
The API keeps the last 1,000 scored transactions and `GET /drift` compares them with the training distribution using the Population Stability Index (PSI) per feature (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant). It returns the overall status and features ranked by drift, and reports `insufficient_data` until 100 transactions have been scored. Reference bins are stored with the model at train time.

Offline demo (`python -m src.drift --data real`): held-out data scores max PSI 0.0003 (stable); the same data shifted by +1 std scores 8.3 (significant, led by `Amount`).

## Model registry
Each training run registers a new version of `fraud-detector` in the MLflow registry and tags it `candidate`. Promotion to `production` is a separate, gated step:

```bash
python -m src.train --data real      # registers a new candidate version
python -m src.registry promote       # gate, set `production` alias, export to models/model.joblib
python -m src.registry list          # versions, PR-AUC, cost, aliases
```
The gate refuses a candidate whose PR-AUC is lower than the current production model, or that was trained on a different dataset, unless you pass `--force`. The API only serves the promoted model (`models/model.joblib`), never a raw training output.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data real   # or --data synthetic (default)
python -m src.registry promote    # serve the new model
python -m src.plot                 # regenerate docs/results.png
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn src.api:app --reload
pytest
```
```bash
curl localhost:8000/drift      # PSI drift report on recent traffic
curl localhost:8000/example    # sample fraud + legitimate request bodies for /predict
curl localhost:8000/features   # feature names the loaded model expects
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"features":{"amount":900,"hour":2,"merchant_risk":0.9,"distance_from_home":300,"txn_last_24h":8,"is_foreign":1}}'
```

## Deploy
`render.yaml` is a Render blueprint: it builds the Dockerfile with `DATA=real` (trains on the real dataset during the build) and serves the API, with `/health` as the health check. Interactive docs are at `/docs`.

## Next steps
Automated retraining when drift is significant, a persistent drift buffer (Redis).
