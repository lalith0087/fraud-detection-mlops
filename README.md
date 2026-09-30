# Fraud Detection MLOps

End-to-end fraud detection: data generation → training → REST API → Docker → CI.

## Overview
- **Data**: synthetic, ~1% fraud rate, with overlapping classes (`src/data.py`). Swap in the Kaggle credit-card dataset by replacing `make_transactions`.
- **Model**: `HistGradientBoostingClassifier` with balanced class weights, evaluated on PR-AUC (the right metric for imbalanced data).
- **Serving**: FastAPI `POST /predict` with validated input and a tunable threshold.
- **Ops**: Dockerfile trains the model at build time; GitHub Actions runs tests and builds the image.

## Results (held-out 20%)
| PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.586 | 0.791 | 0.457 | 0.567 | 0.506 |

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --reload
pytest
```
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"amount":900,"hour":2,"merchant_risk":0.9,"distance_from_home":300,"txn_last_24h":8,"is_foreign":1}'
```

## Next steps
MLflow experiment tracking, threshold tuning by cost, drift monitoring, real dataset.
