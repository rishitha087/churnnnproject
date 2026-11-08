# Architecture — Telco Churn Prediction MLOps Project

## Overview

This project implements a complete MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset. It covers the full lifecycle: raw data → preprocessing → training → tracking → registry → quality gate → serving → containerization.

---

## Project Structure

```
churnn project/
├── .github/workflows/        # Lab 8: CI automation (GitHub Actions)
│   ├── lab8_ci.yml           # Main CI: triggers on push/PR to main
│   └── lab9_failure_tests.yml
├── api/                      # Lab 10: FastAPI serving
│   ├── main.py               # 4 REST endpoints
│   ├── model_loader.py       # Artifact loading with lifespan
│   ├── preprocessing.py      # Preprocessing helper
│   └── monitoring_logger.py  # Request logging
├── data/
│   ├── raw/                  # Source CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   └── processed/            # Numpy arrays (X_train, X_val, X_test, y_*)
├── deployment/               # Lab 6/11: Production bundle
│   ├── model.pkl             # Production RF model (Version 3)
│   ├── preprocessor.pkl      # Matching ColumnTransformer
│   └── model_metadata.json   # Version + run_id
├── docs/                     # All lab reports + architecture
├── k8s/                      # Lab 11: Kubernetes manifests
│   ├── deployment.yaml       # 1-replica Deployment
│   └── service.yaml          # NodePort Service
├── logs/                     # Pipeline execution logs
├── models/                   # Serialized model artifacts (Labs 2, 6)
├── notebooks/                # Jupyter exploration notebooks
├── outputs/                  # Predictions, false positives/negatives
├── pipelines/                # Labs 1–6 orchestration scripts
├── reports/                  # Evaluation, failure, audit reports
├── src/
│   ├── preprocess.py         # Lab 1: ColumnTransformer preprocessing
│   ├── train.py              # Lab 2: Baseline model training
│   ├── evaluate.py           # Lab 2: Metric computation
│   ├── train_mlflow.py       # Lab 4: MLflow-tracked training
│   ├── validate_data.py      # Lab 5: Schema validation
│   ├── train_registry.py     # Lab 6: Model registration
│   ├── automate_lifecycle.py # Lab 6: Stage transitions
│   └── mlops_pipeline/       # Lab 7: 10-stage integrated pipeline
│       ├── 01_load_data.py
│       ├── 02_validate_data.py
│       ├── 03_preprocessing.py
│       ├── 04_validate_outputs.py
│       ├── 05_train_registry.py
│       ├── 06_model_evaluation.py
│       ├── 07_quality_gate.py
│       ├── 08_automate_lifecycle.py
│       ├── 09_inference.py
│       ├── 10_failure_injection.py
│       ├── 11_prepare_deployment.py
│       └── churn_prediction_pipeline.py  ← MAIN ORCHESTRATOR
├── tests/
│   ├── test_pipeline.py      # 6 pipeline tests
│   ├── test_api.py           # 23 API functional tests
│   ├── test_performance.py   # 6 API performance tests
│   └── direct_client.py      # In-process test client (no server needed)
├── Dockerfile                # Lab 11: Container definition
├── .dockerignore
├── .gitignore
├── config.yaml               # Central config (model params, paths, thresholds)
├── requirements.txt          # Full Python dependencies
├── requirements-api.txt      # API-only dependencies (for Docker)
└── README.md
```

---

## End-to-End Data Flow

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
              │
              ▼
  [Stage 1] 01_load_data.py
       └── EDA: 7,043 rows × 21 columns
              │
              ▼
  [Stage 2] 02_validate_data.py
       └── Schema check: required columns, types, categoricals
              │
              ▼
  [Stage 3] 03_preprocessing.py    ←── models/preprocessor.pkl (saved here)
       └── ColumnTransformer: StandardScaler + OneHotEncoder
       └── Splits: train(5875,30) / val(735,30) / test(1469,30)
              │
              ▼
  [Stage 4] 04_validate_outputs.py
       └── Zero NaN check, shape validation
              │
              ▼
  [Stage 5] 05_train_registry.py   ←── MLflow: logs params + metrics + model
       └── RandomForest: n_estimators=100, max_depth=10, class_weight=balanced
       └── MLflow Registry: Telco_Churn_Production_Model Version 3
              │
              ▼
  [Stage 6] 06_model_evaluation.py
       └── Accuracy=0.7693, Recall=0.7219, ROC-AUC=0.8447
              │
              ▼
  [Stage 7] 07_quality_gate.py     ←── HARD GATE: recall ≥ 0.70
       └── 0.7219 ≥ 0.70 → PASSED
              │
              ▼
  [Stage 8] 08_automate_lifecycle.py
       └── Version 3: None → Staging → Production
              │
              ▼
  [Stage 9] 11_prepare_deployment.py
       └── deployment/model.pkl + preprocessor.pkl + model_metadata.json
              │
              ▼
  [Stage 10] 09_inference.py
       └── outputs/inference_predictions.csv (1,469 predictions)
```

---

## MLflow Architecture

```
mlflow.db (SQLite)                mlruns/ (artifact store)
│                                 │
├── Experiment: Telco_Churn_Prediction
│   ├── Run: RF_Baseline   (run_id: 7d6298...)
│   ├── Run: RF_Shallow
│   ├── Run: LR_Baseline
│   └── Run: DT_Baseline
│
└── Model Registry: Telco_Churn_Production_Model
    ├── Version 1  → Archived
    ├── Version 2  → Archived
    └── Version 3  → Production ← api/main.py loads from deployment/
```

---

## FastAPI Serving Architecture

```
Client Request
    │
    ▼
POST /predict  (or /predict/batch)
    │
    ▼
Pydantic validation (CustomerFeatures schema)
    │  422 on invalid input ──────────────────────► error response
    ▼
preprocessor.transform(X)      ← deployment/preprocessor.pkl
    │
    ▼
model.predict_proba(X)         ← deployment/model.pkl
    │
    ▼
{"prediction": 0/1, "churn_probability": float}
```

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness/readiness probe |
| GET | `/metadata` | Model name, version, feature list |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Multi-customer prediction |

---

## CI/CD Architecture

```
Developer pushes to main
         │
         ▼
GitHub Actions: .github/workflows/lab8_ci.yml
         │
         ├── actions/checkout@v4
         ├── actions/setup-python@v5 (Python 3.11)
         ├── pip install -r requirements.txt
         ├── python churn_prediction_pipeline.py
         │         └── [All 10 stages run + quality gate]
         └── actions/upload-artifact@v4
                   └── reports/, logs/, models/
```

---

## Kubernetes Deployment Architecture

```
kubectl apply -f k8s/
         │
         ▼
Deployment: churn-api
  └── 1 Pod
      └── Container: churn-prediction-api:1.0
          ├── python:3.11-slim base
          ├── COPY api/ ./api/
          ├── COPY deployment/ ./deployment/
          └── CMD uvicorn api.main:app --port 8000
                │
Service: churn-api-service (NodePort)
  └── port 8000 → NodePort (30000-32767)
```

---

## Key Configuration (`config.yaml`)

| Parameter | Value | Used By |
|-----------|-------|---------|
| `n_estimators` | 100 | Train stages |
| `max_depth` | 10 | Train stages |
| `class_weight` | balanced | All classifiers |
| `random_state` | 42 | All random operations |
| `recall_threshold` | 0.70 | Quality gate |
| `test_size` | 0.2 | Train/test split |

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.12.6 |
| ML | scikit-learn | 1.8.0 |
| Tracking | MLflow | 3.16.1 |
| API | FastAPI + uvicorn | 0.141.1 |
| Validation | Pydantic | ≥2.0 |
| Data | pandas + numpy | 3.0.2 |
| Container | Docker | 20.10+ |
| Orchestration | Kubernetes | 1.28+ |
| CI | GitHub Actions | — |
| Testing | pytest | 9.1.1 |
| VCS | Git | 2.53.0 |
