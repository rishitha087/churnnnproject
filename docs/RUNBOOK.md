# Runbook — Telco Churn Prediction MLOps Project

## Prerequisites

### Python Environment
```powershell
python --version          # Requires 3.10+
pip install -r requirements.txt
```

### Verify Key Packages
```powershell
python -c "import sklearn, mlflow, fastapi; print('All OK')"
```

### Optional (for Lab 11)
- Docker Desktop (for container build/run)
- kubectl + minikube (for Kubernetes deployment)

---

## Lab Execution Commands

### Lab 1 — Dataset Preparation
```powershell
cd "churnn project"
python src/preprocess.py
# Output: models/preprocessor.pkl, data/processed/*.npy
```

### Lab 2 — Baseline Models
```powershell
python src/train.py
python src/evaluate.py
# Output: models/random_forest_baseline.pkl, outputs/metrics/baseline_metrics.json
```

### Labs 1–6 (Full Run)
```powershell
python pipelines/run_all.py
# Runs: preprocess → train → evaluate → mlflow → validate → registry
```

### Lab 3 — Git Verification
```powershell
git log --oneline --all
git branch -a
git tag
```

### Lab 4 — MLflow UI
```powershell
# View tracked experiments in browser
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# Then open: http://127.0.0.1:5000
```

### Lab 5 — Data Validation
```powershell
python src/validate_data.py
python src/simulate_data_corruption.py
python src/preprocess_pipeline.py
```

### Lab 6 — Model Registry
```powershell
python src/train_registry.py
python src/automate_lifecycle.py
python src/generate_registry_report.py
# Output: reports/lifecycle_report.json
```

### Lab 7 — Full Pipeline (Recommended)
```powershell
# Run entire 10-stage pipeline (re-runs all stages)
python src/mlops_pipeline/churn_prediction_pipeline.py --force
# Output: models/, deployment/, reports/, logs/churn_prediction_pipeline_run_log.json
```

### Lab 8 — CI Validation
```powershell
# Validate CI YAML locally
python -c "import yaml; yaml.safe_load(open('.github/workflows/lab8_ci.yml')); print('YAML valid')"

# To trigger on GitHub: push this repository to a GitHub remote
git remote add origin https://github.com/<username>/churn-mlops.git
git push -u origin main
# CI will trigger automatically on push to main
```

### Lab 9 — Failure Handling
```powershell
python src/mlops_pipeline/10_failure_injection.py
# Output: reports/failure_report.json (15/15 tests passed)
```

### Lab 10 — API Testing
```powershell
# Run all API + performance tests (no server needed)
pytest tests/test_api.py tests/test_performance.py -v

# Start the API server for manual testing
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Manual curl tests (with server running)
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

### Lab 11 — Docker and Kubernetes
```powershell
# Requires Docker Desktop installed

# Build image
docker build -t churn-prediction-api:1.0 .

# Run container
docker run -d --name churn-api -p 8000:8000 churn-prediction-api:1.0

# Test container API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" `
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.5,"TotalCharges":1026.0}'

# Stop container
docker stop churn-api && docker rm churn-api

# Kubernetes (requires kubectl + cluster)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get services
```

---

## Run All Tests
```powershell
cd "churnn project"
pytest tests/test_pipeline.py tests/test_api.py tests/test_performance.py -v
# Expected: 35 passed
```

---

## Master One-Command Workflow
```powershell
cd "churnn project"
# Step 1: Run Labs 1-6 pipeline
python pipelines/run_all.py
# Step 2: Run full 10-stage integrated pipeline
python src/mlops_pipeline/churn_prediction_pipeline.py --force
# Step 3: Run failure handling tests
python src/mlops_pipeline/10_failure_injection.py
# Step 4: Run all pytest tests
pytest tests/ -v --ignore=tests/test_local_vs_container.py
```

---

## Common Troubleshooting

### ImportError on sklearn model load
**Symptom:** `InconsistentVersionWarning` when loading `.pkl` files  
**Cause:** Model was saved with a different sklearn version  
**Fix:** Retrain by running `python src/mlops_pipeline/churn_prediction_pipeline.py --force`

### MLflow UntrustedTypesFoundException
**Symptom:** `skops.io.exceptions.UntrustedTypesFoundException`  
**Cause:** MLflow 3.x uses skops serialization by default  
**Fix:** Already fixed — all `log_model()` calls use `serialization_format="cloudpickle"`

### Quality Gate FAILED
**Symptom:** `Quality Gate FAILED: recall=X.XXXX < 0.7000`  
**Cause:** Model hyperparameters produce low recall  
**Fix:** Ensure `config.yaml` has `n_estimators: 100, max_depth: 10, class_weight: balanced`

### pytest test_api.py hangs
**Symptom:** Tests run indefinitely  
**Cause:** Starlette TestClient + anyio blocking portal conflict  
**Fix:** Already fixed — `tests/direct_client.py` provides `DirectTestClient` that calls endpoint functions directly without sockets

### Docker not found
**Symptom:** `docker: command not found`  
**Cause:** Docker Desktop not installed  
**Fix:** Install Docker Desktop from https://www.docker.com/products/docker-desktop/  
Then run: `docker build -t churn-prediction-api:1.0 .`

---

## Key File Locations

| Artifact | Path |
|----------|------|
| Main pipeline | `src/mlops_pipeline/churn_prediction_pipeline.py` |
| Production model | `deployment/model.pkl` |
| Preprocessor | `deployment/preprocessor.pkl` |
| Model metadata | `deployment/model_metadata.json` |
| MLflow DB | `mlflow.db` |
| MLflow artifacts | `mlruns/` |
| Config | `config.yaml` |
| CI workflow | `.github/workflows/lab8_ci.yml` |
| Failure report | `reports/failure_report.json` |
| Final audit | `reports/FINAL_MLOPS_AUDIT.md` |
| Test results | `reports/FINAL_TEST_RESULTS.json` |

---

## Environment Versions (Tested)

| Package | Version |
|---------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| MLflow | 3.16.1 |
| FastAPI | 0.141.1 |
| Starlette | 1.7.0 |
| pandas | 3.0.2 |
| pytest | 9.1.1 |
| uvicorn | ≥0.20 |
| pydantic | ≥2.0 |
| joblib | ≥1.3 |
