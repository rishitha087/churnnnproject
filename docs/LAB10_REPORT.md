# LAB 10 – ML Model Serving and API Engineering Using FastAPI

## 1. Aim

To implement a production-ready REST API using FastAPI that serves churn predictions, validates inputs with Pydantic schemas, provides health/metadata endpoints, handles errors gracefully, and meets latency and throughput performance requirements.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| FastAPI | 0.141.1 |
| Starlette | 1.7.0 |
| Pydantic | ≥2.0 |
| uvicorn | ≥0.20 |
| httpx | 0.28.1 |
| scikit-learn | 1.8.0 |
| joblib | ≥1.3 |

---

## 3. Libraries Used

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import uvicorn, joblib, numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional
```

---

## 4. Theory

### REST API Design for ML Serving
A production ML API must expose:
- **Health endpoint** (`GET /health`): Kubernetes liveness/readiness probe
- **Metadata endpoint** (`GET /metadata`): Model version, name, feature schema
- **Prediction endpoint** (`POST /predict`): Single-record inference
- **Batch endpoint** (`POST /predict/batch`): Multi-record inference

### Pydantic Validation
Pydantic enforces schema constraints before prediction:
```python
class CustomerFeatures(BaseModel):
    gender: str           # Must be "Male" or "Female"
    SeniorCitizen: int    # Must be 0 or 1
    tenure: int           # Must be ≥ 0
    MonthlyCharges: float
    TotalCharges: float
```

If validation fails, FastAPI automatically returns HTTP 422 with field-level error messages — no prediction is attempted.

### Model Loading with Lifespan
FastAPI lifespan context manager loads the model once at startup:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and preprocessor
    app.state.model = joblib.load("deployment/model.pkl")
    app.state.preprocessor = joblib.load("deployment/preprocessor.pkl")
    yield
    # Shutdown: cleanup
```

This prevents loading the model on every request (which would be ~200ms overhead per call).

### Train-Serving Consistency
The API uses the **same** `preprocessor.pkl` that was fit during training. This eliminates train-serving skew — the most common source of silent prediction errors in production.

### Performance Requirements
| Metric | Threshold | Actual |
|--------|-----------|--------|
| Single request latency | ≤ 200ms | 75.37ms |
| Average repeated latency | ≤ 50ms | 19.38ms |
| Throughput | ≥ 10 req/s | 51.68 req/s |
| Reliability (50 requests) | 100% | 100% |
| Prediction consistency | Same result for same input | ✅ |

---

## 5. Implementation

### File: `api/main.py`

**4 Endpoints:**

```python
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded}

@app.get("/metadata")
async def metadata():
    return {
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "model_loaded": model_loaded,
        "features": FEATURE_COLUMNS,
    }

@app.post("/predict")
async def predict(customer: CustomerFeatures):
    row = pd.DataFrame([customer.dict()])
    X = preprocessor.transform(row)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "churn_probability": round(prob, 4)}

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    rows = pd.DataFrame([c.dict() for c in request.customers])
    X = preprocessor.transform(rows)
    probs = model.predict_proba(X)[:, 1].tolist()
    preds = model.predict(X).tolist()
    return {"predictions": [{"prediction": p, "churn_probability": round(pr, 4)}
                             for p, pr in zip(preds, probs)]}
```

### File: `api/model_loader.py`
Handles model loading with graceful error logging.

### File: `tests/test_api.py`
23 test cases covering all endpoints and error scenarios.

### File: `tests/test_performance.py`
6 performance tests covering latency, throughput, consistency, reliability.

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI application (4 endpoints) |
| `api/model_loader.py` | Model and preprocessor loading |
| `api/preprocessing.py` | Preprocessing helper |
| `tests/test_api.py` | 23 functional API tests |
| `tests/test_performance.py` | 6 performance/reliability tests |
| `tests/direct_client.py` | In-process test client |
| `deployment/model.pkl` | Production RF model |
| `deployment/preprocessor.pkl` | Matching ColumnTransformer |
| `deployment/model_metadata.json` | Model version metadata |
| `reports/performance_report.json` | Performance test results |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Test endpoints (with server running):
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
       "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No",
       "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":85.5,"TotalCharges":1026.0}'

# Run all API tests (no server needed)
pytest tests/test_api.py tests/test_performance.py -v
```

---

## 8. Actual Outputs

```
=== GET /health ===
{"status": "healthy", "model_loaded": true}

=== GET /metadata ===
{
  "model_name": "Telco_Churn_Production_Model",
  "model_version": 3,
  "model_loaded": true,
  "features": ["gender", "SeniorCitizen", "Partner", ...]
}

=== POST /predict (valid input) ===
{"prediction": 0, "churn_probability": 0.4198}

=== POST /predict (invalid gender "X") ===
{"detail": [{"loc": ["body", "gender"], "msg": "...", "type": "value_error"}]}
HTTP 422 Unprocessable Entity

=== POST /predict/batch (2 customers) ===
{
  "predictions": [
    {"prediction": 0, "churn_probability": 0.4198},
    {"prediction": 1, "churn_probability": 0.7832}
  ]
}

=== Performance Metrics ===
single_request_latency_ms   : 75.37
average_latency_ms          : 19.38
median_latency_ms           : 18.79
minimum_latency_ms          : 17.84
maximum_latency_ms          : 23.89
reliability (50 requests)   : 100% success
batch_latency_ms (20 items) : 21.34
throughput_req_per_sec      : 51.68
prediction_consistency      : true
```

---

## 9. Testing and Validation

```bash
pytest tests/test_api.py tests/test_performance.py -v
```

**Results:**
```
tests/test_api.py::test_health                        PASSED
tests/test_api.py::test_metadata                      PASSED
tests/test_api.py::test_valid_prediction              PASSED
tests/test_api.py::test_invalid_prediction_gender     PASSED
tests/test_api.py::test_invalid_prediction_tenure     PASSED
tests/test_api.py::test_missing_prediction_field      PASSED
tests/test_api.py::test_valid_batch_prediction        PASSED
tests/test_api.py::test_invalid_batch_prediction      PASSED
tests/test_api.py::test_multiple_batch_validation_errors PASSED
tests/test_api.py::test_empty_batch                   PASSED
tests/test_api.py::test_invalid_datatype              PASSED
tests/test_api.py::test_extreme_numeric_values        PASSED
tests/test_api.py::test_null_value                    PASSED
tests/test_api.py::test_malformed_json                PASSED
tests/test_api.py::test_wrong_json_structure          PASSED
tests/test_api.py::test_null_numeric_value            PASSED
tests/test_api.py::test_null_total_charges            PASSED
tests/test_api.py::test_multiple_invalid_datatypes    PASSED
tests/test_api.py::test_invalid_batch_structure       PASSED
tests/test_api.py::test_null_batch_customers          PASSED
tests/test_api.py::test_batch_with_null_customer      PASSED
tests/test_api.py::test_extreme_negative_values       PASSED
tests/test_api.py::test_large_batch                   PASSED
tests/test_performance.py::test_single_request_latency PASSED
tests/test_performance.py::test_repeated_request_latency PASSED
tests/test_performance.py::test_prediction_consistency PASSED
tests/test_performance.py::test_api_reliability       PASSED
tests/test_performance.py::test_batch_prediction_performance PASSED
tests/test_performance.py::test_throughput            PASSED

============================== 29 passed in 10.19s ==============================
```

---

## 10. Result

| Endpoint | Tests | Status |
|---------|-------|--------|
| `GET /health` | 1 | ✅ PASSED |
| `GET /metadata` | 1 | ✅ PASSED |
| `POST /predict` | 15 | ✅ PASSED |
| `POST /predict/batch` | 6 | ✅ PASSED |
| **Functional total** | **23/23** | **✅ 100%** |
| **Performance total** | **6/6** | **✅ 100%** |

| Performance Metric | Threshold | Actual | Status |
|-------------------|-----------|--------|--------|
| Single request | ≤ 200ms | 75.37ms | ✅ |
| Average latency | ≤ 50ms | 19.38ms | ✅ |
| Throughput | ≥ 10 req/s | 51.68 req/s | ✅ |
| Reliability | 100% | 100% | ✅ |

---

## 11. Conclusion

Lab 10 delivered a production-quality FastAPI serving layer with 4 REST endpoints, Pydantic input validation, graceful error handling, and train-serving consistency. The API uses the same preprocessor as training — the single most critical correctness requirement. All 23 functional tests and 6 performance tests passed. The API achieves 51.68 req/s throughput with average latency of 19.38ms — well within acceptable production thresholds. The API is ready for containerization in Lab 11.

