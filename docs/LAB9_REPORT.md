# LAB 9 – Pipeline Failure Handling and Recovery

## 1. Aim

To simulate realistic pipeline failure scenarios, implement structured exception handling and logging, verify that the pipeline correctly detects and reports failures, and validate recovery mechanisms — ensuring robust behavior in production environments.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | ≥1.24 |
| pytest | 9.1.1 |

---

## 3. Libraries Used

```python
import logging, json, traceback, os
import pandas as pd, numpy as np
import joblib
from sklearn.exceptions import NotFittedError
```

---

## 4. Theory

### Why Failure Handling in MLOps?
Production ML pipelines are exposed to real-world instability:
- **Data sources** change schema unexpectedly
- **Storage** systems lose files
- **Network** disruptions interrupt model downloads
- **Data quality** degrades over time

Without proper failure handling:
- Silent failures produce wrong predictions
- Corrupted data trains bad models
- Missing artifacts crash serving APIs

### Failure Categories

| Category | Example | Expected Response |
|----------|---------|------------------|
| **Missing data** | Required CSV column absent | HALT + log error |
| **Invalid datatype** | Numeric column contains strings | HALT + log error |
| **Schema mismatch** | Expected 21 columns, got 15 | HALT + log error |
| **Preprocessing inconsistency** | Pipeline fit on different schema | HALT + log error |
| **Missing model artifact** | `.pkl` file deleted | RECOVER via retraining |

### HALT vs RECOVER Strategy
- **HALT**: Data-quality failures must stop the pipeline — wrong data produces wrong models
- **RECOVER**: Infrastructure failures (missing file) can be recovered by re-triggering the upstream stage

### Exception Handling Pattern
```python
try:
    result = run_stage()
except ValueError as e:
    logger.error(f"Validation error: {e}")
    report_failure(scenario="schema_mismatch", status="HALT")
    raise
except FileNotFoundError as e:
    logger.warning(f"Artifact missing: {e}")
    report_failure(scenario="missing_artifact", status="RECOVERING")
    recover()   # retrain and re-save
```

---

## 5. Implementation

### File: `src/mlops_pipeline/10_failure_injection.py`
Injects controlled failures, validates detection and recovery, runs 3 repetitions × 5 scenarios, generates `reports/failure_report.json`.

### 5 Failure Scenarios Implemented:

**Scenario 1: Missing Data**
```python
# Remove required columns
df_corrupt = df.drop(columns=['tenure', 'MonthlyCharges'])
# Validation must detect: "MISSING COLUMN: tenure"
validate_schema(df_corrupt)  # → raises ValueError
```

**Scenario 2: Invalid Datatype**
```python
# Inject string into numeric column
df_corrupt['tenure'] = 'invalid_string'
# Validation must detect: "TYPE ERROR: tenure"
validate_schema(df_corrupt)  # → raises ValueError
```

**Scenario 3: Schema Mismatch**
```python
# Rename a required column
df_corrupt = df.rename(columns={'Churn': 'ChurnFlag'})
# Validation must detect: "MISSING COLUMN: Churn"
validate_schema(df_corrupt)  # → raises ValueError
```

**Scenario 4: Preprocessing Inconsistency**
```python
# Try to transform data with wrong number of columns
X_bad = X_test[:, :10]   # only 10 of 30 expected features
preprocessor.transform(X_bad)  # → raises ValueError
```

**Scenario 5: Missing Model Artifact**
```python
# Delete model file temporarily
os.rename("models/churn_model.pkl", "models/churn_model.pkl.bak")
try:
    joblib.load("models/churn_model.pkl")  # → FileNotFoundError
except FileNotFoundError:
    # Recovery: retrain and save
    retrain_and_save()   # re-creates the file
    logger.info("Recovery: model retrained successfully")
finally:
    # Restore backup if recovery fails
    if not os.path.exists("models/churn_model.pkl"):
        os.rename("models/churn_model.pkl.bak", "models/churn_model.pkl")
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/mlops_pipeline/10_failure_injection.py` | All 5 failure scenarios + recovery |
| `src/validate_data.py` | Schema validation (used by scenarios 1–3) |
| `src/train.py` | Recovery retraining (scenario 5) |
| `reports/failure_report.json` | Actual test results |
| `.github/workflows/lab9_failure_tests.yml` | CI workflow for failure tests |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Run failure injection and recovery tests
python src/mlops_pipeline/10_failure_injection.py

# View failure report
python -c "import json; r=json.load(open('reports/failure_report.json')); print(json.dumps(r['summary'], indent=2))"
```

---

## 8. Actual Outputs

```
=== Lab 9: Failure Handling and Recovery ===
Running 3 repetitions × 5 scenarios = 15 tests

--- Repetition 1 ---
  [SCENARIO 1] missing_data
    Expected: HALT | Actual: HALT
    Error caught: MISSING COLUMN: tenure
    Result: ✅ PASSED

  [SCENARIO 2] invalid_datatype
    Expected: HALT | Actual: HALT
    Error caught: TYPE ERROR: tenure should be numeric
    Result: ✅ PASSED

  [SCENARIO 3] schema_mismatch
    Expected: HALT | Actual: HALT
    Error caught: MISSING COLUMN: Churn
    Result: ✅ PASSED

  [SCENARIO 4] preprocessing_inconsistency
    Expected: HALT | Actual: HALT
    Error caught: X has 10 features but preprocessor expects 30
    Result: ✅ PASSED

  [SCENARIO 5] missing_model_artifact
    Expected: RECOVER | Actual: RECOVERED
    Failure detected: FileNotFoundError
    Recovery: Model retrained and saved
    Result: ✅ PASSED

--- Repetition 2 --- [All 5 PASSED] ✅
--- Repetition 3 --- [All 5 PASSED] ✅

=== Summary ===
Total tests    : 15
Passed         : 15
Failed         : 0
Success rate   : 100%

Report saved: reports/failure_report.json
```

**Failure report (from `reports/failure_report.json`):**
```json
{
  "stage": "Failure Handling and Recovery",
  "repetitions": 3,
  "runs": [
    {
      "repetition": 1,
      "results": [
        {"scenario": "missing_data",          "expected": "HALT",    "actual": "HALT",      "passed": true},
        {"scenario": "invalid_datatype",       "expected": "HALT",    "actual": "HALT",      "passed": true},
        {"scenario": "schema_mismatch",        "expected": "HALT",    "actual": "HALT",      "passed": true},
        {"scenario": "preprocessing_inconsistency", "expected": "HALT", "actual": "HALT",   "passed": true},
        {"scenario": "missing_model_artifact", "expected": "RECOVER", "actual": "RECOVERED","passed": true}
      ],
      "passed": true
    }
  ]
}
```

---

## 9. Testing and Validation

```bash
pytest tests/test_pipeline.py -v -k "not api"
```

**Results:**
```
tests/test_pipeline.py::test_raw_dataset_exists_and_valid          PASSED
tests/test_pipeline.py::test_preprocessing_produces_clean_arrays   PASSED
tests/test_pipeline.py::test_model_training_and_serialization      PASSED
tests/test_pipeline.py::test_evaluation_metrics_and_error_exports  PASSED
tests/test_pipeline.py::test_predict_pipeline_single_and_batch     PASSED
tests/test_pipeline.py::test_pipeline_reproducibility              PASSED
```

**Validation checks:**
- ✅ All 5 failure scenarios detected correctly (15/15 tests)
- ✅ HALT scenarios raise exceptions and log errors — no silent failures
- ✅ RECOVER scenario re-trains model and restores artifact
- ✅ 3 repetitions confirm consistent behavior (no flakiness)
- ✅ Failure report JSON generated with structured results
- ✅ GitHub Actions CI workflow configured for automated failure testing

---

## 10. Result

| Failure Scenario | Expected | Actual | Passed |
|-----------------|----------|--------|--------|
| Missing data columns | HALT | HALT | ✅ |
| Invalid datatype | HALT | HALT | ✅ |
| Schema mismatch | HALT | HALT | ✅ |
| Preprocessing inconsistency | HALT | HALT | ✅ |
| Missing model artifact | RECOVER | RECOVERED | ✅ |
| **Total (3 repetitions × 5)** | **15** | **15 passed** | **✅ 100%** |

---

## 11. Conclusion

Lab 9 implemented comprehensive failure handling and recovery for all realistic MLOps pipeline failure modes. The structured approach — HALT for data quality failures, RECOVER for infrastructure failures — mirrors production incident response playbooks. All 15 test cases (3 repetitions × 5 scenarios) passed with 100% success rate. The structured `failure_report.json` provides auditable evidence of failure detection capability. This failure handling layer is what distinguishes a research script from a production-grade ML system.

