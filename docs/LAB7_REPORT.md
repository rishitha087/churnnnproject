# LAB 7 – ML Pipeline Engineering

## 1. Aim

To design, implement, and validate a complete, modular, end-to-end MLOps pipeline integrating data loading, validation, preprocessing, training, evaluation, quality gating, model registration, lifecycle management, and production inference into a single reproducible executable workflow.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| MLflow | 3.16.1 |
| pandas | 3.0.2 |
| PyYAML | ≥6.0 |
| joblib | ≥1.3 |

---

## 3. Libraries Used

```python
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_auc_score
import pandas as pd, numpy as np, yaml, json, joblib, logging
```

---

## 4. Theory

### ML Pipeline Architecture
An ML pipeline is a directed acyclic graph (DAG) of stages where each stage's output feeds the next. This structure provides:
- **Modularity**: Each stage can be developed, tested, and replaced independently
- **Reproducibility**: Fixed inputs + fixed code = fixed outputs
- **Automation**: Entire workflow executes from a single command
- **Auditability**: Logs capture what happened at each stage

### Pipeline Stages (10 Stages)

```
Stage 1: Data Loading
    ↓
Stage 2: Raw Data Validation
    ↓
Stage 3: Preprocessing
    ↓
Stage 4: Output Validation
    ↓
Stage 5: Training + MLflow + Registry
    ↓
Stage 6: Model Evaluation
    ↓
Stage 7: Quality Gate
    ↓
Stage 8: Lifecycle Automation
    ↓
Stage 9: Deployment Bundle Preparation
    ↓
Stage 10: Production Inference
```

### Quality Gate
A quality gate is an automated threshold check. If the trained model's recall < 0.70, the pipeline halts and does **not** promote the model to Production. This prevents regression from reaching users.

### Idempotency and `--force` Flag
The pipeline supports `--force` to re-run all stages even if outputs already exist, ensuring fresh results when explicitly triggered. Without `--force`, it skips stages whose outputs are present (caching behavior).

---

## 5. Implementation

### File: `src/mlops_pipeline/churn_prediction_pipeline.py`
The main orchestrator that runs all 10 stages sequentially, with per-stage error handling and logging.

### File: `src/mlops_pipeline/01_load_data.py` through `11_prepare_deployment.py`
Individual stage modules that can also be run independently.

### Key orchestration code:
```python
stages = [
    ("Data Loading",        "01_load_data"),
    ("Raw Validation",      "02_validate_data"),
    ("Preprocessing",       "03_preprocessing"),
    ("Output Validation",   "04_validate_outputs"),
    ("Training+Registry",   "05_train_registry"),
    ("Model Evaluation",    "06_model_evaluation"),
    ("Quality Gate",        "07_quality_gate"),
    ("Lifecycle",           "08_automate_lifecycle"),
    ("Deployment Prep",     "11_prepare_deployment"),
    ("Inference",           "09_inference"),
]

for stage_name, module_name in stages:
    logger.info(f"=== Stage: {stage_name} ===")
    try:
        result = subprocess.run(
            [sys.executable, f"src/mlops_pipeline/{module_name}.py"],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode != 0:
            raise RuntimeError(f"Stage '{stage_name}' failed")
    except Exception as e:
        logger.error(f"Stage {stage_name} failed: {e}")
        sys.exit(1)
```

### Quality Gate (Stage 7):
```python
# Load evaluation report
with open("reports/evaluation_report.json") as f:
    report = json.load(f)

recall = report["recall"]
threshold = 0.70

if recall < threshold:
    raise RuntimeError(
        f"Quality Gate FAILED: recall={recall:.4f} < {threshold}"
    )
print(f"Quality Gate PASSED: recall={recall:.4f} >= {threshold}")
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/mlops_pipeline/churn_prediction_pipeline.py` | Main pipeline orchestrator |
| `src/mlops_pipeline/01_load_data.py` | Stage 1: Data loading |
| `src/mlops_pipeline/02_validate_data.py` | Stage 2: Raw validation |
| `src/mlops_pipeline/03_preprocessing.py` | Stage 3: Preprocessing |
| `src/mlops_pipeline/04_validate_outputs.py` | Stage 4: Output validation |
| `src/mlops_pipeline/05_train_registry.py` | Stage 5: Training + MLflow |
| `src/mlops_pipeline/06_model_evaluation.py` | Stage 6: Evaluation |
| `src/mlops_pipeline/07_quality_gate.py` | Stage 7: Quality gate |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Stage 8: Lifecycle |
| `src/mlops_pipeline/11_prepare_deployment.py` | Stage 9: Deployment |
| `src/mlops_pipeline/09_inference.py` | Stage 10: Inference |
| `logs/churn_prediction_pipeline_run_log.json` | Pipeline execution log |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Run full 10-stage pipeline (cached)
python src/mlops_pipeline/churn_prediction_pipeline.py

# Run with --force to re-execute all stages
python src/mlops_pipeline/churn_prediction_pipeline.py --force

# Run individual stages
python src/mlops_pipeline/01_load_data.py
python src/mlops_pipeline/07_quality_gate.py
```

---

## 8. Actual Outputs

```
=== Churn Prediction MLOps Pipeline ===
Running with --force: all stages will re-execute

[Stage 1] Data Loading         → 7343 rows loaded ✅
[Stage 2] Raw Validation       → Schema: PASSED ✅
[Stage 3] Preprocessing        → (5875,30) train, (1469,30) test ✅
[Stage 4] Output Validation    → No NaNs, shape correct ✅
[Stage 5] Training+Registry    → RF Model Version 3 registered ✅
           MLflow Run ID: 7d6298bea41640cb9a4e4b6128df82d4
[Stage 6] Model Evaluation     → Accuracy=0.7693, Recall=0.7219 ✅
[Stage 7] Quality Gate         → recall=0.7219 >= 0.70 PASSED ✅
[Stage 8] Lifecycle            → Version 3 → Production ✅
[Stage 9] Deployment Prep      → deployment/ bundle saved ✅
[Stage 10] Inference           → 1469 predictions completed ✅

=== Final Metrics ===
Accuracy : 0.7693
Precision: 0.5499
Recall   : 0.7219
F1-Score : 0.6243
ROC-AUC  : 0.8447

=== Pipeline Complete ===
Duration: ~3 minutes
Log: logs/churn_prediction_pipeline_run_log.json
```

---

## 9. Testing and Validation

```bash
pytest tests/test_pipeline.py -v
```

**Results:**
```
tests/test_pipeline.py::test_raw_dataset_exists_and_valid          PASSED
tests/test_pipeline.py::test_preprocessing_produces_clean_arrays   PASSED
tests/test_pipeline.py::test_model_training_and_serialization      PASSED
tests/test_pipeline.py::test_evaluation_metrics_and_error_exports  PASSED
tests/test_pipeline.py::test_predict_pipeline_single_and_batch     PASSED
tests/test_pipeline.py::test_pipeline_reproducibility              PASSED

============================== 6 passed in 36.94s ==============================
```

**Validation checks:**
- ✅ All 10 pipeline stages complete without error
- ✅ Quality gate recall threshold (0.70) met: 0.7219
- ✅ Model Version 3 in Production stage
- ✅ Deployment bundle saved to `deployment/`
- ✅ Inference predictions saved to `outputs/inference_predictions.csv`
- ✅ Pipeline log saved to `logs/churn_prediction_pipeline_run_log.json`
- ✅ 6/6 pytest tests pass

---

## 10. Result

| Pipeline Stage | Status | Key Output |
|---------------|--------|-----------|
| Data Loading | ✅ PASSED | 7,343 rows |
| Raw Validation | ✅ PASSED | Schema valid |
| Preprocessing | ✅ PASSED | (5875,30) train |
| Output Validation | ✅ PASSED | Zero NaNs |
| Training + Registry | ✅ PASSED | Version 3 registered |
| Evaluation | ✅ PASSED | ROC-AUC=0.8447 |
| Quality Gate | ✅ PASSED | Recall=0.7219 ≥ 0.70 |
| Lifecycle | ✅ PASSED | Production promoted |
| Deployment Prep | ✅ PASSED | Bundle saved |
| Inference | ✅ PASSED | 1,469 predictions |

---

## 11. Conclusion

Lab 7 delivered a fully integrated, modular, 10-stage MLOps pipeline that takes raw data from a CSV file and produces a production-ready model deployment bundle — entirely from one command. The quality gate prevents model regressions from reaching production. Stage modularity allows individual stages to be re-run or replaced without affecting others. The pipeline's log provides a complete audit trail of every execution. This pipeline forms the backbone of the CI automation in Lab 8 and the failure handling in Lab 9.
