# LAB 4 – Experiment Tracking and Reproducibility Using MLflow

## 1. Aim

To instrument ML training runs with MLflow tracking, compare multiple experiment configurations, validate reproducibility across runs, and explore the MLflow UI for experiment comparison and artifact management.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| MLflow | 3.16.1 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| SQLite | (built-in) |

---

## 3. Libraries Used

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
```

---

## 4. Theory

### MLflow Components

| Component | Purpose |
|-----------|---------|
| **MLflow Tracking** | Log parameters, metrics, and artifacts per run |
| **MLflow Projects** | Package code as reproducible units |
| **MLflow Models** | Standard model packaging format |
| **MLflow Registry** | Centralized model lifecycle management |

### Experiment vs Run
- An **Experiment** is a named collection of runs (e.g., `Telco_Churn_Prediction`)
- A **Run** is one training execution with its own parameters, metrics, and artifacts
- Multiple runs in one experiment allow hyperparameter comparison

### Reproducibility
MLflow achieves reproducibility by logging:
1. All hyperparameters (`log_param`)
2. All evaluation metrics (`log_metric`)
3. The serialized model (`log_model`)
4. The git commit hash (`mlflow.set_tag`)
5. The random seed used

Re-running with the same parameters and seed produces identical metrics.

### Serialization Format
MLflow 3.16.1 uses `skops` serialization by default. This project uses `serialization_format="cloudpickle"` for broader compatibility:
```python
mlflow.sklearn.log_model(
    model, "model",
    serialization_format="cloudpickle"
)
```

---

## 5. Implementation

### File: `src/train_mlflow.py`
Runs all three baseline models inside MLflow tracking context, logging parameters and metrics.

### File: `src/mlops_pipeline/05_train_registry.py`
Pipeline-integrated training with MLflow tracking + model registration.

### File: `src/validate_reproducibility.py`
Runs training twice with identical seed and verifies metric equality.

### Key MLflow tracking code:
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telco_Churn_Prediction")

with mlflow.start_run(run_name="RF_Baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)

    model.fit(X_train, y_train)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall",   recall)
    mlflow.log_metric("f1",       f1)
    mlflow.log_metric("roc_auc",  roc_auc)

    mlflow.sklearn.log_model(
        model, "model",
        serialization_format="cloudpickle"
    )
```

### Experiment Runs Logged:

| Run Name | n_estimators | max_depth | Accuracy | Recall | F1 | ROC-AUC |
|----------|-------------|-----------|----------|--------|----|---------|
| RF_Baseline | 100 | 10 | 0.7693 | 0.7219 | 0.6243 | 0.8447 |
| RF_Shallow | 50 | 5 | 0.7612 | 0.7098 | 0.6187 | 0.8321 |
| LR_Baseline | — | — | 0.7368 | 0.7598 | 0.6050 | 0.8347 |
| DT_Baseline | — | 5 | 0.7261 | 0.6948 | 0.5696 | 0.7260 |

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/train_mlflow.py` | MLflow-instrumented training |
| `src/mlops_pipeline/05_train_registry.py` | Pipeline-stage training + registry |
| `src/validate_reproducibility.py` | Reproducibility validation |
| `pipelines/run_lab4_tracking.py` | Lab 4 orchestration |
| `mlflow.db` | SQLite tracking backend |
| `mlruns/` | Artifact storage directory |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Run MLflow-tracked training
python src/train_mlflow.py

# Run Lab 4 pipeline stage
python pipelines/run_lab4_tracking.py

# Validate reproducibility
python src/validate_reproducibility.py

# Launch MLflow UI (requires browser)
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# Then open: http://127.0.0.1:5000
```

---

## 8. Actual Outputs

```
=== MLflow Experiment: Telco_Churn_Prediction ===
Tracking URI: sqlite:///mlflow.db
Artifact root: mlruns/

Logged 4 runs:
  Run: RF_Baseline    | recall=0.7219 | roc_auc=0.8447
  Run: RF_Shallow     | recall=0.7098 | roc_auc=0.8321
  Run: LR_Baseline    | recall=0.7598 | roc_auc=0.8347
  Run: DT_Baseline    | recall=0.6948 | roc_auc=0.7260

=== Reproducibility Check ===
Run 1 metrics: {'accuracy': 0.7693, 'recall': 0.7219, 'f1': 0.6243, 'roc_auc': 0.8447}
Run 2 metrics: {'accuracy': 0.7693, 'recall': 0.7219, 'f1': 0.6243, 'roc_auc': 0.8447}
Reproducibility: 100% ✅

=== Best Run ===
Run ID: 7d6298bea41640cb9a4e4b6128df82d4
Model Version 3 registered as: Telco_Churn_Production_Model
```

---

## 9. Testing and Validation

**Reproducibility test:**
```bash
python src/validate_reproducibility.py
```

**Validation checks:**
- ✅ Experiment `Telco_Churn_Prediction` created in SQLite DB
- ✅ 4 runs tracked with parameters and metrics
- ✅ Model artifacts saved to `mlruns/<experiment_id>/<run_id>/artifacts/`
- ✅ Reproducibility 100% (identical metrics across two runs with same seed)
- ✅ MLflow UI accessible at http://127.0.0.1:5000

---

## 10. Result

| Tracking Metric | Value |
|----------------|-------|
| Experiment name | `Telco_Churn_Prediction` |
| Total runs logged | 4 |
| Best model | Random Forest (RF_Baseline) |
| Best ROC-AUC | 0.8447 |
| Best Recall | 0.7219 |
| Reproducibility | 100% |
| Tracking backend | SQLite (`mlflow.db`) |
| Serialization format | cloudpickle |

---

## 11. Conclusion

Lab 4 demonstrated full MLflow experiment tracking for the Churn prediction project. All four model configurations were logged with their hyperparameters and metrics in a persistent SQLite database. Reproducibility was verified at 100% — re-running with the same seed produces byte-identical metrics. The MLflow UI provides a visual interface for experiment comparison and artifact inspection, essential for collaborative ML workflows. The best run (RF_Baseline) was automatically promoted to the Model Registry in Lab 6.
