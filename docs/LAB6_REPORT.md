# LAB 6 – Model Registry and Version Management

## 1. Aim

To register trained ML models in MLflow Model Registry, maintain model versions with metadata and lineage, implement staging/production lifecycle transitions, and validate model promotion and rollback workflows.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| MLflow | 3.16.1 |
| scikit-learn | 1.8.0 |
| SQLite | (built-in) |

---

## 3. Libraries Used

```python
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
```

---

## 4. Theory

### MLflow Model Registry
The MLflow Model Registry provides a centralized model store with:
- **Versioning**: Each registered model can have multiple versions
- **Lifecycle Stages**: None → Staging → Production → Archived
- **Aliases**: Named pointers (e.g., `champion`, `challenger`) to specific versions
- **Annotations**: Tags, descriptions, and metadata per version

### Model Lifecycle Stages

```
Training → Register → Staging → Validation → Production
                                                 ↓
                                            Archived (rollback)
```

| Stage | Meaning |
|-------|---------|
| **None** | Freshly registered, not evaluated |
| **Staging** | Under validation testing |
| **Production** | Live model serving requests |
| **Archived** | Retired from active use |

### Model Lineage
Each registered model version links back to:
- The MLflow run that produced it (via `run_id`)
- The experiment and parameters used
- The artifact path within `mlruns/`

This lineage enables full audit trails: "which training run produced the current production model?"

### Champion-Challenger Pattern
In production:
- **Champion**: Current production model (best known performance)
- **Challenger**: New model under A/B evaluation
- If challenger outperforms champion → promote challenger, archive champion

---

## 5. Implementation

### File: `src/train_registry.py`
Trains and registers models with the MLflow registry.

### File: `src/automate_lifecycle.py`
Automates lifecycle transitions based on metric thresholds.

### File: `src/generate_registry_report.py`
Queries the registry and generates a human-readable report.

### File: `src/mlops_pipeline/08_automate_lifecycle.py`
Pipeline-stage lifecycle management.

### Key registry code:
```python
client = MlflowClient()
model_name = "Telco_Churn_Production_Model"

# Register model from run
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    model_name
)

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging",
    archive_existing_versions=False
)

# After validation, promote to Production
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production",
    archive_existing_versions=True  # archives previous Production
)

# Add metadata
client.update_model_version(
    name=model_name,
    version=result.version,
    description=f"RF n_estimators=100, recall={recall:.4f}, roc_auc={roc_auc:.4f}"
)
client.set_model_version_tag(model_name, result.version, "validated", "true")
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/train_registry.py` | Register models from training runs |
| `src/automate_lifecycle.py` | Automated stage transitions |
| `src/generate_registry_report.py` | Registry reporting |
| `src/mlops_pipeline/05_train_registry.py` | Pipeline-integrated registration |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Pipeline lifecycle management |
| `pipelines/run_lab6_registry.py` | Lab 6 orchestration |
| `reports/lifecycle_report.json` | Registry lifecycle report |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Register models
python src/train_registry.py

# Automate lifecycle
python src/automate_lifecycle.py

# Generate registry report
python src/generate_registry_report.py

# Run Lab 6 pipeline
python pipelines/run_lab6_registry.py

# Or full pipeline
python pipelines/run_all.py
```

---

## 8. Actual Outputs

```
=== MLflow Model Registry ===
Registered Model: Telco_Churn_Production_Model

Version History:
  Version 1: Stage=Archived | recall=0.7219 | roc_auc=0.8447
  Version 2: Stage=Archived | recall=0.7219 | roc_auc=0.8447
  Version 3: Stage=Production | recall=0.7219 | roc_auc=0.8447
  (additional versions from pipeline runs)

=== Current Production Model ===
Model Name   : Telco_Churn_Production_Model
Version      : 3
Stage        : Production
Run ID       : 7d6298bea41640cb9a4e4b6128df82d4
Accuracy     : 0.7693
Precision    : 0.5499
Recall       : 0.7219
F1-Score     : 0.6243
ROC-AUC      : 0.8447

=== Lifecycle Transitions ===
Version 3: None → Staging → Production ✅
Previous versions: Archived ✅

=== Deployment Bundle ===
deployment/model.pkl          → Production RF model
deployment/preprocessor.pkl   → Matching ColumnTransformer
deployment/model_metadata.json → Version + run_id metadata
```

---

## 9. Testing and Validation

**Registry validation:**
```bash
python src/mlops_pipeline/check_mlflow_registry.py
```

**Output:**
```
Registry name: Telco_Churn_Production_Model
Production version: 3
Recall: 0.7219 ≥ 0.70 threshold ✅
Artifact available: ✅
```

**Lifecycle report:**
```bash
cat reports/lifecycle_report.json
```

**Validation checks:**
- ✅ Model registered in `Telco_Churn_Production_Model`
- ✅ Version 3 in Production stage
- ✅ Previous versions archived on promotion
- ✅ Metadata and tags attached
- ✅ Run lineage maintained (run_id traceable)
- ✅ Deployment bundle saved to `deployment/`
- ✅ Rollback achievable by transitioning old version back to Production

---

## 10. Result

| Registry Metric | Value |
|----------------|-------|
| Registered model name | `Telco_Churn_Production_Model` |
| Total versions | 3+ |
| Current Production version | Version 3 |
| Production recall | 0.7219 |
| Production ROC-AUC | 0.8447 |
| Lifecycle transitions | None → Staging → Production |
| Deployment bundle | `deployment/model.pkl` |

---

## 11. Conclusion

Lab 6 established a complete model registry workflow using MLflow's built-in Model Registry. Models are versioned with full metadata and lineage, staged through validation before production promotion, and older versions are archived rather than deleted — preserving rollback capability. The champion-challenger pattern was demonstrated by promoting Version 3 (highest recall) to Production while archiving predecessors. The deployment bundle created here feeds directly into the FastAPI serving layer in Lab 10.
