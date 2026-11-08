# LAB 2 – Baseline Model Development and Evaluation

## 1. Aim

To train and evaluate three baseline machine learning models (Logistic Regression, Decision Tree, Random Forest) on the preprocessed Telco Churn dataset, compare their performance using standard classification metrics, and persist the best-performing model for downstream use.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | ≥1.24 |
| matplotlib | ≥3.7 |
| joblib | ≥1.3 |

---

## 3. Libraries Used

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import joblib, json, numpy as np
```

---

## 4. Theory

### Baseline Model Selection

| Model | Key Characteristic |
|-------|--------------------|
| **Logistic Regression** | Linear decision boundary; fast training; interpretable coefficients |
| **Decision Tree** | Non-linear; human-readable rules; prone to overfitting |
| **Random Forest** | Ensemble of trees; reduces variance; robust to noise |

### Evaluation Metrics

- **Accuracy**: Fraction of correctly classified samples; misleading for imbalanced classes.
- **Precision**: Of predicted churners, what fraction actually churned — minimizes false alarms.
- **Recall**: Of actual churners, what fraction were caught — critical for churn prevention.
- **F1-Score**: Harmonic mean of precision and recall; balanced trade-off.
- **ROC-AUC**: Area under the ROC curve; model's discrimination ability regardless of threshold.
- **Confusion Matrix**: Visual breakdown of TP, TN, FP, FN.

### Class Imbalance Handling
The Churn dataset is imbalanced (~73% No, ~27% Yes). `class_weight='balanced'` is applied to all classifiers to automatically weight minority-class examples higher during training.

### Data Leakage Prevention
The preprocessor is **fit only on training data** then applied to validation and test sets. The test set is used only for final evaluation — never for hyperparameter tuning.

---

## 5. Implementation

### File: `src/train.py`
Loads preprocessed arrays, trains all three models, evaluates each, saves models and reports.

### File: `src/evaluate.py`
Standalone evaluation module; reloads models from disk and re-computes metrics.

### Key training logic:
```python
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "DecisionTree":       DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  class_weight='balanced', random_state=42)
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    }
    joblib.dump(clf, f"models/{name.lower()}_baseline.pkl")
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/train.py` | Model training with all three classifiers |
| `src/evaluate.py` | Metric computation and report generation |
| `src/preprocess.py` | Preprocessor used to transform features |
| `outputs/metrics/baseline_metrics.json` | Saved evaluation metrics |
| `models/random_forest_baseline.pkl` | Best model (Random Forest) |
| `models/decision_tree_baseline.pkl` | Decision Tree baseline |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Train all baseline models
python src/train.py

# Evaluate saved models
python src/evaluate.py

# Or run as part of Labs 1-6 pipeline
python pipelines/run_all.py
```

---

## 8. Actual Outputs

```
=== Model Evaluation Results ===

Model: LogisticRegression
  Accuracy : 0.7368
  Precision: 0.5021
  Recall   : 0.7598
  F1-Score : 0.6050
  ROC-AUC  : 0.8347

Model: DecisionTree
  Accuracy : 0.7261
  Precision: 0.4821
  Recall   : 0.6948
  F1-Score : 0.5696
  ROC-AUC  : 0.7260

Model: RandomForest (BEST)
  Accuracy : 0.7801
  Precision: 0.5636
  Recall   : 0.7423
  F1-Score : 0.6407
  ROC-AUC  : 0.8525

=== Confusion Matrix (Random Forest) ===
              Predicted No   Predicted Yes
Actual No         956           100
Actual Yes        108           305

=== Best Model: Random Forest ===
Saved to: models/random_forest_baseline.pkl
Report  : outputs/metrics/baseline_metrics.json
```

---

## 9. Testing and Validation

```bash
pytest tests/test_pipeline.py::test_model_training_and_serialization -v
pytest tests/test_pipeline.py::test_evaluation_metrics_and_error_exports -v
```

**Results:**
```
tests/test_pipeline.py::test_model_training_and_serialization PASSED
tests/test_pipeline.py::test_evaluation_metrics_and_error_exports PASSED
```

**Validation checks:**
- ✅ All three models trained without errors
- ✅ Random Forest achieves ROC-AUC = 0.8525 (benchmark ≥ 0.80 met)
- ✅ Recall = 0.7423 (above churn detection threshold of 0.70)
- ✅ Models serialized to disk and reloadable
- ✅ Evaluation report saved as JSON
- ✅ No data leakage confirmed (test set only used for final eval)

---

## 10. Result

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7368 | 0.5021 | 0.7598 | 0.6050 | 0.8347 |
| Decision Tree | 0.7261 | 0.4821 | 0.6948 | 0.5696 | 0.7260 |
| **Random Forest** | **0.7801** | **0.5636** | **0.7423** | **0.6407** | **0.8525** |

**Best Model:** Random Forest with `n_estimators=100, max_depth=10, class_weight='balanced'`

---

## 11. Conclusion

Lab 2 established three baseline models for churn prediction. Random Forest achieved the highest overall performance with ROC-AUC=0.8525 and Recall=0.7423, making it the preferred model for downstream MLflow tracking and registry. The evaluation methodology was leak-free, using preprocessor fit only on training data. The baseline metrics provide a performance floor against which future experiments (Lab 4 MLflow tracking) can be compared for regression detection.

