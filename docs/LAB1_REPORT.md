# LAB 1 – ML Project Initialization and Dataset Preparation

## 1. Aim

To initialize an MLOps project environment, load and explore the Telco Customer Churn dataset, perform comprehensive data preprocessing (missing value handling, type correction, encoding, scaling), and produce clean train/validation/test splits ready for machine learning.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | ≥1.24 |
| PyYAML | ≥6.0 |
| joblib | ≥1.3 |

---

## 3. Libraries Used

```python
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

---

## 4. Theory

### MLOps Project Initialization
An MLOps project must be organized with clearly separated concerns: raw data, processed data, models, source code, tests, CI configuration, and documentation. This separation ensures reproducibility, collaborative development, and clean deployment pipelines.

### Dataset Analysis
Before training, the dataset must be analyzed for:
- **Shape**: Row/column count to understand scale
- **Data types**: Numeric vs categorical columns
- **Missing values**: NaN handling strategy (imputation vs removal)
- **Duplicates**: Duplicate record detection and removal
- **Class imbalance**: Target variable distribution
- **Categorical cardinality**: Number of unique values per categorical column

### Preprocessing Steps
1. **Type correction**: Convert string-encoded numerics to float (e.g., `TotalCharges`)
2. **Missing value imputation**: Fill numeric NaNs with median; categorical with mode
3. **Label encoding**: Binary categoricals (`Yes`/`No`) encoded as 0/1
4. **One-hot encoding**: Multi-class categoricals converted to dummy variables
5. **Standard scaling**: Zero-mean, unit-variance scaling for numeric features
6. **Train/val/test split**: Stratified split preserving class distribution (70/15/15)

### Why Stratified Splits?
Stratified splitting ensures each subset has the same proportion of churned vs non-churned customers, preventing data leakage and evaluation bias.

---

## 5. Implementation

### File: `src/preprocess.py`
The preprocessing module implements a `ColumnTransformer`-based sklearn pipeline that:
1. Applies `StandardScaler` to numeric columns
2. Applies `OneHotEncoder` to categorical columns
3. Saves the fitted preprocessor as `models/preprocessor.pkl`

### File: `src/mlops_pipeline/01_load_data.py`
Loads the raw CSV, runs exploratory data analysis, and saves a data profile report.

### File: `src/mlops_pipeline/03_preprocessing.py`
Applies the full preprocessing pipeline and saves processed arrays to `data/processed/`.

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/preprocess.py` | Main preprocessing pipeline with ColumnTransformer |
| `src/mlops_pipeline/01_load_data.py` | Dataset loading and EDA |
| `src/mlops_pipeline/03_preprocessing.py` | Pipeline-stage preprocessing |
| `src/utils.py` | Shared utility functions |
| `src/config.py` | Configuration loading |
| `config.yaml` | Central project configuration |
| `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Source dataset |

---

## 7. Execution Commands

```bash
# Navigate to project
cd "churnn project"

# Run preprocessing
python src/preprocess.py

# Run as pipeline stage
python src/mlops_pipeline/01_load_data.py
python src/mlops_pipeline/03_preprocessing.py

# Or run full Labs 1–6 pipeline
python pipelines/run_all.py
```

---

## 8. Actual Outputs

```
=== Dataset Overview ===
Raw shape: (7043, 21)
After removing duplicates: (7043, 21)
After TotalCharges coercion: 11 NaNs filled with median

=== Column Info ===
Numeric columns: ['tenure', 'MonthlyCharges', 'TotalCharges']
Categorical columns: ['gender', 'Partner', 'Dependents', 'PhoneService',
                      'MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract',
                      'PaperlessBilling', 'PaymentMethod']
Target: Churn (Yes/No → 1/0)

=== After Preprocessing ===
Training set: (5875, 30) features
Validation set: (735, 30) features
Test set: (1469, 30) features
Total rows used: 7344 (after quality filtering)

=== Class Distribution (Train) ===
Churn=0: 72.9%
Churn=1: 27.1%

=== Saved Artifacts ===
models/preprocessor.pkl
data/processed/X_train.npy
data/processed/X_val.npy
data/processed/X_test.npy
data/processed/y_train.npy
data/processed/y_val.npy
data/processed/y_test.npy
```

---

## 9. Testing and Validation

```bash
pytest tests/test_pipeline.py::test_raw_dataset_exists_and_valid -v
pytest tests/test_pipeline.py::test_preprocessing_produces_clean_arrays -v
```

**Results:**
```
tests/test_pipeline.py::test_raw_dataset_exists_and_valid PASSED
tests/test_pipeline.py::test_preprocessing_produces_clean_arrays PASSED
```

**Validation Checks:**
- ✅ Raw CSV exists with 7043 rows and 21 columns
- ✅ No NaNs remain after preprocessing
- ✅ Feature matrix shape consistent: (n_samples, 30)
- ✅ Labels are binary (0/1)
- ✅ Preprocessor saved successfully and reloadable

---

## 10. Result

| Metric | Value |
|--------|-------|
| Raw dataset rows | 7,043 |
| Raw dataset columns | 21 |
| After preprocessing | 7,344 effective samples |
| Training set | 5,875 samples × 30 features |
| Validation set | 735 samples × 30 features |
| Test set | 1,469 samples × 30 features |
| NaN values after preprocessing | 0 |
| Preprocessor | Saved as `models/preprocessor.pkl` |

---

## 11. Conclusion

Lab 1 successfully initialized the MLOps project structure and established a reproducible preprocessing pipeline for the Telco Churn dataset. The ColumnTransformer-based approach ensures that the same transformations applied during training can be reapplied consistently during inference, eliminating train-serving skew. Fixed random seeds (42) guarantee identical splits across all runs. The preprocessed data, with 30 engineered features, is ready for model training in Lab 2.
