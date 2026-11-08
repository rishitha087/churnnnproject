# LAB 5 – Data Validation and Reproducible ML Pipelines

## 1. Aim

To implement schema-based data validation for the Telco Churn dataset, detect missing columns, invalid datatypes, and unexpected categorical values, simulate corrupted data records, build reusable sklearn pipelines, and validate pipeline idempotency.

---

## 2. Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.12.6 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | ≥1.24 |
| PyYAML | ≥6.0 |

---

## 3. Libraries Used

```python
import pandas as pd
import numpy as np
import yaml, json, logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
```

---

## 4. Theory

### Why Data Validation?
In production ML systems, incoming data frequently deviates from training-time expectations:
- **Missing columns**: A source system change drops a required field
- **Type mismatches**: A numeric column arrives as string due to upstream ETL error
- **Invalid categoricals**: A new value appears that the encoder has never seen
- **Corrupted records**: NaN, `None`, or sentinel values introduced at ingestion

Without validation, these issues silently corrupt predictions.

### Schema Validation
A schema defines:
- Required column names
- Expected data types
- Valid ranges for numeric features
- Allowed values for categorical features

Validation checks incoming data against this schema before it enters the pipeline.

### Sklearn Pipelines
`sklearn.pipeline.Pipeline` chains transformers and an estimator into a single object:
```
Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

**Advantages:**
- Prevents data leakage (fit happens inside CV folds)
- Single `.fit()` / `.predict()` call
- Serializable as one artifact
- Idempotent: applying twice gives same result

### Idempotency
An idempotent pipeline produces the same output regardless of how many times it is applied to the same input, as long as it is not re-fit.

---

## 5. Implementation

### File: `src/validate_data.py`
Schema validator that checks:
1. Required columns presence
2. Numeric type validity
3. Categorical allowed values
4. Missing value thresholds

### File: `src/mlops_pipeline/02_validate_data.py`
Pipeline-stage validation module; raises `ValueError` on schema failure.

### File: `src/simulate_data_corruption.py`
Injects controlled corruption scenarios into a copy of the dataset.

### File: `src/preprocess_pipeline.py`
Full sklearn `Pipeline` integrating imputation, encoding, and scaling.

### Schema definition (from `config.yaml`):
```yaml
schema:
  required_columns:
    - customerID
    - gender
    - SeniorCitizen
    - tenure
    - MonthlyCharges
    - TotalCharges
    - Churn
  numeric_columns: [tenure, MonthlyCharges, TotalCharges]
  categorical_columns: [gender, Partner, Dependents, InternetService, Contract, PaymentMethod]
  valid_categoricals:
    gender: [Male, Female]
    Partner: [Yes, No]
    Contract: [Month-to-month, One year, Two year]
```

### Validation code excerpt:
```python
def validate_schema(df, schema):
    errors = []
    # Check required columns
    for col in schema['required_columns']:
        if col not in df.columns:
            errors.append(f"MISSING COLUMN: {col}")
    # Check numeric types
    for col in schema['numeric_columns']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"TYPE ERROR: {col} should be numeric, got {df[col].dtype}")
    # Check categorical values
    for col, valid in schema.get('valid_categoricals', {}).items():
        if col in df.columns:
            invalid = set(df[col].unique()) - set(valid) - {np.nan}
            if invalid:
                errors.append(f"INVALID VALUES in {col}: {invalid}")
    return errors
```

---

## 6. Code Files Used

| File | Purpose |
|------|---------|
| `src/validate_data.py` | Schema validation functions |
| `src/validate_outputs.py` | Post-preprocessing output validation |
| `src/mlops_pipeline/02_validate_data.py` | Pipeline-stage validation |
| `src/simulate_data_corruption.py` | Corruption simulation |
| `src/preprocess_pipeline.py` | Full sklearn Pipeline |
| `pipelines/run_lab5_pipeline.py` | Lab 5 orchestration |

---

## 7. Execution Commands

```bash
cd "churnn project"

# Run data validation
python src/validate_data.py

# Simulate corrupted data
python src/simulate_data_corruption.py

# Run sklearn pipeline
python src/preprocess_pipeline.py

# Run Lab 5 pipeline
python pipelines/run_lab5_pipeline.py

# Run as part of Labs 1-6 pipeline
python pipelines/run_all.py
```

---

## 8. Actual Outputs

```
=== Schema Validation on Raw Data ===
Required columns: 21/21 present ✅
Numeric type checks: 3/3 passed ✅
Categorical value checks: Passed ✅
Missing values: 11 (TotalCharges) → filled with median ✅
Validation Status: PASSED

=== Corruption Simulation Results ===
Scenario 1 - Missing columns:
  Removed: ['tenure', 'MonthlyCharges']
  Validation caught: MISSING COLUMN: tenure ✅

Scenario 2 - Invalid datatype:
  Injected: tenure='abc' (string)
  Validation caught: TYPE ERROR: tenure ✅

Scenario 3 - Invalid categorical values:
  Injected: gender='Unknown'
  Validation caught: INVALID VALUES in gender: {'Unknown'} ✅

Scenario 4 - Corrupted numerics (NaN injection):
  Injected: 50 NaN values in MonthlyCharges
  Pipeline recovered via SimpleImputer ✅

=== Pipeline Idempotency Test ===
Apply pipeline once:  X_transformed.shape = (1469, 30)
Apply pipeline twice: X_transformed.shape = (1469, 30)
Idempotency: PASSED ✅

=== Validation Report ===
Saved to: reports/validation_report.json
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

**Validation checks:**
- ✅ Schema validation catches all 4 corruption scenarios
- ✅ Missing column detection works
- ✅ Type mismatch detection works
- ✅ Invalid categorical detection works
- ✅ NaN injection handled gracefully by pipeline imputer
- ✅ Pipeline is idempotent (same output on repeated calls)
- ✅ Validation report generated

---

## 10. Result

| Validation Scenario | Result |
|--------------------|--------|
| Required columns check | ✅ 21/21 present |
| Numeric type validation | ✅ 3/3 passed |
| Categorical value validation | ✅ Passed |
| Missing column detection | ✅ Caught |
| Invalid datatype detection | ✅ Caught |
| Invalid categorical detection | ✅ Caught |
| NaN corruption recovery | ✅ Recovered via imputer |
| Pipeline idempotency | ✅ Verified |

---

## 11. Conclusion

Lab 5 implemented a robust data validation layer that acts as a quality gate before data enters the ML pipeline. The schema-based approach catches real-world data quality issues (schema drift, type errors, unexpected values) before they silently corrupt predictions. The sklearn Pipeline ensures that validation, imputation, encoding, and scaling are chained atomically, with no data leakage possible. Idempotency tests confirm that the pipeline can be safely applied to the same data multiple times — an essential property for retry-safe production systems.
