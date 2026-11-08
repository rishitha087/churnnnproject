import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH, BASELINE_MODEL_PATH, SCALER_PATH, OHE_PATH
)
from src.preprocess import run_preprocessing
from src.train import run_training
from src.evaluate import run_evaluation
from src.predict import ChurnPredictor, get_sample_customer
from src.validate_reproducibility import run_deterministic_test

def test_raw_dataset_exists_and_valid():
    """Verify raw dataset exists and contains expected columns and non-empty rows."""
    assert os.path.exists(str(RAW_DATA_PATH)), f"Raw data missing at {RAW_DATA_PATH}"
    df = pd.read_csv(str(RAW_DATA_PATH))
    assert df.shape[0] > 7000, f"Expected >7000 rows, found {df.shape[0]}"
    assert df.shape[1] == 21, f"Expected 21 columns, found {df.shape[1]}"
    assert "Churn" in df.columns
    assert "MonthlyCharges" in df.columns
    assert "TotalCharges" in df.columns

def test_preprocessing_produces_clean_arrays():
    """Verify preprocessing produces correctly shaped, NaN-free NumPy arrays."""
    X_train, X_test, y_train, y_test = run_preprocessing()
    
    assert os.path.exists(str(X_TRAIN_PATH))
    assert os.path.exists(str(X_TEST_PATH))
    assert os.path.exists(str(Y_TRAIN_PATH))
    assert os.path.exists(str(Y_TEST_PATH))
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Feature count must be consistent between train and test
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
    # Telco dataset produces 30 features after OHE (4 num + 26 encoded cat)
    assert X_train.shape[1] >= 20, f"Unexpectedly few features: {X_train.shape[1]}"
    assert not np.isnan(X_train).any(), "NaN values found in X_train"
    assert not np.isnan(X_test).any(), "NaN values found in X_test"

def test_model_training_and_serialization():
    """Verify model training produces a valid scikit-learn model artifact."""
    model = run_training()
    assert os.path.exists(str(BASELINE_MODEL_PATH))
    loaded_model = joblib.load(str(BASELINE_MODEL_PATH))
    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "predict_proba")

def test_evaluation_metrics_and_error_exports():
    """Verify model evaluation calculates realistic classification metrics and error CSVs."""
    metrics = run_evaluation()
    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] > 0.65
    assert metrics["recall"] > 0.65
    assert os.path.exists("outputs/false_negatives.csv")
    assert os.path.exists("outputs/false_positives.csv")

def test_predict_pipeline_single_and_batch():
    """Verify inference pipeline operates correctly on dictionary and DataFrame."""
    predictor = ChurnPredictor()
    sample = get_sample_customer()
    
    # Test single customer dict
    res = predictor.predict(sample)
    assert len(res) == 1
    assert res[0]["churn_prediction"] in [0, 1]
    assert 0.0 <= res[0]["churn_probability"] <= 1.0
    
    # Test DataFrame batch
    df_batch = pd.DataFrame([sample, sample])
    res_batch = predictor.predict(df_batch)
    assert len(res_batch) == 2

def test_pipeline_reproducibility():
    """Verify pipeline produces identical deterministic results with fixed random state."""
    is_reproducible = run_deterministic_test()
    assert is_reproducible is True
