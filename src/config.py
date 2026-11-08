import os
from pathlib import Path

# Base Paths (Dynamically resolved so scripts work anywhere)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_final.npy"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_final.npy"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

# Models Paths
MODELS_DIR = BASE_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
OHE_PATH = MODELS_DIR / "ohe.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "random_forest_baseline.pkl"
TRACKED_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

# Outputs Paths
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FALSE_NEGATIVES_PATH = OUTPUTS_DIR / "false_negatives.csv"
FALSE_POSITIVES_PATH = OUTPUTS_DIR / "false_positives.csv"

# Artifacts Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.png"

# Logs
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"

# Experiment Tracking
MLFLOW_EXPERIMENT_NAME = "Telco_Churn_Prediction"
REGISTERED_MODEL_NAME = "Telco_Churn_Production_Model"

# Model & Split Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Ensure all essential directories exist
for path in [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
]:
    os.makedirs(path, exist_ok=True)
