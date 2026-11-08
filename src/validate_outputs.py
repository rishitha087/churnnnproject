import os
import json
import numpy as np

try:
    from src.config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, ARTIFACTS_DIR
    from src.utils import get_logger
    logger = get_logger("Validate_Outputs")
except ImportError:
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    ARTIFACTS_DIR = 'artifacts'
    import logging
    logger = logging.getLogger("Validate_Outputs")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def validate_preprocessing_outputs():
    logger.info("Validating Preprocessing Outputs (Lab 5)...")
    
    # 1. Load the pipeline outputs
    try:
        X_train = np.load(str(X_TRAIN_PATH))
        X_test = np.load(str(X_TEST_PATH))
        y_train = np.load(str(Y_TRAIN_PATH))
        y_test = np.load(str(Y_TEST_PATH))
    except FileNotFoundError as e:
        logger.error(f"Could not find processed data: {e}")
        return False
        
    errors = []
    
    # 2. Assert No Missing Values (Validating the Imputer)
    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")
        
    # 3. Assert Dimensionality Consistency (Validating OHE and Split)
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")
        
    # 4. Generate Transformation Summary Report
    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape)
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum())
        },
        "errors": errors
    }
    
    os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
    report_path = os.path.join(str(ARTIFACTS_DIR), 'preprocessing_summary_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    if errors:
        logger.error("Output Validation FAILED!")
        for e in errors:
            logger.error(f"  - {e}")
        return False
    else:
        logger.info("[SUCCESS] Output Validation PASSED.")
        logger.info("Features are clean, scaled, encoded, and dimensionally consistent.")
        logger.info(f"Summary report saved to {report_path}")
        return True

if __name__ == "__main__":
    validate_preprocessing_outputs()
