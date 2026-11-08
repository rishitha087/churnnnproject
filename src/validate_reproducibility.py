import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from src.config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, ARTIFACTS_DIR, RANDOM_STATE
    from src.utils import get_logger
    logger = get_logger("Reproducibility_Validation")
except ImportError:
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    ARTIFACTS_DIR = 'artifacts'
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Reproducibility_Validation")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_deterministic_test():
    logger.info("Running Reproducibility Validation (Lab 4)...")
    
    # Load Data
    X_train = np.load(str(X_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    y_test = np.load(str(Y_TEST_PATH))
    
    # Fixed Parameters for strict reproducibility
    params = {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE, "class_weight": "balanced"}
    
    # Execution 1
    model_1 = RandomForestClassifier(**params)
    model_1.fit(X_train, y_train)
    preds_1 = model_1.predict(X_test)
    score_1 = float(f1_score(y_test, preds_1))
    
    # Execution 2 (Simulating a re-run)
    model_2 = RandomForestClassifier(**params)
    model_2.fit(X_train, y_train)
    preds_2 = model_2.predict(X_test)
    score_2 = float(f1_score(y_test, preds_2))
    
    # Validate consistency
    is_reproducible = (score_1 == score_2) and np.array_equal(preds_1, preds_2)
    
    # Generate Report
    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_f1": score_1,
        "execution_2_f1": score_2,
        "predictions_identical": bool(np.array_equal(preds_1, preds_2)),
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }
    
    os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
    report_file = os.path.join(str(ARTIFACTS_DIR), "reproducibility_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Execution 1 F1: {score_1:.6f}")
    logger.info(f"Execution 2 F1: {score_2:.6f}")
    if is_reproducible:
        logger.info(f"SUCCESS: Pipeline is 100% reproducible. Report saved to '{report_file}'.")
    else:
        logger.error("FAILED: Pipeline is non-deterministic.")
        
    return is_reproducible

if __name__ == "__main__":
    run_deterministic_test()
