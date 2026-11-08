import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

try:
    from src.config import (
        X_TEST_PATH, Y_TEST_PATH, BASELINE_MODEL_PATH, RAW_DATA_PATH,
        OUTPUTS_DIR, METRICS_DIR, FALSE_NEGATIVES_PATH, FALSE_POSITIVES_PATH,
        RANDOM_STATE, TEST_SIZE
    )
    from src.utils import get_logger
    logger = get_logger("Evaluate_Baseline")
except ImportError:
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    BASELINE_MODEL_PATH = 'models/random_forest_baseline.pkl'
    RAW_DATA_PATH = 'data/raw/churn.csv'
    OUTPUTS_DIR = 'outputs'
    METRICS_DIR = 'outputs/metrics'
    FALSE_NEGATIVES_PATH = 'outputs/false_negatives.csv'
    FALSE_POSITIVES_PATH = 'outputs/false_positives.csv'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    import logging
    logger = logging.getLogger("Evaluate_Baseline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_evaluation():
    logger.info("Starting Model Evaluation (Lab 2)...")
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
    os.makedirs(str(METRICS_DIR), exist_ok=True)
    
    # 1. Load test data and the trained model
    if not os.path.exists(str(X_TEST_PATH)) or not os.path.exists(str(Y_TEST_PATH)):
        logger.error("Processed test data not found!")
        raise FileNotFoundError("Processed test data not found.")
    if not os.path.exists(str(BASELINE_MODEL_PATH)):
        logger.error("Trained model not found at models/random_forest_baseline.pkl!")
        raise FileNotFoundError(f"Model not found at {BASELINE_MODEL_PATH}")
        
    X_test_final = np.load(str(X_TEST_PATH))
    y_test = np.load(str(Y_TEST_PATH))
    model = joblib.load(str(BASELINE_MODEL_PATH))
    
    # 2. Make predictions
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 3. Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
    
    logger.info("--- Model Evaluation Report ---")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1-Score : {f1:.4f}")
    logger.info(f"ROC-AUC  : {roc_auc:.4f}")
    logger.info("-------------------------------")
    
    # Save metrics JSON
    metrics = {
        "model": "RandomForest_Baseline",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    metrics_file = os.path.join(str(METRICS_DIR), "baseline_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics summary to '{metrics_file}'")
    
    # 4. Error Analysis Export
    errors_df = None
    processed_dir = os.path.dirname(str(X_TEST_PATH))
    raw_test_csv = os.path.join(processed_dir, "X_test_raw.csv")
    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")

    # First preference: check if raw test CSV was saved by preprocessing pipeline and matches y_test length
    if os.path.exists(raw_test_csv):
        try:
            cand_df = pd.read_csv(raw_test_csv)
            if len(cand_df) == len(y_test):
                errors_df = cand_df.copy()
        except Exception:
            pass

    # Second preference: check if raw data + test IDs match y_test length
    if errors_df is None and os.path.exists(str(RAW_DATA_PATH)):
        try:
            df = pd.read_csv(str(RAW_DATA_PATH))
            if os.path.exists(test_ids_path) and "customerID" in df.columns:
                test_ids = set(pd.read_csv(test_ids_path)["customerID"].astype(str))
                test_mask = df["customerID"].astype(str).isin(test_ids)
                if test_mask.sum() == len(y_test):
                    errors_df = df.loc[test_mask].drop(columns=['Churn'], errors='ignore').copy().reset_index(drop=True)

            if errors_df is None:
                _, X_test_raw = train_test_split(
                    df.drop('Churn', axis=1, errors='ignore'), 
                    test_size=TEST_SIZE, 
                    random_state=RANDOM_STATE, 
                    stratify=df['Churn'] if 'Churn' in df.columns else None
                )
                if len(X_test_raw) == len(y_test):
                    errors_df = X_test_raw.copy().reset_index(drop=True)
                else:
                    errors_df = X_test_raw.iloc[:len(y_test)].copy().reset_index(drop=True)
        except Exception:
            pass

    # Fallback: create DataFrame directly matching y_test length
    if errors_df is None or len(errors_df) != len(y_test):
        errors_df = pd.DataFrame(index=range(len(y_test)))

    errors_df = errors_df.reset_index(drop=True)
    errors_df['Actual_Churn'] = y_test
    errors_df['Predicted_Churn'] = y_pred
    if y_proba is not None:
        errors_df['Predicted_Probability'] = y_proba

    false_negatives = errors_df[(errors_df['Actual_Churn'] == 1) & (errors_df['Predicted_Churn'] == 0)]
    false_positives = errors_df[(errors_df['Actual_Churn'] == 0) & (errors_df['Predicted_Churn'] == 1)]

    false_negatives.to_csv(str(FALSE_NEGATIVES_PATH), index=False)
    false_positives.to_csv(str(FALSE_POSITIVES_PATH), index=False)
    logger.info(f"Exported error analysis: {len(false_negatives)} false negatives, {len(false_positives)} false positives.")
    logger.info(f"Saved: {FALSE_NEGATIVES_PATH} & {FALSE_POSITIVES_PATH}")

    return metrics

if __name__ == "__main__":
    run_evaluation()
