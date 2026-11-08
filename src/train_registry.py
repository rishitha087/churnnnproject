import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        PREPROCESSOR_PATH, MLFLOW_EXPERIMENT_NAME, REGISTERED_MODEL_NAME, RANDOM_STATE
    )
    from src.utils import get_logger
    logger = get_logger("Train_Registry")
except ImportError:
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    PREPROCESSOR_PATH = 'models/preprocessor.pkl'
    MLFLOW_EXPERIMENT_NAME = "Telco_Churn_Prediction"
    REGISTERED_MODEL_NAME = "Telco_Churn_Production_Model"
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Train_Registry")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def train_and_register_model(version_tag="V2", params=None):
    logger.info(f"--- Starting MLflow Run with Model Registry ({version_tag}) ---")
    
    # 1. Set up MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 2. Load Processed Data
    try:
        X_train = np.load(str(X_TRAIN_PATH))
        X_test = np.load(str(X_TEST_PATH))
        y_train = np.load(str(Y_TRAIN_PATH))
        y_test = np.load(str(Y_TEST_PATH))
    except FileNotFoundError:
        logger.error("[ERROR] Processed data not found. Please run preprocessing first.")
        return None
    
    # 3. Define Model Parameters
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced"
        }
    
    run_name = f"RandomForest_Registry_{version_tag}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("version_tag", version_tag)
        
        # 4. Train Model
        logger.info(f"Training RandomForest model ({run_name})...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # 5. Evaluate
        logger.info("Evaluating model on test partition...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba))
        }
        
        mlflow.log_metrics(metrics)
        
        # 6. Attach Preprocessor Pipeline Artifact (Lineage)
        if os.path.exists(str(PREPROCESSOR_PATH)):
            logger.info("Attaching preprocessing pipeline to model artifacts...")
            mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessing_pipeline")
        
        # 7. Log and Register Model in MLflow Registry
        logger.info(f"Pushing model to MLflow Registry under name: '{REGISTERED_MODEL_NAME}'...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        
        logger.info(f"[SUCCESS] Metrics logged: Recall = {metrics['recall']:.4f} | F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        logger.info(f"[SUCCESS] Model successfully registered under name: '{REGISTERED_MODEL_NAME}'")
        return run.info.run_id, metrics

if __name__ == "__main__":
    train_and_register_model()
