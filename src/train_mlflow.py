import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

try:
    from src.config import (
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        DATASET_METADATA_PATH, TRACKED_MODEL_PATH, ARTIFACTS_DIR,
        PLOTS_DIR, MLFLOW_EXPERIMENT_NAME, RANDOM_STATE
    )
    from src.utils import get_logger
    logger = get_logger("Train_MLflow")
except ImportError:
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    DATASET_METADATA_PATH = 'data/processed/dataset_metadata.json'
    TRACKED_MODEL_PATH = 'models/random_forest_model.pkl'
    ARTIFACTS_DIR = 'artifacts'
    PLOTS_DIR = 'artifacts/plots'
    MLFLOW_EXPERIMENT_NAME = "Telco_Churn_Prediction"
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Train_MLflow")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def train_and_track_model(model, run_name: str, model_family: str, params: dict):
    """Trains a given model, logs all metrics, parameters, and diagnostic plots to MLflow."""
    logger.info(f"\n--- Starting MLflow Run: {run_name} ({model_family}) ---")
    
    # 1. Load Data
    X_train = np.load(str(X_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    y_test = np.load(str(Y_TEST_PATH))

    # Set MLflow Experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_family)

        # 3. Train Model
        logger.info(f"Fitting {model_family}...")
        model.fit(X_train, y_train)

        # 4. Evaluate Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        }

        # 5. Log Metrics to MLflow
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics logged: Accuracy={metrics['accuracy']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

        # 6. Generate and Log Diagnostic Plots
        os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
        os.makedirs(str(PLOTS_DIR), exist_ok=True)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = os.path.join(str(PLOTS_DIR), f"confusion_matrix_{run_name}.png")
        fig_cm.savefig(cm_path, bbox_inches="tight")
        # Also save default for backward compatibility
        fig_cm.savefig(os.path.join(str(ARTIFACTS_DIR), "confusion_matrix.png"), bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {run_name}")
        roc_path = os.path.join(str(PLOTS_DIR), f"roc_curve_{run_name}.png")
        fig_roc.savefig(roc_path, bbox_inches="tight")
        # Also save default for backward compatibility
        fig_roc.savefig(os.path.join(str(ARTIFACTS_DIR), "roc_curve.png"), bbox_inches="tight")
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 7. Log Preprocessing Metadata for Lineage
        if os.path.exists(str(DATASET_METADATA_PATH)):
            mlflow.log_artifact(str(DATASET_METADATA_PATH), artifact_path="metadata")

        # 8. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model, 
            name="model", 
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        # Save local backup for baseline random forest
        if "RandomForest" in run_name:
            joblib.dump(model, str(TRACKED_MODEL_PATH))
            
        logger.info(f"Run '{run_name}' successfully tracked in MLflow!")
        return metrics

def run_all_experiments():
    """Runs standard suite of experiments for comparison in MLflow."""
    logger.info("Executing Full MLflow Experiment Suite (Lab 4)...")
    
    experiments = [
        {
            "name": "RandomForest_Baseline",
            "family": "RandomForest",
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE, "class_weight": "balanced"},
            "model_cls": RandomForestClassifier
        },
        {
            "name": "RandomForest_Shallow",
            "family": "RandomForest",
            "params": {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE, "class_weight": "balanced"},
            "model_cls": RandomForestClassifier
        },
        {
            "name": "LogisticRegression_Baseline",
            "family": "LogisticRegression",
            "params": {"C": 1.0, "max_iter": 1000, "random_state": RANDOM_STATE, "class_weight": "balanced"},
            "model_cls": LogisticRegression
        },
        {
            "name": "DecisionTree_Baseline",
            "family": "DecisionTree",
            "params": {"max_depth": 5, "random_state": RANDOM_STATE, "class_weight": "balanced"},
            "model_cls": DecisionTreeClassifier
        }
    ]
    
    results = {}
    for exp in experiments:
        model = exp["model_cls"](**exp["params"])
        metrics = train_and_track_model(model, exp["name"], exp["family"], exp["params"])
        results[exp["name"]] = metrics
        
    return results

if __name__ == "__main__":
    run_all_experiments()
