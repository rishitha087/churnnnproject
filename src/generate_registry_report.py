import os
import json
import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.config import REGISTERED_MODEL_NAME, ARTIFACTS_DIR
    from src.utils import get_logger
    logger = get_logger("Registry_Report")
except ImportError:
    REGISTERED_MODEL_NAME = "Telco_Churn_Production_Model"
    ARTIFACTS_DIR = "artifacts"
    import logging
    logger = logging.getLogger("Registry_Report")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def generate_registry_report(model_name: str = None):
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME
        
    logger.info(f"Generating Model Registry Report for '{model_name}'...")
    client = MlflowClient()
    
    try:
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)
        
        if not champion:
            logger.warning("[WARNING] No model currently in Production stage! Taking latest version for report.")
            champion = production_models[-1] if production_models else None
            
        if not champion:
            logger.error("[ERROR] No models found in registry!")
            return False
            
        run = client.get_run(champion.run_id)
        
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision")
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl"
        }
        
        os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
        report_path = os.path.join(str(ARTIFACTS_DIR), "production_model_report.json")
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        logger.info(f"[INFO] Saved to: {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate report: {e}")
        return False

if __name__ == "__main__":
    generate_registry_report()
