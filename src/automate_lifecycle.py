import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.config import REGISTERED_MODEL_NAME
    from src.utils import get_logger
    logger = get_logger("Automate_Lifecycle")
except ImportError:
    REGISTERED_MODEL_NAME = "Telco_Churn_Production_Model"
    import logging
    logger = logging.getLogger("Automate_Lifecycle")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def automate_champion_challenger(model_name: str = None, metric_to_optimize: str = "recall"):
    """Evaluates Staging challenger models against current Production champion based on target metric and transitions stages."""
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME
        
    logger.info(f"Starting Automated Model Lifecycle Manager for '{model_name}'...")
    logger.info(f"Optimization Metric: '{metric_to_optimize}'")
    
    client = MlflowClient()
    
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        # 1. Discover untracked models (Stage: None) and push them to Staging
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]
        if new_models:
            logger.info(f"Found {len(new_models)} new model(s). Moving them to Staging...")
            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                logger.info(f"  -> Version {mv.version} is now in Staging.")
                
        # Refresh registry data
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
        
        if not staging_models:
            logger.info("No models in Staging to evaluate. Exiting.")
            return True
            
        # 2. Evaluate Staging Models (Find the Best Challenger)
        logger.info("Evaluating Staging candidates...")
        best_challenger = None
        best_challenger_score = -1.0
        
        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            logger.info(f"  -> Staging Candidate: Version {mv.version} | {metric_to_optimize}: {score:.4f}")
            
            if score > best_challenger_score:
                best_challenger_score = score
                best_challenger = mv
                
        logger.info(f"Best Challenger is Version {best_challenger.version} ({metric_to_optimize}: {best_challenger_score:.4f})")
        
        # 3. Find the Current Champion in Production
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None
        promote_challenger = False
        
        if not current_champion:
            logger.info("No model currently in Production. Challenger wins by default!")
            promote_challenger = True
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = champ_run.data.metrics.get(metric_to_optimize, 0.0)
            logger.info(f"Current Production Champion: Version {current_champion.version} | {metric_to_optimize}: {champ_score:.4f}")
            
            # 4. The Face-Off
            if best_challenger_score >= champ_score:
                logger.info(f"[SUCCESS] Challenger (Version {best_challenger.version}) beat or matched Champion (Version {current_champion.version})!")
                promote_challenger = True
            else:
                logger.info("[INFO] Challenger failed to beat Champion. Champion retains title.")
                
        # 5. Execute Lifecycle Promotions & Rollbacks
        if promote_challenger:
            logger.info(f"Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            if current_champion and current_champion.version != best_challenger.version:
                logger.info(f"Archiving defeated Champion (Version {current_champion.version})...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )
                
        # 6. Clean up Staging (Archive all remaining losers)
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]
        
        if remaining_staging:
            logger.info("Cleaning up remaining candidates in Staging...")
            for mv in remaining_staging:
                logger.info(f"  -> Archiving Version {mv.version}")
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
                )
                
        logger.info("[SUCCESS] Automated Model Lifecycle execution complete!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to automate lifecycle: {e}")
        return False

if __name__ == "__main__":
    automate_champion_challenger()
