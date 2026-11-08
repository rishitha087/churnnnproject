import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from src.config import X_TRAIN_PATH, Y_TRAIN_PATH, BASELINE_MODEL_PATH, MODELS_DIR, RANDOM_STATE
    from src.utils import get_logger
    logger = get_logger("Train_Baseline")
except ImportError:
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    BASELINE_MODEL_PATH = 'models/random_forest_baseline.pkl'
    MODELS_DIR = 'models'
    RANDOM_STATE = 42
    import logging
    logger = logging.getLogger("Train_Baseline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_training():
    logger.info("Starting Baseline Model Training (Lab 2)...")
    os.makedirs(str(MODELS_DIR), exist_ok=True)
    
    # 1. Load processed training data
    if not os.path.exists(str(X_TRAIN_PATH)) or not os.path.exists(str(Y_TRAIN_PATH)):
        logger.error("Processed training data not found! Please run preprocessing first.")
        raise FileNotFoundError("Processed training data not found.")
        
    X_train_final = np.load(str(X_TRAIN_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    logger.info(f"Loaded training data: X_train shape={X_train_final.shape}, y_train shape={y_train.shape}")
    
    # 2. Initialize and train the baseline Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=RANDOM_STATE, 
        class_weight='balanced'
    )
    
    logger.info("Fitting RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')...")
    model.fit(X_train_final, y_train)
    
    # 3. Save the trained model
    joblib.dump(model, str(BASELINE_MODEL_PATH))
    logger.info(f"Baseline model training complete and saved to '{BASELINE_MODEL_PATH}'!")
    return model

if __name__ == "__main__":
    run_training()
