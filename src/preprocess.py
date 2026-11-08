import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from src.config import (
        RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        SCALER_PATH, OHE_PATH, DATASET_METADATA_PATH,
        RANDOM_STATE, TEST_SIZE
    )
    from src.utils import get_logger
    logger = get_logger("Preprocess_Lab1")
except ImportError:
    RAW_DATA_PATH = 'data/raw/churn.csv'
    PROCESSED_DATA_DIR = 'data/processed'
    MODELS_DIR = 'models'
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    SCALER_PATH = 'models/scaler.pkl'
    OHE_PATH = 'models/ohe.pkl'
    DATASET_METADATA_PATH = 'data/processed/dataset_metadata.json'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    import logging
    logger = logging.getLogger("Preprocess_Lab1")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_preprocessing():
    logger.info("Starting Preprocessing Pipeline (Lab 1)...")
    
    # Ensure directories exist
    os.makedirs(str(PROCESSED_DATA_DIR), exist_ok=True)
    os.makedirs(str(MODELS_DIR), exist_ok=True)
    
    # 1. Load the raw data
    if not os.path.exists(str(RAW_DATA_PATH)):
        logger.error(f"Raw data file not found at: {RAW_DATA_PATH}")
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")
        
    df = pd.read_csv(str(RAW_DATA_PATH))
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    
    # 2. Data Cleaning
    # Handle blank strings in TotalCharges by coercing to NaN and filling with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Unconditional string to binary integer conversion
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Train-Test Split (done before scaling/encoding to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples)")
    
    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 5. Scale & Encode (Fit strictly on train set)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    
    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    
    # 6. Save Artifacts with explicit integer type casting
    np.save(str(X_TRAIN_PATH), X_train_final)
    np.save(str(X_TEST_PATH), X_test_final)
    np.save(str(Y_TRAIN_PATH), y_train.to_numpy(dtype=np.int64))
    np.save(str(Y_TEST_PATH), y_test.to_numpy(dtype=np.int64))
    
    joblib.dump(scaler, str(SCALER_PATH))
    joblib.dump(ohe, str(OHE_PATH))
    
    # Save Metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }
    with open(str(DATASET_METADATA_PATH), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Preprocessing completed successfully! Final train shape: {X_train_final.shape}, test shape: {X_test_final.shape}")
    return X_train_final, X_test_final, y_train.to_numpy(), y_test.to_numpy()

if __name__ == "__main__":
    run_preprocessing()
