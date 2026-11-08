import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

try:
    from src.config import (
        RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        PREPROCESSOR_PATH, DATASET_METADATA_PATH,
        RANDOM_STATE, TEST_SIZE
    )
    from src.utils import get_logger
    logger = get_logger("Preprocess_Pipeline_Lab5")
except ImportError:
    RAW_DATA_PATH = 'data/raw/churn.csv'
    PROCESSED_DATA_DIR = 'data/processed'
    MODELS_DIR = 'models'
    X_TRAIN_PATH = 'data/processed/X_train_final.npy'
    X_TEST_PATH = 'data/processed/X_test_final.npy'
    Y_TRAIN_PATH = 'data/processed/y_train.npy'
    Y_TEST_PATH = 'data/processed/y_test.npy'
    PREPROCESSOR_PATH = 'models/preprocessor.pkl'
    DATASET_METADATA_PATH = 'data/processed/dataset_metadata.json'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    import logging
    logger = logging.getLogger("Preprocess_Pipeline_Lab5")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def run_preprocessing():
    logger.info("Starting Sklearn Modular Pipeline Preprocessing (Lab 5)...")
    
    # 1. Load the raw data
    data_path = str(RAW_DATA_PATH)
    if not os.path.exists(data_path):
        logger.error(f"Raw data file not found at: {data_path}")
        raise FileNotFoundError(f"Raw data not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Basic Data Formatting
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Coerce TotalCharges to numeric (blanks become NaN, which the pipeline imputer handles)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Target string to binary integer conversion
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Train-Test Split (Done before transformations to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 5. Build Modular Scikit-Learn Pipelines
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    # 6. Execute Pipeline
    logger.info("Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    
    logger.info("Transforming test data with fitted preprocessor...")
    X_test_final = preprocessor.transform(X_test)
    
    # 7. Save Artifacts
    os.makedirs(str(PROCESSED_DATA_DIR), exist_ok=True)
    os.makedirs(str(MODELS_DIR), exist_ok=True)
    
    np.save(str(X_TRAIN_PATH), X_train_final)
    np.save(str(X_TEST_PATH), X_test_final)
    np.save(str(Y_TRAIN_PATH), y_train.to_numpy(dtype=np.int64))
    np.save(str(Y_TEST_PATH), y_test.to_numpy(dtype=np.int64))
    
    # Save the unified preprocessor artifact
    joblib.dump(preprocessor, str(PREPROCESSOR_PATH))
    
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
        
    logger.info(f"[SUCCESS] Preprocessing completed! Unified pipeline saved to '{PREPROCESSOR_PATH}'.")
    return preprocessor, X_train_final, X_test_final, y_train.to_numpy(), y_test.to_numpy()

if __name__ == "__main__":
    run_preprocessing()
