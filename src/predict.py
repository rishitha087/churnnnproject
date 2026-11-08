import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List

try:
    from src.config import (
        MODELS_DIR, PREPROCESSOR_PATH, SCALER_PATH, OHE_PATH,
        BASELINE_MODEL_PATH, TRACKED_MODEL_PATH, DATASET_METADATA_PATH
    )
    from src.utils import get_logger
    logger = get_logger("Predict_Pipeline")
except ImportError:
    MODELS_DIR = 'models'
    PREPROCESSOR_PATH = 'models/preprocessor.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    OHE_PATH = 'models/ohe.pkl'
    BASELINE_MODEL_PATH = 'models/random_forest_baseline.pkl'
    TRACKED_MODEL_PATH = 'models/random_forest_model.pkl'
    DATASET_METADATA_PATH = 'data/processed/dataset_metadata.json'
    import logging
    logger = logging.getLogger("Predict_Pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

class ChurnPredictor:
    """Production inference class for Telco Customer Churn."""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        # Locate Model
        if model_path is None:
            if os.path.exists(str(TRACKED_MODEL_PATH)):
                model_path = str(TRACKED_MODEL_PATH)
            elif os.path.exists(str(BASELINE_MODEL_PATH)):
                model_path = str(BASELINE_MODEL_PATH)
            else:
                raise FileNotFoundError("No trained model found in models/ directory.")
                
        self.model = joblib.load(model_path)
        logger.info(f"Loaded trained model from: {model_path}")
        
        # Locate Preprocessing artifacts
        self.use_column_transformer = False
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            self.use_column_transformer = True
            logger.info(f"Loaded ColumnTransformer preprocessor from: {preprocessor_path}")
        elif os.path.exists(str(PREPROCESSOR_PATH)):
            self.preprocessor = joblib.load(str(PREPROCESSOR_PATH))
            self.use_column_transformer = True
            logger.info(f"Loaded ColumnTransformer preprocessor from: {PREPROCESSOR_PATH}")
        elif os.path.exists(str(SCALER_PATH)) and os.path.exists(str(OHE_PATH)):
            self.scaler = joblib.load(str(SCALER_PATH))
            self.ohe = joblib.load(str(OHE_PATH))
            self.use_column_transformer = False
            logger.info("Loaded separate scaler and one-hot encoder from models/")
        else:
            raise FileNotFoundError("Preprocessing artifacts not found in models/ directory.")

    def _prepare_dataframe(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")
            
        # Drop ID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        # Drop Churn if present (e.g. testing data)
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
            
        # Clean numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if self.use_column_transformer:
            return self.preprocessor.transform(df)
        else:
            cat_cols = self.ohe.feature_names_in_
            num_cols = self.scaler.feature_names_in_
            
            scaled_num = self.scaler.transform(df[num_cols])
            encoded_cat = self.ohe.transform(df[cat_cols])
            return np.hstack((scaled_num, encoded_cat))

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        df = self._prepare_dataframe(data)
        features = self.transform_features(df)
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1] if hasattr(self.model, "predict_proba") else [None] * len(predictions)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "record_index": i,
                "churn_prediction": int(pred),
                "churn_label": "Churn" if pred == 1 else "No Churn",
                "churn_probability": round(float(prob), 4) if prob is not None else None
            })
            
        return results

def get_sample_customer() -> Dict[str, Any]:
    """Returns a representative high-risk customer profile for inference demonstration."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
    }

if __name__ == "__main__":
    sample = get_sample_customer()
    predictor = ChurnPredictor()
    pred_res = predictor.predict(sample)
    print("\n" + "=" * 50)
    print("CUSTOMER CHURN INFERENCE DEMO")
    print("=" * 50)
    print(f"Customer Input Profile:\n  Tenure: {sample['tenure']} month(s)\n  Contract: {sample['Contract']}\n  MonthlyCharges: ${sample['MonthlyCharges']}")
    print(f"Prediction Result: {pred_res[0]['churn_label']} (Class: {pred_res[0]['churn_prediction']})")
    print(f"Churn Risk Probability: {pred_res[0]['churn_probability'] * 100:.2f}%")
    print("=" * 50 + "\n")
