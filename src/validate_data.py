import os
import pandas as pd
try:
    import pandera as pa
    from pandera import Column, Check
except ImportError:
    pa = None

try:
    from src.config import RAW_DATA_PATH, ARTIFACTS_DIR
    from src.utils import get_logger
    logger = get_logger("Validate_Data")
except ImportError:
    RAW_DATA_PATH = 'data/raw/churn.csv'
    ARTIFACTS_DIR = 'artifacts'
    import logging
    logger = logging.getLogger("Validate_Data")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def get_telco_schema():
    """Defines the strict schema specification for the Telco Churn dataset."""
    if pa is None:
        raise ImportError("Pandera is required for schema validation.")
    return pa.DataFrameSchema({
        "customerID": Column(pa.String),
        "gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(pa.Int, Check.isin([0, 1])),
        "Partner": Column(pa.String, Check.isin(["Yes", "No"])),
        "Dependents": Column(pa.String, Check.isin(["Yes", "No"])),
        "tenure": Column(pa.Int, Check.ge(0)),
        "PhoneService": Column(pa.String, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(pa.String, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(pa.String, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "OnlineBackup": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "DeviceProtection": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "TechSupport": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(pa.String, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(pa.String, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(pa.String, Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])),
        "MonthlyCharges": Column(pa.Float, Check.ge(0.0)),
        "TotalCharges": Column(pa.String, nullable=True),
        "Churn": Column(pa.String, Check.isin(["Yes", "No"]))
    }, strict=True)

def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    logger.info(f"Validating schema (Records: {len(df)})...")
    schema = get_telco_schema()
    
    try:
        schema.validate(df, lazy=True)
        logger.info("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    
    except pa.errors.SchemaErrors as err:
        logger.error("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        logger.error(f"\n{failures.to_string()}")
        
        os.makedirs(str(ARTIFACTS_DIR), exist_ok=True)
        report_path = os.path.join(str(ARTIFACTS_DIR), output_report_name)
        failures.to_csv(report_path, index=False)
        logger.info(f"Detailed failure report saved to '{report_path}'.")
        return False

if __name__ == "__main__":
    data_path = str(RAW_DATA_PATH)
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        logger.error(f"Target file not found at: {data_path}")
