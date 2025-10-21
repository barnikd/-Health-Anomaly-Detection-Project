"""
src/data_loader.py
Utility functions to load and validate the health insurance dataset.
"""

import os
import pandas as pd

REQUIRED_COLUMNS = [
    "id", "age", "gender", "bmi", "daily_steps", "sleep_hours",
    "water_intake_l", "calories_consumed", "smoker", "alcohol",
    "resting_hr", "systolic_bp", "diastolic_bp", "cholesterol",
    "family_history", "disease_risk"
]

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV into pandas DataFrame. Raises FileNotFoundError if not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df

def validate_columns(df: pd.DataFrame, required=REQUIRED_COLUMNS) -> bool:
    """
    Check that the dataframe contains the required columns.
    Returns True if valid, otherwise raises ValueError with missing columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

def save_processed(df: pd.DataFrame, out_path: str):
    """
    Save processed DataFrame to CSV (creates directory if needed).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    # quick smoke test if run as a script
    sample_path = "data/health_insurance.csv"
    try:
        df = load_csv(sample_path)
        validate_columns(df)
        print("Loaded and validated:", sample_path)
        print(df.head().to_string())
    except Exception as e:
        print("Error:", e)
