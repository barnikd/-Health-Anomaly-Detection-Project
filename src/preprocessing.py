"""
src/preprocessing.py
Preprocess and feature-engineer health insurance dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.data_loader import validate_columns

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove duplicates, handle missing values"""
    df = df.drop_duplicates()
    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    validate_columns(df)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns"""
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    return df, le_gender

def scale_features(df: pd.DataFrame, feature_cols: list):
    """Scale numeric features"""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

# === FEATURE ENGINEERING ===
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for ML"""
    # BMI Category
    def bmi_category(bmi):
        if bmi < 18.5: return 0  # underweight
        elif bmi < 25: return 1  # normal
        elif bmi < 30: return 2  # overweight
        return 3  # obese
    df['bmi_cat'] = df['bmi'].apply(bmi_category)

    # Activity Level
    def activity_level(steps):
        if steps < 4000: return 0  # low
        elif steps < 8000: return 1  # medium
        return 2  # high
    df['activity_level'] = df['daily_steps'].apply(activity_level)

    # Interaction Feature Example: bmi * cholesterol
    df['bmi_chol'] = df['bmi'] * df['cholesterol']

    return df

# === FULL PIPELINE ===
def preprocess(df: pd.DataFrame, scale_numeric=True):
    df = clean_df(df)
    df = add_features(df)
    df, le_gender = encode_features(df)
    feature_cols = [
        "age", "bmi", "daily_steps", "sleep_hours", "water_intake_l",
        "calories_consumed", "resting_hr", "systolic_bp", "diastolic_bp",
        "cholesterol", "bmi_cat", "activity_level", "bmi_chol",
        "smoker", "alcohol", "family_history"
    ]
    if scale_numeric:
        df, scaler = scale_features(df, feature_cols)
    else:
        scaler = None
    return df, le_gender, scaler, feature_cols

# Test run
if __name__ == "__main__":
    from src.data_loader import load_csv
    df = load_csv("data/health_insurance.csv")
    df_processed, le_gender, scaler, features = preprocess(df)
    print("Preprocessing done. Sample data:")
    print(df_processed.head())
