"""
Single-user anomaly detection and advisory system.

Compares user input against the population in the CSV dataset,
detects anomalies, and gives health risks & recommendations.

Usage:
    python -m src.predict
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Configuration constants for the anomaly detection system."""
    DATA_PATHS = [
        "data/health_insurance.csv",       # if inside data folder
        "health_insurance.csv",            # if in project root
        r"D:\vc cdode\insurance\health_insurance.csv"  # absolute path
    ]
    Z_SCORE_THRESHOLD = 2.0
    HIGH_RISK_THRESHOLD = 5
    MEDIUM_RISK_THRESHOLD = 2
    MIN_STD = 1e-9  # Prevent division by zero
    
    # Feature thresholds
    AGE_RISK_THRESHOLD = 45
    NORMAL_STEPS_MIN = 7000
    NORMAL_SLEEP_MIN = 7
    NORMAL_SLEEP_MAX = 9


@dataclass
class HealthAdvice:
    """Container for health issue and recommendation."""
    feature: str
    value: Any
    issue: str
    recommendation: str
    z_score: Optional[float] = None
    population_mean: Optional[float] = None
    population_std: Optional[float] = None


class HealthAdvisorySystem:
    """Rule-based health advisory system."""
    
    @staticmethod
    def get_advice(feature: str, value: Any, z_score: Optional[float] = None) -> Optional[Tuple[str, str]]:
        advisories = {
            "bmi": lambda v, z: ("Abnormal BMI", "Consult dietitian, adjust diet/exercise plan") 
                if z and abs(z) > Config.Z_SCORE_THRESHOLD else None,
            "daily_steps": lambda v, z: ("Insufficient physical activity", f"Aim for {Config.NORMAL_STEPS_MIN}-10,000 steps daily") 
                if z and z < -Config.Z_SCORE_THRESHOLD else None,
            "sleep_hours": lambda v, z: ("Sleep pattern concern", f"Maintain {Config.NORMAL_SLEEP_MIN}-{Config.NORMAL_SLEEP_MAX} hours of quality sleep") 
                if z and abs(z) > Config.Z_SCORE_THRESHOLD else None,
            "cholesterol": lambda v, z: ("Elevated cholesterol levels", "Improve diet, increase exercise, schedule lipid panel") 
                if z and z > Config.Z_SCORE_THRESHOLD else None,
            "systolic_bp": lambda v, z: ("High systolic blood pressure", "Reduce sodium intake, monitor BP regularly, consult physician") 
                if z and z > Config.Z_SCORE_THRESHOLD else None,
            "diastolic_bp": lambda v, z: ("High diastolic blood pressure", "Manage stress, monitor BP, consider lifestyle changes") 
                if z and z > Config.Z_SCORE_THRESHOLD else None,
            "resting_hr": lambda v, z: ("Abnormal resting heart rate", "Improve cardiovascular fitness, consult physician") 
                if z and abs(z) > Config.Z_SCORE_THRESHOLD else None,
            "smoker": lambda v, z: ("Smoking risk factor", "Seek smoking cessation program immediately") 
                if v == 1 else None,
            "alcohol": lambda v, z: ("Alcohol consumption risk", "Reduce alcohol intake, consider counseling") 
                if v == 1 else None,
            "family_history": lambda v, z: ("Genetic predisposition", "Schedule regular preventive screenings") 
                if v == 1 else None,
            "age": lambda v, z: ("Age-related health risks", "Increase screening frequency, maintain preventive care") 
                if v > Config.AGE_RISK_THRESHOLD else None,
        }
        if feature in advisories:
            return advisories[feature](value, z_score)
        return None


class AnomalyDetector:
    """Detects health anomalies by comparing user data to population statistics."""
    
    def __init__(self, data_paths: Optional[list] = None):
        self.data_paths = data_paths or Config.DATA_PATHS
        self.df = self._load_data(self.data_paths)
        self.numeric_features = self.df.select_dtypes(include=np.number).columns.tolist()
        
    def _load_data(self, paths: list) -> pd.DataFrame:
        """Load dataset from available paths."""
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                df = pd.read_csv(path)
                if df.empty:
                    continue
                logger.info(f"Loaded dataset from: {path} ({len(df)} records)")
                return df
        raise FileNotFoundError(f"Could not find any valid dataset in paths: {paths}")
    
    def _validate_user_data(self, user_data: Dict[str, Any]) -> None:
        if not user_data:
            raise ValueError("User data cannot be empty")
        validations = {
            "age": (0, 120),
            "bmi": (10, 60),
            "daily_steps": (0, 50000),
            "sleep_hours": (0, 24),
            "systolic_bp": (60, 250),
            "diastolic_bp": (40, 150),
            "cholesterol": (100, 400),
            "resting_hr": (30, 200),
        }
        for feature, (min_val, max_val) in validations.items():
            if feature in user_data:
                value = user_data[feature]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{feature} must be numeric")
                if not min_val <= value <= max_val:
                    raise ValueError(f"{feature} value {value} outside valid range [{min_val}, {max_val}]")
    
    def _calculate_z_score(self, feature: str, value: float) -> Optional[float]:
        if feature not in self.numeric_features:
            return None
        mean = self.df[feature].mean()
        std = self.df[feature].std()
        return (value - mean) / (std + Config.MIN_STD)
    
    def analyze_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_user_data(user_data)
        advisories = []
        for feature, value in user_data.items():
            z_score = self._calculate_z_score(feature, value) if feature in self.numeric_features else None
            advice_tuple = HealthAdvisorySystem.get_advice(feature, value, z_score)
            if advice_tuple:
                issue, recommendation = advice_tuple
                advisories.append(HealthAdvice(
                    feature=feature,
                    value=value,
                    issue=issue,
                    recommendation=recommendation,
                    z_score=z_score,
                    population_mean=float(self.df[feature].mean()) if feature in self.numeric_features else None,
                    population_std=float(self.df[feature].std()) if feature in self.numeric_features else None,
                ))
        risk_level = self._calculate_risk_level(len(advisories))
        return {
            "advisories": advisories,
            "overall_risk": risk_level,
            "total_issues": len(advisories),
            "analyzed_features": len(user_data),
        }
    
    def _calculate_risk_level(self, issue_count: int) -> str:
        if issue_count >= Config.HIGH_RISK_THRESHOLD:
            return "High"
        elif issue_count >= Config.MEDIUM_RISK_THRESHOLD:
            return "Medium"
        return "Low"


def format_report(results: Dict[str, Any]) -> str:
    """Convert results dictionary to readable text report."""
    lines = [
        "\n" + "="*70,
        "HEALTH ANOMALY DETECTION REPORT",
        "="*70,
        f"\nAnalyzed Features: {results['analyzed_features']}",
        f"Issues Detected: {results['total_issues']}",
        f"Overall Risk Level: {results['overall_risk']}",
        "\n" + "-"*70
    ]
    
    if results['advisories']:
        lines.append("\nDETAILED FINDINGS:\n")
        for i, adv in enumerate(results['advisories'], 1):
            lines.extend([
                f"{i}. {adv.feature.upper().replace('_', ' ')}",
                f"   Current Value: {adv.value}",
            ])
            if adv.population_mean is not None:
                lines.append(f"   Population Mean: {adv.population_mean:.2f} (±{adv.population_std:.2f})")
            if adv.z_score is not None:
                lines.append(f"   Z-Score: {adv.z_score:.2f}")
            lines.extend([
                f"   ⚠ Issue: {adv.issue}",
                f"   ✓ Recommendation: {adv.recommendation}",
                "",
            ])
    else:
        lines.append("\n✓ No significant health anomalies detected.")
        lines.append("  Continue maintaining healthy lifestyle habits.\n")
    
    lines.append("="*70 + "\n")
    return "\n".join(lines)


def main():
    """Run a sample user analysis."""
    try:
        detector = AnomalyDetector()
        
        example_user = {
            "age": 50,
            "gender": 1,  # 1 = Male
            "bmi": 27.5,
            "daily_steps": 3000,
            "sleep_hours": 5,
            "water_intake_l": 1.5,
            "calories_consumed": 2500,
            "smoker": 1,
            "alcohol": 1,
            "resting_hr": 80,
            "systolic_bp": 140,
            "diastolic_bp": 85,
            "cholesterol": 250,
            "family_history": 1
        }
        
        results = detector.analyze_user(example_user)
        print(format_report(results))
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
