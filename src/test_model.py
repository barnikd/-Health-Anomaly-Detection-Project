"""
Test script for health anomaly detection system.

Uses sample user data to evaluate the AnomalyDetector from predict.py.
"""

from src.predict import AnomalyDetector, format_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize detector (ensure CSV path inside predict.py points to your dataset)
        detector = AnomalyDetector()

        # Sample test users (values based on typical health data from online sources)
        test_users = [
            {
                "age": 35,
                "gender": 0,  # 0 = Female
                "bmi": 22.5,
                "daily_steps": 5000,
                "sleep_hours": 6,
                "water_intake_l": 2.0,
                "calories_consumed": 2000,
                "smoker": 0,
                "alcohol": 0,
                "resting_hr": 70,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "cholesterol": 180,
                "family_history": 0
            },
            {
                "age": 55,
                "gender": 1,  # 1 = Male
                "bmi": 30.2,
                "daily_steps": 2500,
                "sleep_hours": 5,
                "water_intake_l": 1.2,
                "calories_consumed": 3000,
                "smoker": 1,
                "alcohol": 1,
                "resting_hr": 85,
                "systolic_bp": 145,
                "diastolic_bp": 90,
                "cholesterol": 240,
                "family_history": 1
            },
        ]

        for idx, user in enumerate(test_users, 1):
            logger.info(f"\nAnalyzing Test User {idx}...")
            results = detector.analyze_user(user)
            report = format_report(results)
            print(report)

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        raise

if __name__ == "__main__":
    main()
