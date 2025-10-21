# Health Anomaly Detection Project

A lightweight health/anomaly detection & advisory system for insurance/lifestyle risk. 
- Baseline: Logistic Regression; production candidate: XGBoost.
- Rule-based advisory with z-score comparison to a population CSV.
- Streamlit UI for uploading CSV/Excel and receiving per-user reports and recommendations.

Run:
1. Create venv: `python -m venv .venv` & `.\.venv\Scripts\activate`
2. Install: `pip install -r requirements.txt`
3. Preprocess & train: `python -m src.preprocessing && python -m src.train_model`
4. Test: `python -m src.test_model`
5. Run UI: `streamlit run src/app.py`
