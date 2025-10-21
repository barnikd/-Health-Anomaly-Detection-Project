import streamlit as st
import pandas as pd
import os

# Use relative import since app.py is in src/
from predict import AnomalyDetector, format_report

# Page config
st.set_page_config(
    page_title="Health Anomaly Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header { text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem; color: white; }
.upload-section { background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border: 2px dashed #667eea; text-align: center; margin: 2rem 0; }
.user-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 5px solid #667eea; }
.metric-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 8px; color: white; text-align: center; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 Health Anomaly Detection & Advisory System</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem;">AI-powered health monitoring for personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/health-graph.png", width=80)
    st.header("📊 About")
    st.markdown("""
    This system analyzes health metrics to:
    - 🔍 Detect anomalies
    - 📈 Predict health trends
    - 💡 Provide recommendations
    - ⚠️ Alert on critical values
    """)
    st.divider()
    st.header("📋 Instructions")
    st.markdown("""
    1. Upload your CSV/Excel file
    2. Review the data preview
    3. Get instant analysis
    4. Download results
    """)
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 8px;">
        <p style="margin: 0; color: #666;">Powered by AI</p>
        <p style="margin: 0; font-size: 0.9rem; color: #999;">Secure & Confidential</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Health Data")
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xlsx"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Use fallback sample data if no file uploaded
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ {uploaded_file.name} loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ No file uploaded. Using sample data for testing.")
    df = pd.DataFrame({
        "AGE": [50, 30, 40],
        "SMOKER": [1, 0, 0],
        "ALCOHOL": [1, 0, 1],
        "FAMILY_HISTORY": [1, 0, 1]
    })

# Preview data
with st.expander("🔍 Preview Data", expanded=True):
    st.dataframe(df)

# Start analysis
if st.button("🔬 Start Analysis"):
    with st.spinner("🧠 Running AI analysis..."):
        detector = AnomalyDetector()
        all_reports = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        for idx, row in df.iterrows():
            progress_bar.progress((idx + 1) / len(df))
            status_text.text(f"Analyzing user {idx + 1} of {len(df)}...")
            user_data = row.to_dict()
            results = detector.analyze_user(user_data)
            report = format_report(results)
            all_reports.append(report)
            with results_container:
                with st.expander(f"👤 User {idx + 1} - Analysis Report", expanded=(idx < 3)):
                    st.markdown(f"```\n{report}\n```")

        progress_bar.empty()
        status_text.empty()
        st.balloons()
        st.success("🎉 Analysis completed successfully!")

        # Download full report
        full_report = "\n\n" + "="*60 + "\n\n".join([f"USER {i+1}\n{rep}" for i, rep in enumerate(all_reports)])
        st.download_button(
            label="📥 Download Full Report",
            data=full_report,
            file_name="health_analysis_report.txt",
            mime="text/plain"
        )
