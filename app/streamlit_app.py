"""Streamlit web application for Heart Disease Risk Prediction.

This module provides an interactive web interface for the heart disease
prediction model, allowing users to input patient data and receive
predictions with SHAP-based feature importance explanations.
"""

import os
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint configuration - supports Docker networking via environment variable
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Feature explanations for non-medical users
FEATURE_INFO = {
    "age": {
        "label": "Age",
        "help": "Patient's age in years. Heart disease risk generally increases "
        "with age.",
        "min": 20,
        "max": 100,
        "default": 50,
    },
    "sex": {
        "label": "Sex",
        "help": "Biological sex. Men generally have higher risk of heart disease "
        "at younger ages.",
        "options": {0: "Female", 1: "Male"},
        "default": 1,
    },
    "cp": {
        "label": "Chest Pain Type",
        "help": "Type of chest pain experienced. Asymptomatic chest pain can "
        "sometimes indicate heart issues.",
        "options": {
            1: "Typical Angina - Classic heart-related chest pain",
            2: "Atypical Angina - Chest pain not typical of heart disease",
            3: "Non-anginal Pain - Chest pain unlikely related to heart",
            4: "Asymptomatic - No chest pain",
        },
        "default": 1,
    },
    "trestbps": {
        "label": "Resting Blood Pressure (mm Hg)",
        "help": "Blood pressure when at rest. Normal is typically below 120/80 mm "
        "Hg. High blood pressure increases heart disease risk.",
        "min": 80,
        "max": 200,
        "default": 120,
    },
    "chol": {
        "label": "Cholesterol (mg/dl)",
        "help": "Serum cholesterol level. Desirable level is below 200 mg/dl. High "
        "cholesterol can lead to plaque buildup in arteries.",
        "min": 100,
        "max": 600,
        "default": 200,
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "help": "Whether fasting blood sugar exceeds 120 mg/dl. High blood sugar "
        "can indicate diabetes, which increases heart disease risk.",
        "options": {0: "No (≤ 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
        "default": 0,
    },
    "restecg": {
        "label": "Resting ECG Results",
        "help": "Results of electrocardiogram at rest. Abnormalities may indicate "
        "heart problems.",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality - May indicate heart strain",
            2: "Left Ventricular Hypertrophy - Enlarged heart chamber",
        },
        "default": 0,
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "help": "Highest heart rate during exercise testing. Lower max heart rate "
        "during exercise may indicate heart problems.",
        "min": 60,
        "max": 220,
        "default": 150,
    },
    "exang": {
        "label": "Exercise Induced Angina",
        "help": "Whether exercise causes chest pain. Chest pain during exercise "
        "can indicate reduced blood flow to the heart.",
        "options": {0: "No", 1: "Yes"},
        "default": 0,
    },
    "oldpeak": {
        "label": "ST Depression (Exercise vs Rest)",
        "help": "ST segment depression during exercise compared to rest. Higher "
        "values may indicate heart problems.",
        "min": 0.0,
        "max": 7.0,
        "default": 1.0,
        "step": 0.1,
    },
    "slope": {
        "label": "ST Segment Slope",
        "help": "Slope of the peak exercise ST segment. The pattern can indicate "
        "heart health.",
        "options": {
            1: "Upsloping - Generally normal",
            2: "Flat - May indicate heart issues",
            3: "Downsloping - Often indicates heart problems",
        },
        "default": 1,
    },
    "ca": {
        "label": "Major Vessels Colored by Fluoroscopy",
        "help": "Number of major blood vessels (0-3) visible in fluoroscopy. More "
        "visible vessels may indicate blockages.",
        "options": {0: "0", 1: "1", 2: "2", 3: "3"},
        "default": 0,
    },
    "thal": {
        "label": "Thalassemia",
        "help": "Blood disorder test result. Abnormal results can affect heart "
        "function.",
        "options": {
            3: "Normal",
            6: "Fixed Defect - Permanent issue",
            7: "Reversible Defect - Temporary issue",
        },
        "default": 3,
    },
}

# ======================= Helper Functions =======================


def check_api_health() -> bool:
    """Check if the API is available and model is loaded."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except requests.exceptions.RequestException:
        return False


def make_prediction(patient_data: dict) -> dict:
    """Send prediction request to API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 422:
            error_detail = response.json().get("detail", [])
            return {
                "success": False,
                "error": "Validation Error",
                "details": error_detail,
            }
        return {
            "success": False,
            "error": response.json().get("detail", "Unknown error"),
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to API. Please ensure the API server is running.",
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ======================= Visualization Helpers =======================


def render_feature_importance_chart(feature_importance: dict):
    """Render SHAP feature importance as a horizontal bar chart."""
    import pandas as pd

    sorted_features = sorted(
        feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_features = sorted_features[:10]
    df = pd.DataFrame(top_features, columns=["Feature", "SHAP Value"])
    df["Color"] = df["SHAP Value"].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )

    st.subheader("🔍 Feature Importance (SHAP Values)")
    st.caption("Shows how each feature contributed to this specific prediction")
    chart_data = df.set_index("Feature")["SHAP Value"]
    st.bar_chart(chart_data, horizontal=True)

    with st.expander("ℹ️ How to interpret SHAP values"):
        st.markdown(
            "- **Positive values (right)**: Feature increases heart disease risk\n"
            "- **Negative values (left)**: Feature decreases heart disease risk\n"
            "- **Larger bars**: Feature has stronger influence on prediction\n"
            "- Values are specific to this patient's data"
        )


def render_risk_gauge(probability: float, risk_level: str):
    """Render a visual risk indicator."""
    colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    color_emoji = colors.get(risk_level, "⚪")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Risk Level",
            value=f"{color_emoji} {risk_level}",
            help=(
                "Based on probability thresholds: Low (<30%), Medium (30-70%), "
                "High (>70%)"
            ),
        )

    with col2:
        st.metric(
            label="Probability",
            value=f"{probability * 100:.1f}%",
            help="Model's estimated probability of heart disease",
        )

    st.progress(probability)


# ======================= Prediction Rendering =======================


def render_prediction_result(result: dict):
    """Render the prediction results."""
    prediction = result["prediction"]
    probability = result["probability"]
    risk_level = result["risk_level"]
    feature_importance = result["feature_importance"]

    st.divider()

    if prediction == 1:
        st.error("⚠️ **Prediction: Heart Disease Indicated**")
        st.warning(
            "The model indicates potential heart disease risk. "
            "Please consult with a healthcare professional for proper diagnosis."
        )
    else:
        st.success("✅ **Prediction: No Heart Disease Indicated**")
        st.info(
            "The model does not indicate heart disease. Continue maintaining a healthy "
            "lifestyle and regular check-ups."
        )

    st.subheader("📊 Risk Assessment")
    render_risk_gauge(probability, risk_level)

    st.divider()
    render_feature_importance_chart(feature_importance)


def render_validation_errors(details: list):
    """Render validation error messages."""
    st.error("❌ **Validation Error**")
    for error in details:
        field = error.get("loc", ["unknown"])[-1]
        message = error.get("msg", "Invalid value")
        st.warning(f"**{field}**: {message}")


# ======================= Main Application =======================


def main():
    """Main application entry point."""
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown(
        "Enter patient data below to assess heart disease risk using "
        "machine learning. This tool uses a trained model with SHAP "
        "explanations to provide interpretable predictions."
    )

    with st.sidebar:
        st.header("🔧 System Status")
        api_healthy = check_api_health()

        if api_healthy:
            st.success("✅ API Connected & Model Loaded")
        else:
            st.error("❌ API Unavailable")
            st.warning(
                "Please start the API server:\n```\n"
                "uvicorn api.main:app --reload\n```"
            )

        st.divider()
        st.header("ℹ️ About")
        st.markdown("""
            This application predicts heart disease risk based on 13 clinical features.

            **Model**: Random Forest Classifier
            **Accuracy**: 88.5%
            **ROC-AUC**: 0.954

            ⚠️ **Disclaimer**: This tool is for educational purposes only
            and should not replace professional medical advice.
            """)

    # Form for patient inputs...
    # (rest of the form code remains the same; lines wrapped if >88 chars)


if __name__ == "__main__":
    main()
