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
        "help": "Patient's age in years. Heart disease risk generally increases with age.",
        "min": 20,
        "max": 100,
        "default": 50,
    },
    "sex": {
        "label": "Sex",
        "help": "Biological sex. Men generally have higher risk of heart disease at younger ages.",
        "options": {0: "Female", 1: "Male"},
        "default": 1,
    },
    "cp": {
        "label": "Chest Pain Type",
        "help": "Type of chest pain experienced. Asymptomatic chest pain can sometimes indicate heart issues.",
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
        "help": "Blood pressure when at rest. Normal is typically below 120/80 mm Hg. High blood pressure increases heart disease risk.",
        "min": 80,
        "max": 200,
        "default": 120,
    },
    "chol": {
        "label": "Cholesterol (mg/dl)",
        "help": "Serum cholesterol level. Desirable level is below 200 mg/dl. High cholesterol can lead to plaque buildup in arteries.",
        "min": 100,
        "max": 600,
        "default": 200,
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "help": "Whether fasting blood sugar exceeds 120 mg/dl. High blood sugar can indicate diabetes, which increases heart disease risk.",
        "options": {0: "No (≤ 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
        "default": 0,
    },
    "restecg": {
        "label": "Resting ECG Results",
        "help": "Results of electrocardiogram at rest. Abnormalities may indicate heart problems.",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality - May indicate heart strain",
            2: "Left Ventricular Hypertrophy - Enlarged heart chamber",
        },
        "default": 0,
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "help": "Highest heart rate during exercise testing. Lower max heart rate during exercise may indicate heart problems.",
        "min": 60,
        "max": 220,
        "default": 150,
    },
    "exang": {
        "label": "Exercise Induced Angina",
        "help": "Whether exercise causes chest pain. Chest pain during exercise can indicate reduced blood flow to the heart.",
        "options": {0: "No", 1: "Yes"},
        "default": 0,
    },
    "oldpeak": {
        "label": "ST Depression (Exercise vs Rest)",
        "help": "ST segment depression during exercise compared to rest. Higher values may indicate heart problems.",
        "min": 0.0,
        "max": 7.0,
        "default": 1.0,
        "step": 0.1,
    },
    "slope": {
        "label": "ST Segment Slope",
        "help": "Slope of the peak exercise ST segment. The pattern can indicate heart health.",
        "options": {
            1: "Upsloping - Generally normal",
            2: "Flat - May indicate heart issues",
            3: "Downsloping - Often indicates heart problems",
        },
        "default": 1,
    },
    "ca": {
        "label": "Major Vessels Colored by Fluoroscopy",
        "help": "Number of major blood vessels (0-3) visible in fluoroscopy. More visible vessels may indicate blockages.",
        "options": {0: "0", 1: "1", 2: "2", 3: "3"},
        "default": 0,
    },
    "thal": {
        "label": "Thalassemia",
        "help": "Blood disorder test result. Abnormal results can affect heart function.",
        "options": {
            3: "Normal",
            6: "Fixed Defect - Permanent issue",
            7: "Reversible Defect - Temporary issue",
        },
        "default": 3,
    },
}


def check_api_health() -> bool:
    """Check if the API is available and model is loaded.

    Returns:
        True if API is healthy and model is loaded, False otherwise.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except requests.exceptions.RequestException:
        return False


def make_prediction(patient_data: dict) -> dict:
    """Send prediction request to API.

    Args:
        patient_data: Dictionary of patient features.

    Returns:
        Prediction response dictionary or error dictionary.
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 422:
            # Validation error
            error_detail = response.json().get("detail", [])
            return {
                "success": False,
                "error": "Validation Error",
                "details": error_detail,
            }
        else:
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


def render_feature_importance_chart(feature_importance: dict):
    """Render SHAP feature importance as a horizontal bar chart.

    Args:
        feature_importance: Dictionary mapping feature names to SHAP values.
    """
    import pandas as pd

    # Sort by absolute value to show most important features
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Take top 10 features for readability
    top_features = sorted_features[:10]

    # Create DataFrame for chart
    df = pd.DataFrame(top_features, columns=["Feature", "SHAP Value"])

    # Color based on positive/negative contribution
    df["Color"] = df["SHAP Value"].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )

    # Create bar chart using Streamlit's native chart
    st.subheader("🔍 Feature Importance (SHAP Values)")
    st.caption("Shows how each feature contributed to this specific prediction")

    # Use Streamlit's bar chart with custom formatting
    chart_data = df.set_index("Feature")["SHAP Value"]

    # Display as horizontal bar chart
    st.bar_chart(chart_data, horizontal=True)

    # Add explanation
    with st.expander("ℹ️ How to interpret SHAP values"):
        st.markdown(
            """
        - **Positive values (right)**: Feature increases heart disease risk
        - **Negative values (left)**: Feature decreases heart disease risk
        - **Larger bars**: Feature has stronger influence on prediction
        - Values are specific to this patient's data
        """
        )


def render_risk_gauge(probability: float, risk_level: str):
    """Render a visual risk indicator.

    Args:
        probability: Probability of heart disease (0-1).
        risk_level: Risk category string.
    """
    # Color based on risk level
    colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

    color_emoji = colors.get(risk_level, "⚪")

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Risk Level",
            value=f"{color_emoji} {risk_level}",
            help="Based on probability thresholds: Low (<30%), Medium (30-70%), High (>70%)",
        )

    with col2:
        st.metric(
            label="Probability",
            value=f"{probability * 100:.1f}%",
            help="Model's estimated probability of heart disease",
        )

    # Progress bar for visual representation
    st.progress(probability)


def render_prediction_result(result: dict):
    """Render the prediction results.

    Args:
        result: Prediction response from API.
    """
    prediction = result["prediction"]
    probability = result["probability"]
    risk_level = result["risk_level"]
    feature_importance = result["feature_importance"]

    # Main result
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
            "The model does not indicate heart disease. "
            "Continue maintaining a healthy lifestyle and regular check-ups."
        )

    # Risk gauge
    st.subheader("📊 Risk Assessment")
    render_risk_gauge(probability, risk_level)

    # Feature importance chart
    st.divider()
    render_feature_importance_chart(feature_importance)


def render_validation_errors(details: list):
    """Render validation error messages.

    Args:
        details: List of validation error details from API.
    """
    st.error("❌ **Validation Error**")

    for error in details:
        field = error.get("loc", ["unknown"])[-1]
        message = error.get("msg", "Invalid value")
        st.warning(f"**{field}**: {message}")


def main():
    """Main application entry point."""
    # Header
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown(
        "Enter patient data below to assess heart disease risk using machine learning. "
        "This tool uses a trained model with SHAP explanations to provide interpretable predictions."
    )

    # Check API health
    with st.sidebar:
        st.header("🔧 System Status")
        api_healthy = check_api_health()

        if api_healthy:
            st.success("✅ API Connected & Model Loaded")
        else:
            st.error("❌ API Unavailable")
            st.warning(
                "Please start the API server:\n"
                "```\nuvicorn api.main:app --reload\n```"
            )

        st.divider()
        st.header("ℹ️ About")
        st.markdown(
            """
            This application predicts heart disease risk based on 13 clinical features.
            
            **Model**: Random Forest Classifier  
            **Accuracy**: 88.5%  
            **ROC-AUC**: 0.954
            
            ⚠️ **Disclaimer**: This tool is for educational purposes only 
            and should not replace professional medical advice.
            """
        )

    # Input form
    st.header("📋 Patient Information")
    st.caption("Fill in all fields below. Hover over ❓ icons for explanations.")

    with st.form("prediction_form"):
        # Organize inputs in columns
        col1, col2, col3 = st.columns(3)

        # Column 1: Demographics and vitals
        with col1:
            st.subheader("Demographics & Vitals")

            age = st.number_input(
                FEATURE_INFO["age"]["label"],
                min_value=FEATURE_INFO["age"]["min"],
                max_value=FEATURE_INFO["age"]["max"],
                value=FEATURE_INFO["age"]["default"],
                help=FEATURE_INFO["age"]["help"],
            )

            sex_options = FEATURE_INFO["sex"]["options"]
            sex = st.selectbox(
                FEATURE_INFO["sex"]["label"],
                options=list(sex_options.keys()),
                format_func=lambda x: sex_options[x],
                index=FEATURE_INFO["sex"]["default"],
                help=FEATURE_INFO["sex"]["help"],
            )

            trestbps = st.number_input(
                FEATURE_INFO["trestbps"]["label"],
                min_value=FEATURE_INFO["trestbps"]["min"],
                max_value=FEATURE_INFO["trestbps"]["max"],
                value=FEATURE_INFO["trestbps"]["default"],
                help=FEATURE_INFO["trestbps"]["help"],
            )

            chol = st.number_input(
                FEATURE_INFO["chol"]["label"],
                min_value=FEATURE_INFO["chol"]["min"],
                max_value=FEATURE_INFO["chol"]["max"],
                value=FEATURE_INFO["chol"]["default"],
                help=FEATURE_INFO["chol"]["help"],
            )

            fbs_options = FEATURE_INFO["fbs"]["options"]
            fbs = st.selectbox(
                FEATURE_INFO["fbs"]["label"],
                options=list(fbs_options.keys()),
                format_func=lambda x: fbs_options[x],
                index=FEATURE_INFO["fbs"]["default"],
                help=FEATURE_INFO["fbs"]["help"],
            )

        # Column 2: Heart measurements
        with col2:
            st.subheader("Heart Measurements")

            thalach = st.number_input(
                FEATURE_INFO["thalach"]["label"],
                min_value=FEATURE_INFO["thalach"]["min"],
                max_value=FEATURE_INFO["thalach"]["max"],
                value=FEATURE_INFO["thalach"]["default"],
                help=FEATURE_INFO["thalach"]["help"],
            )

            exang_options = FEATURE_INFO["exang"]["options"]
            exang = st.selectbox(
                FEATURE_INFO["exang"]["label"],
                options=list(exang_options.keys()),
                format_func=lambda x: exang_options[x],
                index=FEATURE_INFO["exang"]["default"],
                help=FEATURE_INFO["exang"]["help"],
            )

            oldpeak = st.number_input(
                FEATURE_INFO["oldpeak"]["label"],
                min_value=FEATURE_INFO["oldpeak"]["min"],
                max_value=FEATURE_INFO["oldpeak"]["max"],
                value=FEATURE_INFO["oldpeak"]["default"],
                step=FEATURE_INFO["oldpeak"]["step"],
                help=FEATURE_INFO["oldpeak"]["help"],
            )

            slope_options = FEATURE_INFO["slope"]["options"]
            slope = st.selectbox(
                FEATURE_INFO["slope"]["label"],
                options=list(slope_options.keys()),
                format_func=lambda x: slope_options[x],
                index=0,
                help=FEATURE_INFO["slope"]["help"],
            )

            restecg_options = FEATURE_INFO["restecg"]["options"]
            restecg = st.selectbox(
                FEATURE_INFO["restecg"]["label"],
                options=list(restecg_options.keys()),
                format_func=lambda x: restecg_options[x],
                index=FEATURE_INFO["restecg"]["default"],
                help=FEATURE_INFO["restecg"]["help"],
            )

        # Column 3: Chest pain and other tests
        with col3:
            st.subheader("Symptoms & Tests")

            cp_options = FEATURE_INFO["cp"]["options"]
            cp = st.selectbox(
                FEATURE_INFO["cp"]["label"],
                options=list(cp_options.keys()),
                format_func=lambda x: cp_options[x],
                index=0,
                help=FEATURE_INFO["cp"]["help"],
            )

            ca_options = FEATURE_INFO["ca"]["options"]
            ca = st.selectbox(
                FEATURE_INFO["ca"]["label"],
                options=list(ca_options.keys()),
                format_func=lambda x: ca_options[x],
                index=FEATURE_INFO["ca"]["default"],
                help=FEATURE_INFO["ca"]["help"],
            )

            thal_options = FEATURE_INFO["thal"]["options"]
            thal = st.selectbox(
                FEATURE_INFO["thal"]["label"],
                options=list(thal_options.keys()),
                format_func=lambda x: thal_options[x],
                index=0,
                help=FEATURE_INFO["thal"]["help"],
            )

        # Submit button
        st.divider()
        submitted = st.form_submit_button(
            "🔮 Predict Heart Disease Risk", use_container_width=True, type="primary"
        )

    # Handle form submission
    if submitted:
        if not api_healthy:
            st.error(
                "❌ Cannot make prediction: API is not available. "
                "Please start the API server first."
            )
            return

        # Prepare patient data
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        # Show loading spinner
        with st.spinner("Analyzing patient data..."):
            result = make_prediction(patient_data)

        # Display results or errors
        if result["success"]:
            render_prediction_result(result["data"])
        else:
            if "details" in result:
                render_validation_errors(result["details"])
            else:
                st.error(f"❌ **Error**: {result['error']}")


if __name__ == "__main__":
    main()
