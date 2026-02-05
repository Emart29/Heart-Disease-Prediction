"""Property-based tests for model prediction module.

This module tests the HeartDiseasePredictor class, validating:
- Property 1: Valid Input Produces Complete Response
- Property 4: Risk Level Consistency
- Property 5: SHAP Values Completeness

**Feature: portfolio-enhancement**
**Validates: Requirements 2.2, 1.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.models.predict import HeartDiseasePredictor
from src.features.engineering import get_feature_columns


# Valid value strategies for each field (matching validation schemas)
VALID_AGE = st.integers(min_value=20, max_value=100)
VALID_SEX = st.sampled_from([0, 1])
VALID_CP = st.sampled_from([1, 2, 3, 4])
VALID_TRESTBPS = st.integers(min_value=80, max_value=200)
VALID_CHOL = st.integers(min_value=100, max_value=600)
VALID_FBS = st.sampled_from([0, 1])
VALID_RESTECG = st.sampled_from([0, 1, 2])
VALID_THALACH = st.integers(min_value=60, max_value=220)
VALID_EXANG = st.sampled_from([0, 1])
VALID_OLDPEAK = st.floats(min_value=0.0, max_value=7.0, allow_nan=False)
VALID_SLOPE = st.sampled_from([1, 2, 3])
VALID_CA = st.sampled_from([0, 1, 2, 3])
VALID_THAL = st.sampled_from([3, 6, 7])


def create_valid_patient_data():
    """Strategy to generate valid patient data dictionaries."""
    return st.fixed_dictionaries(
        {
            "age": VALID_AGE,
            "sex": VALID_SEX,
            "cp": VALID_CP,
            "trestbps": VALID_TRESTBPS,
            "chol": VALID_CHOL,
            "fbs": VALID_FBS,
            "restecg": VALID_RESTECG,
            "thalach": VALID_THALACH,
            "exang": VALID_EXANG,
            "oldpeak": VALID_OLDPEAK,
            "slope": VALID_SLOPE,
            "ca": VALID_CA,
            "thal": VALID_THAL,
        }
    )


# Module-level predictor instance to avoid reloading model for each test
_predictor = None


def get_predictor():
    """Get or create a singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = HeartDiseasePredictor()
    return _predictor


class TestProperty1ValidInputProducesCompleteResponse:
    """Property 1: Valid Input Produces Complete Response.

    For any valid patient data (all fields present and within specified ranges),
    the prediction API SHALL return a response containing prediction (0 or 1),
    probability (0.0-1.0), risk_level ("Low", "Medium", or "High"), and
    feature_importance dictionary.

    **Feature: portfolio-enhancement, Property 1: Valid Input Produces Complete Response**
    **Validates: Requirements 2.2**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prediction_contains_all_required_fields(self, patient_data):
        """For any valid patient data, response SHALL contain all required fields.

        Validates: Requirement 2.2 - return prediction, probability, and risk level
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        # Check all required fields are present
        assert "prediction" in result, "Response missing 'prediction' field"
        assert "probability" in result, "Response missing 'probability' field"
        assert "risk_level" in result, "Response missing 'risk_level' field"
        assert (
            "feature_importance" in result
        ), "Response missing 'feature_importance' field"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prediction_is_binary(self, patient_data):
        """For any valid patient data, prediction SHALL be 0 or 1.

        Validates: Requirement 2.2 - return prediction
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        assert result["prediction"] in [
            0,
            1,
        ], f"Prediction {result['prediction']} is not binary (0 or 1)"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_probability_in_valid_range(self, patient_data):
        """For any valid patient data, probability SHALL be between 0.0 and 1.0.

        Validates: Requirement 2.2 - return probability
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        assert (
            0.0 <= result["probability"] <= 1.0
        ), f"Probability {result['probability']} is not in range [0.0, 1.0]"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_level_is_valid_category(self, patient_data):
        """For any valid patient data, risk_level SHALL be Low, Medium, or High.

        Validates: Requirement 2.2 - return risk level
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        valid_risk_levels = ["Low", "Medium", "High"]
        assert (
            result["risk_level"] in valid_risk_levels
        ), f"Risk level '{result['risk_level']}' is not in {valid_risk_levels}"


class TestProperty4RiskLevelConsistency:
    """Property 4: Risk Level Consistency.

    For any prediction result, the risk_level SHALL be "Low" if probability < 0.3,
    "Medium" if 0.3 <= probability < 0.7, and "High" if probability >= 0.7.

    **Feature: portfolio-enhancement, Property 4: Risk Level Consistency**
    **Validates: Requirements 2.2**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_risk_level_matches_probability_thresholds(self, patient_data):
        """For any prediction, risk_level SHALL match probability thresholds.

        Validates: Requirement 2.2 - risk level based on probability
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        probability = result["probability"]
        risk_level = result["risk_level"]

        if probability < 0.3:
            expected_risk = "Low"
        elif probability < 0.7:
            expected_risk = "Medium"
        else:
            expected_risk = "High"

        assert risk_level == expected_risk, (
            f"Risk level '{risk_level}' does not match expected '{expected_risk}' "
            f"for probability {probability}"
        )


class TestProperty5SHAPValuesCompleteness:
    """Property 5: SHAP Values Completeness.

    For any valid prediction, the feature_importance dictionary SHALL contain
    exactly one entry for each of the 22 model features.

    **Feature: portfolio-enhancement, Property 5: SHAP Values Completeness**
    **Validates: Requirements 1.3**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_importance_has_22_entries(self, patient_data):
        """For any valid prediction, feature_importance SHALL have exactly 22 entries.

        Validates: Requirement 1.3 - SHAP-based feature importance
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        feature_importance = result["feature_importance"]

        assert (
            len(feature_importance) == 22
        ), f"Feature importance has {len(feature_importance)} entries, expected 22"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_importance_contains_all_features(self, patient_data):
        """For any valid prediction, feature_importance SHALL contain all model features.

        Validates: Requirement 1.3 - SHAP-based feature importance
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        feature_importance = result["feature_importance"]
        expected_features = set(get_feature_columns())
        actual_features = set(feature_importance.keys())

        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features

        assert (
            missing_features == set()
        ), f"Missing features in feature_importance: {missing_features}"
        assert (
            extra_features == set()
        ), f"Unexpected features in feature_importance: {extra_features}"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_importance_values_are_numeric(self, patient_data):
        """For any valid prediction, all SHAP values SHALL be numeric (float).

        Validates: Requirement 1.3 - SHAP-based feature importance
        """
        predictor = get_predictor()
        result = predictor.predict(patient_data)

        feature_importance = result["feature_importance"]

        for feature_name, shap_value in feature_importance.items():
            assert isinstance(
                shap_value, (int, float)
            ), f"SHAP value for '{feature_name}' is not numeric: {type(shap_value)}"
