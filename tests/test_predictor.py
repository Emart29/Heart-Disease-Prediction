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
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

    **Feature: portfolio-enhancement, Property 1: Valid Input Produces**
    **Complete Response**
    **Validates: Requirements 2.2**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
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
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
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
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
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
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
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
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_feature_importance_contains_all_features(self, patient_data):
        """For any valid prediction, feature_importance SHALL contain all
        model features.

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
    @settings(
        max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
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


class TestPredictorEdgeCases:
    """Test edge cases and error handling in predictor."""

    def test_predictor_with_custom_model_path(self, tmp_path):
        """Test predictor initialization with custom model path."""
        # Create a mock model file
        model_data = {
            "model": RandomForestClassifier(n_estimators=5, random_state=42),
            "scaler": StandardScaler(),
            "feature_names": get_feature_columns(),
        }

        # Fit the model and scaler with dummy data
        X_dummy = [[1, 2] + [0] * 20]  # 22 features total
        y_dummy = [1]
        model_data["model"].fit(X_dummy, y_dummy)
        model_data["scaler"].fit(X_dummy)

        model_path = tmp_path / "custom_model.pkl"
        joblib.dump(model_data, model_path)

        # Test predictor with custom path
        predictor = HeartDiseasePredictor(model_path=model_path)
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_predictor_with_missing_feature_names(self, tmp_path):
        """Test predictor when model file doesn't contain feature_names."""
        # Create model file without feature_names
        model_data = {
            "model": RandomForestClassifier(n_estimators=5, random_state=42),
            "scaler": StandardScaler(),
            # Missing 'feature_names' key
        }

        # Fit the model and scaler
        X_dummy = [[1, 2] + [0] * 20]
        y_dummy = [1]
        model_data["model"].fit(X_dummy, y_dummy)
        model_data["scaler"].fit(X_dummy)

        model_path = tmp_path / "model_no_features.pkl"
        joblib.dump(model_data, model_path)

        # Should use default feature names
        predictor = HeartDiseasePredictor(model_path=model_path)
        assert predictor.feature_names == get_feature_columns()

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_shap_values_different_formats(self, patient_data):
        """Test handling of different SHAP value formats."""
        predictor = get_predictor()

        # Test with valid patient data
        result = predictor.predict(patient_data)

        # Should handle SHAP values correctly regardless of format
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == 22
        assert all(isinstance(v, float) for v in result["feature_importance"].values())


class TestPredictorInitialization:
    """Test predictor initialization edge cases."""

    def test_predictor_file_not_found(self):
        """Test predictor initialization with non-existent model file."""
        with pytest.raises(FileNotFoundError):
            HeartDiseasePredictor(model_path=Path("nonexistent_model.pkl"))

    def test_predictor_invalid_model_file(self, tmp_path):
        """Test predictor initialization with invalid model file."""
        # Create an invalid model file
        invalid_file = tmp_path / "invalid_model.pkl"
        invalid_file.write_text("invalid content")

        with pytest.raises(
            Exception
        ):  # Could be various exceptions depending on joblib version
            HeartDiseasePredictor(model_path=invalid_file)


class TestShapValueHandling:
    """Test different SHAP value formats."""

    def test_shap_values_list_format(self):
        """Test handling of SHAP values in list format."""
        predictor = get_predictor()

        # Mock the explainer to return list format
        with patch.object(predictor.explainer, "shap_values") as mock_shap:
            # Simulate old SHAP format: list of arrays
            mock_shap.return_value = [
                np.zeros(22),  # class 0 values
                np.ones(22),  # class 1 values
            ]

            patient_data = {
                "age": 50,
                "sex": 1,
                "cp": 1,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 3,
            }

            result = predictor.predict(patient_data)

            # Should use class 1 values (positive class)
            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 22
            assert all(v == 1.0 for v in result["feature_importance"].values())

    def test_shap_values_3d_array_format(self):
        """Test handling of SHAP values in 3D array format."""
        predictor = get_predictor()

        # Mock the explainer to return 3D array format
        with patch.object(predictor.explainer, "shap_values") as mock_shap:
            # Simulate new SHAP format: 3D array (n_samples, n_features, n_classes)
            shap_array = np.zeros((1, 22, 2))
            shap_array[0, :, 1] = 2.0  # Set class 1 values to 2.0
            mock_shap.return_value = shap_array

            patient_data = {
                "age": 50,
                "sex": 1,
                "cp": 1,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 3,
            }

            result = predictor.predict(patient_data)

            # Should use class 1 values (positive class)
            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 22
            assert all(v == 2.0 for v in result["feature_importance"].values())

    def test_shap_values_2d_array_format(self):
        """Test handling of SHAP values in 2D array format."""
        predictor = get_predictor()

        # Mock the explainer to return 2D array format
        with patch.object(predictor.explainer, "shap_values") as mock_shap:
            # Simulate 2D array format: (n_samples, n_features)
            shap_array = np.full((1, 22), 3.0)
            mock_shap.return_value = shap_array

            patient_data = {
                "age": 50,
                "sex": 1,
                "cp": 1,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 3,
            }

            result = predictor.predict(patient_data)

            assert "feature_importance" in result
            assert len(result["feature_importance"]) == 22
            assert all(v == 3.0 for v in result["feature_importance"].values())


class TestRiskLevelEdgeCases:
    """Test risk level calculation edge cases."""

    def test_risk_level_boundary_values(self):
        """Test risk level calculation at boundary values."""
        predictor = get_predictor()

        # Test exact boundary values
        assert predictor._get_risk_level(0.0) == "Low"
        assert predictor._get_risk_level(0.29999) == "Low"
        assert predictor._get_risk_level(0.3) == "Medium"
        assert predictor._get_risk_level(0.69999) == "Medium"
        assert predictor._get_risk_level(0.7) == "High"
        assert predictor._get_risk_level(1.0) == "High"
