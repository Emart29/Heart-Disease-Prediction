"""Property-based tests for feature engineering module.

This module tests that feature engineering preserves data integrity,
validating Requirements 3.3.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.features.engineering import (
    create_age_group,
    create_risk_features,
    encode_categorical,
    get_feature_columns,
    prepare_features,
)


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


class TestProperty3FeatureEngineeringPreservesDataIntegrity:
    """Property 3: Feature Engineering Preserves Data Integrity.

    For any valid patient data dictionary, the feature preparation function
    SHALL produce a feature array of exactly 22 elements with no NaN values.

    **Feature: portfolio-enhancement, Property 3: Feature Engineering Preserves Data Integrity**
    **Validates: Requirements 3.3**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prepare_features_produces_22_elements(self, patient_data):
        """For any valid patient data, prepare_features SHALL produce exactly 22 elements.

        Validates: Requirement 3.3 - reusable functions for feature engineering
        """
        result = prepare_features(patient_data)

        assert result.shape == (1, 22), f"Expected shape (1, 22), got {result.shape}"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prepare_features_produces_no_nan_values(self, patient_data):
        """For any valid patient data, prepare_features SHALL produce no NaN values.

        Validates: Requirement 3.3 - reusable functions for feature engineering
        """
        result = prepare_features(patient_data)

        assert not np.isnan(
            result
        ).any(), f"Feature array contains NaN values: {result}"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_columns_count_matches_output(self, patient_data):
        """The number of feature columns SHALL match the output array size.

        Validates: Requirement 3.3 - reusable functions for feature engineering
        """
        feature_cols = get_feature_columns()
        result = prepare_features(patient_data)

        assert len(feature_cols) == result.shape[1], (
            f"Feature columns count ({len(feature_cols)}) doesn't match "
            f"output array size ({result.shape[1]})"
        )

    @given(age=VALID_AGE)
    @settings(max_examples=100)
    def test_create_age_group_returns_valid_category(self, age):
        """For any valid age, create_age_group SHALL return a category 0-3.

        Validates: Requirement 3.3 - reusable functions for feature engineering
        """
        result = create_age_group(age)

        assert result in [
            0,
            1,
            2,
            3,
        ], f"Age group {result} not in valid range [0, 1, 2, 3] for age {age}"

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prepare_features_produces_numeric_values(self, patient_data):
        """For any valid patient data, all output values SHALL be numeric.

        Validates: Requirement 3.3 - reusable functions for feature engineering
        """
        result = prepare_features(patient_data)

        assert np.issubdtype(
            result.dtype, np.number
        ), f"Feature array dtype {result.dtype} is not numeric"
