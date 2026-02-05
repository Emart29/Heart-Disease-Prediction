"""Property-based tests for feature engineering module.

This module tests that feature engineering preserves data integrity,
validating Requirements 3.3.
"""

import numpy as np
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

    **Feature: portfolio-enhancement, Property 3: Feature Engineering**
    **Preserves Data Integrity**
    **Validates: Requirements 3.3**
    """

    @given(patient_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prepare_features_produces_22_elements(self, patient_data):
        """For any valid patient data, prepare_features SHALL produce exactly
        22 elements.

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


class TestCreateAgeGroup:
    """Test age group creation function."""

    def test_create_age_group_single_values(self):
        """Test age group creation for single values."""
        assert create_age_group(35) == 0  # young adult
        assert create_age_group(45) == 1  # middle age
        assert create_age_group(60) == 2  # senior
        assert create_age_group(75) == 3  # elderly

    def test_create_age_group_edge_cases(self):
        """Test age group creation for edge cases."""
        assert create_age_group(0) == 0  # below first bin
        assert create_age_group(40) == 0  # exactly at boundary
        assert create_age_group(100) == 3  # at upper boundary

    def test_create_age_group_series(self):
        """Test age group creation for pandas Series."""
        import pandas as pd

        ages = pd.Series([35, 45, 60, 75])
        result = create_age_group(ages)
        expected = pd.Series([0, 1, 2, 3])
        pd.testing.assert_series_equal(result, expected)


class TestCreateRiskFeatures:
    """Test risk feature creation function."""

    def test_create_risk_features(self):
        """Test creation of derived risk features."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "age": [50, 60],
                "chol": [200, 250],
                "trestbps": [120, 150],
                "thalach": [150, 140],
            }
        )

        result = create_risk_features(df)

        # Check new columns exist
        assert "age_group" in result.columns
        assert "chol_risk" in result.columns
        assert "bp_risk" in result.columns
        assert "heart_rate_reserve" in result.columns

        # Check values
        assert result["chol_risk"].iloc[0] == 0  # 200 <= 240
        assert result["chol_risk"].iloc[1] == 1  # 250 > 240
        assert result["bp_risk"].iloc[0] == 0  # 120 <= 140
        assert result["bp_risk"].iloc[1] == 1  # 150 > 140


class TestEncodeCategorical:
    """Test categorical encoding function."""

    def test_encode_categorical_basic(self):
        """Test basic categorical encoding."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "cp": [1, 2, 3],
                "restecg": [0, 1, 2],
                "slope": [1, 2, 3],
                "thal": [3, 6, 7],
            }
        )

        result = encode_categorical(df)

        # Check that original columns are removed
        assert "cp" not in result.columns
        assert "restecg" not in result.columns

        # Check that encoded columns exist
        assert "cp_2.0" in result.columns
        assert "cp_3.0" in result.columns
        assert "restecg_1.0" in result.columns

    def test_encode_categorical_missing_columns(self):
        """Test encoding when some categorical columns are missing."""
        import pandas as pd

        df = pd.DataFrame({"cp": [1, 2], "other_col": [10, 20]})

        result = encode_categorical(df)

        # Should only encode existing categorical columns
        assert "other_col" in result.columns
        assert "cp_2.0" in result.columns


class TestCreateAgeGroupEdgeCases:
    """Test edge cases in age group creation."""

    def test_create_age_group_exact_zero(self):
        """Test age group creation for age exactly 0."""
        result = create_age_group(0)
        assert result == 0

    def test_create_age_group_negative_age(self):
        """Test age group creation for negative age."""
        result = create_age_group(-5)
        assert result == 0

    def test_create_age_group_very_high_age(self):
        """Test age group creation for very high age."""
        result = create_age_group(150)
        assert result == 3


class TestPrepareFeaturesMissingColumns:
    """Test prepare_features with missing columns."""

    def test_prepare_features_ensures_all_columns(self):
        """Test that prepare_features ensures all expected columns exist."""
        # Create patient data that will result in missing encoded columns
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

        result = prepare_features(patient_data)
        expected_cols = get_feature_columns()

        # Should have exactly the right number of features
        assert result.shape == (1, len(expected_cols))

        # All values should be numeric (no NaN)
        assert not np.isnan(result).any()


class TestGetFeatureColumns:
    """Test feature column names function."""

    def test_get_feature_columns_count(self):
        """Test that get_feature_columns returns exactly 22 columns."""
        columns = get_feature_columns()
        assert len(columns) == 22

    def test_get_feature_columns_types(self):
        """Test that all feature column names are strings."""
        columns = get_feature_columns()
        assert all(isinstance(col, str) for col in columns)

    def test_get_feature_columns_no_duplicates(self):
        """Test that feature column names have no duplicates."""
        columns = get_feature_columns()
        assert len(columns) == len(set(columns))
