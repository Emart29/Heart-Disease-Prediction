"""Property-based tests for data validation schemas.

This module tests that invalid inputs produce validation errors,
validating Requirements 4.1, 4.2, 4.3, 4.4.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from pydantic import ValidationError

from src.validation.schemas import PatientData


# Valid value strategies for each field
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


# Invalid value strategies
INVALID_AGE_LOW = st.integers(max_value=19)
INVALID_AGE_HIGH = st.integers(min_value=101)
INVALID_SEX = st.integers().filter(lambda x: x not in [0, 1])
INVALID_CP = st.integers().filter(lambda x: x not in [1, 2, 3, 4])
INVALID_TRESTBPS_LOW = st.integers(max_value=79)
INVALID_TRESTBPS_HIGH = st.integers(min_value=201)
INVALID_CHOL_LOW = st.integers(max_value=99)
INVALID_CHOL_HIGH = st.integers(min_value=601)
INVALID_FBS = st.integers().filter(lambda x: x not in [0, 1])
INVALID_RESTECG = st.integers().filter(lambda x: x not in [0, 1, 2])
INVALID_THALACH_LOW = st.integers(max_value=59)
INVALID_THALACH_HIGH = st.integers(min_value=221)
INVALID_EXANG = st.integers().filter(lambda x: x not in [0, 1])
INVALID_OLDPEAK_LOW = st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False)
INVALID_OLDPEAK_HIGH = st.floats(min_value=7.01, allow_nan=False, allow_infinity=False)
INVALID_SLOPE = st.integers().filter(lambda x: x not in [1, 2, 3])
INVALID_CA = st.integers().filter(lambda x: x not in [0, 1, 2, 3])
INVALID_THAL = st.integers().filter(lambda x: x not in [3, 6, 7])


class TestProperty2InvalidInputProducesValidationError:
    """Property 2: Invalid Input Produces Validation Error.

    For any patient data with missing required fields OR numeric values
    outside valid ranges OR invalid categorical values, the Data_Validator
    SHALL reject the input and return an error message identifying the
    specific invalid field(s).

    **Feature: portfolio-enhancement, Property 2: Invalid Input Produces Validation Error**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @given(valid_data=create_valid_patient_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_input_is_accepted(self, valid_data):
        """For any valid patient data, validation should succeed."""
        patient = PatientData(**valid_data)
        assert patient.age == valid_data["age"]
        assert patient.sex == valid_data["sex"]

    @given(
        valid_data=create_valid_patient_data(),
        field_to_remove=st.sampled_from(
            [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]
        ),
    )
    @settings(max_examples=100)
    def test_missing_required_field_produces_error(self, valid_data, field_to_remove):
        """For any missing required field, validation SHALL fail.

        Validates: Requirement 4.1 - validate all required fields are present
        """
        data = valid_data.copy()
        del data[field_to_remove]

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        # Verify error identifies the specific missing field
        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert field_to_remove in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_age=INVALID_AGE_LOW)
    @settings(max_examples=100)
    def test_age_below_range_produces_error(self, valid_data, invalid_age):
        """For any age below valid range, validation SHALL fail.

        Validates: Requirement 4.2 - validate numeric fields within expected ranges
        """
        data = valid_data.copy()
        data["age"] = invalid_age

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "age" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_age=INVALID_AGE_HIGH)
    @settings(max_examples=100)
    def test_age_above_range_produces_error(self, valid_data, invalid_age):
        """For any age above valid range, validation SHALL fail.

        Validates: Requirement 4.2 - validate numeric fields within expected ranges
        """
        data = valid_data.copy()
        data["age"] = invalid_age

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "age" in error_fields

    @given(
        valid_data=create_valid_patient_data(), invalid_trestbps=INVALID_TRESTBPS_LOW
    )
    @settings(max_examples=100)
    def test_trestbps_below_range_produces_error(self, valid_data, invalid_trestbps):
        """For any blood pressure below valid range, validation SHALL fail.

        Validates: Requirement 4.2 - validate numeric fields within expected ranges
        """
        data = valid_data.copy()
        data["trestbps"] = invalid_trestbps

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "trestbps" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_chol=INVALID_CHOL_HIGH)
    @settings(max_examples=100)
    def test_chol_above_range_produces_error(self, valid_data, invalid_chol):
        """For any cholesterol above valid range, validation SHALL fail.

        Validates: Requirement 4.2 - validate numeric fields within expected ranges
        """
        data = valid_data.copy()
        data["chol"] = invalid_chol

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "chol" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_oldpeak=INVALID_OLDPEAK_HIGH)
    @settings(max_examples=100)
    def test_oldpeak_above_range_produces_error(self, valid_data, invalid_oldpeak):
        """For any oldpeak above valid range, validation SHALL fail.

        Validates: Requirement 4.2 - validate numeric fields within expected ranges
        """
        data = valid_data.copy()
        data["oldpeak"] = invalid_oldpeak

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "oldpeak" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_sex=INVALID_SEX)
    @settings(max_examples=100)
    def test_invalid_sex_produces_error(self, valid_data, invalid_sex):
        """For any invalid sex value, validation SHALL fail.

        Validates: Requirement 4.3 - validate categorical fields contain valid values
        """
        data = valid_data.copy()
        data["sex"] = invalid_sex

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "sex" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_cp=INVALID_CP)
    @settings(max_examples=100)
    def test_invalid_cp_produces_error(self, valid_data, invalid_cp):
        """For any invalid chest pain type, validation SHALL fail.

        Validates: Requirement 4.3 - validate categorical fields contain valid values
        """
        data = valid_data.copy()
        data["cp"] = invalid_cp

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "cp" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_thal=INVALID_THAL)
    @settings(max_examples=100)
    def test_invalid_thal_produces_error(self, valid_data, invalid_thal):
        """For any invalid thalassemia value, validation SHALL fail.

        Validates: Requirement 4.3 - validate categorical fields contain valid values
        """
        data = valid_data.copy()
        data["thal"] = invalid_thal

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "thal" in error_fields

    @given(valid_data=create_valid_patient_data(), invalid_slope=INVALID_SLOPE)
    @settings(max_examples=100)
    def test_invalid_slope_produces_error(self, valid_data, invalid_slope):
        """For any invalid slope value, validation SHALL fail.

        Validates: Requirement 4.3 - validate categorical fields contain valid values
        """
        data = valid_data.copy()
        data["slope"] = invalid_slope

        with pytest.raises(ValidationError) as exc_info:
            PatientData(**data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "slope" in error_fields
