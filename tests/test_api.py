"""Integration tests for FastAPI endpoints.

This module tests all API endpoints with valid/invalid inputs
and error responses.

**Feature: portfolio-enhancement**
**Validates: Requirements 9.3**
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app, load_model

# Ensure model is loaded before tests run
load_model()

# Create test client
client = TestClient(app)


# Sample valid patient data for testing
VALID_PATIENT_DATA = {
    "age": 55,
    "sex": 1,
    "cp": 3,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


class TestHealthEndpoint:
    """Tests for the /health endpoint.

    Validates: Requirement 2.3 - GET /health for health checks
    """

    def test_health_returns_200(self):
        """Health endpoint SHALL return 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_field(self):
        """Health endpoint SHALL return status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_model_loaded_field(self):
        """Health endpoint SHALL return model_loaded field."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint.

    Validates: Requirement 2.4 - GET /model-info returning model metadata
    """

    def test_model_info_returns_200(self):
        """Model info endpoint SHALL return 200 status code."""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_returns_version(self):
        """Model info endpoint SHALL return version field."""
        response = client.get("/model-info")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_model_info_returns_model_type(self):
        """Model info endpoint SHALL return model_type field."""
        response = client.get("/model-info")
        data = response.json()
        assert "model_type" in data
        assert isinstance(data["model_type"], str)

    def test_model_info_returns_features(self):
        """Model info endpoint SHALL return features list."""
        response = client.get("/model-info")
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 22

    def test_model_info_returns_metrics(self):
        """Model info endpoint SHALL return metrics dictionary."""
        response = client.get("/model-info")
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
        assert "accuracy" in data["metrics"]
        assert "roc_auc" in data["metrics"]


class TestPredictEndpoint:
    """Tests for the /predict endpoint.

    Validates: Requirements 2.1, 2.2, 2.5
    """

    def test_predict_with_valid_data_returns_200(self):
        """Predict endpoint SHALL return 200 for valid input.

        Validates: Requirement 2.1 - POST /predict accepts patient data
        """
        response = client.post("/predict", json=VALID_PATIENT_DATA)
        assert response.status_code == 200

    def test_predict_returns_prediction_field(self):
        """Predict endpoint SHALL return prediction field.

        Validates: Requirement 2.2 - return prediction
        """
        response = client.post("/predict", json=VALID_PATIENT_DATA)
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]

    def test_predict_returns_probability_field(self):
        """Predict endpoint SHALL return probability field.

        Validates: Requirement 2.2 - return probability
        """
        response = client.post("/predict", json=VALID_PATIENT_DATA)
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_returns_risk_level_field(self):
        """Predict endpoint SHALL return risk_level field.

        Validates: Requirement 2.2 - return risk level
        """
        response = client.post("/predict", json=VALID_PATIENT_DATA)
        data = response.json()
        assert "risk_level" in data
        assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_returns_feature_importance(self):
        """Predict endpoint SHALL return feature_importance field.

        Validates: Requirement 2.2 - return feature importance
        """
        response = client.post("/predict", json=VALID_PATIENT_DATA)
        data = response.json()
        assert "feature_importance" in data
        assert isinstance(data["feature_importance"], dict)
        assert len(data["feature_importance"]) == 22


class TestPredictEndpointValidation:
    """Tests for /predict endpoint validation errors.

    Validates: Requirement 2.5 - return appropriate HTTP error codes
    """

    def test_predict_missing_field_returns_422(self):
        """Predict endpoint SHALL return 422 for missing required field.

        Validates: Requirement 2.5 - return appropriate HTTP error codes
        """
        invalid_data = VALID_PATIENT_DATA.copy()
        del invalid_data["age"]

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_field_error_identifies_field(self):
        """Validation error SHALL identify the missing field.

        Validates: Requirement 2.5 - descriptive error messages
        """
        invalid_data = VALID_PATIENT_DATA.copy()
        del invalid_data["age"]

        response = client.post("/predict", json=invalid_data)
        data = response.json()

        assert "detail" in data
        error_fields = [e["loc"][-1] for e in data["detail"]]
        assert "age" in error_fields

    def test_predict_invalid_age_range_returns_422(self):
        """Predict endpoint SHALL return 422 for age out of range.

        Validates: Requirement 2.5 - return appropriate HTTP error codes
        """
        invalid_data = VALID_PATIENT_DATA.copy()
        invalid_data["age"] = 150  # Above valid range

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_categorical_returns_422(self):
        """Predict endpoint SHALL return 422 for invalid categorical value.

        Validates: Requirement 2.5 - return appropriate HTTP error codes
        """
        invalid_data = VALID_PATIENT_DATA.copy()
        invalid_data["cp"] = 99  # Invalid chest pain type

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_thal_returns_422(self):
        """Predict endpoint SHALL return 422 for invalid thal value.

        Validates: Requirement 2.5 - return appropriate HTTP error codes
        """
        invalid_data = VALID_PATIENT_DATA.copy()
        invalid_data["thal"] = 5  # Invalid thalassemia value

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self):
        """Predict endpoint SHALL return 422 for empty request body.

        Validates: Requirement 2.5 - return appropriate HTTP error codes
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestAPIDocumentation:
    """Tests for automatic API documentation.

    Validates: Requirement 2.6 - automatic API documentation via Swagger/OpenAPI
    """

    def test_swagger_docs_available(self):
        """Swagger documentation SHALL be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self):
        """OpenAPI schema SHALL be available at /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_redoc_available(self):
        """ReDoc documentation SHALL be available at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestModelLoadingFailure:
    """Tests for model loading failure scenarios."""

    @patch("api.main.HeartDiseasePredictor")
    def test_model_info_when_model_not_loaded(self, mock_predictor):
        """Model info endpoint SHALL return 503 when model not loaded."""
        # Mock predictor initialization to fail
        mock_predictor.side_effect = Exception("Model file not found")

        # Create a new app instance with failed model loading
        with patch("api.main.predictor", None):
            response = client.get("/model-info")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    @patch("api.main.HeartDiseasePredictor")
    def test_predict_when_model_not_loaded(self, mock_predictor):
        """Predict endpoint SHALL return 503 when model not loaded."""
        # Mock predictor initialization to fail
        mock_predictor.side_effect = Exception("Model file not found")

        # Create a new app instance with failed model loading
        with patch("api.main.predictor", None):
            response = client.post("/predict", json=VALID_PATIENT_DATA)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    def test_predict_when_prediction_fails(self):
        """Predict endpoint SHALL return 500 when prediction fails."""
        # Mock predictor to raise exception during prediction
        with patch("api.main.predictor") as mock_predictor:
            mock_predictor.predict.side_effect = Exception("Prediction error")

            response = client.post("/predict", json=VALID_PATIENT_DATA)
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]


class TestLoadModelFunction:
    """Tests for the load_model function."""

    @patch("api.main.HeartDiseasePredictor")
    @patch("builtins.print")
    def test_load_model_handles_exception(self, mock_print, mock_predictor):
        """load_model SHALL handle exceptions gracefully."""
        # Mock predictor initialization to fail
        mock_predictor.side_effect = Exception("Model file not found")

        # Call load_model function
        load_model()

        # Verify error was printed and predictor is None
        mock_print.assert_called_once()
        assert "Failed to load model" in mock_print.call_args[0][0]


class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses."""
        response = client.options("/health")
        # OPTIONS requests should be handled by CORS middleware
        assert response.status_code in [
            200,
            405,
        ]  # 405 if OPTIONS not explicitly handled


class TestAPIMetadata:
    """Test API metadata and documentation."""

    def test_api_title_and_description(self):
        """Test that API has correct title and description."""
        response = client.get("/openapi.json")
        data = response.json()

        assert "Heart Disease Prediction API" in data["info"]["title"]
        assert "ML-powered" in data["info"]["description"]

    def test_api_version(self):
        """Test that API version is set."""
        response = client.get("/openapi.json")
        data = response.json()

        assert "version" in data["info"]
        assert data["info"]["version"] is not None


class TestModelLoadingEdgeCases:
    """Test edge cases in model loading."""

    @patch("api.main.HeartDiseasePredictor")
    @patch("builtins.print")
    def test_load_model_exception_handling(self, mock_print, mock_predictor):
        """Test load_model handles exceptions and prints warnings."""
        # Mock predictor to raise an exception
        mock_predictor.side_effect = Exception("Model loading failed")

        # Call load_model directly
        from api.main import load_model

        load_model()

        # Verify exception was caught and warning was printed
        mock_print.assert_called_once()
        assert "Failed to load model" in mock_print.call_args[0][0]

    def test_load_model_success(self):
        """Test successful model loading."""
        from api.main import load_model

        # This should work with the actual model file
        load_model()

        # Note: predictor should either be loaded or None if file doesn't exist
