"""Tests for model artifact loading."""

import joblib

from src.config import settings
from src.models.predict import HeartDiseasePredictor


def test_model_artifact_exists():
    """Ensure the model artifact exists on disk."""
    assert settings.model_path.exists()


def test_model_artifact_contains_required_keys():
    """Ensure the model artifact contains required keys."""
    artifacts = joblib.load(settings.model_path)

    assert "model" in artifacts
    assert "scaler" in artifacts


def test_predictor_initializes_from_artifacts():
    """Ensure the predictor loads model and scaler artifacts."""
    predictor = HeartDiseasePredictor()

    assert predictor.model is not None
    assert predictor.scaler is not None
