"""Centralized configuration management using Pydantic Settings.

This module provides type-safe configuration for the Heart Disease
Prediction application, including paths, model settings, API settings,
and feature validation ranges.
"""

from pathlib import Path
from typing import Tuple

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration.

    Attributes:
        model_path: Path to the trained model pickle file.
        data_path: Path to the heart disease dataset.
        model_version: Semantic version of the model.
        random_state: Random seed for reproducibility.
        test_size: Fraction of data to use for testing.
        api_host: Host address for the API server.
        api_port: Port number for the API server.
        age_range: Valid range for patient age.
        trestbps_range: Valid range for resting blood pressure.
        chol_range: Valid range for serum cholesterol.
        thalach_range: Valid range for maximum heart rate.
        oldpeak_range: Valid range for ST depression.
    """

    # Paths
    model_path: Path = Path("models/heart_disease_model.pkl")
    data_path: Path = Path("data/heart_disease.csv")

    # Model settings
    model_version: str = "1.0.0"
    random_state: int = 42
    test_size: float = 0.2

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Feature validation ranges
    age_range: Tuple[int, int] = (20, 100)
    trestbps_range: Tuple[int, int] = (80, 200)
    chol_range: Tuple[int, int] = (100, 600)
    thalach_range: Tuple[int, int] = (60, 220)
    oldpeak_range: Tuple[float, float] = (0.0, 7.0)

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


# Global settings instance
settings = Settings()
