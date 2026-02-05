"""Pydantic validation schemas for Heart Disease Prediction API.

This module defines input validation schemas with detailed error messages
and output schemas for API responses.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):
    """Input schema for patient health data.

    Validates all 13 patient features required for heart disease prediction.
    Each field includes range constraints and descriptive error messages.
    """

    age: int = Field(..., ge=20, le=100, description="Patient age in years (20-100)")
    sex: Literal[0, 1] = Field(..., description="Biological sex (0=Female, 1=Male)")
    cp: Literal[1, 2, 3, 4] = Field(
        ...,
        description="Chest pain type (1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic)",
    )
    trestbps: int = Field(
        ..., ge=80, le=200, description="Resting blood pressure in mm Hg (80-200)"
    )
    chol: int = Field(
        ..., ge=100, le=600, description="Serum cholesterol in mg/dl (100-600)"
    )
    fbs: Literal[0, 1] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (0=False, 1=True)"
    )
    restecg: Literal[0, 1, 2] = Field(
        ...,
        description="Resting ECG results (0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy)",
    )
    thalach: int = Field(
        ..., ge=60, le=220, description="Maximum heart rate achieved (60-220)"
    )
    exang: Literal[0, 1] = Field(
        ..., description="Exercise induced angina (0=No, 1=Yes)"
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=7.0,
        description="ST depression induced by exercise relative to rest (0.0-7.0)",
    )
    slope: Literal[1, 2, 3] = Field(
        ...,
        description="Slope of peak exercise ST segment (1=Upsloping, 2=Flat, 3=Downsloping)",
    )
    ca: Literal[0, 1, 2, 3] = Field(
        ..., description="Number of major vessels colored by fluoroscopy (0-3)"
    )
    thal: Literal[3, 6, 7] = Field(
        ..., description="Thalassemia (3=Normal, 6=Fixed defect, 7=Reversible defect)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    """Output schema for prediction results.

    Contains the model prediction, probability score, risk level,
    and SHAP-based feature importance values.
    """

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction result (0=No heart disease, 1=Heart disease)",
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of heart disease (0.0-1.0)"
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., description="Risk category based on probability thresholds"
    )
    feature_importance: Dict[str, float] = Field(
        ..., description="SHAP values for each feature in this prediction"
    )


class HealthResponse(BaseModel):
    """Health check response schema.

    Used by the /health endpoint to report API and model status.
    """

    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(
        ..., description="Whether the ML model is loaded and ready"
    )


class ModelInfoResponse(BaseModel):
    """Model metadata response schema.

    Used by the /model-info endpoint to provide model details.
    """

    version: str = Field(..., description="Model version string")
    model_type: str = Field(..., description="Type of ML model used")
    features: List[str] = Field(
        ..., description="List of feature names used by the model"
    )
    training_date: str = Field(..., description="Date when the model was trained")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
