"""FastAPI application for Heart Disease Prediction API.

This module provides REST API endpoints for heart disease prediction,
including health checks, model information, and prediction endpoints.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models.predict import HeartDiseasePredictor
from src.validation.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PatientData,
    PredictionResponse,
)

# Global predictor instance
predictor: HeartDiseasePredictor = None


def load_model():
    """Load the ML model."""
    global predictor
    try:
        predictor = HeartDiseasePredictor()
    except Exception as e:
        # Log error but don't crash - health endpoint will report model not loaded
        print(f"Warning: Failed to load model: {e}")
        predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: Load the model
    load_model()
    yield
    # Shutdown: cleanup if needed
    pass


# Create FastAPI application with lifespan
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered heart disease risk assessment API. "
    "Provides predictions with probability scores and SHAP-based feature importance.",
    version=settings.model_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and model health status.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Check API and model health.

    Returns:
        HealthResponse with status and model_loaded flag.
    """
    return HealthResponse(status="healthy", model_loaded=predictor is not None)


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get metadata about the loaded ML model.",
    tags=["Model"],
)
async def model_info() -> ModelInfoResponse:
    """Get model metadata.

    Returns:
        ModelInfoResponse with version, type, features, and metrics.

    Raises:
        HTTPException: If model is not loaded.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Service unavailable."
        )

    return ModelInfoResponse(
        version=settings.model_version,
        model_type="RandomForestClassifier",
        features=predictor.feature_names,
        training_date="2025-12-29",
        metrics={
            "accuracy": 0.885,
            "roc_auc": 0.954,
            "precision": 0.862,
            "recall": 0.893,
            "f1_score": 0.877,
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Heart Disease Risk",
    description="Make a heart disease prediction based on patient data. "
    "Returns prediction, probability, risk level, and feature importance.",
    tags=["Prediction"],
)
async def predict(patient: PatientData) -> PredictionResponse:
    """Make heart disease prediction.

    Args:
        patient: Validated patient data with all 13 required features.

    Returns:
        PredictionResponse with prediction, probability, risk_level,
        and feature_importance.

    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Service unavailable."
        )

    try:
        result = predictor.predict(patient.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
