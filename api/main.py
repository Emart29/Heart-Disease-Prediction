"""FastAPI application for Heart Disease Prediction API.

This module provides REST API endpoints for heart disease prediction,
including health checks, model information, and prediction endpoints.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

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

# Custom business metrics
prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of ML predictions made",
    ["prediction_result", "risk_level"],
)

prediction_confidence = Histogram(
    "ml_prediction_confidence",
    "Confidence score of ML predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

model_accuracy_gauge = Gauge(
    "ml_model_accuracy", "Current model accuracy from training"
)

feature_importance_gauge = Gauge(
    "ml_feature_importance", "Feature importance scores", ["feature_name"]
)

# Set initial model accuracy (from training)
model_accuracy_gauge.set(0.885)


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

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


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

        # Track business metrics
        prediction_result = "positive" if result["prediction"] == 1 else "negative"
        risk_level = result["risk_level"].lower()

        # Update counters
        prediction_counter.labels(
            prediction_result=prediction_result, risk_level=risk_level
        ).inc()

        # Track confidence/probability
        prediction_confidence.observe(result["probability"])

        # Update feature importance metrics (top 5 features)
        feature_importance = result.get("feature_importance", {})
        top_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        for feature_name, importance in top_features:
            feature_importance_gauge.labels(feature_name=feature_name).set(
                abs(importance)
            )

        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
