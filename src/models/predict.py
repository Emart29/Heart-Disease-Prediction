"""Heart disease prediction module with SHAP explanations.

This module provides the HeartDiseasePredictor class for making
predictions with the trained heart disease model, including
SHAP-based feature importance explanations.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import shap

from src.config import settings
from src.features.engineering import get_feature_columns, prepare_features


class HeartDiseasePredictor:
    """Heart disease prediction with SHAP explanations.

    This class loads a trained model and scaler, and provides methods
    for making predictions with SHAP-based feature importance values.

    Attributes:
        model: The trained scikit-learn model.
        scaler: The fitted StandardScaler for feature normalization.
        explainer: SHAP explainer for computing feature importance.
        feature_names: List of feature column names.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """Load model artifacts from disk.

        Args:
            model_path: Path to the model pickle file. If None, uses
                the default path from settings.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            KeyError: If the model file doesn't contain expected keys.
        """
        path = model_path or settings.model_path

        # Load model artifacts
        artifacts = joblib.load(path)
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]

        # Get feature names from artifacts or use default
        self.feature_names = artifacts.get("feature_names", get_feature_columns())

        # Initialize SHAP explainer based on model type
        # Use TreeExplainer for tree-based models (RandomForest, etc.)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, patient_data: Dict) -> Dict:
        """Make prediction with SHAP explanation.

        Takes validated patient data and returns a prediction with
        probability, risk level, and feature importance values.

        Args:
            patient_data: Dictionary of patient features matching
                the PatientData schema fields.

        Returns:
            Dictionary containing:
                - prediction: 0 (no heart disease) or 1 (heart disease)
                - probability: Probability of heart disease (0.0-1.0)
                - risk_level: "Low", "Medium", or "High"
                - feature_importance: Dict mapping feature names to SHAP values

        Raises:
            ValueError: If feature preparation fails.
        """
        # Prepare features
        features = prepare_features(patient_data)

        # Scale features using DataFrame to avoid sklearn warning about feature names
        import pandas as pd

        features_df = pd.DataFrame(features, columns=self.feature_names)
        features_scaled = self.scaler.transform(features_df)

        # Make prediction
        prediction = int(self.model.predict(features_scaled)[0])
        probability = float(self.model.predict_proba(features_scaled)[0][1])

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features_scaled)

        # Handle different SHAP output formats
        # TreeExplainer can return different formats depending on version:
        # - List of arrays [class_0, class_1] for older versions
        # - 3D array (n_samples, n_features, n_classes) for newer versions
        if isinstance(shap_values, list):
            # Take values for positive class (heart disease)
            shap_values_class1 = (
                shap_values[1] if len(shap_values) > 1 else shap_values[0]
            )
        elif shap_values.ndim == 3:
            # Shape is (n_samples, n_features, n_classes)
            # Take values for positive class (index 1)
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

        # Flatten to 1D array (shape is (1, n_features) -> (n_features,))
        if shap_values_class1.ndim > 1:
            shap_values_class1 = shap_values_class1[0]

        # Create feature importance dictionary
        feature_importance = dict(
            zip(self.feature_names, [float(v) for v in shap_values_class1])
        )

        # Determine risk level based on probability thresholds
        risk_level = self._get_risk_level(probability)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "feature_importance": feature_importance,
        }

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level from probability.

        Args:
            probability: Probability of heart disease (0.0-1.0).

        Returns:
            Risk level string: "Low", "Medium", or "High".
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
