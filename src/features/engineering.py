"""Feature engineering functions for Heart Disease Prediction.

This module provides reusable functions for data transformation,
feature creation, and encoding for the heart disease prediction model.
"""

from typing import List, Union

import numpy as np
import pandas as pd


def create_age_group(age: Union[int, pd.Series]) -> Union[int, pd.Series]:
    """Categorize age into risk groups.

    Age groups are defined as:
    - 0: 20-40 years (young adult)
    - 1: 41-55 years (middle age)
    - 2: 56-70 years (senior)
    - 3: 71-100 years (elderly)

    Args:
        age: Patient age(s) in years. Can be a single integer or pandas Series.

    Returns:
        Age group category (0-3) as int or Series matching input type.

    Examples:
        >>> create_age_group(35)
        0
        >>> create_age_group(60)
        2
    """
    bins = [0, 40, 55, 70, 100]
    labels = [0, 1, 2, 3]

    if isinstance(age, pd.Series):
        return pd.cut(age, bins=bins, labels=labels).astype(int)

    # Handle single value
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if low < age <= high:
            return labels[i]

    # Edge case: age exactly 0 or below first bin
    if age <= bins[0]:
        return labels[0]

    return labels[-1]


def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived risk indicator features.

    Creates the following features:
    - age_group: Categorized age (0-3)
    - chol_risk: High cholesterol indicator (>240 mg/dl)
    - bp_risk: High blood pressure indicator (>140 mm Hg)
    - heart_rate_reserve: Difference between max predicted HR and achieved HR

    Args:
        df: DataFrame with patient data containing 'age', 'chol',
            'trestbps', and 'thalach' columns.

    Returns:
        DataFrame with additional risk feature columns added.
    """
    df = df.copy()
    df["age_group"] = create_age_group(df["age"])
    df["chol_risk"] = (df["chol"] > 240).astype(int)
    df["bp_risk"] = (df["trestbps"] > 140).astype(int)
    df["heart_rate_reserve"] = 220 - df["age"] - df["thalach"]
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical variables.

    Encodes the following categorical columns:
    - cp: Chest pain type (1-4)
    - restecg: Resting ECG results (0-2)
    - slope: ST segment slope (1-3)
    - thal: Thalassemia type (3, 6, 7)

    Uses drop_first=False to maintain all categories, then manually
    drops the first category to match training data format.

    Note: Categorical columns are converted to float before encoding
    to produce column names with '.0' suffix (e.g., 'cp_2.0') matching
    the training data format.

    Args:
        df: DataFrame with categorical columns to encode.

    Returns:
        DataFrame with one-hot encoded features replacing original
        categorical columns.
    """
    df = df.copy()
    categorical_cols = ["cp", "restecg", "slope", "thal"]

    # Only encode columns that exist in the dataframe
    cols_to_encode = [col for col in categorical_cols if col in df.columns]

    if cols_to_encode:
        # Convert to float to match training data column naming (e.g., 'cp_2.0')
        for col in cols_to_encode:
            df[col] = df[col].astype(float)
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    return df


def get_feature_columns() -> List[str]:
    """Return ordered list of feature column names.

    Returns the exact feature names in the order expected by the
    trained model. This includes:
    - Original numeric features (9)
    - Derived risk features (4)
    - One-hot encoded categorical features (9)

    Note: One-hot encoded column names include '.0' suffix to match
    the training data format produced by pandas get_dummies with
    float-typed categorical columns.

    Returns:
        List of 22 feature column names in model-expected order.
    """
    return [
        # Original numeric features
        "age",
        "sex",
        "trestbps",
        "chol",
        "fbs",
        "thalach",
        "exang",
        "oldpeak",
        "ca",
        # Derived risk features
        "age_group",
        "chol_risk",
        "bp_risk",
        "heart_rate_reserve",
        # One-hot encoded categorical features (drop_first=True)
        # Note: '.0' suffix matches training data format
        "cp_2.0",
        "cp_3.0",
        "cp_4.0",  # cp: 1 is dropped
        "restecg_1.0",
        "restecg_2.0",  # restecg: 0 is dropped
        "slope_2.0",
        "slope_3.0",  # slope: 1 is dropped
        "thal_6.0",
        "thal_7.0",  # thal: 3 is dropped
    ]


def prepare_features(patient_data: dict) -> np.ndarray:
    """Prepare single patient data for prediction.

    Takes a dictionary of patient features and transforms it into
    a feature array ready for model input. This includes:
    1. Creating a DataFrame from the input
    2. Adding derived risk features
    3. One-hot encoding categorical variables
    4. Ensuring all expected columns exist
    5. Ordering columns to match model expectations

    Args:
        patient_data: Dictionary of patient features with keys matching
            the PatientData schema fields.

    Returns:
        NumPy array of shape (1, 22) with features in model-expected order.
        All values are numeric with no NaN values.

    Raises:
        KeyError: If required fields are missing from patient_data.
    """
    # Create DataFrame from single patient record
    df = pd.DataFrame([patient_data])

    # Add derived risk features
    df = create_risk_features(df)

    # One-hot encode categorical variables
    df = encode_categorical(df)

    # Get expected feature columns
    expected_cols = get_feature_columns()

    # Ensure all expected columns exist (fill missing with 0)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Select and order columns to match model expectations
    feature_array = df[expected_cols].values

    return feature_array
