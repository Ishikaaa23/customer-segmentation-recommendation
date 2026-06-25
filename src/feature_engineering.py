"""
Feature Engineering Module
Scales and prepares customer-level features for PCA and clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder


NUMERIC_FEATURES = [
    "Recency", "Frequency", "Monetary",
    "AvgOrderValue", "AvgItemsPerOrder",
    "UniqueProducts", "UniqueCategories",
    "PurchaseSpan", "PreferredHour", "CountryCount",
]


def select_numeric_features(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return only the numeric features available in the profile,
    filling any missing values with the column median.
    """
    available = [c for c in NUMERIC_FEATURES if c in profile.columns]
    X = profile[available].copy()
    X.fillna(X.median(), inplace=True)
    return X


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to right-skewed columns (Monetary, Frequency, UniqueProducts, etc.)
    to reduce the effect of outliers before scaling.
    """
    skewed_cols = ["Recency","Monetary", "Frequency", "AvgOrderValue",
                   "UniqueProducts", "AvgItemsPerOrder", "PurchaseSpan"]
    X = X.copy()
    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    """
    Scale features using RobustScaler (resistant to outliers).
    Returns (scaled_array, fitted_scaler).
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def prepare_features(profile: pd.DataFrame) -> tuple[np.ndarray, list[str], RobustScaler]:
    """
    End-to-end feature preparation:
      1. Select numeric features
      2. Log-transform skewed columns
      3. RobustScale

    Returns (X_scaled, feature_names, scaler).
    """
    X = select_numeric_features(profile)
    feature_names = list(X.columns)
    X = log_transform(X)
    X_scaled, scaler = scale_features(X)
    print(f"[FeatureEngineering] Prepared {X_scaled.shape[0]:,} samples × "
          f"{X_scaled.shape[1]} features.")
    return X_scaled, feature_names, scaler
