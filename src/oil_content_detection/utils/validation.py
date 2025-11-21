"""Data validation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when data validation fails."""

    pass


def validate_spectral_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame contains required spectral data columns.

    Args:
        df: DataFrame to validate

    Raises:
        DataValidationError: If validation fails
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty")

    if "oil_content" not in df.columns:
        raise DataValidationError("Missing required column: 'oil_content'")

    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    if len(feature_cols) == 0:
        raise DataValidationError("No wavelength columns found (columns starting with 'wl_')")

    # Validate oil content range
    oil_content = df["oil_content"].values
    if np.any(oil_content < 0):
        raise DataValidationError("Oil content values must be non-negative")

    if np.any(oil_content > 100):
        raise DataValidationError("Oil content values must be <= 100%")

    # Check for NaN values
    if df[feature_cols].isnull().any().any():
        raise DataValidationError("Wavelength columns contain NaN values")

    if df["oil_content"].isnull().any():
        raise DataValidationError("Oil content column contains NaN values")


def validate_spectral_array(X: np.ndarray, y: np.ndarray) -> None:
    """Validate spectral feature matrix and target vector.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)

    Raises:
        DataValidationError: If validation fails
    """
    if X.ndim != 2:
        raise DataValidationError(f"X must be 2D array, got shape {X.shape}")

    if y.ndim != 1:
        raise DataValidationError(f"y must be 1D array, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise DataValidationError(f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}")

    if X.shape[0] == 0:
        raise DataValidationError("X contains zero samples")

    if X.shape[1] == 0:
        raise DataValidationError("X contains zero features")

    if np.any(np.isnan(X)):
        raise DataValidationError("X contains NaN values")

    if np.any(np.isnan(y)):
        raise DataValidationError("y contains NaN values")

    if np.any(y < 0) or np.any(y > 100):
        raise DataValidationError("Oil content (y) must be in range [0, 100]")


__all__ = ["DataValidationError", "validate_spectral_dataframe", "validate_spectral_array"]