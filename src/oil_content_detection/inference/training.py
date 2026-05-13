"""Training helpers for the deployable Huajiao serving artifact."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oil_content_detection.preprocessing import PreprocessStep
from oil_content_detection.preprocessing.sklearn import PLSScoreTransformer, SpectralPreprocessor

DEPLOY_QUALITY_COLUMNS = ["coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio"]
FORBIDDEN_TRAINING_COLUMNS = ["distill_ml", "oil_ml_per_gram", "oil_ml_per_100g", "weight_g", "batch", "sid_num"]


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    """Return wl_<nm> columns sorted by wavelength number."""

    return sorted([c for c in df.columns if c.startswith("wl_") and c[3:].isdigit()], key=lambda c: int(c.split("_")[1]))


def prepare_deployment_training_frame(
    df: pd.DataFrame,
    *,
    target: str = "oil_ml_per_100g",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build the no-weight/no-batch deployment training matrix.

    The returned features are restricted to values that mode-B serving can derive
    from uploaded RGB image + ENVI cube: wavelength means and ROI quality fields.
    """

    if target not in df.columns:
        raise ValueError(f"Target column missing: {target}")
    wl_cols = wavelength_columns(df)
    if not wl_cols:
        raise ValueError("No wl_<nm> wavelength columns found")
    required = ["coverage_ratio", "valid_pixel_count", "pixel_count"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing ROI quality source columns: {missing_required}")

    out = df.copy()
    out["log_valid_pixel_count"] = np.log1p(out["valid_pixel_count"].astype(float))
    out["valid_pixel_ratio"] = out["valid_pixel_count"].astype(float) / out["pixel_count"].astype(float)
    feature_columns = wl_cols + DEPLOY_QUALITY_COLUMNS
    forbidden_present = [c for c in FORBIDDEN_TRAINING_COLUMNS if c in feature_columns]
    if forbidden_present:
        raise ValueError(f"Forbidden deployment features selected: {forbidden_present}")
    X = out[feature_columns].astype(float)
    y = out[target].to_numpy(dtype=float)
    feature_schema: dict[str, Any] = {
        "feature_columns": feature_columns,
        "wavelength_columns": wl_cols,
        "quality_columns": list(DEPLOY_QUALITY_COLUMNS),
        "forbidden_columns": list(FORBIDDEN_TRAINING_COLUMNS),
    }
    return X, y, feature_schema


def build_deployment_pipeline(
    *,
    wavelength_columns: list[str],
    quality_columns: list[str] | None = None,
    n_components: int = 10,
    regressor_type: str = "huber",
    huber_alpha: float = 0.001,
    huber_max_iter: int = 2000,
    ridge_alpha: float = 10.0,
) -> Pipeline:
    """Build the report-derived deployable model family.

    Model family: SNV + SG first derivative (window=13, polyorder=2) + PLS score
    representation + ROI quality features + a compact linear regressor.
    """

    quality_columns = quality_columns or list(DEPLOY_QUALITY_COLUMNS)
    spectral_branch = Pipeline(
        steps=[
            (
                "preprocess",
                SpectralPreprocessor(
                    [
                        PreprocessStep("snv"),
                        PreprocessStep("savgol", {"window_length": 13, "polyorder": 2, "deriv": 1}),
                    ]
                ),
            ),
            ("pls", PLSScoreTransformer(n_components=n_components, scale=False)),
        ]
    )
    features = ColumnTransformer(
        transformers=[
            ("spectral", spectral_branch, list(wavelength_columns)),
            ("quality", "passthrough", list(quality_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if regressor_type == "huber":
        regressor = HuberRegressor(alpha=huber_alpha, max_iter=huber_max_iter)
    elif regressor_type == "ridge":
        regressor = Ridge(alpha=ridge_alpha)
    else:
        raise ValueError(f"Unsupported regressor_type={regressor_type!r}; expected 'huber' or 'ridge'.")
    return Pipeline(
        steps=[
            ("features", features),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics used in the experiment reports."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
