"""Weight regression model for estimating sample mass from ROI-derived features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class WeightRegressorConfig:
    """Configuration for the weight regression model."""

    model_type: Literal["linear", "pls", "random_forest"] = "linear"
    pls_components: int = 2
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None
    random_state: int = 2024


class WeightRegressor:
    """Regressor that predicts weight_g from simple ROI features."""

    def __init__(self, config: WeightRegressorConfig | None = None) -> None:
        self.config = config or WeightRegressorConfig()
        self.model: LinearRegression | PLSRegression | RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples")

        model_type = self.config.model_type
        if model_type == "linear":
            model: LinearRegression | PLSRegression | RandomForestRegressor = LinearRegression()
        elif model_type == "pls":
            n_components = min(
                self.config.pls_components,
                X_arr.shape[1],
                max(1, X_arr.shape[0] - 1),
            )
            model = PLSRegression(n_components=n_components, scale=False)
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_arr, y_arr)
        self.model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("WeightRegressor is not fitted")
        X_arr = np.asarray(X, dtype=float)
        pred = self.model.predict(X_arr)
        return np.asarray(pred, dtype=float).ravel()


__all__ = ["WeightRegressor", "WeightRegressorConfig"]

