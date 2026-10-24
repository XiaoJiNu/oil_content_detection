"""Advanced deployable training helpers for Huajiao oil-content inference.

The serving contract still exposes only RGB/ENVI-derived base columns:
``wl_*`` wavelength means plus ROI quality fields.  This module keeps that
contract and performs any extra spectral summary engineering inside the fitted
model object, so the web predictor can continue passing the same one-row
``DataFrame`` produced from an uploaded cube.
"""
from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from oil_content_detection.inference.training import (
    DEPLOY_QUALITY_COLUMNS,
    FORBIDDEN_TRAINING_COLUMNS,
)
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline

SpectralFeatureSet = Literal["none", "summary", "rich"]
RegressorType = Literal["ridge", "huber", "elasticnet", "svr", "random_forest", "extra_trees", "gbr"]
PreprocessName = Literal["none", "snv", "sg1", "snv_sg1", "snv_sg0", "msc_sg1"]


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric ``wl_<nm>`` columns sorted by wavelength."""

    return sorted(
        [c for c in df.columns if c.startswith("wl_") and c[3:].isdigit()],
        key=lambda c: int(c.split("_", maxsplit=1)[1]),
    )


def _as_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("AdvancedHuajiaoRegressor expects a pandas DataFrame input")
    return X


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=np.abs(den) > 1e-12)


def _bin_indices(n_features: int, n_bins: int) -> list[np.ndarray]:
    bins = max(1, min(int(n_bins), int(n_features)))
    return [idx.astype(int) for idx in np.array_split(np.arange(n_features), bins) if idx.size > 0]


class AdvancedHuajiaoRegressor(BaseEstimator, RegressorMixin):
    """Deployable regressor with internal spectral summary features.

    Parameters are intentionally sklearn-style so the class can be cloned during
    grid search.  The public input schema remains the base deployable schema; no
    weight, batch, sample-id or target-derived field is accepted as a feature.
    """

    def __init__(
        self,
        *,
        n_components: int = 12,
        preprocess: PreprocessName = "snv_sg1",
        spectral_feature_set: SpectralFeatureSet = "summary",
        n_summary_bins: int = 8,
        regressor_type: RegressorType = "ridge",
        ridge_alpha: float = 1.0,
        huber_alpha: float = 0.001,
        huber_epsilon: float = 1.35,
        huber_max_iter: int = 3000,
        elasticnet_alpha: float = 0.01,
        elasticnet_l1_ratio: float = 0.2,
        svr_c: float = 1.0,
        svr_epsilon: float = 0.05,
        svr_gamma: str | float = "scale",
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 20260514,
    ) -> None:
        self.n_components = n_components
        self.preprocess = preprocess
        self.spectral_feature_set = spectral_feature_set
        self.n_summary_bins = n_summary_bins
        self.regressor_type = regressor_type
        self.ridge_alpha = ridge_alpha
        self.huber_alpha = huber_alpha
        self.huber_epsilon = huber_epsilon
        self.huber_max_iter = huber_max_iter
        self.elasticnet_alpha = elasticnet_alpha
        self.elasticnet_l1_ratio = elasticnet_l1_ratio
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon
        self.svr_gamma = svr_gamma
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Iterable[float]) -> "AdvancedHuajiaoRegressor":
        X_df = _as_frame(X).copy()
        forbidden = sorted(set(FORBIDDEN_TRAINING_COLUMNS) & set(X_df.columns))
        if forbidden:
            raise ValueError(f"forbidden deployment feature columns found: {forbidden}")
        wl_cols = wavelength_columns(X_df)
        if not wl_cols:
            raise ValueError("no wl_<nm> wavelength columns found")
        quality_cols = [c for c in DEPLOY_QUALITY_COLUMNS if c in X_df.columns]
        missing_quality = [c for c in DEPLOY_QUALITY_COLUMNS if c not in X_df.columns]
        if missing_quality:
            raise ValueError(f"missing deployable ROI quality columns: {missing_quality}")

        y_arr = np.asarray(list(y), dtype=float).reshape(-1)
        if len(X_df) != y_arr.shape[0]:
            raise ValueError(f"X/y row count mismatch: {len(X_df)} != {y_arr.shape[0]}")
        if y_arr.shape[0] < 3:
            raise ValueError("at least 3 samples are required")

        self.base_feature_columns_ = list(X_df.columns)
        self.wavelength_columns_ = wl_cols
        self.wavelength_numbers_ = np.asarray([int(c.split("_", maxsplit=1)[1]) for c in wl_cols], dtype=float)
        self.quality_columns_ = quality_cols

        raw = self._raw_spectra(X_df)
        proc = self._preprocess_spectra(raw)
        effective_n = max(1, min(int(self.n_components), proc.shape[1], y_arr.shape[0] - 1))
        self.effective_n_components_ = effective_n
        self.pls_ = PLSRegression(n_components=effective_n, scale=False)
        self.pls_.fit(proc, y_arr)

        features, names = self._engineered_matrix(X_df, raw=raw, proc=proc)
        self.engineered_feature_names_ = names
        self.scaler_ = StandardScaler()
        features_scaled = self.scaler_.fit_transform(features)
        self.regressor_ = self._make_regressor()
        self.regressor_.fit(features_scaled, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["base_feature_columns_", "pls_", "scaler_", "regressor_"])
        X_df = _as_frame(X)
        missing = [c for c in self.base_feature_columns_ if c not in X_df.columns]
        if missing:
            raise ValueError(f"missing required feature columns: {missing}")
        X_ordered = X_df[self.base_feature_columns_].copy()
        raw = self._raw_spectra(X_ordered)
        proc = self._preprocess_spectra(raw)
        features, _ = self._engineered_matrix(X_ordered, raw=raw, proc=proc)
        pred = self.regressor_.predict(self.scaler_.transform(features))
        return np.asarray(pred, dtype=float).reshape(-1)

    def _raw_spectra(self, X: pd.DataFrame) -> np.ndarray:
        return _finite(X[self.wavelength_columns_].to_numpy(dtype=float))

    def _preprocess_steps(self) -> list[PreprocessStep]:
        if self.preprocess == "none":
            return []
        if self.preprocess == "snv":
            return [PreprocessStep("snv")]
        if self.preprocess == "sg1":
            return [PreprocessStep("savgol", {"window_length": 13, "polyorder": 2, "deriv": 1})]
        if self.preprocess == "snv_sg1":
            return [
                PreprocessStep("snv"),
                PreprocessStep("savgol", {"window_length": 13, "polyorder": 2, "deriv": 1}),
            ]
        if self.preprocess == "snv_sg0":
            return [
                PreprocessStep("snv"),
                PreprocessStep("savgol", {"window_length": 13, "polyorder": 2, "deriv": 0}),
            ]
        if self.preprocess == "msc_sg1":
            return [
                PreprocessStep("msc"),
                PreprocessStep("savgol", {"window_length": 13, "polyorder": 2, "deriv": 1}),
            ]
        raise ValueError(f"unsupported preprocess={self.preprocess!r}")

    def _preprocess_spectra(self, raw: np.ndarray) -> np.ndarray:
        steps = self._preprocess_steps()
        if not steps:
            return raw.copy()
        return _finite(apply_preprocessing_pipeline(raw, steps))

    def _engineered_matrix(
        self,
        X: pd.DataFrame,
        *,
        raw: np.ndarray,
        proc: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        parts: list[np.ndarray] = []
        names: list[str] = []

        pls = _finite(self.pls_.transform(proc))
        parts.append(pls)
        names.extend([f"pls_{idx + 1}" for idx in range(pls.shape[1])])

        quality = _finite(X[self.quality_columns_].to_numpy(dtype=float))
        parts.append(quality)
        names.extend(self.quality_columns_)
        quality_extra, quality_names = self._quality_features(X)
        parts.append(quality_extra)
        names.extend(quality_names)

        if self.spectral_feature_set in {"summary", "rich"}:
            summary, summary_names = self._spectral_summary_features(raw, proc)
            parts.append(summary)
            names.extend(summary_names)
        elif self.spectral_feature_set != "none":
            raise ValueError(
                f"unsupported spectral_feature_set={self.spectral_feature_set!r}; expected 'none', 'summary', or 'rich'"
            )

        return np.hstack(parts).astype(float), names

    def _quality_features(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        coverage = _finite(X["coverage_ratio"].to_numpy(dtype=float)).reshape(-1, 1)
        log_valid = _finite(X["log_valid_pixel_count"].to_numpy(dtype=float)).reshape(-1, 1)
        valid_ratio = _finite(X["valid_pixel_ratio"].to_numpy(dtype=float)).reshape(-1, 1)
        features = np.hstack(
            [
                coverage**2,
                log_valid**2,
                coverage * log_valid,
                coverage * valid_ratio,
            ]
        )
        names = [
            "coverage_ratio_sq",
            "log_valid_pixel_count_sq",
            "coverage_x_log_valid_pixel_count",
            "coverage_x_valid_pixel_ratio",
        ]
        return features, names

    def _spectral_summary_features(self, raw: np.ndarray, proc: np.ndarray) -> tuple[np.ndarray, list[str]]:
        features: list[np.ndarray] = []
        names: list[str] = []

        def add_column(name: str, values: np.ndarray) -> None:
            features.append(_finite(values).reshape(-1, 1))
            names.append(name)

        for prefix, arr in [("raw", raw), ("proc", proc)]:
            add_column(f"{prefix}_mean", np.mean(arr, axis=1))
            add_column(f"{prefix}_std", np.std(arr, axis=1))
            add_column(f"{prefix}_q10", np.quantile(arr, 0.10, axis=1))
            add_column(f"{prefix}_q50", np.quantile(arr, 0.50, axis=1))
            add_column(f"{prefix}_q90", np.quantile(arr, 0.90, axis=1))
            add_column(f"{prefix}_ptp", np.ptp(arr, axis=1))
            add_column(f"{prefix}_abs_mean", np.mean(np.abs(arr), axis=1))
            add_column(f"{prefix}_abs_max", np.max(np.abs(arr), axis=1))

        for bin_idx, idx in enumerate(_bin_indices(raw.shape[1], self.n_summary_bins), start=1):
            for prefix, arr in [("raw", raw), ("proc", proc)]:
                block = arr[:, idx]
                add_column(f"{prefix}_bin{bin_idx}_mean", np.mean(block, axis=1))
                add_column(f"{prefix}_bin{bin_idx}_std", np.std(block, axis=1))

        if self.spectral_feature_set == "rich":
            rich_features, rich_names = self._rich_ratio_features(raw)
            features.append(rich_features)
            names.extend(rich_names)

        return np.hstack(features), names

    def _nearest_wavelength_index(self, wavelength: float) -> int:
        return int(np.argmin(np.abs(self.wavelength_numbers_ - float(wavelength))))

    def _rich_ratio_features(self, raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
        ratio_pairs = [(550, 700), (650, 800), (700, 900), (800, 970), (550, 970)]
        slope_pairs = [(550, 700), (700, 900), (900, 970)]
        parts: list[np.ndarray] = []
        names: list[str] = []
        for left, right in ratio_pairs:
            i = self._nearest_wavelength_index(left)
            j = self._nearest_wavelength_index(right)
            parts.append(_safe_divide(raw[:, j], raw[:, i]).reshape(-1, 1))
            names.append(f"raw_ratio_{int(self.wavelength_numbers_[j])}_over_{int(self.wavelength_numbers_[i])}")
        for left, right in slope_pairs:
            i = self._nearest_wavelength_index(left)
            j = self._nearest_wavelength_index(right)
            denom = float(self.wavelength_numbers_[j] - self.wavelength_numbers_[i]) or 1.0
            parts.append(((raw[:, j] - raw[:, i]) / denom).reshape(-1, 1))
            names.append(f"raw_slope_{int(self.wavelength_numbers_[i])}_{int(self.wavelength_numbers_[j])}")
        return np.hstack(parts), names

    def _make_regressor(self):
        if self.regressor_type == "ridge":
            return Ridge(alpha=float(self.ridge_alpha))
        if self.regressor_type == "huber":
            return HuberRegressor(
                alpha=float(self.huber_alpha),
                epsilon=float(self.huber_epsilon),
                max_iter=int(self.huber_max_iter),
            )
        if self.regressor_type == "elasticnet":
            return ElasticNet(
                alpha=float(self.elasticnet_alpha),
                l1_ratio=float(self.elasticnet_l1_ratio),
                max_iter=10000,
                random_state=int(self.random_state),
            )
        if self.regressor_type == "svr":
            return SVR(C=float(self.svr_c), epsilon=float(self.svr_epsilon), gamma=self.svr_gamma)
        if self.regressor_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=int(self.n_estimators),
                max_depth=self.max_depth,
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=int(self.random_state),
                n_jobs=-1,
            )
        if self.regressor_type == "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=int(self.n_estimators),
                max_depth=self.max_depth,
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=int(self.random_state),
                n_jobs=-1,
            )
        if self.regressor_type == "gbr":
            return GradientBoostingRegressor(
                n_estimators=int(self.n_estimators),
                max_depth=2 if self.max_depth is None else int(self.max_depth),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=int(self.random_state),
            )
        raise ValueError(f"unsupported regressor_type={self.regressor_type!r}")


__all__ = ["AdvancedHuajiaoRegressor", "wavelength_columns"]
