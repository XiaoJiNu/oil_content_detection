"""Compatibility sklearn transformers for historical spectral models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression

from .spectral import PreprocessStep, apply_preprocessing_pipeline


@dataclass
class _FittedStep:
    """Stores per-step fitted state for preprocessing pipelines."""

    step: PreprocessStep
    state: Dict[str, Any] = field(default_factory=dict)


class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around spectral preprocessing steps."""

    def __init__(self, steps: Sequence[PreprocessStep | str] = ()) -> None:
        # Keep the constructor parameter unchanged so sklearn.clone can rebuild
        # this transformer inside Pipeline/ColumnTransformer artifacts.
        self.steps = steps
        self.fitted_steps_: list[_FittedStep] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpectralPreprocessor":
        current = np.asarray(X, dtype=float)
        self.fitted_steps_ = []
        for raw_step in self.steps:
            step = raw_step if isinstance(raw_step, PreprocessStep) else PreprocessStep(name=str(raw_step))
            name = step.name.lower()
            if name == "msc":
                ref = np.nanmean(current, axis=0, keepdims=True)
                self.fitted_steps_.append(_FittedStep(step=step, state={"reference": ref}))
                current = self._apply_step(current, self.fitted_steps_[-1])
            else:
                fitted = _FittedStep(step=step, state={})
                self.fitted_steps_.append(fitted)
                current = self._apply_step(current, fitted)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        current = np.asarray(X, dtype=float)
        for fitted in self.fitted_steps_:
            current = self._apply_step(current, fitted)
        return current

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, **fit_params: Any) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X)

    @staticmethod
    def _apply_step(X: np.ndarray, fitted: _FittedStep) -> np.ndarray:
        name = fitted.step.name.lower()
        if name == "msc":
            ref = np.asarray(fitted.state["reference"], dtype=float)
            corrected = np.empty_like(X, dtype=float)
            ref_col = ref.reshape(-1, 1)
            ref_vec = ref.reshape(-1)
            for i, spectrum in enumerate(X):
                coeffs, *_ = np.linalg.lstsq(ref_col, spectrum, rcond=None)
                slope = float(coeffs[0]) if np.ndim(coeffs) > 0 else float(coeffs)
                intercept = float(np.nanmean(spectrum - slope * ref_vec))
                corrected[i] = (spectrum - intercept) / max(slope, 1e-8)
            return corrected
        return apply_preprocessing_pipeline(X, [fitted.step])


class PLSScoreTransformer(BaseEstimator, TransformerMixin):
    """Projects spectra to latent PLS scores for downstream models."""

    def __init__(self, n_components: int = 8, scale: bool = False) -> None:
        self.n_components = n_components
        self.scale = scale
        self.pls_: PLSRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSScoreTransformer":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        centered = X_arr - np.nanmean(X_arr, axis=0, keepdims=True)
        rank = int(np.linalg.matrix_rank(np.nan_to_num(centered)))
        n_components = min(self.n_components, max(1, X_arr.shape[0] - 1), X_arr.shape[1], max(1, rank))
        self.pls_ = PLSRegression(n_components=n_components, scale=self.scale)
        self.pls_.fit(X_arr, y_arr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pls_ is None:
            raise RuntimeError("PLSScoreTransformer must be fitted before transform")
        transformed = self.pls_.transform(np.asarray(X, dtype=float))
        if isinstance(transformed, tuple):
            return np.asarray(transformed[0], dtype=float)
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, **fit_params: Any) -> np.ndarray:
        if y is None:
            raise ValueError("PLSScoreTransformer requires y during fit_transform")
        return self.fit(X, y, **fit_params).transform(X)
