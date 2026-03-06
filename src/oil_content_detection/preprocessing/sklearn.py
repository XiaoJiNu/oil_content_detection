"""scikit-learn compatible spectral preprocessing transformers.

The existing functional API in :mod:`oil_content_detection.preprocessing.spectral`
is convenient for offline experiments, but some steps (e.g. MSC) require fitting
statistics on the training split to avoid cross-validation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression

from oil_content_detection.preprocessing.spectral import (
    PreprocessStep,
    detrend,
    normalize,
    savgol,
    snv,
)


@dataclass
class _FittedStep:
    name: str
    params: Dict[str, Any]


def _msc_with_reference(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Apply MSC using a fixed reference spectrum fitted on training data."""
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float).reshape(1, -1)
    if X.shape[1] != ref.shape[1]:
        raise ValueError("MSC reference must have same feature dimension as X")

    corrected = np.empty_like(X)
    ref_col = ref.reshape(-1, 1)
    ref_vec = ref.reshape(-1)
    for i, spectrum in enumerate(X):
        coeffs, *_ = np.linalg.lstsq(ref_col, spectrum, rcond=None)
        slope = float(coeffs[0]) if np.ndim(coeffs) > 0 else float(coeffs)
        intercept = float(np.nanmean(spectrum - slope * ref_vec))
        corrected[i] = (spectrum - intercept) / max(slope, 1e-8)
    return corrected


class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    """Apply a fixed sequence of spectral preprocessing steps.

    Notes
    -----
    - SNV/SG/normalize/detrend are stateless (fit is a no-op).
    - MSC is stateful: it fits a reference spectrum on the training split and
      reuses it for all subsequent transforms.
    """

    def __init__(self, steps: Sequence[PreprocessStep | str] = ()) -> None:
        # IMPORTANT: do not copy/mutate parameters here; sklearn.clone requires
        # estimator parameters to be stored as-is.
        self.steps = steps

    def _to_steps(self) -> List[PreprocessStep]:
        parsed: List[PreprocessStep] = []
        for step in self.steps:
            if isinstance(step, PreprocessStep):
                parsed.append(step)
            elif isinstance(step, str):
                parsed.append(PreprocessStep(name=step))
            else:
                raise TypeError(f"Unsupported preprocess step type: {type(step)}")
        return parsed

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpectralPreprocessor":
        X = np.asarray(X, dtype=float)
        fitted_steps: List[_FittedStep] = []

        cur = X
        for step in self._to_steps():
            name = step.name.lower()
            params = dict(step.params or {})
            if name == "snv":
                cur = snv(cur)
                fitted_steps.append(_FittedStep(name="snv", params={}))
            elif name == "msc":
                ref = np.nanmean(cur, axis=0, keepdims=True)
                fitted_steps.append(_FittedStep(name="msc", params={"ref": ref}))
                cur = _msc_with_reference(cur, ref)
            elif name in {"sg", "savgol"}:
                cur = savgol(cur, **params)
                fitted_steps.append(_FittedStep(name="sg", params=params))
            elif name == "normalize":
                cur = normalize(cur)
                fitted_steps.append(_FittedStep(name="normalize", params={}))
            elif name == "detrend":
                cur = detrend(cur)
                fitted_steps.append(_FittedStep(name="detrend", params={}))
            else:
                raise ValueError(f"Unknown preprocess step: {step.name}")

        self.fitted_steps_ = fitted_steps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_steps_"):
            raise RuntimeError(
                "SpectralPreprocessor.fit must be called before transform"
            )

        X = np.asarray(X, dtype=float)
        cur = X
        for step in self.fitted_steps_:
            name = step.name
            if name == "snv":
                cur = snv(cur)
            elif name == "msc":
                ref = np.asarray(step.params["ref"], dtype=float)
                cur = _msc_with_reference(cur, ref)
            elif name == "sg":
                cur = savgol(cur, **(step.params or {}))
            elif name == "normalize":
                cur = normalize(cur)
            elif name == "detrend":
                cur = detrend(cur)
            else:
                raise ValueError(f"Unknown fitted step: {name}")
        return cur


class PLSScoreTransformer(BaseEstimator, TransformerMixin):
    """Project X to PLS latent scores (X-scores) for downstream models.

    scikit-learn's ``PLSRegression.fit_transform`` returns a tuple
    ``(X_scores, Y_scores)``, which is not compatible with :class:`~sklearn.pipeline.Pipeline`.
    This wrapper exposes a transformer API that always returns X-scores.
    """

    def __init__(self, n_components: int = 10, *, scale: bool = False) -> None:
        self.n_components = int(n_components)
        self.scale = bool(scale)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSScoreTransformer":
        self.model_ = PLSRegression(
            n_components=int(self.n_components), scale=bool(self.scale)
        )
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError(
                "PLSScoreTransformer.fit must be called before transform"
            )
        return self.model_.transform(X)


__all__ = ["PLSScoreTransformer", "SpectralPreprocessor"]
