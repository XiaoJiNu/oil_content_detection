"""光谱预处理工具集，支持串联多个步骤（SNV、MSC、SG、一阶导、归一化、去趋势等）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from scipy import signal  # type: ignore
    from scipy.signal import savgol_filter  # type: ignore
except Exception:  # pragma: no cover
    signal = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]

from oil_content_detection.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessStep:
    """描述单个预处理步骤。"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


def snv(X: np.ndarray) -> np.ndarray:
    """标准正态变量变换（逐样本零均值、单位方差）。"""
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def msc(X: np.ndarray) -> np.ndarray:
    """多元散射校正，使用整体平均光谱作为参考。"""
    X = np.asarray(X, dtype=float)
    ref = np.nanmean(X, axis=0, keepdims=True)
    corrected = np.empty_like(X)
    for i, spectrum in enumerate(X):
        coeffs, *_ = np.linalg.lstsq(ref.T, spectrum, rcond=None)
        slope = coeffs[0] if np.ndim(coeffs) > 0 else coeffs
        intercept = np.nanmean(spectrum - slope * ref)
        corrected[i] = (spectrum - intercept) / max(slope, 1e-8)
    return corrected


def normalize(X: np.ndarray) -> np.ndarray:
    """按每条光谱的最大值进行归一化。"""
    X = np.asarray(X, dtype=float)
    max_val = np.nanmax(X, axis=1, keepdims=True)
    max_val = np.where(max_val == 0, 1.0, max_val)
    return X / max_val


def detrend(X: np.ndarray) -> np.ndarray:
    """按波长维做一次线性去趋势。"""
    X = np.asarray(X, dtype=float)
    if signal is not None:
        return signal.detrend(X, axis=1, type="linear")

    # Fallback: vectorized least-squares detrend (linear) along axis=1.
    n_features = X.shape[1]
    if n_features <= 1:
        return X.copy()
    x = np.arange(n_features, dtype=float)
    x_mean = float(x.mean())
    x_var = float(((x - x_mean) ** 2).mean())
    if x_var == 0:
        return X.copy()

    y_mean = np.nanmean(X, axis=1, keepdims=True)
    cov = np.nanmean((X - y_mean) * (x - x_mean), axis=1, keepdims=True)
    slope = cov / x_var
    intercept = y_mean - slope * x_mean
    trend = slope * x.reshape(1, -1) + intercept
    return X - trend


def _savgol_coeffs(window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")
    if deriv < 0:
        raise ValueError("deriv must be non-negative")
    if deriv > polyorder:
        return np.zeros((window_length,), dtype=float)

    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coeffs = pinv[deriv] * float(factorial(deriv))
    return coeffs.astype(float)


def savgol(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """Savitzky-Golay 平滑/导数."""
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    if window_length > n_features:
        window_length = n_features if n_features % 2 == 1 else n_features - 1
    if window_length < 3:
        logger.warning("window_length adjusted to minimum odd value; returning original data")
        return X.copy()
    if window_length % 2 == 0:
        window_length += 1
    if savgol_filter is not None:
        return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)

    coeffs = _savgol_coeffs(window_length=window_length, polyorder=polyorder, deriv=deriv)
    half = window_length // 2
    padded = np.pad(X, ((0, 0), (half, half)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_length, axis=1)
    return np.tensordot(windows, coeffs, axes=([2], [0]))


def _to_steps(steps: Sequence[PreprocessStep | str]) -> List[PreprocessStep]:
    parsed: List[PreprocessStep] = []
    for step in steps:
        if isinstance(step, PreprocessStep):
            parsed.append(step)
        elif isinstance(step, str):
            parsed.append(PreprocessStep(name=step))
        else:
            raise TypeError(f"Unsupported preprocess step type: {type(step)}")
    return parsed


def apply_preprocessing_pipeline(X: np.ndarray, steps: Sequence[PreprocessStep | str]) -> np.ndarray:
    """按顺序应用预处理步骤。"""
    result = np.asarray(X, dtype=float)
    for step in _to_steps(steps):
        name = step.name.lower()
        params = step.params or {}
        if name == "snv":
            result = snv(result)
        elif name == "msc":
            result = msc(result)
        elif name in {"sg", "savgol"}:
            result = savgol(result, **params)
        elif name == "normalize":
            result = normalize(result)
        elif name == "detrend":
            result = detrend(result)
        else:
            raise ValueError(f"Unknown preprocess step: {step.name}")
    return result


__all__ = [
    "PreprocessStep",
    "apply_preprocessing_pipeline",
    "detrend",
    "msc",
    "normalize",
    "savgol",
    "snv",
]
