"""光谱预处理工具集，支持串联多个步骤（SNV、MSC、SG、一阶导、归一化、去趋势等）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter

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
    return signal.detrend(X, axis=1, type="linear")


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
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)


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
