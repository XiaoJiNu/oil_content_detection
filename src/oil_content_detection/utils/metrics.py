"""Metric helpers.

尽量保持依赖轻量：这里的指标使用 NumPy 直接实现，避免在仅做 ROI/可视化等任务时
因为引入 scikit-learn/scipy 而导致额外的二进制依赖问题。
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}")
    diff = y_true_arr - y_pred_arr
    mse = float(np.mean(diff * diff))
    return float(np.sqrt(mse))


__all__ = ["rmse"]
