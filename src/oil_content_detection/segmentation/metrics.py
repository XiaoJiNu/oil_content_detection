"""Metrics for binary ROI segmentation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _as_bool_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == np.bool_:
        return arr
    return arr > 0.5


def binary_segmentation_metrics(pred_mask, target_mask) -> dict[str, float]:
    """Compute binary segmentation metrics for one prediction/target pair."""

    pred = _as_bool_array(pred_mask).astype(bool)
    target = _as_bool_array(target_mask).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "pred_coverage": float(pred.mean()),
        "target_coverage": float(target.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate_metric_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {
        f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in keys
    }


def binary_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1).float()
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()
