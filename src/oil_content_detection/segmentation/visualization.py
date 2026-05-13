"""Visualization helpers for ROI segmentation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_mask_png(mask: np.ndarray, path: str | Path) -> Path:
    """Save a boolean mask as an 8-bit PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask).astype(bool).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)
    return path


def save_overlay(
    image_path: str | Path,
    mask: np.ndarray,
    path: str | Path,
    color: tuple[int, int, int] = (255, 32, 32),
    alpha: float = 0.45,
) -> Path:
    """Save an RGB image with the ROI mask overlaid."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != (image.height, image.width):
        mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255).resize(
            image.size, resample=Image.Resampling.NEAREST
        )
        mask_arr = np.asarray(mask_img) > 0
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    overlay[mask_arr] = (1.0 - alpha) * overlay[mask_arr] + alpha * np.asarray(
        color, dtype=np.float32
    )
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)
    return path
