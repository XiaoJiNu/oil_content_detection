"""ROI segmentation utilities for huajiao (peppercorn) images.

This module provides two segmentation modes:

- ``legacy``: a lightweight color-based heuristic that segments paper, label,
  and pepper pixels. This is mainly kept for compatibility and unit tests.
- ``gray_otsu``: the "gemini-灰度方案" described in the reference docs:
  CLAHE enhancement + Otsu thresholding inside a provided ROI mask, followed by
  morphology and small-hole filling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PepperROIConfig:
    """Configuration for pepper ROI segmentation."""

    segmentation_mode: str = "legacy"  # {"legacy", "gray_otsu"}

    # --- legacy mode (color/texture) ---
    use_texture: bool = False
    texture_min_pixels: int = 300
    texture_min_keep_ratio: float = 0.02

    paper_gray_threshold: int = 30
    bg_dilate_size: int = 3

    label_fallback_enabled: bool = True
    label_max_area_ratio: float = 0.12
    label_gray_threshold: int = 220
    label_close_size: int = 7
    label_dilate_size: int = 11
    label_final_dilate_size: int = 21

    pepper_open_size: int = 3
    pepper_close_size: int = 5
    min_component_area: int = 64

    # --- gray_otsu mode (gemini) ---
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    otsu_invert_area_ratio: float = 0.5
    gray_close_size: int = 5
    gray_open_size: int = 5
    gray_final_close_size: int = 5
    max_hole_area: int = 300
    min_keep_area: int = 64


@dataclass(frozen=True)
class PepperSegmentationResult:
    paper_mask: np.ndarray  # bool, HxW
    label_mask: np.ndarray  # bool, HxW
    pepper_mask: np.ndarray  # bool, HxW


def _as_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.shape != shape:
        raise ValueError(f"Mask shape {m.shape} does not match image shape {shape}")
    return m


def _kernel(size: int) -> Optional[np.ndarray]:
    if size is None or size <= 1:
        return None
    return np.ones((size, size), dtype=np.uint8)


def _keep_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = [i + 1 for i, c in enumerate(counts) if c >= min_area]
    return np.isin(labels, keep)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    best = int(np.argmax(counts) + 1)
    return labels == best


def _otsu_threshold_uint8(values: np.ndarray) -> int:
    """Compute Otsu threshold for uint8 values (1D/ND)."""
    vals = np.asarray(values, dtype=np.uint8).ravel()
    if vals.size == 0:
        return 0
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    total = float(vals.size)
    sum_total = float(np.dot(np.arange(256), hist))

    sum_bg = 0.0
    weight_bg = 0.0
    best_var = -1.0
    best_t = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += float(t) * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return int(best_t)


def _segment_gray_otsu(image_bgr: np.ndarray, roi_mask: np.ndarray, cfg: PepperROIConfig) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for gray_otsu segmentation") from exc

    h, w = image_bgr.shape[:2]
    roi_mask = _as_bool_mask(roi_mask, (h, w))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=tuple(cfg.clahe_tile_grid_size))
    gray_enhanced = clahe.apply(gray)

    roi_vals = gray_enhanced[roi_mask]
    thr = _otsu_threshold_uint8(roi_vals)

    pepper = (gray_enhanced > thr) & roi_mask
    area_ratio = pepper.sum() / float(max(roi_mask.sum(), 1))
    if area_ratio > cfg.otsu_invert_area_ratio:
        pepper = (~pepper) & roi_mask

    # Morphology: close -> open -> close (within ROI).
    for op, k in (
        (cv2.MORPH_CLOSE, cfg.gray_close_size),
        (cv2.MORPH_OPEN, cfg.gray_open_size),
        (cv2.MORPH_CLOSE, cfg.gray_final_close_size),
    ):
        ker = _kernel(k)
        if ker is None:
            continue
        pepper_u8 = (pepper.astype(np.uint8) * 255)
        pepper_u8 = cv2.morphologyEx(pepper_u8, op, ker)
        pepper = (pepper_u8 > 0) & roi_mask

    # Fill small holes only.
    filled = ndimage.binary_fill_holes(pepper)
    holes = filled & (~pepper) & roi_mask
    if cfg.max_hole_area and holes.any():
        hole_labels, hole_n = ndimage.label(holes)
        if hole_n > 0:
            hole_sizes = ndimage.sum(holes, hole_labels, index=np.arange(1, hole_n + 1))
            small = [i + 1 for i, s in enumerate(hole_sizes) if s <= cfg.max_hole_area]
            if small:
                pepper = pepper | np.isin(hole_labels, small)

    pepper = _keep_components(pepper, min_area=cfg.min_keep_area)
    return pepper & roi_mask


def _segment_legacy(image_bgr: np.ndarray, roi_mask: np.ndarray, cfg: PepperROIConfig) -> PepperSegmentationResult:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for legacy segmentation") from exc

    h, w = image_bgr.shape[:2]
    roi_mask = _as_bool_mask(roi_mask, (h, w))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Label detection (white label + red border).
    if cfg.label_fallback_enabled:
        b, g, r = cv2.split(image_bgr)
        white = (gray >= cfg.label_gray_threshold) & roi_mask
        red_border = (r >= 220) & (g <= 90) & (b <= 90) & roi_mask
        label = white | red_border

        ker_close = _kernel(cfg.label_close_size)
        if ker_close is not None:
            label_u8 = cv2.morphologyEx(label.astype(np.uint8) * 255, cv2.MORPH_CLOSE, ker_close)
            label = label_u8 > 0

        ker_dilate = _kernel(cfg.label_dilate_size)
        if ker_dilate is not None:
            label_u8 = cv2.dilate(label.astype(np.uint8) * 255, ker_dilate)
            label = label_u8 > 0

        label = label & roi_mask
        area_ratio = label.sum() / float(max(roi_mask.sum(), 1))
        if cfg.label_max_area_ratio and area_ratio > cfg.label_max_area_ratio:
            label = np.zeros_like(label, dtype=bool)
    else:
        label = np.zeros((h, w), dtype=bool)

    # Pepper pixel detection (simple color rules covering green and purple).
    b, g, r = cv2.split(image_bgr)
    green = (g >= 150) & (r <= 220) & (b <= 220)
    purple = (b >= 80) & (r >= 80) & (g <= 80)
    pepper = (green | purple) & roi_mask & (~label)

    ker_open = _kernel(cfg.pepper_open_size)
    if ker_open is not None:
        pepper_u8 = cv2.morphologyEx(pepper.astype(np.uint8) * 255, cv2.MORPH_OPEN, ker_open)
        pepper = pepper_u8 > 0
    ker_close = _kernel(cfg.pepper_close_size)
    if ker_close is not None:
        pepper_u8 = cv2.morphologyEx(pepper.astype(np.uint8) * 255, cv2.MORPH_CLOSE, ker_close)
        pepper = pepper_u8 > 0

    pepper = pepper & roi_mask & (~label)
    pepper = _keep_components(pepper, min_area=cfg.min_component_area)

    # Optional texture refinement (kept minimal; must not crash).
    if cfg.use_texture and pepper.sum() >= cfg.texture_min_pixels:
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_abs = np.abs(lap)
        vals = lap_abs[pepper]
        if vals.size:
            thr = np.quantile(vals, max(0.0, 1.0 - cfg.texture_min_keep_ratio))
            pepper = pepper & (lap_abs >= thr)
            pepper = _keep_components(pepper, min_area=cfg.min_component_area)

    # Final dilation of label to guarantee exclusion.
    if cfg.label_fallback_enabled and cfg.label_final_dilate_size and label.any():
        ker_final = _kernel(cfg.label_final_dilate_size)
        if ker_final is not None:
            label_u8 = cv2.dilate(label.astype(np.uint8) * 255, ker_final)
            label = (label_u8 > 0) & roi_mask
            pepper = pepper & (~label)

    return PepperSegmentationResult(paper_mask=roi_mask, label_mask=label, pepper_mask=pepper)


def segment_pepper_in_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    config: PepperROIConfig | None = None,
) -> PepperSegmentationResult:
    """Segment pepper pixels inside a provided ROI mask.

    Args:
        image_bgr: BGR image array (H, W, 3).
        roi_mask: boolean/0-1 mask selecting the ROI (H, W).
        config: segmentation configuration.
    """
    cfg = config or PepperROIConfig()
    if cfg.segmentation_mode.lower() == "gray_otsu":
        pepper = _segment_gray_otsu(image_bgr, roi_mask, cfg)
        label = np.zeros_like(pepper, dtype=bool)
        return PepperSegmentationResult(paper_mask=_as_bool_mask(roi_mask, pepper.shape), label_mask=label, pepper_mask=pepper)
    return _segment_legacy(image_bgr, roi_mask, cfg)


def segment_pepper_roi(image_bgr: np.ndarray, config: PepperROIConfig | None = None) -> PepperSegmentationResult:
    """Segment pepper pixels from a full image by first locating the paper ROI."""
    cfg = config or PepperROIConfig()
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for ROI segmentation") from exc

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper = gray >= int(cfg.paper_gray_threshold)

    ker = _kernel(cfg.bg_dilate_size)
    if ker is not None:
        paper_u8 = cv2.morphologyEx(paper.astype(np.uint8) * 255, cv2.MORPH_CLOSE, ker)
        paper = paper_u8 > 0

    paper = ndimage.binary_fill_holes(paper)
    paper = _largest_component(paper)
    return segment_pepper_in_mask(image_bgr, paper, config=cfg)


__all__ = [
    "PepperROIConfig",
    "PepperSegmentationResult",
    "segment_pepper_roi",
    "segment_pepper_in_mask",
]

