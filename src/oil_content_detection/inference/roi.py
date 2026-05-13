"""Traditional ROI inference wrapper for RGB Huajiao images."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oil_content_detection.preprocessing.roi_segmentation import (
    PepperROIConfig,
    PepperROIResult,
    segment_pepper_in_mask,
    segment_pepper_roi,
)

from .schema import PredictionError


_ROI_CONFIG_FIELD_NAMES = {field.name for field in fields(PepperROIConfig)}


def pepper_config_from_dict(config: dict[str, Any] | None) -> PepperROIConfig:
    """Build PepperROIConfig from artifact JSON, ignoring serving-only keys."""

    raw = dict(config or {})
    segmentation = raw.get("segmentation")
    if isinstance(segmentation, dict):
        raw = dict(segmentation)
    kwargs = {k: v for k, v in raw.items() if k in _ROI_CONFIG_FIELD_NAMES}
    return PepperROIConfig(**kwargs)


def load_rgb_image_bgr(image_path: str | Path) -> np.ndarray:
    """Read an RGB/BGR display image for ROI segmentation."""

    path = Path(image_path)
    if not path.exists():
        raise PredictionError("INPUT_FILE_NOT_FOUND", f"RGB image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PredictionError("INPUT_FILE_NOT_FOUND", f"Cannot read RGB image: {path}")
    return image


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def _save_overlay(image_bgr: np.ndarray, mask: np.ndarray, overlay_path: Path) -> None:
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    color = np.zeros_like(image_bgr, dtype=np.uint8)
    color[:, :, 1] = 255
    blended = cv2.addWeighted(image_bgr, 0.70, color, 0.30, 0.0)
    out = image_bgr.copy()
    out[mask.astype(bool)] = blended[mask.astype(bool)]
    cv2.imwrite(str(overlay_path), out)


def save_roi_debug_outputs(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    mask_path: str | Path | None,
    overlay_path: str | Path | None,
) -> None:
    """Save final ROI mask and overlay debug files when paths are provided."""

    if mask_path is not None:
        mask_file = Path(mask_path)
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_file), _mask_u8(mask))
    if overlay_path is not None:
        _save_overlay(image_bgr, mask, Path(overlay_path))


def _dinov3_post_refinement_config(
    roi_config: dict[str, Any] | None
) -> dict[str, Any] | None:
    cfg = dict(roi_config or {})
    dinov3_cfg = cfg.get("dinov3")
    raw_post = None
    if isinstance(dinov3_cfg, dict) and isinstance(
        dinov3_cfg.get("post_refinement"), dict
    ):
        raw_post = dinov3_cfg["post_refinement"]
    elif isinstance(cfg.get("post_refinement"), dict):
        raw_post = cfg["post_refinement"]
    if not isinstance(raw_post, dict) or not bool(raw_post.get("enabled", False)):
        return None

    post = dict(raw_post)
    mode = post.pop("mode", None)
    if mode is not None and "segmentation_mode" not in post:
        post["segmentation_mode"] = mode
    post.pop("enabled", None)
    return post


def refine_mask_in_roi_if_configured(
    *,
    image_bgr: np.ndarray,
    base_mask: np.ndarray,
    roi_config: dict[str, Any] | None,
    base_mask_source: str = "dinov3_roi_segmentation",
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Optionally refine a DINOv3 coarse polygon mask using an in-mask ROI segmenter."""

    mask = np.asarray(base_mask, dtype=bool)
    post = _dinov3_post_refinement_config(roi_config)
    if post is None:
        return mask, {}, base_mask_source

    mode = str(post.get("segmentation_mode", "color_texture"))
    try:
        result = segment_pepper_in_mask(
            image_bgr=image_bgr,
            roi_mask=mask,
            config=pepper_config_from_dict(post),
        )
    except ValueError as exc:
        raise PredictionError("ROI_SEGMENTATION_FAILED", str(exc)) from exc

    refined = np.asarray(result.pepper_mask, dtype=bool)
    info = dict(result.info)
    info.update(
        {
            "post_refinement_enabled": 1.0,
            "post_refinement_mode": mode,
            "dinov3_polygon_coverage_ratio": float(mask.mean()),
            "post_refined_coverage_ratio": float(refined.mean()),
        }
    )
    return refined, info, f"dinov3_roi_{mode}_refined"


def segment_image_roi(
    *,
    image_bgr: np.ndarray,
    image_stem: str,
    roi_config: dict[str, Any] | None = None,
    debug_dir: str | Path | None = None,
) -> tuple[PepperROIResult, str | None, str | None]:
    """Run traditional ROI segmentation and optionally save mask/overlay files."""

    try:
        result = segment_pepper_roi(
            image_bgr, config=pepper_config_from_dict(roi_config)
        )
    except ValueError as exc:
        raise PredictionError("ROI_SEGMENTATION_FAILED", str(exc)) from exc

    if int(result.pepper_mask.sum()) <= 0:
        raise PredictionError(
            "ROI_EMPTY", "Traditional ROI segmentation returned an empty pepper mask."
        )

    mask_path: str | None = None
    overlay_path: str | None = None
    if debug_dir is not None:
        debug_root = Path(debug_dir)
        debug_root.mkdir(parents=True, exist_ok=True)
        mask_file = debug_root / f"{image_stem}_pepper_mask.png"
        overlay_file = debug_root / f"{image_stem}_roi_overlay.png"
        cv2.imwrite(str(mask_file), _mask_u8(result.pepper_mask))
        _save_overlay(image_bgr, result.pepper_mask, overlay_file)
        mask_path = str(mask_file)
        overlay_path = str(overlay_file)
    return result, mask_path, overlay_path


def enforce_roi_quality(
    roi_pixels: int, coverage_ratio: float, roi_config: dict[str, Any] | None
) -> list[str]:
    """Apply serving ROI quality gates; return non-fatal warnings."""

    cfg = dict(roi_config or {})
    min_pixels = int(cfg.get("min_pepper_pixels", 1))
    min_coverage = float(cfg.get("min_coverage_ratio", 0.0))
    max_coverage = float(cfg.get("max_coverage_ratio", 1.0))
    if roi_pixels < min_pixels:
        raise PredictionError(
            "ROI_LOW_COVERAGE",
            f"ROI has too few pepper pixels: {roi_pixels} < {min_pixels}",
        )
    if coverage_ratio < min_coverage:
        raise PredictionError(
            "ROI_LOW_COVERAGE",
            f"ROI coverage is too low: {coverage_ratio:.6f} < {min_coverage:.6f}",
        )
    warnings: list[str] = []
    if coverage_ratio > max_coverage:
        warnings.append(
            f"ROI coverage is higher than configured max: {coverage_ratio:.6f} > {max_coverage:.6f}"
        )
    return warnings
