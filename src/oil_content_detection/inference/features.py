"""Feature extraction for mode-B Huajiao inference (ENVI cube + RGB ROI)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import PredictionError, ROIMetrics


QUALITY_FEATURES = {"coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio"}


def validate_same_view_shape(image_shape_hw: tuple[int, int], cube_shape_hw: tuple[int, int]) -> None:
    """Require ROI mask/image plane and hyperspectral cube to share height/width."""

    if tuple(image_shape_hw) != tuple(cube_shape_hw):
        raise PredictionError(
            "CUBE_IMAGE_SHAPE_MISMATCH",
            "ROI mask and hyperspectral cube must have the same height/width after RGB/cube alignment.",
            {"image_shape_hw": tuple(image_shape_hw), "cube_shape_hw": tuple(cube_shape_hw)},
        )


def _validate_columns(feature_columns: Iterable[str], wavelength_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    feature_cols = list(feature_columns)
    wl_cols = list(wavelength_columns)
    if not wl_cols:
        raise PredictionError("WAVELENGTH_SCHEMA_MISMATCH", "No wavelength columns were provided by the model artifact.")
    if len(set(feature_cols)) != len(feature_cols):
        raise PredictionError("WAVELENGTH_SCHEMA_MISMATCH", "Feature columns contain duplicates.")
    missing = [c for c in wl_cols if c not in feature_cols]
    if missing:
        raise PredictionError("WAVELENGTH_SCHEMA_MISMATCH", "Wavelength columns must be part of feature_columns.", {"missing": missing})
    unsupported = [c for c in feature_cols if not c.startswith("wl_") and c not in QUALITY_FEATURES]
    if unsupported:
        raise PredictionError(
            "UNSUPPORTED_FEATURE_SCHEMA",
            "This deployment supports only wl_* and ROI quality features derived from uploaded files.",
            {"unsupported_features": unsupported},
        )
    return feature_cols, wl_cols


def extract_roi_spectrum_features(
    *,
    cube: np.ndarray,
    mask: np.ndarray,
    wavelength_columns: list[str],
    feature_columns: list[str],
    mask_path: str | None = None,
    overlay_path: str | None = None,
    mask_source: str = "traditional_roi_segmentation",
    roi_info: dict | None = None,
) -> tuple[dict[str, float], ROIMetrics]:
    """Aggregate ROI pixels into one model feature row."""

    feature_cols, wl_cols = _validate_columns(feature_columns, wavelength_columns)
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise PredictionError("UNSUPPORTED_SPECTRAL_FORMAT", f"Expected cube with shape (height,width,bands), got {cube.shape}")
    mask = np.asarray(mask, dtype=bool)
    validate_same_view_shape(mask.shape[:2], cube.shape[:2])
    if cube.shape[2] != len(wl_cols):
        raise PredictionError(
            "WAVELENGTH_SCHEMA_MISMATCH",
            "Cube band count does not match model wavelength schema.",
            {"cube_bands": int(cube.shape[2]), "schema_wavelength_count": len(wl_cols)},
        )
    pixel_count = int(mask.sum())
    if pixel_count <= 0:
        raise PredictionError("ROI_EMPTY", "Traditional ROI segmentation returned an empty pepper mask.")

    pixels = np.asarray(cube[mask], dtype=float)
    finite = np.isfinite(pixels).all(axis=1)
    valid_pixels = pixels[finite]
    valid_pixel_count = int(valid_pixels.shape[0])
    if valid_pixel_count <= 0:
        raise PredictionError("ROI_EMPTY", "No finite spectral pixels remained inside the ROI mask.")

    spectrum = np.nanmean(valid_pixels, axis=0)
    coverage_ratio = float(pixel_count) / float(mask.size)
    valid_pixel_ratio = float(valid_pixel_count) / float(pixel_count)
    row: dict[str, float] = {col: float(value) for col, value in zip(wl_cols, spectrum, strict=True)}
    row.update(
        {
            "coverage_ratio": coverage_ratio,
            "log_valid_pixel_count": float(np.log1p(valid_pixel_count)),
            "valid_pixel_ratio": valid_pixel_ratio,
        }
    )
    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise PredictionError("UNSUPPORTED_FEATURE_SCHEMA", "Cannot produce required feature(s) from cube + RGB image.", {"missing": missing})

    roi = ROIMetrics(
        mask_source=mask_source,
        pepper_pixels=pixel_count,
        pixel_count=pixel_count,
        valid_pixel_count=valid_pixel_count,
        coverage_ratio=coverage_ratio,
        valid_pixel_ratio=valid_pixel_ratio,
        mask_path=mask_path,
        overlay_path=overlay_path,
        info=dict(roi_info or {}),
    )
    return {c: row[c] for c in feature_cols}, roi


def feature_row_to_frame(row: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    """Convert one feature row to a one-row DataFrame in model order."""

    missing = [c for c in feature_columns if c not in row]
    if missing:
        raise PredictionError("UNSUPPORTED_FEATURE_SCHEMA", "Feature row is missing model columns.", {"missing": missing})
    return pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)
