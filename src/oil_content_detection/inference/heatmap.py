"""Heatmap generation for Huajiao oil-content model-response visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .artifact import ModelArtifact
from .features import QUALITY_FEATURES
from .schema import ROIMetrics, _json_safe


@dataclass(frozen=True)
class HeatmapResult:
    """Files and statistics produced for one model-response heatmap."""

    heatmap_path: Path
    overlay_path: Path
    stats_path: Path
    stats: dict[str, Any]
    kind: str = "model_response_heatmap"
    note: str = (
        "Heatmap shows relative model response from ROI pixel spectra; it is not "
        "independently calibrated pixel-level oil-content ground truth."
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "kind": self.kind,
                "heatmap_path": str(self.heatmap_path),
                "overlay_path": str(self.overlay_path),
                "stats_path": str(self.stats_path),
                "stats": self.stats,
                "note": self.note,
            }
        )


def _ensure_same_hw(
    name: str, shape_hw: tuple[int, int], expected_hw: tuple[int, int]
) -> None:
    if tuple(shape_hw) != tuple(expected_hw):
        raise ValueError(
            f"{name} must have the same shape as cube height/width: {shape_hw} != {expected_hw}"
        )


def _select_roi_pixels(
    mask: np.ndarray, max_pixels: int | None
) -> tuple[np.ndarray, bool]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot generate heatmap for an empty ROI mask.")
    if max_pixels is None or max_pixels <= 0 or len(coords) <= max_pixels:
        return coords, False
    positions = np.linspace(0, len(coords) - 1, int(max_pixels), dtype=int)
    return coords[positions], True


def _feature_frame_for_pixels(
    *,
    spectra: np.ndarray,
    artifact: ModelArtifact,
    roi: ROIMetrics,
) -> pd.DataFrame:
    feature_columns = artifact.feature_columns
    wavelength_columns = artifact.wavelength_columns
    if spectra.shape[1] != len(wavelength_columns):
        raise ValueError(
            "Cube band count does not match model wavelength schema for heatmap generation: "
            f"{spectra.shape[1]} vs {len(wavelength_columns)}"
        )

    data: dict[str, np.ndarray] = {}
    for idx, column in enumerate(wavelength_columns):
        data[column] = spectra[:, idx].astype(float, copy=False)

    quality_values = {
        "coverage_ratio": float(roi.coverage_ratio),
        "log_valid_pixel_count": float(np.log1p(roi.valid_pixel_count)),
        "valid_pixel_ratio": float(roi.valid_pixel_ratio),
    }
    for column in feature_columns:
        if column.startswith("wl_"):
            continue
        if column not in QUALITY_FEATURES:
            raise ValueError(f"Unsupported heatmap feature column: {column}")
        data[column] = np.full(spectra.shape[0], quality_values[column], dtype=float)

    return pd.DataFrame({column: data[column] for column in feature_columns})


def _predict_in_batches(model: Any, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        pred = np.asarray(model.predict(batch), dtype=float).reshape(-1)
        chunks.append(pred)
    if not chunks:
        return np.empty((0,), dtype=float)
    return np.concatenate(chunks)


def _stats(
    values: np.ndarray, *, roi_pixel_count: int, pixels_used: int, sampled: bool
) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Heatmap model predictions contain no finite values.")
    return {
        "roi_pixel_count": int(roi_pixel_count),
        "pixels_used": int(pixels_used),
        "sampled": bool(sampled),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def _colored_heatmap(
    prediction_map: np.ndarray, p05: float, p95: float
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(prediction_map)
    denom = float(p95 - p05)
    normalized = np.zeros(prediction_map.shape, dtype=np.float32)
    if abs(denom) < 1e-12:
        normalized[finite_mask] = 0.5
    else:
        normalized[finite_mask] = np.clip(
            (prediction_map[finite_mask] - p05) / denom, 0.0, 1.0
        )
    color_bgr = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heatmap_bgra = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2BGRA)
    heatmap_bgra[:, :, 3] = np.where(finite_mask, 255, 0).astype(np.uint8)
    return color_bgr, heatmap_bgra


def _overlay_heatmap(
    image_bgr: np.ndarray, color_bgr: np.ndarray, finite_mask: np.ndarray, alpha: float
) -> np.ndarray:
    overlay = image_bgr.copy()
    if finite_mask.any():
        base = overlay[finite_mask].astype(np.float32)
        color = color_bgr[finite_mask].astype(np.float32)
        blended = base * float(1.0 - alpha) + color * float(alpha)
        overlay[finite_mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay


def generate_oil_response_heatmap(
    *,
    cube: np.ndarray,
    mask: np.ndarray,
    artifact: ModelArtifact,
    roi: ROIMetrics,
    image_bgr: np.ndarray,
    output_dir: str | Path,
    max_pixels: int | None = 200_000,
    batch_size: int = 50_000,
    overlay_alpha: float = 0.45,
) -> HeatmapResult:
    """Generate a model-response heatmap from ROI pixel spectra.

    This visualizes how the sample-level regression model responds to individual ROI
    pixel spectra. It is useful for demos and QC, but it is not pixel-level ground truth.
    """

    cube = np.asarray(cube)
    mask = np.asarray(mask, dtype=bool)
    image_bgr = np.asarray(image_bgr)
    if cube.ndim != 3:
        raise ValueError(f"cube must be a 3D array, got shape {cube.shape}")
    _ensure_same_hw("mask", mask.shape[:2], cube.shape[:2])
    _ensure_same_hw("image_bgr", image_bgr.shape[:2], cube.shape[:2])

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    coords_all = np.argwhere(mask)
    coords, sampled = _select_roi_pixels(mask, max_pixels)
    spectra = np.asarray(cube[coords[:, 0], coords[:, 1], :], dtype=float)
    finite_rows = np.isfinite(spectra).all(axis=1)
    coords = coords[finite_rows]
    spectra = spectra[finite_rows]
    if len(coords) == 0:
        raise ValueError("No finite spectral pixels are available inside the ROI mask.")

    frame = _feature_frame_for_pixels(spectra=spectra, artifact=artifact, roi=roi)
    predictions = _predict_in_batches(artifact.model, frame, batch_size=batch_size)

    prediction_map = np.full(mask.shape, np.nan, dtype=np.float32)
    prediction_map[coords[:, 0], coords[:, 1]] = predictions.astype(np.float32)
    stats = _stats(
        predictions,
        roi_pixel_count=int(coords_all.shape[0]),
        pixels_used=int(len(coords)),
        sampled=sampled or int(len(coords)) < int(coords_all.shape[0]),
    )

    color_bgr, heatmap_bgra = _colored_heatmap(
        prediction_map, stats["p05"], stats["p95"]
    )
    overlay = _overlay_heatmap(
        image_bgr, color_bgr, np.isfinite(prediction_map), overlay_alpha
    )

    heatmap_path = output_root / "oil_heatmap.png"
    overlay_path = output_root / "oil_heatmap_overlay.png"
    stats_path = output_root / "oil_heatmap_stats.json"
    cv2.imwrite(str(heatmap_path), heatmap_bgra)
    cv2.imwrite(str(overlay_path), overlay)

    result = HeatmapResult(
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
        stats_path=stats_path,
        stats=stats,
    )
    stats_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["HeatmapResult", "generate_oil_response_heatmap"]
