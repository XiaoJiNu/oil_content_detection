from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from oil_content_detection.inference.artifact import load_model_artifact
from oil_content_detection.inference.heatmap import generate_oil_response_heatmap
from oil_content_detection.inference.schema import ROIMetrics

WL_COLUMNS = ["wl_400", "wl_500"]
QUALITY_COLUMNS = ["coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio"]
FEATURE_COLUMNS = WL_COLUMNS + QUALITY_COLUMNS


def _write_linear_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    X = np.array(
        [
            [0.0, 0.0, 0.25, np.log1p(4), 1.0],
            [1.0, 0.0, 0.25, np.log1p(4), 1.0],
            [0.0, 1.0, 0.25, np.log1p(4), 1.0],
            [1.0, 1.0, 0.25, np.log1p(4), 1.0],
        ],
        dtype=float,
    )
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    model = LinearRegression().fit(pd.DataFrame(X, columns=FEATURE_COLUMNS), y)
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "wavelength_columns": WL_COLUMNS,
                "quality_columns": QUALITY_COLUMNS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": "heatmap-test-model",
                "target_name": "oil_ml_per_100g",
                "unit": "ml/100g",
                "input_mode": "envi_cube_with_rgb_image",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "roi_config.json").write_text("{}", encoding="utf-8")
    return artifact_dir


def test_generate_oil_response_heatmap_writes_png_overlay_and_strict_json(
    tmp_path: Path,
):
    artifact = load_model_artifact(_write_linear_artifact(tmp_path))
    cube = np.zeros((4, 5, 2), dtype=np.float32)
    cube[..., 0] = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    cube[..., 1] = np.linspace(1.0, 0.0, 20, dtype=np.float32).reshape(4, 5)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    image_bgr = np.full((4, 5, 3), 80, dtype=np.uint8)
    roi = ROIMetrics(
        mask_source="unit_test_mask",
        pepper_pixels=int(mask.sum()),
        pixel_count=int(mask.sum()),
        valid_pixel_count=int(mask.sum()),
        coverage_ratio=float(mask.mean()),
        valid_pixel_ratio=1.0,
    )

    result = generate_oil_response_heatmap(
        cube=cube,
        mask=mask,
        artifact=artifact,
        roi=roi,
        image_bgr=image_bgr,
        output_dir=tmp_path / "visualizations",
        max_pixels=1000,
    )

    payload = result.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["kind"] == "model_response_heatmap"
    assert payload["stats"]["roi_pixel_count"] == int(mask.sum())
    assert payload["stats"]["pixels_used"] == int(mask.sum())
    assert payload["stats"]["sampled"] is False
    assert Path(payload["heatmap_path"]).exists()
    assert Path(payload["overlay_path"]).exists()
    assert Path(payload["stats_path"]).exists()
    assert (
        cv2.imread(payload["heatmap_path"], cv2.IMREAD_UNCHANGED).shape[:2]
        == mask.shape
    )
    assert (
        cv2.imread(payload["overlay_path"], cv2.IMREAD_UNCHANGED).shape[:2]
        == mask.shape
    )


def test_generate_oil_response_heatmap_samples_large_roi_and_keeps_shape(
    tmp_path: Path,
):
    artifact = load_model_artifact(_write_linear_artifact(tmp_path))
    cube = np.ones((6, 6, 2), dtype=np.float32)
    mask = np.ones((6, 6), dtype=bool)
    image_bgr = np.full((6, 6, 3), 120, dtype=np.uint8)
    roi = ROIMetrics(
        mask_source="unit_test_mask",
        pepper_pixels=36,
        pixel_count=36,
        valid_pixel_count=36,
        coverage_ratio=1.0,
        valid_pixel_ratio=1.0,
    )

    result = generate_oil_response_heatmap(
        cube=cube,
        mask=mask,
        artifact=artifact,
        roi=roi,
        image_bgr=image_bgr,
        output_dir=tmp_path / "sampled_visualizations",
        max_pixels=9,
    )

    assert result.stats["roi_pixel_count"] == 36
    assert result.stats["pixels_used"] <= 9
    assert result.stats["sampled"] is True
    assert (
        cv2.imread(str(result.heatmap_path), cv2.IMREAD_UNCHANGED).shape[:2]
        == mask.shape
    )


def test_generate_oil_response_heatmap_rejects_mask_cube_shape_mismatch(tmp_path: Path):
    artifact = load_model_artifact(_write_linear_artifact(tmp_path))
    roi = ROIMetrics(
        mask_source="unit_test_mask",
        pepper_pixels=1,
        pixel_count=1,
        valid_pixel_count=1,
        coverage_ratio=0.25,
        valid_pixel_ratio=1.0,
    )

    with pytest.raises(ValueError, match="same shape"):
        generate_oil_response_heatmap(
            cube=np.ones((4, 4, 2), dtype=np.float32),
            mask=np.ones((3, 4), dtype=bool),
            artifact=artifact,
            roi=roi,
            image_bgr=np.ones((4, 4, 3), dtype=np.uint8),
            output_dir=tmp_path / "bad",
        )
