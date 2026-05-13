from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from oil_content_detection.inference import HuajiaoOilPredictor
from oil_content_detection.inference import predictor as predictor_module
from oil_content_detection.inference import roi as roi_module
from oil_content_detection.inference.artifact import load_model_artifact
from oil_content_detection.inference.features import (
    extract_roi_spectrum_features,
    validate_same_view_shape,
)
from oil_content_detection.inference.schema import PredictionError, ROIMetrics


WL_COLUMNS = ["wl_400", "wl_500"]
QUALITY_COLUMNS = ["coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio"]
FEATURE_COLUMNS = WL_COLUMNS + QUALITY_COLUMNS


def _write_envi_cube(
    tmp_path: Path, cube: np.ndarray, wavelengths: list[float] | None = None
) -> Path:
    hdr_path = tmp_path / "REFLECTANCE_TEST.hdr"
    dat_path = hdr_path.with_suffix(".dat")
    cube.astype(np.float32).tofile(dat_path)
    wl_text = ", ".join(str(v) for v in (wavelengths or [400, 500]))
    hdr_path.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {cube.shape[1]}",
                f"lines = {cube.shape[0]}",
                f"bands = {cube.shape[2]}",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bip",
                "byte order = 0",
                f"wavelength = {{{wl_text}}}",
            ]
        ),
        encoding="utf-8",
    )
    return hdr_path


def _write_synthetic_rgb(tmp_path: Path, height: int, width: int) -> Path:
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.rectangle(
        image, (10, 10), (width - 11, height - 11), (30, 30, 150), thickness=-1
    )
    cv2.rectangle(
        image, (18, height - 42), (62, height - 18), (245, 245, 245), thickness=-1
    )
    cv2.rectangle(image, (18, height - 42), (62, height - 18), (0, 0, 255), thickness=2)
    rng = np.random.default_rng(7)
    for _ in range(80):
        cx = int(rng.integers(75, width - 35))
        cy = int(rng.integers(35, height - 60))
        cv2.circle(image, (cx, cy), 3, (0, 180, 0), thickness=-1)
    image_path = tmp_path / "TEST.png"
    cv2.imwrite(str(image_path), image)
    return image_path


def _write_artifact(
    tmp_path: Path,
    *,
    feature_columns: list[str] | None = None,
    constant: float = 1.23,
    roi_config: dict | None = None,
) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    columns = feature_columns or FEATURE_COLUMNS
    model = DummyRegressor(strategy="constant", constant=constant)
    model.fit(np.zeros((2, len(columns))), np.array([constant, constant]))
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": columns,
                "wavelength_columns": [c for c in columns if c.startswith("wl_")],
                "quality_columns": [c for c in columns if not c.startswith("wl_")],
                "forbidden_columns": ["weight_g", "batch", "sid_num"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": "huajiao-oil-content-test",
                "target_name": "oil_ml_per_100g",
                "unit": "ml/100g",
                "input_mode": "envi_cube_with_rgb_image",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "roi_config.json").write_text(
        json.dumps(
            roi_config
            or {
                "backend": "traditional",
                "min_pepper_pixels": 1,
                "min_coverage_ratio": 0.0,
                "max_coverage_ratio": 1.0,
                "require_same_shape": True,
                "segmentation": {
                    "use_texture": False,
                    "pepper_open_size": 1,
                    "pepper_close_size": 1,
                    "min_component_area": 1,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir


def test_load_model_artifact_rejects_weight_batch_and_sid_features(tmp_path: Path):
    artifact_dir = _write_artifact(tmp_path, feature_columns=WL_COLUMNS + ["weight_g"])

    with pytest.raises(ValueError, match="forbidden feature"):
        load_model_artifact(artifact_dir)


def test_roi_metrics_to_dict_replaces_non_finite_info_values():
    roi = ROIMetrics(
        mask_source="unit_test",
        pepper_pixels=1,
        pixel_count=4,
        valid_pixel_count=1,
        coverage_ratio=0.25,
        valid_pixel_ratio=1.0,
        info={
            "texture_thr": float("nan"),
            "nested": {"positive_inf": float("inf"), "ok": 1.25},
            "values": [float("-inf"), 3.0],
        },
    )

    payload = roi.to_dict()

    assert payload["info"]["texture_thr"] is None
    assert payload["info"]["nested"] == {"positive_inf": None, "ok": 1.25}
    assert payload["info"]["values"] == [None, 3.0]
    json.dumps(payload, allow_nan=False)


def test_validate_same_view_shape_reports_structured_error():
    with pytest.raises(PredictionError) as excinfo:
        validate_same_view_shape(image_shape_hw=(10, 12), cube_shape_hw=(10, 11))

    assert excinfo.value.code == "CUBE_IMAGE_SHAPE_MISMATCH"
    assert "same view" in excinfo.value.message


def test_extract_roi_spectrum_features_uses_mask_and_quality_columns():
    cube = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=float,
    )
    mask = np.array([[True, False], [True, False]])

    row, roi = extract_roi_spectrum_features(
        cube=cube,
        mask=mask,
        wavelength_columns=WL_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
    )

    assert row["wl_400"] == pytest.approx(2.0)
    assert row["wl_500"] == pytest.approx(20.0)
    assert row["coverage_ratio"] == pytest.approx(0.5)
    assert row["valid_pixel_ratio"] == pytest.approx(1.0)
    assert row["log_valid_pixel_count"] == pytest.approx(np.log1p(2))
    assert roi.pepper_pixels == 2
    assert roi.valid_pixel_count == 2


def test_predictor_api_accepts_envi_cube_and_same_size_rgb_image(tmp_path: Path):
    cube = np.ones((180, 140, 2), dtype=np.float32)
    cube[..., 0] *= 3.0
    cube[..., 1] *= 7.0
    hdr_path = _write_envi_cube(tmp_path, cube)
    image_path = _write_synthetic_rgb(tmp_path, height=180, width=140)
    artifact_dir = _write_artifact(tmp_path, constant=1.23)

    predictor = HuajiaoOilPredictor.from_artifact(artifact_dir)
    result = predictor.predict(
        image_path=image_path,
        hdr_path=hdr_path,
        sample_id="TEST",
        debug_dir=tmp_path / "debug",
    )

    payload = result.to_dict()
    assert payload["sample_id"] == "TEST"
    assert payload["prediction"] == pytest.approx(1.23)
    assert payload["unit"] == "ml/100g"
    assert payload["input_mode"] == "envi_cube_with_rgb_image"
    assert payload["qc"]["status"] == "ok"
    assert payload["roi"]["mask_source"] == "traditional_roi_segmentation"
    assert payload["roi"]["pepper_pixels"] > 0
    assert Path(payload["roi"]["overlay_path"]).exists()


def test_predictor_can_use_dinov3_roi_backend_with_relative_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cube = np.ones((120, 120, 2), dtype=np.float32)
    cube[:60, :, 0] = 4.0
    cube[:60, :, 1] = 8.0
    hdr_path = _write_envi_cube(tmp_path, cube)
    image_path = _write_synthetic_rgb(tmp_path, height=120, width=120)
    artifact_dir = _write_artifact(
        tmp_path,
        constant=3.45,
        roi_config={
            "backend": "dinov3",
            "require_same_shape": True,
            "min_pepper_pixels": 1,
            "min_coverage_ratio": 0.0,
            "max_coverage_ratio": 1.0,
            "dinov3": {"artifact_dir": "roi_model", "device": "cpu"},
        },
    )
    (artifact_dir / "roi_model").mkdir()

    class FakeDinoV3ROIPredictor:
        loaded_artifact_dir: Path | None = None
        loaded_device: str | None = None

        @classmethod
        def from_artifact(
            cls, artifact_dir: str | Path, device: str | None = None
        ) -> "FakeDinoV3ROIPredictor":
            cls.loaded_artifact_dir = Path(artifact_dir)
            cls.loaded_device = device
            return cls()

        def predict_image(
            self,
            image_path: str | Path,
            sample_id: str | None = None,
            out_mask_path: str | Path | None = None,
            overlay_path: str | Path | None = None,
        ) -> SimpleNamespace:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            assert image is not None
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[: image.shape[0] // 2, :] = True
            if out_mask_path is not None:
                cv2.imwrite(str(out_mask_path), (mask.astype(np.uint8) * 255))
            if overlay_path is not None:
                cv2.imwrite(str(overlay_path), image)
            return SimpleNamespace(
                sample_id=sample_id,
                mask=mask,
                probability=mask.astype(np.float32),
                threshold=0.5,
                coverage_ratio=float(mask.mean()),
                mask_path=str(out_mask_path) if out_mask_path is not None else None,
                overlay_path=str(overlay_path) if overlay_path is not None else None,
            )

    monkeypatch.setattr(
        predictor_module,
        "_load_dinov3_predictor_class",
        lambda: FakeDinoV3ROIPredictor,
    )

    predictor = HuajiaoOilPredictor.from_artifact(artifact_dir)
    result = predictor.predict(
        image_path=image_path,
        hdr_path=hdr_path,
        sample_id="TEST_DINO",
        debug_dir=tmp_path / "debug_dino",
    )

    payload = result.to_dict()
    assert FakeDinoV3ROIPredictor.loaded_artifact_dir == artifact_dir / "roi_model"
    assert FakeDinoV3ROIPredictor.loaded_device == "cpu"
    assert payload["prediction"] == pytest.approx(3.45)
    assert payload["roi"]["mask_source"] == "dinov3_roi_segmentation"
    assert payload["roi"]["pepper_pixels"] == 7200
    assert payload["roi"]["coverage_ratio"] == pytest.approx(0.5)
    assert payload["roi"]["info"]["dinov3_threshold"] == pytest.approx(0.5)
    assert Path(payload["roi"]["mask_path"]).exists()
    assert Path(payload["roi"]["overlay_path"]).exists()


def test_predictor_applies_dinov3_post_refinement_inside_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cube = np.ones((120, 120, 2), dtype=np.float32)
    hdr_path = _write_envi_cube(tmp_path, cube)
    image_path = _write_synthetic_rgb(tmp_path, height=120, width=120)
    artifact_dir = _write_artifact(
        tmp_path,
        constant=4.56,
        roi_config={
            "backend": "dinov3",
            "require_same_shape": True,
            "min_pepper_pixels": 1,
            "min_coverage_ratio": 0.0,
            "max_coverage_ratio": 1.0,
            "dinov3": {
                "artifact_dir": "roi_model",
                "device": "cpu",
                "post_refinement": {
                    "enabled": True,
                    "segmentation_mode": "gray_otsu",
                    "pepper_open_size": 5,
                    "pepper_close_size": 5,
                    "hole_fill_max_area": 300,
                    "label_fallback_enabled": False,
                    "label_max_area_ratio": 0.0,
                },
            },
        },
    )
    (artifact_dir / "roi_model").mkdir()

    class FakeDinoV3ROIPredictor:
        @classmethod
        def from_artifact(
            cls, artifact_dir: str | Path, device: str | None = None
        ) -> "FakeDinoV3ROIPredictor":
            return cls()

        def predict_image(
            self,
            image_path: str | Path,
            sample_id: str | None = None,
            out_mask_path: str | Path | None = None,
            overlay_path: str | Path | None = None,
        ) -> SimpleNamespace:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            assert image is not None
            coarse_mask = np.ones(image.shape[:2], dtype=bool)
            return SimpleNamespace(
                sample_id=sample_id,
                mask=coarse_mask,
                probability=coarse_mask.astype(np.float32),
                threshold=0.5,
                coverage_ratio=float(coarse_mask.mean()),
                mask_path=str(out_mask_path) if out_mask_path is not None else None,
                overlay_path=str(overlay_path) if overlay_path is not None else None,
            )

    calls: dict[str, object] = {}

    def fake_segment_pepper_in_mask(image_bgr, roi_mask, config):
        calls["roi_mask_pixels"] = int(roi_mask.sum())
        calls["config_mode"] = config.segmentation_mode
        refined_mask = np.zeros_like(roi_mask, dtype=bool)
        refined_mask[:3, :3] = True
        return SimpleNamespace(
            pepper_mask=refined_mask,
            info={"segmentation_strategy": "gray_otsu", "pepper_pixels": 9.0},
        )

    monkeypatch.setattr(
        predictor_module,
        "_load_dinov3_predictor_class",
        lambda: FakeDinoV3ROIPredictor,
    )
    monkeypatch.setattr(
        roi_module, "segment_pepper_in_mask", fake_segment_pepper_in_mask
    )

    predictor = HuajiaoOilPredictor.from_artifact(artifact_dir)
    result = predictor.predict(
        image_path=image_path,
        hdr_path=hdr_path,
        sample_id="TEST_DINO_REFINED",
        debug_dir=tmp_path / "debug_dino_refined",
    )

    payload = result.to_dict()
    assert calls == {"roi_mask_pixels": 14400, "config_mode": "gray_otsu"}
    assert payload["prediction"] == pytest.approx(4.56)
    assert payload["roi"]["mask_source"] == "dinov3_roi_gray_otsu_refined"
    assert payload["roi"]["pepper_pixels"] == 9
    assert payload["roi"]["coverage_ratio"] == pytest.approx(9 / 14400)
    assert payload["roi"]["info"]["dinov3_polygon_coverage_ratio"] == pytest.approx(1.0)
    assert payload["roi"]["info"]["post_refined_coverage_ratio"] == pytest.approx(
        9 / 14400
    )
    assert Path(payload["roi"]["mask_path"]).exists()
    assert Path(payload["roi"]["overlay_path"]).exists()


def test_huajiao_predict_cli_writes_prediction_json(tmp_path: Path):
    cube = np.ones((180, 140, 2), dtype=np.float32)
    hdr_path = _write_envi_cube(tmp_path, cube)
    image_path = _write_synthetic_rgb(tmp_path, height=180, width=140)
    artifact_dir = _write_artifact(tmp_path, constant=2.34)
    out_path = tmp_path / "prediction.json"

    script = Path(__file__).resolve().parents[2] / "scripts" / "huajiao_predict.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-dir",
            str(artifact_dir),
            "--image",
            str(image_path),
            "--hdr",
            str(hdr_path),
            "--sample-id",
            "TEST_CLI",
            "--out",
            str(out_path),
            "--debug-dir",
            str(tmp_path / "debug_cli"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists(), completed.stdout + completed.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "TEST_CLI"
    assert payload["prediction"] == pytest.approx(2.34)
    assert payload["qc"]["status"] == "ok"
