from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from oil_content_detection.inference.schema import (
    PredictionError,
    PredictionResult,
    QCResult,
    ROIMetrics,
)
from oil_content_detection.service.app import create_app
from oil_content_detection.service.settings import HuajiaoWebSettings


class FakePredictor:
    def __init__(self, prediction: float = 1.23) -> None:
        self.artifact = type(
            "Artifact",
            (),
            {
                "model_version": "fake-huajiao-model",
                "model_sha256": "abc123",
                "artifact_dir": Path("fake-artifact"),
            },
        )()
        self.prediction = prediction
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        *,
        image_path: str | Path,
        hdr_path: str | Path,
        sample_id: str | None = None,
        debug_dir: str | Path | None = None,
    ) -> PredictionResult:
        self.calls.append(
            {
                "image_path": Path(image_path),
                "hdr_path": Path(hdr_path),
                "sample_id": sample_id,
                "debug_dir": Path(debug_dir) if debug_dir is not None else None,
            }
        )
        debug_root = Path(debug_dir or Path(image_path).parent)
        debug_root.mkdir(parents=True, exist_ok=True)
        mask_path = debug_root / f"{sample_id or 'sample'}_dinov3_pepper_mask.png"
        overlay_path = debug_root / f"{sample_id or 'sample'}_dinov3_roi_overlay.png"
        cv2.imwrite(str(mask_path), np.full((4, 5), 255, dtype=np.uint8))
        cv2.imwrite(str(overlay_path), np.full((4, 5, 3), 128, dtype=np.uint8))
        return PredictionResult(
            sample_id=sample_id,
            target_name="oil_ml_per_100g",
            prediction=self.prediction,
            unit="ml/100g",
            model_version="fake-huajiao-model",
            artifact_sha256="abc123",
            input_mode="envi_cube_with_rgb_image",
            roi=ROIMetrics(
                mask_source="dinov3_roi_gray_otsu_refined",
                pepper_pixels=20,
                pixel_count=20,
                valid_pixel_count=20,
                coverage_ratio=1.0,
                valid_pixel_ratio=1.0,
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
            ),
            qc=QCResult(status="ok", warnings=[]),
        )


def _settings(tmp_path: Path) -> HuajiaoWebSettings:
    return HuajiaoWebSettings(
        model_dir=tmp_path / "artifact",
        upload_root=tmp_path / "uploads",
        output_root=tmp_path / "outputs",
        max_upload_mb=4,
        heatmap_max_pixels=100,
    )


def _write_upload_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.full((4, 5, 3), 80, dtype=np.uint8))
    hdr_path = tmp_path / "REFLECTANCE_sample.hdr"
    hdr_path.write_text(
        "\n".join(
            [
                "ENVI",
                "samples = 5",
                "lines = 4",
                "bands = 2",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bip",
                "byte order = 0",
                "wavelength = {400, 500}",
            ]
        ),
        encoding="utf-8",
    )
    dat_path = tmp_path / "uploaded_name.raw"
    np.ones((4, 5, 2), dtype=np.float32).tofile(dat_path)
    return image_path, hdr_path, dat_path


def _multipart_files(image_path: Path, hdr_path: Path, dat_path: Path):
    return {
        "image": (image_path.name, image_path.read_bytes(), "image/png"),
        "hdr": (hdr_path.name, hdr_path.read_bytes(), "text/plain"),
        "dat": (dat_path.name, dat_path.read_bytes(), "application/octet-stream"),
    }


def test_health_reports_model_loaded_and_version(tmp_path: Path):
    predictor = FakePredictor()
    app = create_app(
        settings=_settings(tmp_path), predictor_factory=lambda _: predictor
    )
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "fake-huajiao-model"


def test_health_failure_returns_sanitized_error_message(tmp_path: Path):
    def broken_factory(_model_dir: Path):
        raise RuntimeError("internal path /very/secret/model/file.pt failed")

    app = create_app(settings=_settings(tmp_path), predictor_factory=broken_factory)
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["model_loaded"] is False
    assert payload["error"] == "model health check failed"
    assert "/very/secret" not in json.dumps(payload)


def test_predict_upload_saves_job_outputs_and_returns_visualization_urls(
    tmp_path: Path,
):
    predictor = FakePredictor(prediction=2.34)
    app = create_app(
        settings=_settings(tmp_path), predictor_factory=lambda _: predictor
    )
    client = TestClient(app)
    image_path, hdr_path, dat_path = _write_upload_files(tmp_path)

    response = client.post(
        "/api/predict",
        data={"sample_id": "demo-001", "generate_heatmap": "false"},
        files=_multipart_files(image_path, hdr_path, dat_path),
    )

    assert response.status_code == 200
    payload = response.json()
    json.dumps(payload, allow_nan=False)
    assert payload["job_id"]
    assert payload["prediction"]["prediction"] == pytest.approx(2.34)
    assert payload["prediction"]["sample_id"] == "demo-001"
    assert payload["visualizations"]["mask_url"].endswith("_dinov3_pepper_mask.png")
    assert payload["visualizations"]["overlay_url"].endswith("_dinov3_roi_overlay.png")
    assert payload["visualizations"]["heatmap_url"] is None
    assert payload["downloads"]["prediction_json_url"].endswith("prediction.json")
    assert (tmp_path / "outputs" / payload["job_id"] / "prediction.json").exists()
    assert (tmp_path / "outputs" / payload["job_id"] / "response.json").exists()
    assert predictor.calls[0]["sample_id"] == "demo-001"
    assert Path(predictor.calls[0]["hdr_path"]).with_suffix(".dat").exists()


def test_predict_heatmap_uses_aligned_image_path_when_predictor_resizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from types import SimpleNamespace

    from oil_content_detection.service import app as app_module

    class AlignedImagePredictor(FakePredictor):
        def predict(self, **kwargs):  # type: ignore[no-untyped-def]
            result = super().predict(**kwargs)
            debug_dir = Path(kwargs["debug_dir"])
            aligned_image_path = debug_dir / "demo-heatmap_aligned_to_cube.png"
            cv2.imwrite(str(aligned_image_path), np.full((4, 5, 3), 90, dtype=np.uint8))
            return PredictionResult(
                sample_id=result.sample_id,
                target_name=result.target_name,
                prediction=result.prediction,
                unit=result.unit,
                model_version=result.model_version,
                artifact_sha256=result.artifact_sha256,
                input_mode=result.input_mode,
                roi=ROIMetrics(
                    mask_source=result.roi.mask_source,
                    pepper_pixels=result.roi.pepper_pixels,
                    pixel_count=result.roi.pixel_count,
                    valid_pixel_count=result.roi.valid_pixel_count,
                    coverage_ratio=result.roi.coverage_ratio,
                    valid_pixel_ratio=result.roi.valid_pixel_ratio,
                    mask_path=result.roi.mask_path,
                    overlay_path=result.roi.overlay_path,
                    info={
                        "shape_alignment": {
                            "resized": True,
                            "aligned_image_path": str(aligned_image_path),
                        }
                    },
                ),
                qc=result.qc,
            )

    captured: dict[str, tuple[int, int]] = {}

    def fake_generate_oil_response_heatmap(**kwargs):  # type: ignore[no-untyped-def]
        captured["image_shape_hw"] = kwargs["image_bgr"].shape[:2]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_dir / "oil_heatmap.png"
        overlay_path = output_dir / "oil_heatmap_overlay.png"
        stats_path = output_dir / "oil_heatmap_stats.json"
        cv2.imwrite(str(heatmap_path), np.full((4, 5, 4), 255, dtype=np.uint8))
        cv2.imwrite(str(overlay_path), np.full((4, 5, 3), 128, dtype=np.uint8))
        stats_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            to_dict=lambda: {
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
                "stats_path": str(stats_path),
            }
        )

    monkeypatch.setattr(
        app_module, "generate_oil_response_heatmap", fake_generate_oil_response_heatmap
    )
    app = create_app(
        settings=_settings(tmp_path), predictor_factory=lambda _: AlignedImagePredictor()
    )
    client = TestClient(app)
    image_path, hdr_path, dat_path = _write_upload_files(tmp_path)
    cv2.imwrite(str(image_path), np.full((5, 5, 3), 80, dtype=np.uint8))

    response = client.post(
        "/api/predict",
        data={"sample_id": "demo-heatmap", "generate_heatmap": "true"},
        files=_multipart_files(image_path, hdr_path, dat_path),
    )

    assert response.status_code == 200
    assert captured["image_shape_hw"] == (4, 5)
    payload = response.json()
    assert payload["visualizations"]["heatmap_url"].endswith("oil_heatmap.png")


def test_predict_returns_structured_prediction_error(tmp_path: Path):
    class FailingPredictor(FakePredictor):
        def predict(self, **kwargs):  # type: ignore[no-untyped-def]
            raise PredictionError(
                "CUBE_IMAGE_SHAPE_MISMATCH",
                "RGB image and hyperspectral cube must be the same view; automatic RGB resize is allowed only when height and width differ by no more than 5 pixels.",
                {"image_shape_hw": [4, 5], "cube_shape_hw": [5, 5]},
            )

    app = create_app(
        settings=_settings(tmp_path), predictor_factory=lambda _: FailingPredictor()
    )
    client = TestClient(app)
    image_path, hdr_path, dat_path = _write_upload_files(tmp_path)

    response = client.post(
        "/api/predict",
        data={"sample_id": "bad-sample", "generate_heatmap": "false"},
        files=_multipart_files(image_path, hdr_path, dat_path),
    )

    assert response.status_code == 400
    payload = response.json()
    json.dumps(payload, allow_nan=False)
    assert payload["job_id"]
    assert payload["qc"]["status"] == "error"
    assert payload["error"]["code"] == "CUBE_IMAGE_SHAPE_MISMATCH"
    assert (tmp_path / "outputs" / payload["job_id"] / "error.json").exists()


def test_predict_rejects_unsupported_upload_extension(tmp_path: Path):
    app = create_app(
        settings=_settings(tmp_path), predictor_factory=lambda _: FakePredictor()
    )
    client = TestClient(app)
    image_path, hdr_path, dat_path = _write_upload_files(tmp_path)

    response = client.post(
        "/api/predict",
        data={"sample_id": "bad-ext", "generate_heatmap": "false"},
        files={
            "image": ("image.gif", image_path.read_bytes(), "image/gif"),
            "hdr": (hdr_path.name, hdr_path.read_bytes(), "text/plain"),
            "dat": (dat_path.name, dat_path.read_bytes(), "application/octet-stream"),
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_UPLOAD"
