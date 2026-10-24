"""FastAPI application for Huajiao oil-content Web demo."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from oil_content_detection.acquisition.envi_reader import load_envi_cube
from oil_content_detection.inference import HuajiaoOilPredictor
from oil_content_detection.inference.heatmap import generate_oil_response_heatmap
from oil_content_detection.inference.schema import PredictionError, _json_safe

from .errors import error_payload, prediction_error_payload
from .settings import HuajiaoWebSettings
from .storage import (
    HDR_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SPECTRAL_DATA_EXTENSIONS,
    copy_input_preview,
    create_upload_job,
    ensure_dat_sibling,
    output_url,
    save_upload_file,
)

PredictorFactory = Callable[[str | Path], Any]
logger = logging.getLogger(__name__)


HEATMAP_NOTE = "heatmap_is_model_response_not_pixel_ground_truth"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _build_visualization_urls(
    *,
    settings: HuajiaoWebSettings,
    image_preview_path: Path | None,
    prediction_payload: dict[str, Any] | None = None,
    heatmap_payload: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    roi_payload = (prediction_payload or {}).get("roi") or {}
    return {
        "image_url": output_url(settings, image_preview_path),
        "mask_url": output_url(settings, roi_payload.get("mask_path")),
        "overlay_url": output_url(settings, roi_payload.get("overlay_path")),
        "heatmap_url": output_url(
            settings, (heatmap_payload or {}).get("heatmap_path")
        ),
        "heatmap_overlay_url": output_url(
            settings, (heatmap_payload or {}).get("overlay_path")
        ),
    }


def _heatmap_image_path(
    *, image_path: Path, prediction_payload: dict[str, Any]
) -> Path:
    """Prefer the predictor-produced aligned RGB image for heatmap generation."""

    roi_payload = prediction_payload.get("roi") or {}
    roi_info = roi_payload.get("info") or {}
    shape_alignment = roi_info.get("shape_alignment") or {}
    aligned_image_path = shape_alignment.get("aligned_image_path")
    if aligned_image_path:
        return Path(str(aligned_image_path))
    return image_path


def create_app(
    *,
    settings: HuajiaoWebSettings | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    ``predictor_factory`` is injectable for tests; production uses
    ``HuajiaoOilPredictor.from_artifact`` and a service-level singleton.
    """

    settings = settings or HuajiaoWebSettings.from_env()
    settings.ensure_directories()
    factory = predictor_factory or HuajiaoOilPredictor.from_artifact

    app = FastAPI(title="Huajiao Oil Content Detection", version="0.1.0")
    app.state.settings = settings
    app.state.predictor_factory = factory
    app.state.predictor = None

    if settings.static_dir.exists():
        app.mount(
            "/static", StaticFiles(directory=str(settings.static_dir)), name="static"
        )
    app.mount(
        "/outputs", StaticFiles(directory=str(settings.output_root)), name="outputs"
    )

    def get_predictor() -> Any:
        if app.state.predictor is None:
            app.state.predictor = app.state.predictor_factory(
                app.state.settings.model_dir
            )
        return app.state.predictor

    @app.get("/api/health")
    def health() -> JSONResponse:
        try:
            predictor = get_predictor()
            artifact = getattr(predictor, "artifact", None)
            model_version = getattr(artifact, "model_version", None)
            roi_config = getattr(artifact, "roi_config", {}) or {}
            device = None
            if isinstance(roi_config, dict):
                device = (
                    (roi_config.get("dinov3") or {}).get("device")
                    if isinstance(roi_config.get("dinov3"), dict)
                    else None
                )
            payload = {
                "status": "ok",
                "model_loaded": True,
                "model_version": model_version,
                "device": device,
            }
            return JSONResponse(_json_safe(payload))
        except Exception:  # pragma: no cover - defensive health boundary
            logger.exception("Huajiao model health check failed")
            payload = {
                "status": "error",
                "model_loaded": False,
                "model_version": None,
                "device": None,
                "error": "model health check failed",
            }
            return JSONResponse(_json_safe(payload), status_code=503)

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        hdr: UploadFile = File(...),
        dat: UploadFile = File(...),
        sample_id: str | None = Form(default=None),
        generate_heatmap: bool = Form(default=True),
    ) -> JSONResponse:
        job = create_upload_job(app.state.settings)
        try:
            image_path = await save_upload_file(
                image,
                destination_dir=job.upload_dir,
                allowed_extensions=IMAGE_EXTENSIONS,
                role="image",
                max_bytes=app.state.settings.max_upload_bytes,
                fallback_name="sample.png",
            )
            hdr_path = await save_upload_file(
                hdr,
                destination_dir=job.upload_dir,
                allowed_extensions=HDR_EXTENSIONS,
                role="hdr",
                max_bytes=app.state.settings.max_upload_bytes,
                fallback_name="REFLECTANCE_sample.hdr",
            )
            data_path = await save_upload_file(
                dat,
                destination_dir=job.upload_dir,
                allowed_extensions=SPECTRAL_DATA_EXTENSIONS,
                role="spectral_data",
                max_bytes=app.state.settings.max_upload_bytes,
                fallback_name="REFLECTANCE_sample.dat",
            )
            ensure_dat_sibling(hdr_path, data_path)
            image_preview_path = copy_input_preview(image_path, job)

            clean_sample_id = (sample_id or "").strip() or image_path.stem
            predictor = get_predictor()
            result = predictor.predict(
                image_path=image_path,
                hdr_path=hdr_path,
                sample_id=clean_sample_id,
                debug_dir=job.debug_dir,
            )
            prediction_payload = result.to_dict()
            _write_json(job.output_dir / "prediction.json", prediction_payload)

            heatmap_payload: dict[str, Any] | None = None
            if generate_heatmap:
                try:
                    mask_path = prediction_payload.get("roi", {}).get("mask_path")
                    if not mask_path:
                        raise ValueError(
                            "ROI mask path is missing; cannot generate heatmap."
                        )
                    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    heatmap_image_path = _heatmap_image_path(
                        image_path=image_path,
                        prediction_payload=prediction_payload,
                    )
                    image_bgr = cv2.imread(str(heatmap_image_path), cv2.IMREAD_COLOR)
                    if mask_image is None or image_bgr is None:
                        raise ValueError(
                            "Failed to read image or ROI mask for heatmap generation."
                        )
                    cube, _ = load_envi_cube(hdr_path, memmap=True)
                    heatmap_result = generate_oil_response_heatmap(
                        cube=cube,
                        mask=mask_image > 0,
                        artifact=predictor.artifact,
                        roi=result.roi,
                        image_bgr=image_bgr,
                        output_dir=job.output_dir / "visualizations",
                        max_pixels=app.state.settings.heatmap_max_pixels,
                    )
                    heatmap_payload = heatmap_result.to_dict()
                except Exception as exc:  # heatmap is a non-critical demo artifact
                    prediction_payload.setdefault("qc", {}).setdefault(
                        "warnings", []
                    ).append(f"HEATMAP_GENERATION_FAILED: {exc}")

            response_payload = {
                "job_id": job.job_id,
                "prediction": prediction_payload,
                "visualizations": _build_visualization_urls(
                    settings=app.state.settings,
                    image_preview_path=image_preview_path,
                    prediction_payload=prediction_payload,
                    heatmap_payload=heatmap_payload,
                ),
                "downloads": {
                    "prediction_json_url": output_url(
                        app.state.settings, job.output_dir / "prediction.json"
                    ),
                    "response_json_url": output_url(
                        app.state.settings, job.output_dir / "response.json"
                    ),
                },
                "notes": [HEATMAP_NOTE],
            }
            _write_json(job.output_dir / "response.json", response_payload)
            return JSONResponse(_json_safe(response_payload))
        except PredictionError as exc:
            payload = prediction_error_payload(exc, job_id=job.job_id)
            _write_json(job.output_dir / "error.json", payload)
            return JSONResponse(payload, status_code=400)
        except ValueError as exc:
            payload = error_payload(
                code="INVALID_UPLOAD", message=str(exc), job_id=job.job_id
            )
            _write_json(job.output_dir / "error.json", payload)
            return JSONResponse(payload, status_code=400)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            payload = error_payload(
                code="UNEXPECTED_ERROR", message=str(exc), job_id=job.job_id
            )
            _write_json(job.output_dir / "error.json", payload)
            return JSONResponse(payload, status_code=500)

    @app.get("/", response_model=None)
    def index():
        index_path = app.state.settings.static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(
            {"status": "ok", "message": "Huajiao oil-content Web service is running."}
        )

    return app


__all__ = ["create_app"]
