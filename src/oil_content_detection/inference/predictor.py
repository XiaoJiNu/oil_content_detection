"""High-level predictor for Huajiao oil-content mode-B inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from oil_content_detection.acquisition.envi_reader import load_envi_cube

from .artifact import ModelArtifact, load_model_artifact
from .features import (
    extract_roi_spectrum_features,
    feature_row_to_frame,
    validate_same_view_shape,
)
from .roi import (
    enforce_roi_quality,
    load_rgb_image_bgr,
    refine_mask_in_roi_if_configured,
    save_roi_debug_outputs,
    segment_image_roi,
)
from .schema import PredictionError, PredictionResult, QCResult

TRADITIONAL_ROI_BACKEND = "traditional"
DINOV3_ROI_BACKEND = "dinov3"


def _load_dinov3_predictor_class() -> type:
    """Import the DINOv3 ROI predictor lazily so traditional inference stays lightweight."""

    from oil_content_detection.segmentation.predictor import DinoV3ROIPredictor

    return DinoV3ROIPredictor


def _roi_backend(roi_config: dict[str, Any] | None) -> str:
    backend = (
        str((roi_config or {}).get("backend", TRADITIONAL_ROI_BACKEND)).strip().lower()
    )
    if backend in {
        "traditional",
        "traditional_cv",
        "color_texture",
        "gray_otsu",
        "hybrid",
    }:
        return TRADITIONAL_ROI_BACKEND
    if backend in {"dinov3", "dino_v3", "dinov3_roi"}:
        return DINOV3_ROI_BACKEND
    raise ValueError(f"Unsupported ROI backend in roi_config.json: {backend}")


def _resolve_artifact_relative_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class HuajiaoOilPredictor:
    """Versioned model predictor for RGB image + ENVI cube inputs."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.roi_backend = _roi_backend(artifact.roi_config)
        self._dinov3_roi_predictor: Any | None = None
        self._dinov3_artifact_dir: Path | None = None
        if self.roi_backend == DINOV3_ROI_BACKEND:
            self._dinov3_roi_predictor = self._build_dinov3_roi_predictor()

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> "HuajiaoOilPredictor":
        return cls(load_model_artifact(artifact_dir))

    def _build_dinov3_roi_predictor(self) -> Any:
        dinov3_cfg = self.artifact.roi_config.get("dinov3", {})
        if not isinstance(dinov3_cfg, dict):
            raise ValueError(
                "roi_config.json field 'dinov3' must be an object when backend='dinov3'"
            )
        raw_artifact_dir = dinov3_cfg.get(
            "artifact_dir"
        ) or self.artifact.roi_config.get("dinov3_artifact_dir")
        if not raw_artifact_dir:
            raise ValueError(
                "roi_config.json backend='dinov3' requires dinov3.artifact_dir"
            )
        roi_artifact_dir = _resolve_artifact_relative_path(
            self.artifact.artifact_dir, raw_artifact_dir
        )
        device = dinov3_cfg.get("device", self.artifact.roi_config.get("roi_device"))
        if device in {"", "auto"}:
            device = None
        predictor_cls = _load_dinov3_predictor_class()
        self._dinov3_artifact_dir = roi_artifact_dir
        return predictor_cls.from_artifact(roi_artifact_dir, device=device)

    def predict(
        self,
        *,
        image_path: str | Path,
        hdr_path: str | Path,
        sample_id: str | None = None,
        debug_dir: str | Path | None = None,
    ) -> PredictionResult:
        """Predict oil content for one sample from same-size RGB image and ENVI cube."""

        image_path = Path(image_path)
        hdr_path = Path(hdr_path)
        if not hdr_path.exists():
            raise PredictionError(
                "INPUT_FILE_NOT_FOUND", f"ENVI header not found: {hdr_path}"
            )
        image_bgr = load_rgb_image_bgr(image_path)
        cube, header = load_envi_cube(hdr_path, memmap=True)
        validate_same_view_shape(image_bgr.shape[:2], cube.shape[:2])

        if self.roi_backend == DINOV3_ROI_BACKEND:
            mask, mask_path, overlay_path, roi_info, mask_source = (
                self._segment_dinov3_roi(
                    image_path=image_path,
                    image_bgr=image_bgr,
                    image_stem=sample_id or image_path.stem,
                    debug_dir=debug_dir,
                )
            )
        else:
            roi_result, mask_path, overlay_path = segment_image_roi(
                image_bgr=image_bgr,
                image_stem=sample_id or image_path.stem,
                roi_config=self.artifact.roi_config,
                debug_dir=debug_dir,
            )
            mask = roi_result.pepper_mask
            roi_info = roi_result.info
            mask_source = "traditional_roi_segmentation"
        row, roi = extract_roi_spectrum_features(
            cube=np.asarray(cube),
            mask=mask,
            wavelength_columns=self.artifact.wavelength_columns,
            feature_columns=self.artifact.feature_columns,
            mask_path=mask_path,
            overlay_path=overlay_path,
            mask_source=mask_source,
            roi_info=roi_info,
        )
        warnings = enforce_roi_quality(
            roi.valid_pixel_count, roi.coverage_ratio, self.artifact.roi_config
        )
        warnings.extend(self._schema_warnings(header_wavelengths=header.wavelengths))
        X = feature_row_to_frame(row, self.artifact.feature_columns)
        pred = np.asarray(self.artifact.model.predict(X), dtype=float).reshape(-1)
        if pred.size != 1:
            raise PredictionError(
                "MODEL_PREDICTION_FAILED",
                f"Expected one prediction, got shape {pred.shape}",
            )
        return PredictionResult(
            sample_id=sample_id,
            target_name=self.artifact.target_name,
            prediction=float(pred[0]),
            unit=self.artifact.unit,
            model_version=self.artifact.model_version,
            artifact_sha256=self.artifact.model_sha256,
            input_mode=self.artifact.input_mode,
            roi=roi,
            qc=QCResult(status="ok", warnings=warnings),
        )

    def _segment_dinov3_roi(
        self,
        *,
        image_path: Path,
        image_bgr: np.ndarray,
        image_stem: str,
        debug_dir: str | Path | None,
    ) -> tuple[np.ndarray, str | None, str | None, dict[str, Any], str]:
        if self._dinov3_roi_predictor is None:
            raise PredictionError(
                "MODEL_ARTIFACT_MISSING", "DINOv3 ROI predictor was not initialized."
            )
        mask_out: Path | None = None
        overlay_out: Path | None = None
        if debug_dir is not None:
            debug_root = Path(debug_dir)
            debug_root.mkdir(parents=True, exist_ok=True)
            mask_out = debug_root / f"{image_stem}_dinov3_pepper_mask.png"
            overlay_out = debug_root / f"{image_stem}_dinov3_roi_overlay.png"
        try:
            prediction = self._dinov3_roi_predictor.predict_image(
                image_path=image_path,
                sample_id=image_stem,
                out_mask_path=mask_out,
                overlay_path=overlay_out,
            )
        except Exception as exc:
            raise PredictionError(
                "ROI_SEGMENTATION_FAILED", f"DINOv3 ROI segmentation failed: {exc}"
            ) from exc
        info = {
            "backend": DINOV3_ROI_BACKEND,
            "dinov3_artifact_dir": (
                str(self._dinov3_artifact_dir)
                if self._dinov3_artifact_dir is not None
                else None
            ),
            "dinov3_threshold": float(getattr(prediction, "threshold", np.nan)),
            "dinov3_coverage_ratio": float(
                getattr(prediction, "coverage_ratio", np.nan)
            ),
        }
        base_mask = np.asarray(prediction.mask, dtype=bool)
        mask, refinement_info, mask_source = refine_mask_in_roi_if_configured(
            image_bgr=image_bgr,
            base_mask=base_mask,
            roi_config=self.artifact.roi_config,
            base_mask_source="dinov3_roi_segmentation",
        )
        info.update(refinement_info)
        save_roi_debug_outputs(
            image_bgr=image_bgr,
            mask=mask,
            mask_path=getattr(prediction, "mask_path", None),
            overlay_path=getattr(prediction, "overlay_path", None),
        )
        return (
            mask,
            getattr(prediction, "mask_path", None),
            getattr(prediction, "overlay_path", None),
            info,
            mask_source,
        )

    def _schema_warnings(self, *, header_wavelengths: list[float] | None) -> list[str]:
        warnings: list[str] = []
        if header_wavelengths is None:
            warnings.append(
                "ENVI header has no wavelength list; bands are mapped to wl_* columns by order."
            )
            return warnings
        if len(header_wavelengths) != len(self.artifact.wavelength_columns):
            warnings.append(
                "ENVI wavelength count differs from artifact schema; bands were still mapped by order "
                f"({len(header_wavelengths)} header wavelengths vs {len(self.artifact.wavelength_columns)} schema columns)."
            )
        return warnings


__all__ = ["HuajiaoOilPredictor"]
