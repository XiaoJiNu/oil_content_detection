"""Structured schemas for Huajiao oil-content inference."""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite numeric values with JSON null."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


class PredictionError(Exception):
    """Structured inference error with a stable machine-readable code."""

    def __init__(
        self, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {"code": self.code, "message": self.message, "details": self.details}
        )


@dataclass(frozen=True)
class QCResult:
    """Quality-control status returned with every successful prediction."""

    status: str = "ok"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ROIMetrics:
    """ROI metrics derived only from the uploaded RGB image and cube."""

    mask_source: str
    pepper_pixels: int
    pixel_count: int
    valid_pixel_count: int
    coverage_ratio: float
    valid_pixel_ratio: float
    mask_path: str | None = None
    overlay_path: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pepper_pixels"] = int(self.pepper_pixels)
        data["pixel_count"] = int(self.pixel_count)
        data["valid_pixel_count"] = int(self.valid_pixel_count)
        data["coverage_ratio"] = float(self.coverage_ratio)
        data["valid_pixel_ratio"] = float(self.valid_pixel_ratio)
        return _json_safe(data)


@dataclass(frozen=True)
class PredictionResult:
    """Successful model prediction payload for application integration."""

    sample_id: str | None
    target_name: str
    prediction: float
    unit: str
    model_version: str
    artifact_sha256: str | None
    input_mode: str
    roi: ROIMetrics
    qc: QCResult = field(default_factory=QCResult)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "sample_id": self.sample_id,
                "target_name": self.target_name,
                "prediction": float(self.prediction),
                "unit": self.unit,
                "model_version": self.model_version,
                "artifact_sha256": self.artifact_sha256,
                "input_mode": self.input_mode,
                "roi": self.roi.to_dict(),
                "qc": self.qc.to_dict(),
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False
        )
