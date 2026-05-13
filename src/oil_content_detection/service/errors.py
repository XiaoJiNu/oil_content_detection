"""Error payload helpers for the Huajiao Web application."""

from __future__ import annotations

from typing import Any

from oil_content_detection.inference.schema import PredictionError, _json_safe


def error_payload(
    *,
    code: str,
    message: str,
    job_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "qc": {"status": "error", "warnings": []},
        "error": {"code": code, "message": message, "details": details or {}},
    }
    if job_id is not None:
        payload["job_id"] = job_id
    return _json_safe(payload)


def prediction_error_payload(
    exc: PredictionError, *, job_id: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "qc": {"status": "error", "warnings": []},
        "error": exc.to_dict(),
    }
    if job_id is not None:
        payload["job_id"] = job_id
    return _json_safe(payload)


__all__ = ["error_payload", "prediction_error_payload"]
