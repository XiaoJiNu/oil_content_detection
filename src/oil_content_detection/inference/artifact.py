"""Model artifact loading and validation for Huajiao inference."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


FORBIDDEN_DEPLOYMENT_FEATURES = {
    "distill_ml",
    "oil_ml_per_gram",
    "oil_ml_per_100g",
    "weight_g",
    "batch",
    "sid_num",
}


@dataclass(frozen=True)
class ModelArtifact:
    """Loaded versioned model artifact."""

    artifact_dir: Path
    model: Any
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    roi_config: dict[str, Any]
    model_sha256: str

    @property
    def feature_columns(self) -> list[str]:
        return list(self.feature_schema["feature_columns"])

    @property
    def wavelength_columns(self) -> list[str]:
        columns = self.feature_schema.get("wavelength_columns")
        if columns is None:
            columns = [c for c in self.feature_columns if str(c).startswith("wl_")]
        return list(columns)

    @property
    def quality_columns(self) -> list[str]:
        columns = self.feature_schema.get("quality_columns")
        if columns is None:
            wl = set(self.wavelength_columns)
            columns = [c for c in self.feature_columns if c not in wl]
        return list(columns)

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", self.artifact_dir.name))

    @property
    def target_name(self) -> str:
        return str(self.manifest.get("target_name", "oil_ml_per_100g"))

    @property
    def unit(self) -> str:
        return str(self.manifest.get("unit", "ml/100g"))

    @property
    def input_mode(self) -> str:
        return str(self.manifest.get("input_mode", "envi_cube_with_rgb_image"))


def sha256_file(path: str | Path) -> str:
    """Compute SHA256 for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_feature_schema(feature_schema: dict[str, Any]) -> None:
    columns = feature_schema.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("feature_schema.json must contain a non-empty feature_columns list")
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if duplicated:
        raise ValueError(f"Duplicate feature columns are not allowed: {duplicated}")
    forbidden = sorted(FORBIDDEN_DEPLOYMENT_FEATURES.intersection(columns))
    if forbidden:
        raise ValueError(
            "Deployment artifact contains forbidden feature(s): "
            + ", ".join(forbidden)
            + ". Repackage without target leakage, weight, batch, or sample-id features."
        )
    if not any(str(c).startswith("wl_") for c in columns):
        raise ValueError("feature_schema.json must include wavelength columns named wl_<nm>")


def load_model_artifact(artifact_dir: str | Path) -> ModelArtifact:
    """Load a model artifact directory and validate its serving contract."""

    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {root}")
    model_path = root / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Required model file missing: {model_path}")
    manifest = _read_json(root / "manifest.json")
    feature_schema = _read_json(root / "feature_schema.json")
    roi_config = _read_json(root / "roi_config.json") if (root / "roi_config.json").exists() else {}
    _validate_feature_schema(feature_schema)
    model = joblib.load(model_path)
    model_sha256 = sha256_file(model_path)
    return ModelArtifact(
        artifact_dir=root,
        model=model,
        manifest=manifest,
        feature_schema=feature_schema,
        roi_config=roi_config,
        model_sha256=model_sha256,
    )
