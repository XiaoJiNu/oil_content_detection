"""Configuration for the Huajiao oil-content Web application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_DIR = Path(
    "artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1"
)
DEFAULT_UPLOAD_ROOT = Path("runtime/huajiao_web/uploads")
DEFAULT_OUTPUT_ROOT = Path("runtime/huajiao_web/outputs")


@dataclass(frozen=True)
class HuajiaoWebSettings:
    """Runtime settings for the Web demo service."""

    model_dir: Path = DEFAULT_MODEL_DIR
    upload_root: Path = DEFAULT_UPLOAD_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    max_upload_mb: int = 2048
    heatmap_max_pixels: int = 200_000
    keep_output_hours: int = 24
    static_dir: Path = Path(__file__).resolve().parent / "static"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    @classmethod
    def from_env(cls) -> "HuajiaoWebSettings":
        """Build settings from environment variables."""

        return cls(
            model_dir=Path(os.getenv("HUAJIAO_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
            upload_root=Path(
                os.getenv("HUAJIAO_UPLOAD_ROOT", str(DEFAULT_UPLOAD_ROOT))
            ),
            output_root=Path(
                os.getenv("HUAJIAO_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
            ),
            max_upload_mb=int(os.getenv("HUAJIAO_MAX_UPLOAD_MB", "2048")),
            heatmap_max_pixels=int(os.getenv("HUAJIAO_HEATMAP_MAX_PIXELS", "200000")),
            keep_output_hours=int(os.getenv("HUAJIAO_KEEP_OUTPUT_HOURS", "24")),
        )

    def ensure_directories(self) -> None:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)


__all__ = ["HuajiaoWebSettings"]
