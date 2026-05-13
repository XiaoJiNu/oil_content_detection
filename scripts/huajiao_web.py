#!/usr/bin/env python
"""Run the Huajiao oil-content Web demo service."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import uvicorn  # noqa: E402

from oil_content_detection.service.app import create_app  # noqa: E402
from oil_content_detection.service.settings import HuajiaoWebSettings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=HuajiaoWebSettings.model_dir,
        help="Path to the packaged Huajiao oil-content model artifact directory.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--upload-root",
        type=Path,
        default=HuajiaoWebSettings.upload_root,
        help="Directory used for uploaded request files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HuajiaoWebSettings.output_root,
        help="Directory used for prediction JSON and visualization outputs.",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=HuajiaoWebSettings.max_upload_mb,
        help="Per-file upload size limit in MiB.",
    )
    parser.add_argument(
        "--heatmap-max-pixels",
        type=int,
        default=HuajiaoWebSettings.heatmap_max_pixels,
        help="Maximum ROI pixels used for heatmap model-response inference; downsampled if larger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = HuajiaoWebSettings(
        model_dir=args.model_dir,
        upload_root=args.upload_root,
        output_root=args.output_root,
        max_upload_mb=args.max_upload_mb,
        heatmap_max_pixels=args.heatmap_max_pixels,
    )
    app = create_app(settings=settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
