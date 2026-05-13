#!/usr/bin/env python3
"""CLI for Huajiao oil-content inference from ENVI cube + RGB image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oil_content_detection.inference import HuajiaoOilPredictor, PredictionError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Huajiao oil content from same-size RGB image and ENVI cube.")
    parser.add_argument("--model-dir", required=True, help="Path to model artifact directory.")
    parser.add_argument("--image", required=True, help="RGB image path (.png/.jpg/.bmp).")
    parser.add_argument("--hdr", required=True, help="ENVI .hdr path; matching .dat must sit next to it.")
    parser.add_argument("--sample-id", default=None, help="Optional sample id to include in output JSON.")
    parser.add_argument("--out", default=None, help="Optional output JSON path. If omitted, prints JSON to stdout.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for ROI mask/overlay debug images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        predictor = HuajiaoOilPredictor.from_artifact(args.model_dir)
        result = predictor.predict(
            image_path=args.image,
            hdr_path=args.hdr,
            sample_id=args.sample_id,
            debug_dir=args.debug_dir,
        )
        payload = result.to_dict()
        exit_code = 0
    except PredictionError as exc:
        payload = {"qc": {"status": "error", "warnings": []}, "error": exc.to_dict()}
        exit_code = 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        payload = {"qc": {"status": "error", "warnings": []}, "error": {"code": "UNEXPECTED_ERROR", "message": str(exc), "details": {}}}
        exit_code = 1

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
