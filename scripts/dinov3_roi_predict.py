#!/usr/bin/env python3
"""Run single-image DINOv3 ROI segmentation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oil_content_detection.segmentation.predictor import DinoV3ROIPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--out-mask", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DinoV3ROIPredictor.from_artifact(args.artifact_dir, device=args.device)
    if args.threshold is not None:
        predictor.threshold = float(args.threshold)
    result = predictor.predict_image(
        args.image,
        sample_id=args.sample_id,
        out_mask_path=args.out_mask,
        overlay_path=args.overlay,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
