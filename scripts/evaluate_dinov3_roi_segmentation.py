#!/usr/bin/env python3
"""Evaluate a DINOv3 huajiao ROI segmentation artifact on an oil-content split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from oil_content_detection.segmentation.data import (
    load_oil_split_samples,
    rasterize_polygons,
)
from oil_content_detection.segmentation.metrics import (
    aggregate_metric_rows,
    binary_segmentation_metrics,
)
from oil_content_detection.segmentation.predictor import DinoV3ROIPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--processed-dir", default="data/processed/huajiao_refined")
    parser.add_argument(
        "--annotation-xml",
        default="/home/yr/yr/data/huajiao_all/annotations_20251231_197.xml",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-overlays", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_manual_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = img.resize(
        (target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    out_dir = (
        Path(args.out_dir) if args.out_dir else artifact_dir / f"eval_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"

    predictor = DinoV3ROIPredictor.from_artifact(artifact_dir, device=args.device)
    if args.threshold is not None:
        predictor.threshold = float(args.threshold)
    samples = load_oil_split_samples(
        args.processed_dir, args.split, args.annotation_xml
    )

    rows = []
    for idx, sample in enumerate(tqdm(samples, desc=f"eval {args.split}", ncols=100)):
        out_mask = mask_dir / f"{sample.sample_id}_mask.png"
        overlay_path = (
            overlay_dir / f"{sample.sample_id}_overlay.png"
            if idx < args.save_overlays
            else None
        )
        pred = predictor.predict_image(
            sample.image_path,
            sample_id=sample.sample_id,
            out_mask_path=out_mask,
            overlay_path=overlay_path,
        )
        target = rasterize_polygons(sample.width, sample.height, sample.polygons)
        target = resize_manual_mask(target, pred.mask.shape)
        metrics = binary_segmentation_metrics(pred.mask, target)
        row = {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            **metrics,
            "mask_path": pred.mask_path,
            "overlay_path": pred.overlay_path,
        }
        rows.append(row)

    predictions_path = out_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    summary = aggregate_metric_rows(rows)
    summary.update(
        {
            "split": args.split,
            "n_samples": len(rows),
            "threshold": predictor.threshold,
            "predictions_csv": str(predictions_path),
        }
    )
    write_json(out_dir / "metrics.json", summary)

    # Also keep the latest eval summary at artifact root for handoff convenience.
    write_json(artifact_dir / f"metrics_{args.split}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
