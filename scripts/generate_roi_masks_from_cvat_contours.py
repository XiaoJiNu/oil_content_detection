#!/usr/bin/env python3
"""Generate pepper masks inside given CVAT polygon contours.

Typical usage (for the provided huajiao_test samples):

  python scripts/generate_roi_masks_from_cvat_contours.py \
    --input-dir /home/yr/yr/data/huajiao_test \
    --output-dir data/processed/ROI_from_contours_20251230 \
    --debug
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.data.cvat_annotations import load_cvat_image_annotations  # noqa: E402
from oil_content_detection.preprocessing.roi_segmentation import (  # noqa: E402
    PepperROIConfig,
    polygons_to_mask,
    segment_pepper_in_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pepper masks within given polygon ROI (CVAT XML).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/yr/yr/data/huajiao_test"),
        help="Directory containing images and annotations.xml",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to CVAT annotations.xml (defaults to <input-dir>/annotations.xml)",
    )
    parser.add_argument("--label", type=str, default="pepper", help="CVAT polygon label name to use as ROI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/ROI_from_contours_20251230"),
        help="Directory to save mask/overlay/debug outputs",
    )
    parser.add_argument("--debug", action="store_true", help="Save intermediate masks into --output-dir/debug/")
    parser.add_argument("--no-texture", action="store_true", help="Disable texture refinement")
    return parser.parse_args()


def _blend_overlay(image_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.6):
    overlay = image_bgr.copy()
    if mask_bool.sum() == 0:
        return overlay
    color = np.array([int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])], dtype=np.float32)
    base = overlay[mask_bool].astype(np.float32)
    blended = base * float(1.0 - alpha) + color * float(alpha)
    overlay[mask_bool] = np.clip(blended, 0, 255).astype(overlay.dtype)
    return overlay


def _draw_roi_polygons(image_bgr: np.ndarray, polygons_xy: List[np.ndarray], color_bgr: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for pts in polygons_xy:
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            continue
        pts_i = np.rint(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)
        cv2.polylines(out, [pts_i.reshape((-1, 1, 2))], isClosed=True, color=color_bgr, thickness=3)
    return out


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    ann_path: Path = args.annotations or (input_dir / "annotations.xml")
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PepperROIConfig(use_texture=not args.no_texture)

    annotations = load_cvat_image_annotations(ann_path)
    items: List[Tuple[str, Path, List[np.ndarray]]] = []
    for image_name, image_ann in annotations.items():
        polys = image_ann.polygons.get(args.label, [])
        if not polys:
            continue
        image_path = input_dir / image_name
        if not image_path.exists():
            print(f"[skip] image not found: {image_path}")
            continue
        items.append((image_name, image_path, polys))

    if not items:
        raise ValueError(f"No images with label='{args.label}' found in {ann_path}")

    summary_rows: List[Dict[str, str]] = []
    print(f"Processing {len(items)} images -> {out_dir}")

    debug_root = out_dir / "debug"
    for image_name, image_path, polys in items:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[skip] cannot read image: {image_path}")
            continue

        roi_mask = polygons_to_mask(polys, img.shape[:2])
        try:
            result = segment_pepper_in_mask(img, roi_mask, config=cfg)
        except Exception as exc:
            print(f"[skip] {image_name} failed: {exc}")
            continue

        stem = Path(image_name).stem
        mask_path = out_dir / f"{stem}_mask.png"
        overlay_path = out_dir / f"{stem}_overlay.png"
        roi_path = out_dir / f"{stem}_roi.png"

        cv2.imwrite(str(mask_path), result.pepper_mask_uint8())
        cv2.imwrite(str(roi_path), (roi_mask.astype(np.uint8) * 255).astype(np.uint8))

        overlay = _blend_overlay(img, result.pepper_mask, (0, 255, 0), alpha=0.55)
        overlay = _blend_overlay(overlay, result.label_mask, (0, 0, 255), alpha=0.55)
        overlay = _draw_roi_polygons(overlay, polys, color_bgr=(255, 0, 0))
        cv2.imwrite(str(overlay_path), overlay)

        if args.debug:
            debug_dir = debug_root / stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{stem}_roi.png"), (roi_mask.astype(np.uint8) * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / f"{stem}_label_mask.png"), (result.label_mask.astype(np.uint8) * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / f"{stem}_red_bg_mask.png"), (result.red_bg_mask.astype(np.uint8) * 255).astype(np.uint8))
            cv2.imwrite(
                str(debug_dir / f"{stem}_pepper_candidate.png"),
                (result.pepper_candidate.astype(np.uint8) * 255).astype(np.uint8),
            )
            if result.texture_mask is not None:
                cv2.imwrite(
                    str(debug_dir / f"{stem}_texture_mask.png"),
                    (result.texture_mask.astype(np.uint8) * 255).astype(np.uint8),
                )
            cv2.imwrite(str(debug_dir / f"{stem}_pepper_mask.png"), result.pepper_mask_uint8())

        roi_pixels = float(roi_mask.sum())
        pepper_pixels = float(result.pepper_mask.sum())
        label_pixels = float(result.label_mask.sum())
        row = {
            "image": image_name,
            "roi_pixels": f"{roi_pixels:.0f}",
            "pepper_pixels": f"{pepper_pixels:.0f}",
            "label_pixels": f"{label_pixels:.0f}",
            "pepper_ratio_in_roi": f"{(pepper_pixels / roi_pixels) if roi_pixels > 0 else 0.0:.6f}",
        }
        for k, v in sorted(result.info.items()):
            row[f"info_{k}"] = f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)
        summary_rows.append(row)

        print(f"[ok] {image_name}: mask={mask_path} overlay={overlay_path}")

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        fieldnames: List[str] = []
        seen = set()
        for row in summary_rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[ok] summary={summary_path}")


if __name__ == "__main__":
    main()

