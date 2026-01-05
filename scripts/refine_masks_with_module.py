#!/usr/bin/env python3
"""Refine manual ROI masks with the gemini gray-otsu method.

Input:
  - CVAT XML polygons (manual contour around the whole pepper region)
  - Split file (train/val txt) to decide which samples to process
  - Image directory used for annotation (copied raw images)

Output:
  - Manual masks:   <manual-out>/<sample_id>_mask.png
  - Refined masks:  <refined-out>/<sample_id>_mask.png
  - Visualizations: <visual-out>/<sample_id>_vis.jpg  (green=pepper, blue=manual contour)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.data.cvat_annotations import CvatImageAnnotations, load_cvat_image_annotations  # noqa: E402
from oil_content_detection.pipelines.huajiao_dataset import normalize_sample_id  # noqa: E402
from oil_content_detection.preprocessing.roi_segmentation import PepperROIConfig, segment_pepper_in_mask  # noqa: E402


@dataclass(frozen=True)
class Paths:
    manual_mask_dir: Path
    refined_mask_dir: Path
    visual_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine manual ROI masks with CLAHE+Otsu gray segmentation")
    parser.add_argument("--split", type=Path, required=True, help="Path to train/val txt listing samples")
    parser.add_argument(
        "--cvat-xml",
        type=Path,
        default=Path("/home/yr/yr/data/huajiao_all/annotations_20251231_197.xml"),
        help="CVAT xml path containing manual polygons",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/home/yr/yr/data/huajiao_all/huajiao_all_pictures"),
        help="Directory containing the images referenced in the CVAT xml",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="peper",
        help="Polygon label name in CVAT xml (default: peper)",
    )
    parser.add_argument(
        "--manual-out",
        type=Path,
        default=Path("data/processed/ROI_masks_manual"),
        help="Directory to write manual contour masks",
    )
    parser.add_argument(
        "--refined-out",
        type=Path,
        default=Path("data/processed/ROI_masks_refined"),
        help="Directory to write refined pepper masks",
    )
    parser.add_argument(
        "--visual-out",
        type=Path,
        default=Path("data/processed/ROI_visual_refined"),
        help="Directory to write refinement visualizations",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def load_split_ids(split_path: Path) -> List[str]:
    ids: List[str] = []
    for line in split_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts:
            continue
        ids.append(normalize_sample_id(parts[0]))
    return ids


def _clip_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
    return pts


def rasterize_manual_mask(ann: CvatImageAnnotations, label: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required to rasterize polygons") from exc

    h, w = int(ann.height), int(ann.width)
    mask = np.zeros((h, w), dtype=np.uint8)
    polygons = ann.polygons.get(label, [])
    if not polygons:
        return mask.astype(bool)

    pts_list = []
    for pts in polygons:
        pts = _clip_points(pts, w=w, h=h)
        pts_i32 = np.round(pts).astype(np.int32)
        if pts_i32.shape[0] < 3:
            continue
        pts_list.append(pts_i32)
    if not pts_list:
        return mask.astype(bool)

    cv2.fillPoly(mask, pts_list, 255)
    return mask > 0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required to save masks") from exc

    _ensure_parent(path)
    mask_u8 = (np.asarray(mask).astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask_u8)


def save_visualization(image_bgr: np.ndarray, manual_mask: np.ndarray, refined_mask: np.ndarray, path: Path) -> None:
    """Green overlay for refined mask + blue contour for manual polygon."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required to save visualization") from exc

    vis = image_bgr.copy()
    manual = np.asarray(manual_mask).astype(bool)
    refined = np.asarray(refined_mask).astype(bool)

    # Green overlay for refined.
    alpha = 0.55
    green = np.zeros_like(vis, dtype=np.uint8)
    green[:, :] = (0, 255, 0)  # BGR
    vis[refined] = (vis[refined] * (1 - alpha) + green[refined] * alpha).astype(np.uint8)

    # Blue contour for manual.
    manual_u8 = (manual.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(manual_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 0, 0), thickness=2)  # blue in BGR

    _ensure_parent(path)
    cv2.imwrite(str(path), vis)


def load_image_bgr(images_dir: Path, image_name: str) -> Optional[np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required to load images") from exc

    path = images_dir / image_name
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def build_paths(args: argparse.Namespace) -> Paths:
    return Paths(manual_mask_dir=args.manual_out, refined_mask_dir=args.refined_out, visual_dir=args.visual_out)


def _sample_image_name(sample_id: str) -> str:
    return f"{sample_id}.png"


def main() -> None:
    args = parse_args()
    paths = build_paths(args)
    sample_ids = load_split_ids(args.split)

    annotations: Dict[str, CvatImageAnnotations] = load_cvat_image_annotations(args.cvat_xml)
    name_to_ann = annotations

    cfg = PepperROIConfig(
        segmentation_mode="gray_otsu",
        # Keep consistent with previous fix: disable label removal inside manual ROI.
        label_fallback_enabled=False,
        label_max_area_ratio=0.0,
        gray_close_size=5,
        gray_open_size=5,
        gray_final_close_size=5,
        max_hole_area=300,
        min_keep_area=64,
    )

    ok = 0
    skipped = 0
    missing_ann = 0
    missing_img = 0

    for sample_id in sample_ids:
        image_name = _sample_image_name(sample_id)
        ann = name_to_ann.get(image_name)
        if ann is None:
            missing_ann += 1
            print(f"[skip] No CVAT annotation for {image_name}")
            continue

        manual_mask_path = paths.manual_mask_dir / f"{sample_id}_mask.png"
        refined_mask_path = paths.refined_mask_dir / f"{sample_id}_mask.png"
        visual_path = paths.visual_dir / f"{sample_id}_vis.jpg"
        if not args.overwrite and manual_mask_path.exists() and refined_mask_path.exists() and visual_path.exists():
            skipped += 1
            continue

        img = load_image_bgr(args.images_dir, image_name)
        if img is None:
            missing_img += 1
            print(f"[skip] Missing image: {args.images_dir / image_name}")
            continue

        manual_mask = rasterize_manual_mask(ann, label=args.label)
        if manual_mask.sum() == 0:
            print(f"[skip] Empty manual mask for {image_name}")
            skipped += 1
            continue

        result = segment_pepper_in_mask(img, manual_mask, config=cfg)
        refined_mask = result.pepper_mask

        save_mask_png(manual_mask, manual_mask_path)
        save_mask_png(refined_mask, refined_mask_path)
        save_visualization(img, manual_mask, refined_mask, visual_path)
        ok += 1

    total = len(sample_ids)
    print("\n=== Summary ===")
    print(f"Split: {args.split} (samples={total})")
    print(f"OK: {ok}, skipped(existing): {skipped}")
    print(f"Missing annotation: {missing_ann}, missing image: {missing_img}")
    print(f"Manual masks: {paths.manual_mask_dir}")
    print(f"Refined masks: {paths.refined_mask_dir}")
    print(f"Visualizations: {paths.visual_dir}")


if __name__ == "__main__":
    main()

