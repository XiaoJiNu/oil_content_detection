#!/usr/bin/env python3
"""Build aggregated spectra/metadata from train/val split txt files with ROI visualization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.pipelines.huajiao_dataset import (
    AggregationConfig,
    HuajiaoROIConfig,
    build_huajiao_dataset_from_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build huajiao dataset from train/val split txt")
    parser.add_argument("--split", type=Path, default="data/labels/train.txt", help="Path to train/val txt file")
    parser.add_argument("--output-dir", type=Path, default="data/processed/huajiao/train", help="Directory to save spectra/metadata")
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=Path("data/processed/ROI"),
        help="Directory to save ROI visualization overlays",
    )
    parser.add_argument(
        "--roi-mask-dir",
        type=Path,
        default=Path("data/processed/ROI"),
        help="Directory containing pre-generated ROI masks (e.g., from generate_roi_masks.py)",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.10, help="Trimmed mean proportion for aggregation")
    parser.add_argument(
        "--include-stats",
        nargs="+",
        default=["trimmed_mean"],
        help="Stats to include (subset of mean, median, trimmed_mean, std)",
    )
    parser.add_argument("--primary-stat", type=str, default="trimmed_mean", help="Primary stat used for wl_<nm> columns")
    parser.add_argument("--nir-target", type=float, default=800.0, help="NIR wavelength for ROI ratio (nm)")
    parser.add_argument("--red-target", type=float, default=650.0, help="Red wavelength for ROI ratio (nm)")
    parser.add_argument("--ratio-quantile", type=float, default=0.90, help="Quantile for NIR/Red ratio threshold")
    parser.add_argument("--ratio-floor", type=float, default=1.05, help="Floor for NIR/Red ratio threshold")
    parser.add_argument("--intensity-quantile", type=float, default=0.15, help="Quantile for intensity threshold")
    parser.add_argument("--closing-size", type=int, default=3, help="Morphological closing kernel size")
    parser.add_argument("--opening-size", type=int, default=3, help="Morphological opening kernel size")
    parser.add_argument("--min-area", type=int, default=64, help="Minimum ROI area (pixels)")
    parser.add_argument("--clip-low", type=float, default=0.01, help="Lower quantile for extreme removal")
    parser.add_argument("--clip-high", type=float, default=0.99, help="Upper quantile for extreme removal")
    parser.add_argument(
        "--hsv-lower",
        nargs=3,
        type=int,
        default=[30, 70, 30],
        help="HSV lower bound (H S V) used for RGB-based ROI mask",
    )
    parser.add_argument(
        "--hsv-upper",
        nargs=3,
        type=int,
        default=[65, 255, 255],
        help="HSV upper bound (H S V) used for RGB-based ROI mask",
    )
    parser.add_argument("--hsv-max-width", type=int, default=400, help="Resize image width before HSV (0/neg to disable)")
    parser.add_argument("--hsv-blur-ksize", type=int, default=5, help="Gaussian blur kernel size before HSV threshold")
    parser.add_argument("--hsv-closing-size", type=int, default=5, help="Morph closing kernel size for HSV mask")
    parser.add_argument("--hsv-opening-size", type=int, default=3, help="Morph opening kernel size for HSV mask")
    parser.add_argument("--hsv-min-area", type=int, default=256, help="Minimum HSV mask area to keep")
    parser.add_argument("--hsv-border-crop", type=float, default=0.02, help="Border crop ratio before HSV masking")
    parser.add_argument("--no-hsv", action="store_true", help="Disable HSV-based masking and fallback to spectral ratio")
    parser.add_argument("--no-save", action="store_true", help="Do not write parquet/CSV outputs")
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[HuajiaoROIConfig, AggregationConfig]:
    roi_cfg = HuajiaoROIConfig(
        nir_target_nm=args.nir_target,
        red_target_nm=args.red_target,
        ratio_quantile=args.ratio_quantile,
        ratio_floor=args.ratio_floor,
        intensity_quantile=args.intensity_quantile,
        closing_size=args.closing_size,
        opening_size=args.opening_size,
        min_area=args.min_area,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        use_hsv=not args.no_hsv,
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
        hsv_blur_ksize=args.hsv_blur_ksize,
        hsv_closing_size=args.hsv_closing_size,
        hsv_opening_size=args.hsv_opening_size,
        hsv_min_area=args.hsv_min_area,
        hsv_border_crop=args.hsv_border_crop,
        hsv_max_width=None if args.hsv_max_width and args.hsv_max_width <= 0 else args.hsv_max_width,
    )
    agg_cfg = AggregationConfig(
        trim_fraction=args.trim_fraction,
        primary_stat=args.primary_stat,
        include_stats=tuple(args.include_stats),
    )
    return roi_cfg, agg_cfg


def main() -> None:
    args = parse_args()
    roi_cfg, agg_cfg = build_configs(args)

    spectra_df, metadata_df = build_huajiao_dataset_from_split(
        split_path=args.split,
        output_dir=args.output_dir,
        roi_config=roi_cfg,
        agg_config=agg_cfg,
        save=not args.no_save,
        roi_visualization_dir=args.roi_dir,
        roi_mask_dir=args.roi_mask_dir,
    )

    print(f"Built dataset with {len(spectra_df)} samples from {args.split}")
    print(f"Spectra columns: {len(spectra_df.columns)}; Metadata rows: {len(metadata_df)}")
    if not args.no_save:
        print(f"Saved spectra/metadata under {args.output_dir}")
    if args.roi_dir:
        print(f"ROI overlays (if generated) saved under {args.roi_dir}")


if __name__ == "__main__":
    main()
