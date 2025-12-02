#!/usr/bin/env python3
"""Generate and save HSV-based ROI masks/overlays for huajiao samples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.preprocessing.hsv_tuner import apply_hsv_mask_simple  # noqa: E402
from oil_content_detection.pipelines.huajiao_dataset import normalize_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROI masks/overlays using HSV thresholds (simple mode matching hsv_debugger)"
    )
    parser.add_argument("--split", type=Path, required=True, help="Path to train/val txt listing samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/ROI"),
        help="Directory to save mask and overlay images",
    )
    parser.add_argument(
        "--hsv-lower",
        nargs=3,
        type=int,
        default=[30, 70, 30],
        help="HSV lower bound (H S V)",
    )
    parser.add_argument(
        "--hsv-upper",
        nargs=3,
        type=int,
        default=[65, 255, 255],
        help="HSV upper bound (H S V)",
    )
    parser.add_argument(
        "--hsv-max-width",
        type=int,
        default=0,
        help="Resize width before HSV (<=0 to disable and use original size, 400 to match hsv_debugger)",
    )
    parser.add_argument("--pattern", type=str, default="*.png", help="Image glob pattern inside each sample dir")
    return parser.parse_args()


def find_image(sample_dir: Path, pattern: str) -> Path | None:
    files: List[Path] = sorted(sample_dir.glob(pattern))
    if not files:
        files = sorted(sample_dir.glob("*.jpg"))
    return files[0] if files else None


def load_split_samples(split_path: Path) -> List[Tuple[str, Path]]:
    """Load sample_id and sample_dir from train/val txt."""
    samples: List[Tuple[str, Path]] = []
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    for line in split_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        sample_id_raw = parts[0]
        sample_dir = Path(" ".join(parts[4:])) if len(parts) > 5 else Path(parts[4])
        sample_id = normalize_sample_id(sample_id_raw)
        samples.append((sample_id, sample_dir))
    return samples


def save_mask_and_overlay(image_path: Path, mask: np.ndarray, out_dir: Path, sample_id: str) -> Tuple[Path, Path]:
    """保存mask和overlay，尺寸与输入mask一致。

    如果在原始尺寸上分割（max_width=0），则mask和overlay都是原始尺寸。
    如果缩放后分割（max_width>0），则mask和overlay都是缩放后的尺寸。
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # 读取原始图像
    img_original = Image.open(image_path).convert("RGB")

    # 如果mask比原图小，说明是缩放后的mask，需要将原图也缩放到相同尺寸
    # 如果mask与原图尺寸一致或更大，则直接使用原图（或crop原图）
    if mask.shape[0] < img_original.height or mask.shape[1] < img_original.width:
        # 将原图缩放到与mask相同的尺寸
        img_resized = img_original.resize((mask.shape[1], mask.shape[0]), resample=Image.LANCZOS)
        img_array = np.asarray(img_resized, dtype=float) / 255.0
    else:
        # mask与原图尺寸一致，直接使用
        img_array = np.asarray(img_original, dtype=float) / 255.0
        # 如果mask更大，需要裁剪到原图大小
        if mask.shape[0] > img_original.height or mask.shape[1] > img_original.width:
            mask = mask[:img_original.height, :img_original.width]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存mask（保持与分割时相同的尺寸）
    mask_path = out_dir / f"{sample_id}_mask.png"
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)

    # 生成overlay（使用对应尺寸的图像）
    overlay = img_array.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6
    overlay_path = out_dir / f"{sample_id}_overlay.png"
    plt.imsave(overlay_path, np.clip(overlay, 0.0, 1.0))

    return mask_path, overlay_path


def main() -> None:
    args = parse_args()
    lower = tuple(args.hsv_lower)
    upper = tuple(args.hsv_upper)
    max_width = None if args.hsv_max_width and args.hsv_max_width <= 0 else args.hsv_max_width
    out_dir = args.output_dir
    samples = load_split_samples(args.split)

    print(f"Using HSV thresholds: lower={lower}, upper={upper}, max_width={max_width}")
    print(f"Processing {len(samples)} samples...")

    for sample_id, sample_dir in samples:
        image_path = find_image(sample_dir, args.pattern)
        if not image_path:
            print(f"[skip] No image for {sample_id} in {sample_dir}")
            continue

        try:
            mask = apply_hsv_mask_simple(
                str(image_path),
                lower_hsv=lower,
                upper_hsv=upper,
                max_width=max_width,
            )
        except Exception as exc:
            print(f"[skip] {sample_id} failed: {exc}")
            continue

        mask_bin = (mask > 0).astype(np.uint8)
        mask_path, overlay_path = save_mask_and_overlay(image_path, mask_bin, out_dir, sample_id)
        print(f"[ok] {sample_id}: mask={mask_path}, overlay={overlay_path}")


if __name__ == "__main__":
    main()
