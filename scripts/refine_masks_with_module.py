#!/usr/bin/env python3
"""
Refine ROI masks by calling the existing roi_segmentation module with gray_otsu mode
within manual contours.
"""
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure local src is importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.pipelines.huajiao_dataset import normalize_sample_id
from oil_content_detection.preprocessing.roi_segmentation import (
    PepperROIConfig,
    segment_pepper_in_mask,
    _mask_to_uint8
)

def parse_args():
    parser = argparse.ArgumentParser(description="Refine masks using roi_segmentation module within manual ROI.")
    parser.add_argument("--manual-mask-dir", type=Path, required=True, help="Directory with manual binary masks")
    parser.add_argument("--raw-root", type=Path, default=Path("/home/yr/yr/data/huajiao_all/huajiao_all_pictures"), help="Directory containing raw images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ROI_masks_refined"), help="Output directory for refined masks")
    parser.add_argument("--visual-dir", type=Path, default=Path("data/processed/ROI_visual_refined"), help="Output directory for visualization")
    return parser.parse_args()

def find_image(sample_id, raw_root):
    # Try common extensions
    for ext in [".png", ".jpg", ".bmp"]:
        p = raw_root / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None

def make_overlay(img_bgr, mask_u8):
    overlay = img_bgr.copy()
    # Green overlay for refined mask
    overlay[mask_u8 > 0] = overlay[mask_u8 > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    return overlay

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    
    manual_masks = sorted(list(args.manual_mask_dir.glob("*_mask.png")))
    print(f"Found {len(manual_masks)} manual masks.")
    
    # Configure the segmentation module to use Gray+Otsu
    # We keep default morphological params as requested in the plan
    # (implied by "retaining morphological optimizations")
    roi_config = PepperROIConfig(
        segmentation_mode="gray_otsu",
        pepper_close_size=5,
        pepper_open_size=5,
        hole_fill_max_area=300,
        label_fallback_enabled=False,  # 禁用标签位置回退（防止切掉左下角）
        label_max_area_ratio=0.0       # 禁用基于面积的标签检测
    )
    
    for mask_path in tqdm(manual_masks):
        sample_id = mask_path.stem.replace("_mask", "")
        img_path = find_image(sample_id, args.raw_root)
        
        if not img_path:
            print(f"Warning: Image not found for {sample_id}")
            continue
            
        img = cv2.imread(str(img_path))
        manual_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize manual mask if needed
        if manual_mask.shape != img.shape[:2]:
            manual_mask = cv2.resize(manual_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Call the module directly
        try:
            result = segment_pepper_in_mask(img, manual_mask, config=roi_config)
            refined_mask_u8 = result.pepper_mask_uint8()
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            # Fallback: empty mask? Or keep manual mask?
            # Let's produce an empty mask to signal failure, or just skip.
            # But skipping might break dataset alignment. 
            # If segmentation fails, it usually means empty ROI or similar.
            refined_mask_u8 = np.zeros_like(manual_mask)

        # Save
        cv2.imwrite(str(args.output_dir / f"{sample_id}_mask.png"), refined_mask_u8)
        
        # Visual Verification
        vis = make_overlay(img, refined_mask_u8)
        # Draw manual contour for comparison (Blue)
        contours, _ = cv2.findContours(manual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
        
        cv2.imwrite(str(args.visual_dir / f"{sample_id}_vis.jpg"), vis)
        
    print("Done.")

if __name__ == "__main__":
    main()
