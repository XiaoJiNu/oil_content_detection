#!/usr/bin/env python3
"""
Refine ROI masks using Gray+Otsu segmentation within manual contours.
Assumes peppers are darker than the background (red paper).
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

def parse_args():
    parser = argparse.ArgumentParser(description="Refine masks using Otsu within manual ROI.")
    parser.add_argument("--manual-mask-dir", type=Path, required=True, help="Directory with manual binary masks")
    parser.add_argument("--raw-root", type=Path, default=Path("/home/yr/yr/data/huajiao_all/huajiao_all_pictures"), help="Directory containing raw images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ROI_masks_refined"), help="Output directory for refined masks")
    parser.add_argument("--visual-dir", type=Path, default=Path("data/processed/ROI_visual_refined"), help="Output directory for visualization")
    # Morphological parameters
    parser.add_argument("--close-size", type=int, default=5, help="Initial close size")
    parser.add_argument("--open-size", type=int, default=5, help="Open size")
    parser.add_argument("--hole-fill", type=int, default=300, help="Max hole size to fill")
    return parser.parse_args()

def find_image(sample_id, raw_root):
    # Try common extensions
    for ext in [".png", ".jpg", ".bmp"]:
        p = raw_root / f"{sample_id}{ext}"
        if p.exists():
            return p
    # Try raw_id logic if needed? Assuming filenames in huajiao_all_pictures are already normalized as sample_id.ext
    # Based on previous steps, they are.
    return None

def refine_mask(img_bgr, manual_mask, args):
    # 1. Gray + CLAHE
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # 2. Otsu in ROI
    roi_pixels = gray_enhanced[manual_mask > 0]
    if roi_pixels.size == 0:
        return np.zeros_like(manual_mask)
        
    thr, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Threshold (Keep Dark)
    # Peppers are dark, background (paper) is light/red.
    refined = (gray_enhanced < thr) & (manual_mask > 0)
    refined_u8 = (refined.astype(np.uint8) * 255)
    
    # 4. Morphology
    # Close: fill internal holes
    if args.close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close_size, args.close_size))
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, k)
        
    # Open: break connections
    if args.open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.open_size, args.open_size))
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_OPEN, k)
        
    # Final Close
    if args.close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close_size, args.close_size))
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, k)
        
    # 5. Fill Holes
    if args.hole_fill > 0:
        # Invert -> Find contours of holes -> fill if small
        inv = cv2.bitwise_not(refined_u8)
        # Note: raw input to connectedComponents is white object on black bg
        # But we want to find holes inside peppers.
        # So we look at 'inv', where holes are now white blobs.
        # But we must mask out the exterior of manual_mask so we don't fill the huge background.
        
        # Simpler approach: find contours of holes
        contours, hierarchy = cv2.findContours(refined_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Hierarchy: [Next, Previous, First_Child, Parent]
        # Holes have parents.
        for i, cnt in enumerate(contours):
            # If it is a hole (has a parent)
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(cnt)
                if area < args.hole_fill:
                    cv2.drawContours(refined_u8, [cnt], -1, 255, -1)
                    
    return refined_u8

def make_overlay(img_bgr, mask_u8):
    overlay = img_bgr.copy()
    # Green overlay for mask
    overlay[mask_u8 > 0] = overlay[mask_u8 > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    return overlay

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    
    manual_masks = sorted(list(args.manual_mask_dir.glob("*_mask.png")))
    print(f"Found {len(manual_masks)} manual masks.")
    
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
            
        refined_mask = refine_mask(img, manual_mask, args)
        
        # Save
        cv2.imwrite(str(args.output_dir / f"{sample_id}_mask.png"), refined_mask)
        
        # Vis
        vis = make_overlay(img, refined_mask)
        # Draw manual contour for comparison (Blue)
        contours, _ = cv2.findContours(manual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
        
        cv2.imwrite(str(args.visual_dir / f"{sample_id}_vis.jpg"), vis)
        
    print("Done.")

if __name__ == "__main__":
    main()
