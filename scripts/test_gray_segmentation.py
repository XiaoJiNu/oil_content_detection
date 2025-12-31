import cv2
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import os

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    for image in root.findall('image'):
        filename = image.get('name')
        polygons = []
        for polygon in image.findall('polygon'):
            points_str = polygon.get('points')
            points = []
            for p in points_str.split(';'):
                x, y = map(float, p.split(','))
                points.append([x, y])
            polygons.append(np.array(points, dtype=np.int32))
        annotations[filename] = polygons
    return annotations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/home/yr/yr/data/huajiao_test")
    parser.add_argument('--xml_path', type=str, default="/home/yr/yr/data/huajiao_test/annotations.xml")
    parser.add_argument('--output_dir', type=str, default="/home/yr/yr/data/huajiao_test/results_gray_test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = parse_annotations(args.xml_path)

    for filename, polygons in annotations.items():
        img_path = input_dir / filename
        if not img_path.exists(): continue
        
        print(f"Processing {filename}...")
        img = cv2.imread(str(img_path))
        if img is None: continue

        h, w = img.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, polygons, 255)

        # Extract ROI content
        masked_img = cv2.bitwise_and(img, img, mask=roi_mask)
        
        # 1. Standard Grayscale
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        
        # We only care about the ROI pixels for thresholding
        roi_pixels = gray[roi_mask > 0]
        
        if roi_pixels.size == 0: continue

        # Otsu on Grayscale
        thresh_val, binary_otsu = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Reconstruct the full mask
        # Note: Otsu returns threshold, we need to apply it
        # Depending on whether peppers are darker or lighter, we might need BINARY or BINARY_INV
        # Usually peppers (green/purple) are darker than the paper (white/reddish)?
        # Or maybe paper is dark red?
        # Let's save both versions.
        
        mask_otsu = np.zeros_like(gray)
        mask_otsu[roi_mask > 0] = np.where(roi_pixels > thresh_val, 255, 0).astype(np.uint8)
        
        mask_otsu_inv = np.zeros_like(gray)
        mask_otsu_inv[roi_mask > 0] = np.where(roi_pixels <= thresh_val, 255, 0).astype(np.uint8)

        # 2. Individual Channels
        b, g, r = cv2.split(masked_img)
        
        # Save visualizations
        cv2.imwrite(str(output_dir / f"{filename}_0_gray.png"), gray)
        cv2.imwrite(str(output_dir / f"{filename}_1_otsu.png"), mask_otsu)
        cv2.imwrite(str(output_dir / f"{filename}_2_otsu_inv.png"), mask_otsu_inv)
        
        # 3. CLAHE Gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        # Apply mask again to zero out background
        gray_clahe = cv2.bitwise_and(gray_clahe, gray_clahe, mask=roi_mask)
        
        roi_pixels_clahe = gray_clahe[roi_mask > 0]
        thresh_val_c, _ = cv2.threshold(roi_pixels_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        mask_otsu_clahe = np.zeros_like(gray)
        mask_otsu_clahe[roi_mask > 0] = np.where(roi_pixels_clahe > thresh_val_c, 255, 0).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"{filename}_3_gray_clahe.png"), gray_clahe)
        cv2.imwrite(str(output_dir / f"{filename}_4_otsu_clahe.png"), mask_otsu_clahe)

if __name__ == "__main__":
    main()
