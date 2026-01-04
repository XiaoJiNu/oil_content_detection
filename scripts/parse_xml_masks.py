#!/usr/bin/env python3
"""
Parse CVAT XML annotations and generate binary masks for each sample.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# Ensure we can import local src if needed (though this script is mostly standalone)
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.pipelines.huajiao_dataset import normalize_sample_id

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CVAT XML polygons to mask images.")
    parser.add_argument(
        "--xml", 
        type=Path, 
        required=True,
        help="Path to CVAT annotations XML file"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("data/processed/ROI_masks_manual"),
        help="Directory to save generated mask images"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="peper",
        help="Polygon label to extract (default: 'peper')"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.xml.exists():
        print(f"Error: XML file not found at {args.xml}")
        sys.exit(1)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing {args.xml}...")
    tree = ET.parse(args.xml)
    root = tree.getroot()
    
    count = 0
    skipped = 0
    
    # Iterate over 'image' tags
    for image_elem in root.findall("image"):
        name = image_elem.get("name") # e.g. "0035_11.png"
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))
        
        # Derive sample_id
        # Assuming name is like "0035_11.png" or "0035_11.jpg"
        stem = Path(name).stem
        sample_id = normalize_sample_id(stem)
        
        # Find polygons for the target label
        polygons = []
        for poly_elem in image_elem.findall("polygon"):
            label = poly_elem.get("label")
            if label == args.label:
                points_str = poly_elem.get("points")
                # points format: "x1,y1;x2,y2;..."
                pts = []
                for pair in points_str.split(';'):
                    x, y = map(float, pair.split(','))
                    pts.append([x, y])
                polygons.append(np.array(pts, dtype=np.int32))
                
        if not polygons:
            # print(f"Skipping {name}: no polygons with label '{args.label}'")
            skipped += 1
            continue
            
        # Create mask
        # Note: height, width order for shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill polygons
        # cv2.fillPoly expects a list of arrays
        cv2.fillPoly(mask, polygons, 255)
        
        # Save mask
        out_path = args.output_dir / f"{sample_id}_mask.png"
        cv2.imwrite(str(out_path), mask)
        count += 1
        
    print(f"Processed {count} images.")
    print(f"Skipped {skipped} images (no matching label).")
    print(f"Masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()
