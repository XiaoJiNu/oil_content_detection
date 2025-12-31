import cv2
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from oil_content_detection.preprocessing.roi_segmentation import (
    segment_pepper_in_mask,
    PepperROIConfig,
    _mask_to_uint8
)

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    
    for image in root.findall('image'):
        filename = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        
        polygons = []
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            # The user mentioned "label paper might exist", but the XML has label="pepper".
            # The user said "this region contains all peppers". 
            # We assume the polygon with label="pepper" (or any polygon if there's only one relevant one) 
            # defines the working area.
            # Looking at the XML snippet, label="pepper" seems to define the large region.
            points_str = polygon.get('points')
            points = []
            for p in points_str.split(';'):
                x, y = map(float, p.split(','))
                points.append([x, y])
            polygons.append(np.array(points, dtype=np.int32))
            
        annotations[filename] = {
            'size': (width, height),
            'polygons': polygons
        }
    return annotations

def main():
    parser = argparse.ArgumentParser(description="Generate ROI masks using XML annotations and advanced segmentation.")
    parser.add_argument('--input_dir', type=str, default="/home/yr/yr/data/huajiao_test", help="Directory with images")
    parser.add_argument('--xml_path', type=str, default="/home/yr/yr/data/huajiao_test/annotations.xml", help="Path to annotations.xml")
    parser.add_argument('--output_dir', type=str, default="/home/yr/yr/data/huajiao_test/results_advanced", help="Output directory")
    parser.add_argument('--debug', action='store_true', help="Save debug masks")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(args.xml_path):
        print(f"Error: XML file not found at {args.xml_path}")
        return

    print(f"Parsing annotations from {args.xml_path}...")
    annotations = parse_annotations(args.xml_path)
    
    print(f"Found {len(annotations)} annotated images.")
    
    # Configure the segmentation
    # Update 4: Switch to Gray+Otsu strategy based on experiment results.
    # This method is more robust for purple/dark peppers and varying lighting.
    # We keep the morphology settings (Close->Open->Close) and hole filling
    # as they effectively clean up the Otsu result.
    config = PepperROIConfig(
        segmentation_mode="gray_otsu",
        label_fallback_enabled=False,
        label_max_area_ratio=0.0,
        pepper_close_size=5,    # Initial/Final Close
        pepper_open_size=5,     # Open to separate
        hole_fill_max_area=300  # Fill internal holes
    )
    
    for filename, data in annotations.items():
        img_path = input_dir / filename
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
            
        print(f"Processing {filename}...")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Create mask from polygons
        # This mask defines the "paper" area (or the area containing peppers)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, data['polygons'], 255)
        
        # Perform segmentation within the mask
        try:
            result = segment_pepper_in_mask(img, roi_mask, config=config)
            
            # Save final result (transparent background or black background)
            # Create a visual result: original image masked by pepper_mask
            final_img = cv2.bitwise_and(img, img, mask=_mask_to_uint8(result.pepper_mask))
            
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), final_img)
            
            # Save debug images if requested
            if args.debug:
                base = str(debug_dir / img_path.stem)
                cv2.imwrite(f"{base}_0_input_mask.png", roi_mask)
                cv2.imwrite(f"{base}_1_label_mask.png", _mask_to_uint8(result.label_mask))
                cv2.imwrite(f"{base}_2_red_bg_mask.png", _mask_to_uint8(result.red_bg_mask))
                if result.texture_mask is not None:
                    cv2.imwrite(f"{base}_3_texture_mask.png", _mask_to_uint8(result.texture_mask))
                cv2.imwrite(f"{base}_4_final_mask.png", _mask_to_uint8(result.pepper_mask))
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()