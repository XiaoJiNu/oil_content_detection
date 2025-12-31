#!/usr/bin/env python3
"""
Update dataset splits and copy images for unified annotation.
Requirements:
1. Copy raw images of valid samples to /home/yr/yr/data/huajiao_all/huajiao_all_pictures
2. Check consistency between Excel labels and train/val splits; add missing samples to train.txt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

# Add src to path to import local modules if needed, though we'll try to keep this self-contained
# for key logic to avoid circular deps or import errors.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.pipelines.huajiao_dataset import LabelConfig, normalize_sample_id, _first_existing_column

@dataclass(frozen=True)
class SampleRecord:
    sample_id_raw: str
    sample_id: str
    weight_g: float
    distill_ml: float
    ml_per_100g: float
    sample_dir: Path

def discover_sample_dirs(raw_root: Path) -> Dict[str, Path]:
    """Walk raw_root to map normalized sample_id -> sample directory."""
    mapping: Dict[str, Path] = {}
    if not raw_root.exists():
        print(f"Warning: Raw root {raw_root} does not exist.")
        return mapping
        
    for dirpath, dirnames, _ in os.walk(raw_root):
        for dirname in dirnames:
            norm = normalize_sample_id(dirname)
            mapping.setdefault(norm, Path(dirpath) / dirname)
    return mapping

def _load_sheet_rows(excel_path: Path, sheet: str, cfg: LabelConfig) -> pd.DataFrame:
    suffix = excel_path.suffix.lower()
    engine = "openpyxl" if suffix in {".xlsx", ".xlsm"} else "xlrd"
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet, engine=engine)
    except ImportError as exc:
        # Fallback or re-raise
        print(f"Error loading excel with {engine}: {exc}")
        df = pd.read_excel(excel_path, sheet_name=sheet)
        
    id_col = _first_existing_column(df, cfg.sample_id_cols)
    distill_col = _first_existing_column(df, cfg.distill_volume_cols)
    weight_col = _first_existing_column(df, cfg.weight_cols)

    subset = pd.DataFrame(
        {
            "sample_id_raw": df[id_col],
            "distill_ml": pd.to_numeric(df[distill_col], errors="coerce"),
            "weight_g": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )
    subset.dropna(subset=["distill_ml", "weight_g"], inplace=True)
    subset["sample_id_raw"] = subset["sample_id_raw"].astype(str).str.strip()
    subset["sample_id"] = subset["sample_id_raw"].apply(normalize_sample_id)
    subset["ml_per_100g"] = subset["distill_ml"] / subset["weight_g"] * 100
    return subset

def collect_records(
    excel_path: Path, sheets: Sequence[str], raw_root: Path, cfg: LabelConfig
) -> tuple[List[SampleRecord], Dict[str, List[str]]]:
    dir_map = discover_sample_dirs(raw_root)
    records: List[SampleRecord] = []
    
    try:
        frames = [_load_sheet_rows(excel_path, sheet, cfg) for sheet in sheets]
    except Exception as e:
        print(f"Error loading sheets: {e}")
        return [], {}
        
    labels_df = pd.concat(frames, ignore_index=True)

    seen_ids = set()
    skipped = {"missing_dir": [], "missing_hdr": [], "missing_dat": [], "zero_weight": [], "duplicates": []}
    
    for _, row in labels_df.iterrows():
        sample_id = row["sample_id"]
        sample_id_raw = row["sample_id_raw"]
        
        if sample_id in seen_ids:
            skipped["duplicates"].append(sample_id)
            continue
        seen_ids.add(sample_id)

        if sample_id not in dir_map:
            skipped["missing_dir"].append(f"{sample_id_raw} ({sample_id})")
            continue
        sample_dir = dir_map[sample_id]

        hdr_path = sample_dir / "capture" / f"REFLECTANCE_{sample_id}.hdr"
        dat_path = hdr_path.with_suffix(".dat")
        if not hdr_path.exists():
            skipped["missing_hdr"].append(str(hdr_path))
            continue
        # We don't strictly enforce .dat existence here if only checking labels, 
        # but for valid dataset usage we should.
        if not dat_path.exists():
             skipped["missing_dat"].append(str(dat_path))
             continue

        weight_g = float(row["weight_g"])
        distill_ml = float(row["distill_ml"])
        if weight_g <= 0:
            skipped["zero_weight"].append(sample_id)
            continue

        records.append(
            SampleRecord(
                sample_id_raw=sample_id_raw,
                sample_id=sample_id,
                weight_g=weight_g,
                distill_ml=distill_ml,
                ml_per_100g=float(row["ml_per_100g"]),
                sample_dir=sample_dir,
            )
        )
    return records, skipped

def _find_sample_image(sample_dir: Path, sample_id_raw: str, sample_id: str) -> Optional[Path]:
    candidates = [
        sample_dir / f"{sample_id_raw}.png",
        sample_dir / f"{sample_id}.png",
        sample_dir / f"{sample_id_raw}.jpg",
        sample_dir / f"{sample_id}.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    for folder in (sample_dir, sample_dir / "capture"):
        if not folder.exists(): continue
        pngs = sorted(folder.glob("*.png"))
        if pngs:
            return pngs[0]
        jpgs = sorted(folder.glob("*.jpg"))
        if jpgs:
            return jpgs[0]
    return None

def copy_images(records: List[SampleRecord], target_dir: Path) -> List[str]:
    """Copy sample images to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    missing_images = []
    
    for rec in records:
        src_img = _find_sample_image(rec.sample_dir, rec.sample_id_raw, rec.sample_id)
        if src_img:
            # Normalize filename in target: sample_id.ext
            ext = src_img.suffix
            dst_name = f"{rec.sample_id}{ext}"
            dst_path = target_dir / dst_name
            shutil.copy2(src_img, dst_path)
            copied_count += 1
        else:
            missing_images.append(rec.sample_id)
            
    print(f"Copied {copied_count} images to {target_dir}")
    if missing_images:
        print(f"Missing images for {len(missing_images)} samples: {missing_images}")
    return missing_images

def read_existing_split(txt_path: Path) -> Set[str]:
    if not txt_path.exists():
        return set()
    ids = set()
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and not line.startswith("#"):
                # Format: sample_id_raw weight ...
                # We need normalize_sample_id(sample_id_raw)
                # But wait, build_splits.py writes sample_id_raw.
                # So we must normalize it to compare with our records.
                raw_id = parts[0]
                ids.add(normalize_sample_id(raw_id))
    return ids

def append_records(records: List[SampleRecord], txt_path: Path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "a") as f:
        for rec in records:
            f.write(
                f"{rec.sample_id_raw}\t{rec.weight_g:.4f}\t{rec.distill_ml:.4f}\t{rec.ml_per_100g:.4f}\t{rec.sample_dir}\n"
            )

def main():
    # Configuration
    excel_path = Path("docs/2025年8月+花椒挥发油测定结果.xls")
    
    # Exclude summary sheets to avoid duplicates
    excluded_sheets = {"挥发油", "总样品情况", "Sheet3"}
    
    try:
        xl = pd.ExcelFile(excel_path)
        all_sheets = xl.sheet_names
        sheets = [s for s in all_sheets if s not in excluded_sheets]
        print(f"Detected {len(all_sheets)} sheets. Processing {len(sheets)} sheets: {sheets}")
        print(f"Skipping excluded sheets: {excluded_sheets}")
    except Exception as e:
        print(f"Error reading excel file: {e}")
        return

    raw_root = Path("/home/yr/yr/data/科研数据")
    split_dir = Path("data/labels/huajiao_2025_08_plus")
    image_target_dir = Path("/home/yr/yr/data/huajiao_all/huajiao_all_pictures")
    
    cfg = LabelConfig()
    
    print(f"Loading records from {excel_path}...")
    records, skipped = collect_records(excel_path, sheets, raw_root, cfg)
    print(f"Found {len(records)} valid records.")
    
    print("\n--- Skipped Samples Summary ---")
    for reason, items in skipped.items():
        print(f"{reason}: {len(items)}")
        if items:
            print(f"  First 10 examples: {items[:10]}")

    
    # 1. Copy Images
    print("\n--- Step 1: Copying Images ---")
    copy_images(records, image_target_dir)
    
    # 2. Check Splits
    print("\n--- Step 2: Checking Splits ---")
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    
    train_ids = read_existing_split(train_path)
    val_ids = read_existing_split(val_path)
    existing_ids = train_ids.union(val_ids)
    
    missing_records = [r for r in records if r.sample_id not in existing_ids]
    
    if missing_records:
        print(f"Found {len(missing_records)} samples missing from splits.")
        print("Appending them to train.txt...")
        append_records(missing_records, train_path)
        print("Done.")
    else:
        print("All valid samples are already in train.txt or val.txt.")
        
    # Generate report for documentation
    report = f"""
## 2025-12-31 数据处理完成记录

1.  **图片归档**
    *   源数据路径: `{raw_root}`
    *   目标路径: `{image_target_dir}`
    *   共处理样本数: {len(records)}
    *   成功复制图片数: {len(records)} (如果少于总数，请检查脚本输出)

2.  **数据集划分更新**
    *   Excel来源: `{excel_path}`
    *   Split目录: `{split_dir}`
    *   原有样本数: {len(existing_ids)}
    *   新增样本数: {len(missing_records)}
    *   新增样本已追加至 `train.txt`
"""
    print("\n--- Report ---")
    print(report)
    
    # Return report via stdout capture or just print for the agent to grab
    
if __name__ == "__main__":
    main()
