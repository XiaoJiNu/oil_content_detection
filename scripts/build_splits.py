#!/usr/bin/env python3
"""Generate train/val txt splits from Excel labels and raw sample directories."""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.pipelines.huajiao_dataset import LabelConfig, _first_existing_column, normalize_sample_id


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
        raise ImportError(f"{engine} is required to read {excel_path}; please install it") from exc
    except Exception as exc:  # fallback attempt
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
    """Load labels from sheets and match to raw directories with HDR/DAT files."""
    dir_map = discover_sample_dirs(raw_root)
    records: List[SampleRecord] = []
    skipped: Dict[str, List[str]] = {
        "missing_dir": [],
        "missing_hdr": [],
        "missing_dat": [],
        "zero_weight": [],
        "duplicates": [],
    }

    frames = [_load_sheet_rows(excel_path, sheet, cfg) for sheet in sheets]
    labels_df = pd.concat(frames, ignore_index=True)

    seen_ids = set()
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


def write_records(records: Iterable[SampleRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(
                f"{rec.sample_id_raw}\t{rec.weight_g:.4f}\t{rec.distill_ml:.4f}\t{rec.ml_per_100g:.4f}\t{rec.sample_dir}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val splits for huajiao hyperspectral data")
    parser.add_argument("--excel", type=Path, default=Path("docs/2025年8月花椒挥发油测定结果.xlsx"), help="Path to label Excel file")
    #                                                       docs/2025年8月花椒挥发油测定结果.xls
    parser.add_argument(
        "--sheets",
        nargs="+",
        default=["云南竹叶椒和云南藤椒1"],
        help="Sheet names to read",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("/home/yr/yr/data/科研数据"), help="Root directory for raw samples")
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"), help="Output directory for txt splits")
    parser.add_argument("--train-file", type=str, default="train.txt", help="Train txt filename")
    parser.add_argument("--val-file", type=str, default="val.txt", help="Val txt filename")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train proportion")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for shuffling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LabelConfig()

    records, skipped = collect_records(args.excel, args.sheets, args.raw_root, cfg)
    if not records:
        raise RuntimeError("No valid samples found; please check Excel and raw data paths")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    split_idx = max(1, int(len(records) * args.train_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    out_train = args.out_dir / args.train_file
    out_val = args.out_dir / args.val_file
    write_records(train_records, out_train)
    write_records(val_records, out_val)

    print(f"Total samples: {len(records)}, train: {len(train_records)}, val: {len(val_records)}")
    print(f"Saved train list to {out_train}")
    print(f"Saved val list to {out_val}")
    print("---- Skipped samples summary ----")
    print(f"Missing directory: {len(skipped['missing_dir'])} -> {skipped['missing_dir']}")
    print(f"Missing HDR: {len(skipped['missing_hdr'])} -> {skipped['missing_hdr']}")
    print(f"Missing DAT: {len(skipped['missing_dat'])} -> {skipped['missing_dat']}")
    print(f"Zero/invalid weight: {len(skipped['zero_weight'])} -> {skipped['zero_weight']}")
    print(f"Duplicate IDs in Excel: {len(skipped['duplicates'])} -> {skipped['duplicates']}")


if __name__ == "__main__":
    main()
