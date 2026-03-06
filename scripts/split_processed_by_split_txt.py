#!/usr/bin/env python3
"""Create new processed train/val tables by splitting an existing processed dataset.

This avoids re-running ROI extraction: it simply concatenates the existing
processed train+val tables, then filters rows by the sample IDs listed in the
new train/val txt files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def _parse_split_ids(path: Path) -> Tuple[List[str], Set[str]]:
    sample_ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split(maxsplit=4)
            if len(parts) < 5:
                raise ValueError(f"Invalid split line: {line}")
            sample_dir = Path(str(parts[4]).strip())
            sid = sample_dir.name
            sample_ids.append(sid)
    return sample_ids, set(sample_ids)


def _load_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"File not found: {path} (or {csv_path})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split processed huajiao dataset by split txt files")
    parser.add_argument(
        "--source-train-dir",
        type=Path,
        default=Path("data/processed/huajiao_refined/train"),
        help="Existing processed train directory",
    )
    parser.add_argument(
        "--source-val-dir",
        type=Path,
        default=Path("data/processed/huajiao_refined/val"),
        help="Existing processed val directory",
    )
    parser.add_argument("--train-split", type=Path, required=True, help="New train.txt")
    parser.add_argument("--val-split", type=Path, required=True, help="New val.txt")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root, will create {train,val} subfolders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_list, train_set = _parse_split_ids(args.train_split)
    val_list, val_set = _parse_split_ids(args.val_split)
    if train_set & val_set:
        overlap = sorted(list(train_set & val_set))[:10]
        raise ValueError(f"Train/val split overlap found, e.g. {overlap}")

    # Load and concatenate existing processed tables.
    train_spectra = _load_table(args.source_train_dir / "huajiao_spectra.parquet")
    val_spectra = _load_table(args.source_val_dir / "huajiao_spectra.parquet")
    full_spectra = pd.concat([train_spectra, val_spectra], ignore_index=True)

    train_meta = _load_table(args.source_train_dir / "huajiao_metadata.parquet")
    val_meta = _load_table(args.source_val_dir / "huajiao_metadata.parquet")
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Filter by sample_id.
    if "sample_id" not in full_spectra.columns or "sample_id" not in full_meta.columns:
        raise KeyError("Both spectra and metadata tables must contain 'sample_id'")

    full_spectra["sample_id"] = full_spectra["sample_id"].astype(str)
    full_meta["sample_id"] = full_meta["sample_id"].astype(str)

    out_train_dir = args.out_root / "train"
    out_val_dir = args.out_root / "val"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_val_dir.mkdir(parents=True, exist_ok=True)

    train_spectra_new = full_spectra[full_spectra["sample_id"].isin(train_set)].copy()
    val_spectra_new = full_spectra[full_spectra["sample_id"].isin(val_set)].copy()
    train_meta_new = full_meta[full_meta["sample_id"].isin(train_set)].copy()
    val_meta_new = full_meta[full_meta["sample_id"].isin(val_set)].copy()

    # Sanity checks (order in txt is not enforced; we just check counts).
    missing_train = sorted(list(train_set - set(train_spectra_new["sample_id"].tolist())))[:10]
    missing_val = sorted(list(val_set - set(val_spectra_new["sample_id"].tolist())))[:10]
    if missing_train:
        raise ValueError(f"Missing train sample_ids in source processed spectra, e.g. {missing_train}")
    if missing_val:
        raise ValueError(f"Missing val sample_ids in source processed spectra, e.g. {missing_val}")

    train_spectra_new.to_parquet(out_train_dir / "huajiao_spectra.parquet", index=False)
    val_spectra_new.to_parquet(out_val_dir / "huajiao_spectra.parquet", index=False)
    train_meta_new.to_parquet(out_train_dir / "huajiao_metadata.parquet", index=False)
    val_meta_new.to_parquet(out_val_dir / "huajiao_metadata.parquet", index=False)

    logger.info(
        "Wrote new split: train=%d val=%d (out=%s)",
        len(train_spectra_new),
        len(val_spectra_new),
        args.out_root,
    )
    print("=== Processed split created ===")
    print(f"Out: {args.out_root}")
    print(f"Train spectra: {len(train_spectra_new)} rows")
    print(f"Val spectra:   {len(val_spectra_new)} rows")


if __name__ == "__main__":
    main()

