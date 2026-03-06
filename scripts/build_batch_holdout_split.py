#!/usr/bin/env python3
"""Create a batch-heldout train/val split from existing label txt files.

This script groups samples by batch (the parent directory name of sample_dir)
and assigns whole batches to either train or val. This is required for
evaluating true cross-batch generalization.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LabelRecord:
    raw_line: str
    sample_id_raw: str
    weight_g: float
    distill_ml: float
    ml_per_100g: float
    sample_dir: Path
    batch: str


def _parse_label_line(line: str) -> Tuple[str, float, float, float, Path]:
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    parts = line.split("\t")
    if len(parts) < 5:
        parts = line.split(maxsplit=4)
    if len(parts) < 5:
        raise ValueError(f"Invalid label line (need 5 fields): {line}")

    sample_id_raw = str(parts[0]).strip()
    weight_g = float(parts[1])
    distill_ml = float(parts[2])
    ml_per_100g = float(parts[3])
    sample_dir = Path(str(parts[4]).strip())
    return sample_id_raw, weight_g, distill_ml, ml_per_100g, sample_dir


def load_label_records(paths: List[Path]) -> List[LabelRecord]:
    records: List[LabelRecord] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw_line = raw.rstrip("\n")
                if not raw_line.strip():
                    continue
                sample_id_raw, weight_g, distill_ml, ml_per_100g, sample_dir = _parse_label_line(raw_line)
                batch = sample_dir.parent.name
                records.append(
                    LabelRecord(
                        raw_line=raw_line,
                        sample_id_raw=sample_id_raw,
                        weight_g=weight_g,
                        distill_ml=distill_ml,
                        ml_per_100g=ml_per_100g,
                        sample_dir=sample_dir,
                        batch=batch,
                    )
                )
    return records


def choose_val_batches_dp(batch_counts: Dict[str, int], target_val_n: int, seed: int) -> List[str]:
    batches = list(batch_counts.keys())
    rng = random.Random(int(seed))
    rng.shuffle(batches)

    counts = [int(batch_counts[b]) for b in batches]
    total = int(sum(counts))
    target = int(max(1, min(target_val_n, total - 1)))

    # dp[s] stores a bitmask of selected batch indices that achieves sum s
    # while maximizing the number of batches. Using a bitmask avoids the
    # 1D-parent-pointer reconstruction bug that can re-use the same batch.
    dp: List[int | None] = [None] * (total + 1)
    dp[0] = 0

    for idx, c in enumerate(counts):
        bit = 1 << idx
        for s in range(total - c, -1, -1):
            if dp[s] is None:
                continue
            if dp[s] & bit:
                continue
            ns = s + c
            candidate = dp[s] | bit
            current = dp[ns]
            if current is None or candidate.bit_count() > current.bit_count():
                dp[ns] = candidate

    best_sum = None
    best_diff = None
    best_batches = -1
    best_mask: int | None = None
    for s, mask in enumerate(dp):
        if mask is None or s == 0 or s == total:
            continue
        diff = abs(s - target)
        batches_n = mask.bit_count()
        if best_sum is None or diff < best_diff or (diff == best_diff and batches_n > best_batches):
            best_sum = s
            best_diff = diff
            best_batches = batches_n
            best_mask = mask

    if best_sum is None or best_mask is None:
        raise RuntimeError("Failed to find a non-empty batch subset for val")

    chosen_batches = [b for i, b in enumerate(batches) if best_mask & (1 << i)]
    return chosen_batches


def write_split(records: List[LabelRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.raw_line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create batch-heldout train/val txt splits")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("data/labels/huajiao_2025_08_plus/train.txt"),
            Path("data/labels/huajiao_2025_08_plus/val.txt"),
        ],
        help="Existing label txt files to combine",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Approx val sample ratio")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for tie-breaking")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/labels/huajiao_2025_08_plus_batchholdout_20260105"),
        help="Output directory",
    )
    parser.add_argument("--train-file", type=str, default="train.txt")
    parser.add_argument("--val-file", type=str, default="val.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_label_records(list(args.inputs))
    if not records:
        raise RuntimeError("No records loaded from inputs")

    batch_to_records: Dict[str, List[LabelRecord]] = {}
    for r in records:
        batch_to_records.setdefault(r.batch, []).append(r)

    batch_counts = {b: len(v) for b, v in batch_to_records.items()}
    total = len(records)
    target_val_n = int(round(total * float(args.val_ratio)))

    val_batches = choose_val_batches_dp(batch_counts, target_val_n=target_val_n, seed=int(args.seed))
    val_batch_set = set(val_batches)

    train_records = [r for r in records if r.batch not in val_batch_set]
    val_records = [r for r in records if r.batch in val_batch_set]

    out_train = args.out_dir / args.train_file
    out_val = args.out_dir / args.val_file
    write_split(train_records, out_train)
    write_split(val_records, out_val)

    payload = {
        "inputs": [str(p) for p in args.inputs],
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "total_samples": int(total),
        "train_samples": int(len(train_records)),
        "val_samples": int(len(val_records)),
        "total_batches": int(len(batch_to_records)),
        "train_batches": int(len({r.batch for r in train_records})),
        "val_batches": int(len(val_batch_set)),
        "val_batches_list": sorted(val_batches),
        "batch_counts": dict(sorted(batch_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    with (args.out_dir / "split_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== Batch-heldout split created ===")
    print(f"Output dir: {args.out_dir}")
    print(f"Train: {len(train_records)} samples, {len({r.batch for r in train_records})} batches")
    print(f"Val:   {len(val_records)} samples, {len(val_batch_set)} batches")
    print(f"Val batches: {sorted(val_batches)}")


if __name__ == "__main__":
    main()
