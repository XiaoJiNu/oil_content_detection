from pathlib import Path
import os
import re

# Helper to normalize IDs (same as in pipeline)
def normalize_sample_id(raw: str) -> str:
    cleaned = str(raw).strip()
    cleaned = cleaned.replace("-", "_").replace("—", "_").replace("－", "_")
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned

raw_root = Path("/home/yr/yr/data/科研数据")
disk_ids = set()

# Scan disk
for root, dirs, files in os.walk(raw_root):
    # We assume structure is root/batch/sample_id
    # So we want the directory names at depth 2 relative to raw_root
    rel = Path(root).relative_to(raw_root)
    if len(rel.parts) == 1: # Inside a batch folder
        for d in dirs:
            disk_ids.add(normalize_sample_id(d))

print(f"Total disk IDs found: {len(disk_ids)}")

# Load matched IDs from the train/val splits we just verified against
matched_ids = set()
for fname in ["train.txt", "val.txt"]:
    fpath = Path("data/labels/huajiao_2025_08_plus") / fname
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    # The file stores raw_id, but we need normalized to compare
                    # Wait, the first column IS raw_id. 
                    # Let's verify what normalize_sample_id does to them.
                    # Actually, better to rely on what the update script found.
                    # But since I can't access that memory easily, let's just re-read the text files
                    # The text file: 0164-11 ...
                    # Normalization: 0164_11
                    matched_ids.add(normalize_sample_id(parts[0]))

print(f"Total matched IDs in splits: {len(matched_ids)}")

unmatched = disk_ids - matched_ids
print(f"On disk but not in splits: {len(unmatched)}")
if unmatched:
    print(list(unmatched))
