"""Data utilities for huajiao ROI segmentation.

The segmentation split is intentionally tied to the oil-content regression split
under ``data/processed/huajiao_refined/{train,val}`` so ROI experiments remain
comparable with downstream oil-content experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from oil_content_detection.data.cvat_annotations import (
    CvatImageAnnotations,
    load_cvat_image_annotations,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class HuajiaoROISample:
    """One RGB image and its manual ROI polygons."""

    sample_id: str
    split: str
    image_path: Path
    width: int
    height: int
    polygons: tuple[np.ndarray, ...]


def rasterize_polygons(
    width: int, height: int, polygons: Sequence[np.ndarray]
) -> np.ndarray:
    """Rasterize polygon points into a boolean mask shaped ``(height, width)``."""

    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    mask = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            continue
        points = [(float(x), float(y)) for x, y in arr]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8).astype(bool)


def _annotation_by_sample_id(
    annotations: dict[str, CvatImageAnnotations]
) -> dict[str, CvatImageAnnotations]:
    return {Path(name).stem: ann for name, ann in annotations.items()}


def _select_polygons(
    ann: CvatImageAnnotations, label_names: Iterable[str]
) -> tuple[np.ndarray, ...]:
    selected: list[np.ndarray] = []
    for label in label_names:
        selected.extend(ann.polygons.get(label, []))
    return tuple(selected)


def resolve_image_path(sample_dir: str | Path, sample_id: str) -> Path:
    """Resolve the RGB image path for a sample directory and sample id."""

    sample_dir = Path(sample_dir)
    candidates = [
        sample_dir / f"{sample_id}.png",
        sample_dir / f"{sample_id}.jpg",
        sample_dir / f"{sample_id}.jpeg",
        sample_dir / f"{sample_id.upper()}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"RGB image not found for sample {sample_id} under {sample_dir}"
    )


def load_oil_split_samples(
    processed_dir: str | Path,
    split: str,
    annotation_xml: str | Path,
    label_names: Sequence[str] = ("peper", "pepper"),
) -> list[HuajiaoROISample]:
    """Load ROI segmentation records following the oil-content split.

    Args:
        processed_dir: Directory containing ``train`` and ``val`` processed metadata.
        split: Oil-content split name, usually ``train`` or ``val``.
        annotation_xml: CVAT XML with manual ROI polygons.
        label_names: Accepted polygon labels. The main 197-image XML uses the
            historical typo ``peper``; a smaller external XML uses ``pepper``.
    """

    processed_dir = Path(processed_dir)
    meta_path = processed_dir / split / "huajiao_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata parquet not found: {meta_path}")

    metadata = pd.read_parquet(meta_path)
    required = {"sample_id", "sample_dir"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata {meta_path} missing columns: {sorted(missing)}")

    annotations = _annotation_by_sample_id(load_cvat_image_annotations(annotation_xml))
    samples: list[HuajiaoROISample] = []
    missing_annotations: list[str] = []
    missing_polygons: list[str] = []

    for row in metadata.itertuples(index=False):
        sample_id = str(getattr(row, "sample_id"))
        ann = annotations.get(sample_id)
        if ann is None:
            missing_annotations.append(sample_id)
            continue
        polygons = _select_polygons(ann, label_names)
        if not polygons:
            missing_polygons.append(sample_id)
            continue
        image_path = resolve_image_path(getattr(row, "sample_dir"), sample_id)
        samples.append(
            HuajiaoROISample(
                sample_id=sample_id,
                split=split,
                image_path=image_path,
                width=ann.width,
                height=ann.height,
                polygons=polygons,
            )
        )

    if missing_annotations:
        raise ValueError(
            f"{len(missing_annotations)} samples missing CVAT annotations: {missing_annotations[:10]}"
        )
    if missing_polygons:
        raise ValueError(
            f"{len(missing_polygons)} samples missing ROI polygons: {missing_polygons[:10]}"
        )
    return samples


class HuajiaoROIDataset(Dataset):
    """PyTorch dataset for binary huajiao ROI segmentation."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        annotation_xml: str | Path,
        image_size: int = 448,
        label_names: Sequence[str] = ("peper", "pepper"),
        augment: bool = False,
        samples: Sequence[HuajiaoROISample] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.annotation_xml = Path(annotation_xml)
        self.image_size = int(image_size)
        self.label_names = tuple(label_names)
        self.augment = bool(augment)
        self.samples = (
            list(samples)
            if samples is not None
            else load_oil_split_samples(
                processed_dir=self.processed_dir,
                split=split,
                annotation_xml=self.annotation_xml,
                label_names=self.label_names,
            )
        )
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, sample: HuajiaoROISample) -> tuple[Image.Image, Image.Image]:
        image = Image.open(sample.image_path).convert("RGB")
        mask_np = rasterize_polygons(sample.width, sample.height, sample.polygons)
        mask = Image.fromarray(mask_np.astype(np.uint8) * 255)
        if image.size != (sample.width, sample.height):
            # CVAT records occasionally differ by one row from PIL metadata. Use
            # the actual image shape for training and resize the annotation mask.
            mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        image, mask = self._load_pair(sample)
        original_width, original_height = image.size

        if self.augment and torch.rand(()) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = (TF.to_tensor(mask) > 0.5).to(torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "original_size": torch.tensor(
                [original_height, original_width], dtype=torch.int64
            ),
        }
