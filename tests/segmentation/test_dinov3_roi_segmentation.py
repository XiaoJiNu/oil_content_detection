from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from oil_content_detection.segmentation.data import (
    HuajiaoROIDataset,
    load_oil_split_samples,
    rasterize_polygons,
)
from oil_content_detection.segmentation.dinov3_model import (
    DinoV3BackboneConfig,
    DinoV3ROISegmenter,
    LinearUpsampleSegHead,
)
from oil_content_detection.segmentation.metrics import binary_segmentation_metrics
from oil_content_detection.segmentation.predictor import (
    postprocess_logits_to_mask,
    sha256_file,
    validate_artifact_hashes,
)
from oil_content_detection.segmentation.visualization import save_mask_png, save_overlay


class FakeDinoBackbone(torch.nn.Module):
    def __init__(self, feature_dim: int = 8, patch_size: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(
            3, feature_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward_features(self, x: torch.Tensor):
        fmap = self.proj(x)
        tokens = fmap.flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens}


def _write_tiny_split(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "sample_0001"
    image_dir.mkdir(parents=True)
    image = Image.new("RGB", (10, 8), color=(30, 20, 10))
    image.save(image_dir / "sample_0001.png")

    xml = tmp_path / "annotations.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="sample_0001.png" width="10" height="8">
    <polygon label="peper" points="2,1;7,1;7,6;2,6"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    processed = tmp_path / "processed"
    split_dir = processed / "train"
    split_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sample_id": "sample_0001",
                "sample_id_raw": "sample_0001",
                "sample_dir": str(image_dir),
                "hdr_path": "unused.hdr",
                "dat_path": "unused.dat",
            }
        ]
    ).to_parquet(split_dir / "huajiao_metadata.parquet")
    return processed, xml


def test_rasterize_polygons_returns_binary_mask():
    poly = np.array([[2, 1], [7, 1], [7, 6], [2, 6]], dtype=np.float32)
    mask = rasterize_polygons(width=10, height=8, polygons=[poly])

    assert mask.shape == (8, 10)
    assert mask.dtype == np.bool_
    assert mask[3, 4]
    assert not mask[0, 0]
    assert mask.sum() > 0


def test_oil_split_samples_match_current_huajiao_split_if_available():
    processed = Path("data/processed/huajiao_refined")
    xml = Path("/home/yr/yr/data/huajiao_all/annotations_20251231_197.xml")
    if not processed.exists() or not xml.exists():
        pytest.skip("local huajiao processed split or CVAT XML is unavailable")

    train = load_oil_split_samples(processed, "train", xml)
    val = load_oil_split_samples(processed, "val", xml)
    train_meta = pd.read_parquet(processed / "train" / "huajiao_metadata.parquet")
    val_meta = pd.read_parquet(processed / "val" / "huajiao_metadata.parquet")

    assert len(train) == len(train_meta) == 157
    assert len(val) == len(val_meta) == 40
    assert {r.sample_id for r in train} == set(train_meta["sample_id"].astype(str))
    assert {r.sample_id for r in val} == set(val_meta["sample_id"].astype(str))


def test_dataset_returns_normalized_image_and_binary_mask(tmp_path: Path):
    processed, xml = _write_tiny_split(tmp_path)
    dataset = HuajiaoROIDataset(
        processed, split="train", annotation_xml=xml, image_size=32
    )

    item = dataset[0]

    assert item["sample_id"] == "sample_0001"
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})
    assert item["mask"].sum() > 0


def test_segmentation_head_upsamples_to_requested_output_size():
    head = LinearUpsampleSegHead(in_channels=8, hidden_channels=16, dropout=0.0)
    fmap = torch.randn(2, 8, 4, 4)

    logits = head(fmap, output_size=(64, 64))

    assert logits.shape == (2, 1, 64, 64)


def test_dinov3_roi_segmenter_forward_with_fake_backbone():
    cfg = DinoV3BackboneConfig(
        repo_dir="unused",
        weights_path="unused",
        model_name="dinov3_vits16",
        image_size=32,
        patch_size=16,
        freeze_backbone=True,
        feature_dim=8,
    )
    model = DinoV3ROISegmenter(
        backbone_config=cfg,
        backbone=FakeDinoBackbone(feature_dim=8, patch_size=16),
        head_channels=16,
        dropout=0.0,
    )

    logits = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 1, 32, 32)


def test_postprocess_logits_to_original_mask_and_metrics():
    logits = torch.full((1, 1, 8, 8), -4.0)
    logits[:, :, 2:6, 2:6] = 4.0

    mask, probability = postprocess_logits_to_mask(
        logits, original_size=(16, 16), threshold=0.5
    )
    metrics = binary_segmentation_metrics(mask, mask)

    assert mask.shape == (16, 16)
    assert probability.shape == (16, 16)
    assert mask.dtype == np.bool_
    assert 0.0 < mask.mean() < 1.0
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)


def test_save_mask_and_overlay_outputs(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 12), color=(20, 20, 20)).save(image_path)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:8, 4:12] = True

    mask_path = tmp_path / "mask.png"
    overlay_path = tmp_path / "overlay.png"
    save_mask_png(mask, mask_path)
    save_overlay(image_path, mask, overlay_path)

    assert mask_path.exists()
    assert overlay_path.exists()
    assert Image.open(mask_path).size == (16, 12)
    assert Image.open(overlay_path).size == (16, 12)


def test_artifact_hash_validation_rejects_modified_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    backbone_weights = tmp_path / "dinov3_backbone.pth"
    checkpoint.write_bytes(b"trusted checkpoint")
    backbone_weights.write_bytes(b"trusted backbone")
    config = {
        "backbone": {"repo_dir": str(tmp_path), "weights_path": str(backbone_weights)},
        "weights_sha256": sha256_file(backbone_weights),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "best_model_sha256": sha256_file(checkpoint),
                "backbone_weights_sha256": sha256_file(backbone_weights),
            }
        ),
        encoding="utf-8",
    )

    validate_artifact_hashes(tmp_path, config)

    checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="best_model.pt sha256 mismatch"):
        validate_artifact_hashes(tmp_path, config)


def test_artifact_hash_validation_rejects_modified_backbone_weights(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    backbone_weights = tmp_path / "dinov3_backbone.pth"
    checkpoint.write_bytes(b"trusted checkpoint")
    backbone_weights.write_bytes(b"trusted backbone")
    config = {
        "backbone": {"repo_dir": str(tmp_path), "weights_path": str(backbone_weights)},
        "weights_sha256": sha256_file(backbone_weights),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "best_model_sha256": sha256_file(checkpoint),
                "backbone_weights_sha256": sha256_file(backbone_weights),
            }
        ),
        encoding="utf-8",
    )

    validate_artifact_hashes(tmp_path, config)

    backbone_weights.write_bytes(b"tampered backbone")
    with pytest.raises(ValueError, match="DINOv3 backbone weights sha256 mismatch"):
        validate_artifact_hashes(tmp_path, config)


def test_artifact_hash_validation_honors_backbone_weight_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "best_model.pt"
    backbone_weights = tmp_path / "runtime_dinov3_backbone.pth"
    checkpoint.write_bytes(b"trusted checkpoint")
    backbone_weights.write_bytes(b"trusted backbone")
    config = {
        "backbone": {
            "repo_dir": str(tmp_path),
            "weights_path": "configured/missing/backbone.pth",
        },
        "weights_sha256": sha256_file(backbone_weights),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "best_model_sha256": sha256_file(checkpoint),
                "backbone_weights_sha256": sha256_file(backbone_weights),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HUAJIAO_DINOV3_WEIGHTS_PATH", str(backbone_weights))

    validate_artifact_hashes(tmp_path, config)


def test_artifact_hash_validation_rejects_missing_backbone_hash(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    backbone_weights = tmp_path / "dinov3_backbone.pth"
    checkpoint.write_bytes(b"trusted checkpoint")
    backbone_weights.write_bytes(b"trusted backbone")
    config = {
        "backbone": {"repo_dir": str(tmp_path), "weights_path": str(backbone_weights)},
        "weights_sha256": sha256_file(backbone_weights),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"best_model_sha256": sha256_file(checkpoint)}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="manifest.json is missing backbone_weights_sha256"
    ):
        validate_artifact_hashes(tmp_path, config)
