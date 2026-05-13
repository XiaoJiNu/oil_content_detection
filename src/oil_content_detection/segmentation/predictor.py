"""Inference wrapper for DINOv3 ROI segmentation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .data import IMAGENET_MEAN, IMAGENET_STD
from .dinov3_model import DinoV3BackboneConfig, DinoV3ROISegmenter
from .visualization import save_mask_png, save_overlay


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 digest for artifact provenance checks."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash_match(path: Path, expected_sha256: str, label: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected_sha256}, got {actual_sha256}. "
            "Refusing to load an untrusted or modified model file."
        )


def _resolve_runtime_path(
    path_value: str | Path,
    *,
    artifact_dir: str | Path,
    env_var: str | None = None,
) -> Path:
    """Resolve artifact paths, allowing deployment-time environment overrides."""

    raw_value = os.getenv(env_var, "") if env_var else ""
    path_text = raw_value.strip() or str(path_value)
    path_text = os.path.expandvars(os.path.expanduser(path_text))
    resolved = Path(path_text)
    if not resolved.is_absolute():
        resolved = Path(artifact_dir) / resolved
    return resolved.resolve()


def _resolve_backbone_paths(
    artifact_dir: str | Path, config: dict[str, Any]
) -> tuple[Path, Path]:
    """Resolve DINOv3 repo and backbone weights for the current runtime."""

    backbone_cfg = dict(config.get("backbone") or {})
    if "repo_dir" not in backbone_cfg or "weights_path" not in backbone_cfg:
        raise ValueError("config.json backbone must include repo_dir and weights_path")
    repo_dir = _resolve_runtime_path(
        backbone_cfg["repo_dir"],
        artifact_dir=artifact_dir,
        env_var="HUAJIAO_DINOV3_REPO_DIR",
    )
    weights_path = _resolve_runtime_path(
        backbone_cfg["weights_path"],
        artifact_dir=artifact_dir,
        env_var="HUAJIAO_DINOV3_WEIGHTS_PATH",
    )
    return repo_dir, weights_path


def validate_artifact_hashes(artifact_dir: str | Path, config: dict[str, Any]) -> None:
    """Validate model files before any PyTorch checkpoint deserialization.

    PyTorch checkpoints are pickle-based. This project only loads internally
    generated artifacts whose checkpoint and DINOv3 backbone hashes match the
    manifest/config metadata. The DINOv3 repo/weights paths may be overridden at
    deployment time via ``HUAJIAO_DINOV3_REPO_DIR`` and
    ``HUAJIAO_DINOV3_WEIGHTS_PATH``; the weights file is still hash-checked.
    """

    artifact_dir = Path(artifact_dir)
    checkpoint_path = artifact_dir / "best_model.pt"
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found: {manifest_path}. Refusing to load artifact without provenance metadata."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = manifest.get("best_model_sha256")
    if not expected_checkpoint_sha:
        raise ValueError("manifest.json is missing best_model_sha256")
    _require_hash_match(checkpoint_path, expected_checkpoint_sha, "best_model.pt")

    expected_backbone_sha = manifest.get("backbone_weights_sha256")
    if not expected_backbone_sha:
        raise ValueError("manifest.json is missing backbone_weights_sha256")
    config_backbone_sha = config.get("weights_sha256")
    if config_backbone_sha and config_backbone_sha != expected_backbone_sha:
        raise ValueError(
            "config.json weights_sha256 does not match manifest.json backbone_weights_sha256"
        )
    weights_path = _resolve_backbone_paths(artifact_dir, config)[1]
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 backbone weights not found: {weights_path}")
    _require_hash_match(
        weights_path,
        expected_backbone_sha,
        "DINOv3 backbone weights",
    )


@dataclass(frozen=True)
class ROIPrediction:
    sample_id: str | None
    mask: np.ndarray
    probability: np.ndarray
    coverage_ratio: float
    threshold: float
    mask_path: str | None = None
    overlay_path: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "coverage_ratio": self.coverage_ratio,
            "threshold": self.threshold,
            "mask_path": self.mask_path,
            "overlay_path": self.overlay_path,
            "height": int(self.mask.shape[0]),
            "width": int(self.mask.shape[1]),
        }


def postprocess_logits_to_mask(
    logits: torch.Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert model logits to original-size boolean mask and probability map."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"expected logits shaped (B,1,H,W), got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits.detach().float().cpu())
    probs = torch.nn.functional.interpolate(
        probs, size=original_size, mode="bilinear", align_corners=False
    )
    prob_np = probs[0, 0].numpy()
    mask = prob_np >= float(threshold)
    return mask.astype(bool), prob_np.astype(np.float32)


class DinoV3ROIPredictor:
    """Single-image predictor that returns an original-size ROI mask."""

    def __init__(
        self,
        model: DinoV3ROISegmenter,
        image_size: int,
        threshold: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.image_size = int(image_size)
        self.threshold = float(threshold)

    @classmethod
    def from_artifact(
        cls, artifact_dir: str | Path, device: str | torch.device | None = None
    ) -> "DinoV3ROIPredictor":
        artifact_dir = Path(artifact_dir)
        config_path = artifact_dir / "config.json"
        checkpoint_path = artifact_dir / "best_model.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"best_model.pt not found: {checkpoint_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_artifact_hashes(artifact_dir, config)
        repo_dir, weights_path = _resolve_backbone_paths(artifact_dir, config)
        backbone_config = dict(config["backbone"])
        backbone_config["repo_dir"] = str(repo_dir)
        backbone_config["weights_path"] = str(weights_path)
        backbone_cfg = DinoV3BackboneConfig(**backbone_config)
        model = DinoV3ROISegmenter(
            backbone_config=backbone_cfg,
            head_channels=int(config["head"].get("head_channels", 128)),
            dropout=float(config["head"].get("dropout", 0.0)),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("head_state_dict", checkpoint)
        model.head.load_state_dict(state)
        return cls(
            model=model,
            image_size=int(config.get("image_size", backbone_cfg.image_size)),
            threshold=float(config.get("threshold", 0.5)),
            device=device,
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def predict_image(
        self,
        image_path: str | Path,
        sample_id: str | None = None,
        out_mask_path: str | Path | None = None,
        overlay_path: str | Path | None = None,
    ) -> ROIPrediction:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        original_size = (image.height, image.width)
        x = self._preprocess(image).to(self.device)
        logits = self.model(x)
        mask, probability = postprocess_logits_to_mask(
            logits, original_size=original_size, threshold=self.threshold
        )
        saved_mask = str(save_mask_png(mask, out_mask_path)) if out_mask_path else None
        saved_overlay = (
            str(save_overlay(image_path, mask, overlay_path)) if overlay_path else None
        )
        return ROIPrediction(
            sample_id=sample_id,
            mask=mask,
            probability=probability,
            coverage_ratio=float(mask.mean()),
            threshold=self.threshold,
            mask_path=saved_mask,
            overlay_path=saved_overlay,
        )
