"""DINOv3 ROI segmentation model wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM_BY_MODEL = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
}


@dataclass(frozen=True)
class DinoV3BackboneConfig:
    repo_dir: str = "/home/yr/yr/code/cv/large_models/dinov3_all/dinov3"
    weights_path: str = (
        "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    model_name: str = "dinov3_vits16"
    image_size: int = 448
    patch_size: int = 16
    freeze_backbone: bool = True
    feature_dim: int | None = None
    trust_repo: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinearUpsampleSegHead(nn.Module):
    """Small binary segmentation head on top of a DINOv3 feature map."""

    def __init__(
        self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, fmap: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.net(fmap)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


class DinoV3ROISegmenter(nn.Module):
    """DINOv3 backbone plus lightweight binary ROI segmentation head."""

    def __init__(
        self,
        backbone_config: DinoV3BackboneConfig | None = None,
        backbone: nn.Module | None = None,
        head_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_config = backbone_config or DinoV3BackboneConfig()
        self.backbone = (
            backbone
            if backbone is not None
            else self._load_backbone(self.backbone_config)
        )
        self.feature_dim = self.backbone_config.feature_dim or FEATURE_DIM_BY_MODEL.get(
            self.backbone_config.model_name
        )
        if self.feature_dim is None:
            raise ValueError(
                f"feature_dim is required for unknown DINOv3 model {self.backbone_config.model_name!r}"
            )
        self.head = LinearUpsampleSegHead(
            self.feature_dim, hidden_channels=head_channels, dropout=dropout
        )
        if self.backbone_config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _load_backbone(config: DinoV3BackboneConfig) -> nn.Module:
        repo_dir = Path(config.repo_dir)
        weights_path = Path(config.weights_path)
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo_dir not found: {repo_dir}")
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv3 weights_path not found: {weights_path}")
        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return torch.hub.load(
            str(repo_dir),
            config.model_name,
            source="local",
            weights=str(weights_path),
            trust_repo=config.trust_repo,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_config.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        context = (
            torch.no_grad()
            if self.backbone_config.freeze_backbone
            else torch.enable_grad()
        )
        with context:
            features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            tokens = features.get("x_norm_patchtokens")
            if tokens is None:
                raise KeyError(
                    "DINOv3 forward_features did not return x_norm_patchtokens"
                )
        else:
            tokens = features
        if tokens.ndim != 3:
            raise ValueError(
                f"expected patch tokens shaped (B,N,C), got {tuple(tokens.shape)}"
            )
        batch, n_tokens, channels = tokens.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: configured {self.feature_dim}, got {channels}"
            )
        h = max(1, x.shape[-2] // self.backbone_config.patch_size)
        w = max(1, x.shape[-1] // self.backbone_config.patch_size)
        if h * w != n_tokens:
            side = int(n_tokens**0.5)
            if side * side != n_tokens:
                raise ValueError(
                    f"cannot reshape {n_tokens} patch tokens into a 2D feature map"
                )
            h = w = side
        return tokens.reshape(batch, h, w, channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.extract_feature_map(x)
        return self.head(fmap, output_size=(x.shape[-2], x.shape[-1]))

    def model_config_dict(
        self, head_channels: int | None = None, dropout: float | None = None
    ) -> dict[str, Any]:
        return {
            "backbone": self.backbone_config.to_dict(),
            "head": {
                "feature_dim": self.feature_dim,
                "head_channels": head_channels,
                "dropout": dropout,
            },
        }
