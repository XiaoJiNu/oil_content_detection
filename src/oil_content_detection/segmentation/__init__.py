"""DINOv3-based ROI segmentation utilities for huajiao images."""

from .data import (
    HuajiaoROIDataset,
    HuajiaoROISample,
    load_oil_split_samples,
    rasterize_polygons,
)
from .dinov3_model import (
    DinoV3BackboneConfig,
    DinoV3ROISegmenter,
    LinearUpsampleSegHead,
)
from .predictor import DinoV3ROIPredictor, postprocess_logits_to_mask

__all__ = [
    "DinoV3BackboneConfig",
    "DinoV3ROIPredictor",
    "DinoV3ROISegmenter",
    "HuajiaoROIDataset",
    "HuajiaoROISample",
    "LinearUpsampleSegHead",
    "load_oil_split_samples",
    "postprocess_logits_to_mask",
    "rasterize_polygons",
]
