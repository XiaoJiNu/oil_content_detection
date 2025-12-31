"""Pipeline utilities for preparing datasets and training models."""

from oil_content_detection.pipelines.huajiao_dataset import (
    AggregationConfig,
    HuajiaoROIConfig,
    LabelConfig,
    build_huajiao_dataset,
    build_huajiao_dataset_from_split,
    clean_mask_extremes,
    create_huajiao_mask,
    discover_huajiao_cubes,
    load_huajiao_labels,
    normalize_sample_id,
    SampleInfo,
)
from oil_content_detection.pipelines.total_oil_pipeline import (
    TotalOilExperimentConfig,
    TotalOilExperimentResult,
    build_shape_feature_matrix,
    run_total_oil_experiment,
    run_total_oil_experiment_with_predictions,
)

__all__ = [
    "AggregationConfig",
    "HuajiaoROIConfig",
    "LabelConfig",
    "build_huajiao_dataset",
    "build_huajiao_dataset_from_split",
    "clean_mask_extremes",
    "create_huajiao_mask",
    "discover_huajiao_cubes",
    "load_huajiao_labels",
    "normalize_sample_id",
    "SampleInfo",
    "TotalOilExperimentConfig",
    "TotalOilExperimentResult",
    "build_shape_feature_matrix",
    "run_total_oil_experiment",
    "run_total_oil_experiment_with_predictions",
]
