"""Visualization module for oil content detection results."""
from oil_content_detection.visualization.plots import (
    plot_all_results,
    plot_ga_history,
    plot_prediction_results,
    plot_spectral_selection,
)
from oil_content_detection.visualization.spatial_distribution import (
    create_summary_grid,
    plot_seed_oil_distribution,
    predict_spatial_distribution,
    visualize_all_seeds,
)

__all__ = [
    "plot_ga_history",
    "plot_spectral_selection",
    "plot_prediction_results",
    "plot_all_results",
    "predict_spatial_distribution",
    "plot_seed_oil_distribution",
    "visualize_all_seeds",
    "create_summary_grid",
]