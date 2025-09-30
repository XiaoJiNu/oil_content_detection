"""Spatial distribution visualization for oil content in camellia seeds.

This module provides functions to visualize the spatial distribution of oil content
within individual seed images using hyperspectral data and trained PLSR models.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from oil_content_detection.utils.logging import get_logger

logger = get_logger(__name__)


def create_oil_content_colormap() -> LinearSegmentedColormap:
    """Create a custom colormap for oil content visualization.

    Returns:
        Custom colormap ranging from blue (low oil) to red (high oil)
    """
    colors = [
        "#2C3E50",  # Dark blue - very low
        "#3498DB",  # Blue - low
        "#1ABC9C",  # Cyan - medium-low
        "#F1C40F",  # Yellow - medium
        "#E67E22",  # Orange - medium-high
        "#E74C3C",  # Red - high
        "#C0392B",  # Dark red - very high
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("oil_content", colors, N=n_bins)
    return cmap


def predict_spatial_distribution(
    cube: np.ndarray,
    roi_mask: np.ndarray,
    model: any,
    support: np.ndarray,
) -> np.ndarray:
    """Predict oil content for each pixel in the hyperspectral cube.

    Args:
        cube: Hyperspectral cube (height, width, n_wavelengths)
        roi_mask: Boolean mask indicating seed region (height, width)
        model: Trained PLSR model
        support: Boolean array indicating selected wavelengths

    Returns:
        2D array of predicted oil content (height, width), with NaN for background
    """
    height, width, n_wavelengths = cube.shape

    # Initialize output with NaN (background)
    oil_map = np.full((height, width), np.nan, dtype=np.float32)

    # Get ROI pixels
    roi_pixels = cube[roi_mask]  # (n_roi_pixels, n_wavelengths)

    # Select features using support mask
    roi_pixels_selected = roi_pixels[:, support]

    # Predict oil content
    predictions = model.predict(roi_pixels_selected).ravel()

    # Fill predictions into the oil map
    oil_map[roi_mask] = predictions

    return oil_map


def plot_seed_oil_distribution(
    cube: np.ndarray,
    roi_mask: np.ndarray,
    oil_map: np.ndarray,
    mean_oil_content: float,
    sample_id: str,
    save_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot oil content spatial distribution for a single seed.

    Args:
        cube: Hyperspectral cube (height, width, n_wavelengths)
        roi_mask: Boolean mask indicating seed region
        oil_map: Predicted oil content map (height, width)
        mean_oil_content: Mean oil content value (for reference)
        sample_id: Sample identifier
        save_path: Optional path to save the figure
        vmin: Minimum value for colorbar (default: min of predictions)
        vmax: Maximum value for colorbar (default: max of predictions)
    """
    # Create RGB-like image from hyperspectral cube for visualization
    # Use wavelengths around R, G, B regions
    # Assuming wavelengths 900-1700nm, we'll use pseudocolor
    rgb_image = np.mean(cube[:, :, :30], axis=2)  # Use first 30 bands as grayscale

    # Normalize for display
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Original image (grayscale)
    ax1 = axes[0]
    ax1.imshow(rgb_image, cmap="gray")
    ax1.contour(roi_mask, colors="yellow", linewidths=2, levels=[0.5])
    ax1.set_title(f"Original Image\nSample: {sample_id}", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Plot 2: ROI mask
    ax2 = axes[1]
    mask_display = np.ma.masked_where(~roi_mask, roi_mask.astype(float))
    ax2.imshow(rgb_image, cmap="gray", alpha=0.5)
    ax2.imshow(mask_display, cmap="Greens", alpha=0.7, vmin=0, vmax=1)
    ax2.set_title(f"ROI Mask\nPixels: {roi_mask.sum()}", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Plot 3: Oil content distribution
    ax3 = axes[2]
    ax3.imshow(rgb_image, cmap="gray", alpha=0.3)

    # Get valid predictions
    valid_predictions = oil_map[~np.isnan(oil_map)]

    if vmin is None:
        vmin = valid_predictions.min()
    if vmax is None:
        vmax = valid_predictions.max()

    # Plot oil content with custom colormap
    oil_cmap = create_oil_content_colormap()
    im = ax3.imshow(oil_map, cmap=oil_cmap, alpha=0.9, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Oil Content (%)", fontsize=11)

    # Add statistics
    std_oil = valid_predictions.std()
    title = (
        f"Oil Content Distribution\n"
        f"Mean: {mean_oil_content:.2f}% | "
        f"Pred Mean: {valid_predictions.mean():.2f}% ± {std_oil:.2f}%\n"
        f"Range: [{valid_predictions.min():.2f}, {valid_predictions.max():.2f}]%"
    )
    ax3.set_title(title, fontsize=12, fontweight="bold")
    ax3.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Oil distribution plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_seeds(
    cube_data_path: Path,
    model_path: Path,
    support_path: Path,
    output_dir: Path,
    sample_indices: Optional[list] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Visualize oil content distribution for all seeds in the dataset.

    Args:
        cube_data_path: Path to NPZ file containing hyperspectral cubes
        model_path: Path to trained PLSR model
        support_path: Path to feature support mask
        output_dir: Directory to save output images
        sample_indices: Optional list of sample indices to process (default: all)
        vmin: Minimum value for colorbar (default: auto)
        vmax: Maximum value for colorbar (default: auto)
    """
    logger.info(f"Loading data from {cube_data_path}")

    # Load data
    data = np.load(cube_data_path)
    cubes = data["cubes"]  # (n_samples, height, width, n_wavelengths)
    roi_masks = data["roi_masks"]  # (n_samples, height, width)
    sample_ids = data["sample_ids"]  # (n_samples,)
    oil_contents = data["oil_content"]  # (n_samples,)

    logger.info(f"Loaded {len(cubes)} samples")

    # Load model and support
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    support = np.load(support_path)
    logger.info(f"Selected {support.sum()} wavelengths")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which samples to process
    if sample_indices is None:
        sample_indices = range(len(cubes))

    # Process each sample
    for idx in sample_indices:
        sample_id = sample_ids[idx]
        logger.info(f"Processing sample {idx + 1}/{len(sample_indices)}: {sample_id}")

        # Predict spatial distribution
        oil_map = predict_spatial_distribution(
            cube=cubes[idx],
            roi_mask=roi_masks[idx],
            model=model,
            support=support,
        )

        # Plot and save
        save_path = output_dir / f"{sample_id}_oil_distribution.png"
        plot_seed_oil_distribution(
            cube=cubes[idx],
            roi_mask=roi_masks[idx],
            oil_map=oil_map,
            mean_oil_content=oil_contents[idx],
            sample_id=sample_id,
            save_path=save_path,
            vmin=vmin,
            vmax=vmax,
        )

    logger.info(f"All visualizations saved to {output_dir}")


def create_summary_grid(
    cube_data_path: Path,
    model_path: Path,
    support_path: Path,
    output_path: Path,
    n_samples: int = 12,
    seed: int = 42,
) -> None:
    """Create a grid showing oil distribution for multiple seeds.

    Args:
        cube_data_path: Path to NPZ file containing hyperspectral cubes
        model_path: Path to trained PLSR model
        support_path: Path to feature support mask
        output_path: Path to save the summary figure
        n_samples: Number of samples to display
        seed: Random seed for sample selection
    """
    logger.info(f"Creating summary grid with {n_samples} samples")

    # Load data
    data = np.load(cube_data_path)
    cubes = data["cubes"]
    roi_masks = data["roi_masks"]
    sample_ids = data["sample_ids"]
    oil_contents = data["oil_content"]

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    support = np.load(support_path)

    # Randomly select samples
    rng = np.random.default_rng(seed)
    n_total = len(cubes)
    n_samples = min(n_samples, n_total)
    selected_indices = rng.choice(n_total, size=n_samples, replace=False)

    # Calculate global vmin/vmax for consistent colorbar
    all_predictions = []
    for idx in selected_indices:
        oil_map = predict_spatial_distribution(cubes[idx], roi_masks[idx], model, support)
        all_predictions.extend(oil_map[~np.isnan(oil_map)])

    vmin = np.percentile(all_predictions, 5)
    vmax = np.percentile(all_predictions, 95)

    # Create grid
    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    oil_cmap = create_oil_content_colormap()

    for i, idx in enumerate(selected_indices):
        ax = axes[i]

        # Predict
        oil_map = predict_spatial_distribution(cubes[idx], roi_masks[idx], model, support)

        # Plot
        rgb_image = np.mean(cubes[idx][:, :, :30], axis=2)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        ax.imshow(rgb_image, cmap="gray", alpha=0.3)
        im = ax.imshow(oil_map, cmap=oil_cmap, alpha=0.9, vmin=vmin, vmax=vmax)

        # Title with statistics
        valid_pred = oil_map[~np.isnan(oil_map)]
        title = f"{sample_ids[idx]}\nMeasured: {oil_contents[idx]:.1f}% | Pred: {valid_pred.mean():.1f}%"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Remove extra subplots
    for i in range(n_samples, len(axes)):
        fig.delaxes(axes[i])

    # Add single colorbar
    fig.colorbar(im, ax=axes[:n_samples], orientation="horizontal", fraction=0.05, pad=0.05, label="Oil Content (%)")

    plt.suptitle("Oil Content Spatial Distribution - Sample Overview", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Summary grid saved to {output_path}")
    plt.close()


__all__ = [
    "predict_spatial_distribution",
    "plot_seed_oil_distribution",
    "visualize_all_seeds",
    "create_summary_grid",
]