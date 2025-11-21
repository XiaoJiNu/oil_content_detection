"""Visualization functions for GA + PLSR results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def plot_ga_history(history_path: Path, save_path: Optional[Path] = None) -> None:
    """Plot GA training history (fitness over generations).

    Args:
        history_path: Path to ga_history.json file
        save_path: Optional path to save the figure
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    generations = [h["generation"] for h in history]
    best_scores = [h["best_score"] for h in history]
    mean_scores = [h["mean_score"] for h in history]
    max_scores = [h["max_score"] for h in history]
    min_scores = [h["min_score"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot fitness scores
    ax1.plot(generations, best_scores, "o-", linewidth=2, markersize=6, label="Best Score", color="#2E86AB")
    ax1.plot(generations, mean_scores, "s--", linewidth=1.5, markersize=4, label="Mean Score", color="#A23B72")
    ax1.fill_between(generations, min_scores, max_scores, alpha=0.2, color="#F18F01", label="Min-Max Range")

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Fitness Score (R²)", fontsize=12)
    ax1.set_title("GA Training History - Fitness Evolution", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot feature count
    best_features = [h["best_features"] for h in history]
    ax2.plot(generations, best_features, "o-", linewidth=2, markersize=6, color="#C73E1D")
    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Number of Selected Features", fontsize=12)
    ax2.set_title("Selected Feature Count Over Generations", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"GA history plot saved to {save_path}")
    else:
        plt.show()


def plot_spectral_selection(
    data_path: Path,
    wavelengths_path: Path,
    save_path: Optional[Path] = None,
) -> None:
    """Plot spectral data with selected wavelengths highlighted.

    Args:
        data_path: Path to CSV file with spectral data
        wavelengths_path: Path to selected_wavelengths.json
        save_path: Optional path to save the figure
    """
    # Load data
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    wavelengths = [int(c.split("_")[1]) for c in feature_cols]

    # Load selected wavelengths
    with open(wavelengths_path, "r") as f:
        selected_data = json.load(f)
        selected_wls = selected_data["wavelengths"]

    # Calculate mean spectrum
    X = df[feature_cols].to_numpy()
    mean_spectrum = X.mean(axis=0)
    std_spectrum = X.std(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot mean spectrum with std band
    ax.plot(wavelengths, mean_spectrum, linewidth=2, color="#2E86AB", label="Mean Spectrum")
    ax.fill_between(
        wavelengths,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        alpha=0.2,
        color="#2E86AB",
        label="±1 Std Dev",
    )

    # Highlight selected wavelengths
    for wl in selected_wls:
        idx = wavelengths.index(wl)
        ax.axvline(wl, color="#C73E1D", alpha=0.6, linestyle="--", linewidth=1)
        ax.plot(wl, mean_spectrum[idx], "o", color="#C73E1D", markersize=8)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title(f"Spectral Data with {len(selected_wls)} GA-Selected Wavelengths", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Spectral selection plot saved to {save_path}")
    else:
        plt.show()


def plot_prediction_results(
    data_path: Path,
    model_path: Path,
    support_path: Path,
    save_path: Optional[Path] = None,
    test_size: float = 0.333,
    random_state: int = 2024,
) -> None:
    """Plot prediction vs actual oil content.

    Args:
        data_path: Path to CSV file with spectral data
        model_path: Path to trained model .pkl file
        support_path: Path to feature support mask .npy file
        save_path: Optional path to save the figure
        test_size: Test set proportion (must match training)
        random_state: Random seed (must match training)
    """
    import pickle

    from sklearn.model_selection import train_test_split

    # Load data
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    X = df[feature_cols].to_numpy()
    y = df["oil_content"].to_numpy()

    # Load model and support
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    support = np.load(support_path)

    # Split data (must match training split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Predict
    y_train_pred = model.predict(X_train[:, support]).ravel()
    y_test_pred = model.predict(X_test[:, support]).ravel()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set
    ax1 = axes[0]
    ax1.scatter(y_train, y_train_pred, alpha=0.6, s=80, color="#2E86AB", edgecolors="white", linewidth=0.5)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    from sklearn.metrics import mean_squared_error, r2_score

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

    ax1.set_xlabel("Actual Oil Content (%)", fontsize=12)
    ax1.set_ylabel("Predicted Oil Content (%)", fontsize=12)
    ax1.set_title(f"Training Set (n={len(y_train)})\nR²={r2_train:.4f}, RMSE={rmse_train:.3f}", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test set
    ax2 = axes[1]
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=80, color="#A23B72", edgecolors="white", linewidth=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

    ax2.set_xlabel("Actual Oil Content (%)", fontsize=12)
    ax2.set_ylabel("Predicted Oil Content (%)", fontsize=12)
    ax2.set_title(f"Test Set (n={len(y_test)})\nR²={r2_test:.4f}, RMSE={rmse_test:.3f}", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction results plot saved to {save_path}")
    else:
        plt.show()


def plot_all_results(
    results_dir: Path,
    data_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate all visualization plots for a complete experiment.

    Args:
        results_dir: Directory containing experiment results
        data_path: Path to original CSV data file
        output_dir: Optional directory to save plots (defaults to results_dir/plots)
    """
    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations for {results_dir}...")

    # Plot 1: GA history
    history_path = results_dir / "ga_history.json"
    if history_path.exists():
        plot_ga_history(history_path, output_dir / "ga_history.png")
    else:
        print(f"Warning: {history_path} not found, skipping GA history plot")

    # Plot 2: Spectral selection
    wavelengths_path = results_dir / "selected_wavelengths.json"
    if wavelengths_path.exists():
        plot_spectral_selection(data_path, wavelengths_path, output_dir / "spectral_selection.png")
    else:
        print(f"Warning: {wavelengths_path} not found, skipping spectral selection plot")

    # Plot 3: Prediction results
    model_path = results_dir / "plsr_model.pkl"
    support_path = results_dir / "feature_support.npy"
    if model_path.exists() and support_path.exists():
        plot_prediction_results(data_path, model_path, support_path, output_dir / "prediction_results.png")
    else:
        print(f"Warning: Model or support file not found, skipping prediction results plot")

    print(f"\nAll plots saved to {output_dir}/")


__all__ = [
    "plot_ga_history",
    "plot_spectral_selection",
    "plot_prediction_results",
    "plot_all_results",
]