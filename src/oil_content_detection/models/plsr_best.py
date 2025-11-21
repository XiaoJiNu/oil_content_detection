"""Pipeline to run GA-selected PLSR on Spectral Set II data.

Uses the simulated dataset in ``data/processed/set_II/mean_spectra.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from oil_content_detection.utils import (
    get_logger,
    save_model,
    save_results_json,
    save_wavelengths,
    setup_single_thread,
    validate_spectral_array,
    validate_spectral_dataframe,
)

setup_single_thread()

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from oil_content_detection.feature_selection.ga_selector import GAConfig, GeneticAlgorithmSelector

logger = get_logger(__name__)


@dataclass
class RunConfig:
    data_path: Path = Path("data/processed/set_II/mean_spectra.csv")
    test_size: float = 34 / 102
    random_state: int = 2024
    ga_generations: int = 10
    ga_population: int = 12
    ga_min_features: int = 10
    ga_max_features: int = 22
    target_features: int = 18
    output_dir: Optional[Path] = None  # If set, save results to this directory
    save_model_file: bool = True  # Whether to save the trained model
    verbose: bool = False  # If True, print GA progress


@dataclass
class RunResult:
    selected_wavelengths: List[int]
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
    ga_score: float
    n_components: int
    model_path: Optional[Path] = None
    results_path: Optional[Path] = None
    wavelengths_path: Optional[Path] = None


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and validate spectral dataset from CSV.

    Args:
        path: Path to CSV file containing spectral data

    Returns:
        Tuple of (feature matrix, target vector, feature column names)

    Raises:
        FileNotFoundError: If file does not exist
        DataValidationError: If data validation fails
    """
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # Validate dataframe structure
    validate_spectral_dataframe(df)

    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["oil_content"].to_numpy(dtype=float)

    # Validate arrays
    validate_spectral_array(X, y)

    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} wavelengths")
    logger.info(f"Oil content range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y, feature_cols


def train_plsr_best(config: RunConfig = RunConfig()) -> RunResult:
    """Train PLSR model with GA-selected wavelengths.

    Args:
        config: Run configuration

    Returns:
        Training results including metrics and selected wavelengths
    """
    logger.info("Starting GA + PLSR training pipeline")

    X, y, feature_cols = load_dataset(config.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    logger.info(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

    ga_cfg = GAConfig(
        population_size=config.ga_population,
        generations=config.ga_generations,
        min_features=config.ga_min_features,
        max_features=config.ga_max_features,
        target_features=config.target_features,
        random_state=config.random_state,
        cv_splits=3,
        verbose=config.verbose,
    )

    logger.info(f"Running GA feature selection: {config.ga_generations} generations, population {config.ga_population}")
    selector = GeneticAlgorithmSelector(ga_cfg)
    selector.fit(X_train, y_train)
    support = selector.get_support()
    selected_indices = selector.selected_indices()
    logger.info(f"GA selected {selector.selected_count()} features with CV score {selector.best_score():.4f}")

    n_components = min(10, max(1, selector.selected_count() // 2))
    logger.info(f"Training PLSR with {n_components} components")
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train[:, support], y_train)

    y_train_pred = model.predict(X_train[:, support]).ravel()
    y_test_pred = model.predict(X_test[:, support]).ravel()

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    logger.info(f"Train R²={train_r2:.4f}, RMSE={train_rmse:.3f}")
    logger.info(f"Test  R²={test_r2:.4f}, RMSE={test_rmse:.3f}")

    wavelengths = [int(feature_cols[idx].split("_")[1]) for idx in selected_indices]

    result = RunResult(
        selected_wavelengths=wavelengths,
        train_r2=train_r2,
        train_rmse=train_rmse,
        test_r2=test_r2,
        test_rmse=test_rmse,
        ga_score=selector.best_score(),
        n_components=n_components,
    )

    # Save results if output directory is specified
    if config.output_dir:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_path = save_results_json(result, output_dir / "results.json", include_timestamp=True)
        result.results_path = results_path

        # Save selected wavelengths
        wavelengths_path = output_dir / "selected_wavelengths.json"
        save_wavelengths(wavelengths, wavelengths_path)
        result.wavelengths_path = wavelengths_path

        # Save trained model
        if config.save_model_file:
            model_path = output_dir / "plsr_model.pkl"
            save_model(model, model_path)
            result.model_path = model_path

        # Save feature support mask
        support_path = output_dir / "feature_support.npy"
        np.save(support_path, support)
        logger.info(f"Feature support mask saved to {support_path}")

        # Save GA training history
        from oil_content_detection.utils.io import _convert_to_serializable
        import json

        history_path = output_dir / "ga_history.json"
        history_data = _convert_to_serializable(selector.history())
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"GA training history saved to {history_path}")

    return result


def run_and_print(config: RunConfig = RunConfig()) -> RunResult:
    result = train_plsr_best(config)
    print("=== GA + PLSR Results ===")
    print(f"Selected wavelengths ({len(result.selected_wavelengths)}): {result.selected_wavelengths}")
    print(f"PLSR components: {result.n_components}")
    print(f"GA cross-val score: {result.ga_score:.4f}")
    print(f"Train R^2: {result.train_r2:.4f}, RMSE: {result.train_rmse:.3f}")
    print(f"Test  R^2: {result.test_r2:.4f}, RMSE: {result.test_rmse:.3f}")
    return result


if __name__ == "__main__":
    run_and_print()
