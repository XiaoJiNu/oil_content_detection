"""Pipeline to run GA-selected PLSR on Spectral Set II data.

Uses the simulated dataset in ``data/processed/set_II/mean_spectra.csv``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from oil_content_detection.feature_selection.ga_selector import GAConfig, GeneticAlgorithmSelector


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


@dataclass
class RunResult:
    selected_wavelengths: List[int]
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
    ga_score: float
    n_components: int


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("wl_")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["oil_content"].to_numpy(dtype=float)
    return X, y, feature_cols


def train_plsr_best(config: RunConfig = RunConfig()) -> RunResult:
    X, y, feature_cols = load_dataset(config.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    ga_cfg = GAConfig(
        population_size=config.ga_population,
        generations=config.ga_generations,
        min_features=config.ga_min_features,
        max_features=config.ga_max_features,
        target_features=config.target_features,
        random_state=config.random_state,
        cv_splits=3,
    )

    selector = GeneticAlgorithmSelector(ga_cfg)
    selector.fit(X_train, y_train)
    support = selector.get_support()
    selected_indices = selector.selected_indices()

    n_components = min(10, max(1, selector.selected_count() // 2))
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train[:, support], y_train)

    y_train_pred = model.predict(X_train[:, support]).ravel()
    y_test_pred = model.predict(X_test[:, support]).ravel()

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    wavelengths = [int(feature_cols[idx].split("_")[1]) for idx in selected_indices]

    return RunResult(
        selected_wavelengths=wavelengths,
        train_r2=train_r2,
        train_rmse=train_rmse,
        test_r2=test_r2,
        test_rmse=test_rmse,
        ga_score=selector.best_score(),
        n_components=n_components,
    )


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
