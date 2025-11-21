"""Tests for PLSR pipeline with GA feature selection."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oil_content_detection.models.plsr_best import (
    RunConfig,
    RunResult,
    load_dataset,
    train_plsr_best,
)


@pytest.fixture
def mock_dataset_csv(tmp_path):
    """Create a mock CSV dataset for testing."""
    # Generate synthetic spectral data
    n_samples = 50
    wavelengths = np.arange(900, 1701, 5)
    n_wavelengths = len(wavelengths)

    # Create random spectral data
    rng = np.random.default_rng(42)
    X = rng.normal(0.5, 0.1, size=(n_samples, n_wavelengths))
    y = 25.0 + 10.0 * X[:, 50:100].mean(axis=1) + rng.normal(0, 2.0, size=n_samples)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"wl_{wl}" for wl in wavelengths])
    df.insert(0, "sample_id", [f"sample_{i:03d}" for i in range(n_samples)])
    df["oil_content"] = y

    # Save to CSV
    csv_path = tmp_path / "test_spectra.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, n_wavelengths, n_samples


def test_load_dataset(mock_dataset_csv):
    """Test dataset loading from CSV."""
    csv_path, n_wavelengths, n_samples = mock_dataset_csv

    X, y, feature_cols = load_dataset(csv_path)

    assert X.shape == (n_samples, n_wavelengths)
    assert y.shape == (n_samples,)
    assert len(feature_cols) == n_wavelengths
    assert all(col.startswith("wl_") for col in feature_cols)


def test_load_dataset_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("/nonexistent/path/data.csv"))


def test_run_config_defaults():
    """Test RunConfig has reasonable defaults."""
    cfg = RunConfig()

    assert cfg.data_path == Path("data/processed/set_II/mean_spectra.csv")
    assert 0 < cfg.test_size < 1
    assert cfg.random_state is not None
    assert cfg.ga_generations > 0
    assert cfg.ga_population > 0
    assert cfg.ga_min_features <= cfg.ga_max_features
    assert cfg.ga_min_features <= cfg.target_features <= cfg.ga_max_features


def test_train_plsr_best_reproducibility(mock_dataset_csv):
    """Test that training produces reproducible results."""
    csv_path, _, _ = mock_dataset_csv

    cfg = RunConfig(
        data_path=csv_path,
        test_size=0.3,
        random_state=2024,
        ga_generations=3,
        ga_population=6,
        ga_min_features=5,
        ga_max_features=15,
        target_features=10,
    )

    result1 = train_plsr_best(cfg)
    result2 = train_plsr_best(cfg)

    # Results should be identical with same seed
    assert result1.selected_wavelengths == result2.selected_wavelengths
    assert result1.train_r2 == result2.train_r2
    assert result1.test_r2 == result2.test_r2
    assert result1.n_components == result2.n_components


def test_train_plsr_best_result_structure(mock_dataset_csv):
    """Test that train_plsr_best returns proper RunResult."""
    csv_path, _, _ = mock_dataset_csv

    cfg = RunConfig(
        data_path=csv_path,
        test_size=0.3,
        random_state=42,
        ga_generations=2,
        ga_population=4,
        ga_min_features=5,
        ga_max_features=50,
    )

    result = train_plsr_best(cfg)

    assert isinstance(result, RunResult)
    assert isinstance(result.selected_wavelengths, list)
    assert len(result.selected_wavelengths) > 0
    assert all(isinstance(wl, int) for wl in result.selected_wavelengths)
    assert isinstance(result.train_r2, float)
    assert isinstance(result.test_r2, float)
    assert isinstance(result.train_rmse, float)
    assert isinstance(result.test_rmse, float)
    assert isinstance(result.n_components, int)
    assert result.n_components > 0


def test_train_plsr_best_metrics_validity(mock_dataset_csv):
    """Test that metrics are in reasonable ranges."""
    csv_path, _, _ = mock_dataset_csv

    cfg = RunConfig(
        data_path=csv_path,
        test_size=0.3,
        random_state=42,
        ga_generations=5,
        ga_population=8,
    )

    result = train_plsr_best(cfg)

    # R² should be between -inf and 1, but typically > 0 for reasonable models
    assert -10 < result.train_r2 <= 1.0
    assert -10 < result.test_r2 <= 1.0

    # RMSE should be positive
    assert result.train_rmse > 0
    assert result.test_rmse > 0

    # GA score should not be -inf
    assert result.ga_score > -np.inf


def test_train_plsr_best_wavelength_selection(mock_dataset_csv):
    """Test that selected wavelengths are within valid range."""
    csv_path, _, _ = mock_dataset_csv

    cfg = RunConfig(
        data_path=csv_path,
        random_state=42,
        ga_generations=3,
        ga_min_features=5,
        ga_max_features=20,
    )

    result = train_plsr_best(cfg)

    # Check wavelength count is within bounds
    assert cfg.ga_min_features <= len(result.selected_wavelengths) <= cfg.ga_max_features

    # Check wavelengths are valid integers in expected range
    assert all(900 <= wl <= 1700 for wl in result.selected_wavelengths)


def test_train_plsr_best_components_calculation(mock_dataset_csv):
    """Test that n_components is calculated correctly."""
    csv_path, _, _ = mock_dataset_csv

    cfg = RunConfig(
        data_path=csv_path,
        random_state=42,
        ga_generations=2,
        ga_min_features=4,
        ga_max_features=10,
    )

    result = train_plsr_best(cfg)

    # n_components should be min(10, selected_count // 2)
    expected_max = min(10, len(result.selected_wavelengths) // 2)
    assert result.n_components <= expected_max
    assert result.n_components >= 1


def test_train_plsr_best_different_test_sizes(mock_dataset_csv):
    """Test training with different test set sizes."""
    csv_path, _, _ = mock_dataset_csv

    for test_size in [0.2, 0.3, 0.4]:
        cfg = RunConfig(
            data_path=csv_path,
            test_size=test_size,
            random_state=42,
            ga_generations=2,
            ga_population=4,
        )

        result = train_plsr_best(cfg)

        # Should successfully train with different splits
        assert result.train_r2 is not None
        assert result.test_r2 is not None


def test_load_dataset_missing_oil_content_column():
    """Test that missing oil_content column raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # CSV without oil_content column
        f.write("sample_id,wl_900,wl_905\n")
        f.write("sample_001,0.5,0.6\n")
        temp_path = Path(f.name)

    try:
        from oil_content_detection.utils import DataValidationError
        with pytest.raises(DataValidationError):
            load_dataset(temp_path)
    finally:
        temp_path.unlink()


def test_load_dataset_no_wavelength_columns():
    """Test that CSV without wl_ columns raises appropriate error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # CSV without any wl_ columns
        f.write("sample_id,oil_content\n")
        f.write("sample_001,25.0\n")
        temp_path = Path(f.name)

    try:
        from oil_content_detection.utils import DataValidationError
        with pytest.raises(DataValidationError):
            load_dataset(temp_path)
    finally:
        temp_path.unlink()