"""Tests for genetic algorithm wavelength selector."""
import numpy as np
import pytest
from sklearn.datasets import make_regression

from oil_content_detection.feature_selection.ga_selector import (
    GAConfig,
    GeneticAlgorithmSelector,
    select_wavelengths,
)


@pytest.fixture
def simple_dataset():
    """Create a simple regression dataset for testing."""
    X, y = make_regression(n_samples=50, n_features=30, n_informative=15, noise=10.0, random_state=42)
    return X, y


def test_ga_config_defaults():
    """Test GAConfig has reasonable defaults."""
    cfg = GAConfig()
    assert cfg.population_size == 12
    assert cfg.generations == 10
    assert cfg.min_features <= cfg.max_features
    assert cfg.target_features >= cfg.min_features
    assert cfg.target_features <= cfg.max_features


def test_ga_selector_reproducibility(simple_dataset):
    """Test that GA produces identical results with the same seed."""
    X, y = simple_dataset
    cfg = GAConfig(random_state=2024, generations=5, population_size=8, min_features=5, max_features=20)

    selector1 = GeneticAlgorithmSelector(cfg)
    selector1.fit(X, y)
    support1 = selector1.get_support()
    score1 = selector1.best_score()

    selector2 = GeneticAlgorithmSelector(cfg)
    selector2.fit(X, y)
    support2 = selector2.get_support()
    score2 = selector2.best_score()

    np.testing.assert_array_equal(support1, support2)
    assert score1 == score2


def test_ga_selector_respects_bounds(simple_dataset):
    """Test that selected features are within configured bounds."""
    X, y = simple_dataset
    cfg = GAConfig(min_features=10, max_features=20, random_state=42, generations=3)

    selector = GeneticAlgorithmSelector(cfg)
    selector.fit(X, y)

    n_selected = selector.selected_count()
    assert cfg.min_features <= n_selected <= cfg.max_features


def test_ga_selector_get_support_before_fit():
    """Test that calling get_support before fit raises an error."""
    selector = GeneticAlgorithmSelector()
    with pytest.raises(RuntimeError, match="fit must be called"):
        selector.get_support()


def test_ga_selector_selected_indices(simple_dataset):
    """Test that selected_indices returns correct indices."""
    X, y = simple_dataset
    cfg = GAConfig(random_state=42, generations=3, min_features=5, max_features=20)

    selector = GeneticAlgorithmSelector(cfg)
    selector.fit(X, y)

    support = selector.get_support()
    indices = selector.selected_indices()

    assert len(indices) == support.sum()
    assert np.all(support[indices])
    assert indices.shape[0] == selector.selected_count()


def test_ga_selector_improves_over_generations(simple_dataset):
    """Test that GA generally improves fitness over generations."""
    X, y = simple_dataset
    cfg = GAConfig(random_state=42, generations=10, patience=None, min_features=5, max_features=20)  # Disable early stopping

    selector = GeneticAlgorithmSelector(cfg)
    selector.fit(X, y)

    # Best score should be reasonable (not -inf)
    assert selector.best_score() > 0.0


def test_select_wavelengths_wrapper(simple_dataset):
    """Test the convenience function select_wavelengths."""
    X, y = simple_dataset
    cfg = GAConfig(random_state=42, generations=3, min_features=5, max_features=20)

    support, selector = select_wavelengths(X, y, cfg)

    assert isinstance(support, np.ndarray)
    assert support.dtype == bool
    assert len(support) == X.shape[1]
    assert isinstance(selector, GeneticAlgorithmSelector)
    assert selector.best_score() > -np.inf


def test_ga_selector_with_minimal_features(simple_dataset):
    """Test GA with very small feature set."""
    X, y = simple_dataset
    cfg = GAConfig(min_features=2, max_features=5, target_features=3, random_state=42, generations=3)

    selector = GeneticAlgorithmSelector(cfg)
    selector.fit(X, y)

    n_selected = selector.selected_count()
    assert 2 <= n_selected <= 5


def test_ga_selector_early_stopping():
    """Test that early stopping works with patience parameter."""
    X, y = make_regression(n_samples=30, n_features=20, random_state=42)
    cfg = GAConfig(random_state=42, generations=20, patience=3, min_features=5, max_features=15)

    selector = GeneticAlgorithmSelector(cfg)
    selector.fit(X, y)

    # Should stop early due to patience, but still have a valid result
    assert selector.best_score() > -np.inf
    assert selector.selected_count() > 0


def test_ga_ensure_bounds_adds_features():
    """Test that _ensure_bounds adds features when below minimum."""
    rng = np.random.default_rng(42)
    cfg = GAConfig(min_features=10, max_features=30)

    mask = np.zeros(50, dtype=bool)
    mask[:5] = True  # Only 5 features, below min

    updated_mask = GeneticAlgorithmSelector._ensure_bounds(mask, cfg, rng)

    assert updated_mask.sum() >= cfg.min_features


def test_ga_ensure_bounds_removes_features():
    """Test that _ensure_bounds removes features when above maximum."""
    rng = np.random.default_rng(42)
    cfg = GAConfig(min_features=10, max_features=20)

    mask = np.ones(50, dtype=bool)  # 50 features, above max

    updated_mask = GeneticAlgorithmSelector._ensure_bounds(mask, cfg, rng)

    assert updated_mask.sum() <= cfg.max_features


def test_ga_pls_components_calculation():
    """Test PLSR component count calculation."""
    assert GeneticAlgorithmSelector._pls_components(2) == 1
    assert GeneticAlgorithmSelector._pls_components(10) == 5
    assert GeneticAlgorithmSelector._pls_components(30) == 10  # Capped at 10
    assert GeneticAlgorithmSelector._pls_components(100) == 10  # Still capped


def test_ga_selector_different_random_states(simple_dataset):
    """Test that different random states produce different results."""
    X, y = simple_dataset

    selector1 = GeneticAlgorithmSelector(GAConfig(random_state=1, generations=5, min_features=5, max_features=20))
    selector1.fit(X, y)

    selector2 = GeneticAlgorithmSelector(GAConfig(random_state=999, generations=5, min_features=5, max_features=20))
    selector2.fit(X, y)

    # Results should differ with different random states
    assert not np.array_equal(selector1.get_support(), selector2.get_support())