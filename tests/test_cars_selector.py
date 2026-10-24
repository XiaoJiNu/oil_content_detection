"""Tests for CARS (Competitive Adaptive Reweighted Sampling) selector."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from oil_content_detection.feature_selection.cars_selector import (
    CARSConfig,
    CARSSelector,
    select_wavelengths,
)


@pytest.fixture
def simple_dataset():
    X, y = make_regression(
        n_samples=60, n_features=40, n_informative=12, noise=8.0, random_state=7
    )
    return X, y


def test_cars_config_defaults():
    cfg = CARSConfig()
    assert cfg.n_mc > 0
    assert 0 < cfg.sample_ratio <= 1.0
    assert cfg.cv_splits >= 2
    assert cfg.pls_n_components >= 1


def test_cars_get_support_before_fit_raises():
    selector = CARSSelector(CARSConfig(n_mc=5, random_state=0))
    with pytest.raises(RuntimeError, match="fit must be called"):
        selector.get_support()


def test_cars_selector_reproducibility(simple_dataset):
    X, y = simple_dataset
    cfg = CARSConfig(
        n_mc=10,
        sample_ratio=0.8,
        cv_splits=3,
        pls_n_components=6,
        min_features=3,
        random_state=2026,
    )

    selector1 = CARSSelector(cfg).fit(X, y)
    selector2 = CARSSelector(cfg).fit(X, y)

    np.testing.assert_array_equal(selector1.get_support(), selector2.get_support())
    assert selector1.best_rmsecv() == selector2.best_rmsecv()
    assert selector1.best_iteration() == selector2.best_iteration()


def test_cars_selector_outputs(simple_dataset):
    X, y = simple_dataset
    cfg = CARSConfig(
        n_mc=8,
        sample_ratio=0.75,
        cv_splits=4,
        pls_n_components=5,
        min_features=2,
        random_state=1,
    )

    selector = CARSSelector(cfg).fit(X, y)
    support = selector.get_support()

    assert support.dtype == bool
    assert support.shape == (X.shape[1],)
    assert selector.selected_count() >= cfg.min_features
    assert selector.selected_count() <= X.shape[1]
    assert selector.history()
    assert len(selector.history()) == cfg.n_mc

    best_rmse = selector.best_rmsecv()
    history_rmse = [float(h["rmsecv"]) for h in selector.history()]
    assert best_rmse == min(history_rmse)


def test_select_wavelengths_wrapper(simple_dataset):
    X, y = simple_dataset
    cfg = CARSConfig(n_mc=6, cv_splits=3, pls_n_components=4, random_state=42)
    support, selector = select_wavelengths(X, y, cfg)

    assert isinstance(support, np.ndarray)
    assert support.dtype == bool
    assert support.shape == (X.shape[1],)
    assert isinstance(selector, CARSSelector)
    assert selector.selected_count() == int(support.sum())
