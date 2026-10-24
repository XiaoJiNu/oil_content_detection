from __future__ import annotations

import numpy as np
import pandas as pd

from oil_content_detection.inference.advanced_training import (
    AdvancedHuajiaoRegressor,
    wavelength_columns,
)
from oil_content_detection.inference.training import FORBIDDEN_TRAINING_COLUMNS


def _make_frame(n_samples: int = 36, n_bands: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260514)
    wavelengths = [f"wl_{400 + idx * 3}" for idx in range(n_bands)]
    base = rng.normal(size=(n_samples, n_bands))
    trend = np.linspace(-1.0, 1.0, n_bands)
    y = 1.5 + 0.35 * (base @ trend) / n_bands + rng.normal(0.0, 0.03, size=n_samples)
    df = pd.DataFrame(base, columns=wavelengths)
    df["coverage_ratio"] = rng.uniform(0.10, 0.24, size=n_samples)
    df["log_valid_pixel_count"] = rng.uniform(10.0, 12.5, size=n_samples)
    df["valid_pixel_ratio"] = 1.0
    return df, y


def test_wavelength_columns_are_numeric_sorted() -> None:
    df = pd.DataFrame(columns=["wl_410", "coverage_ratio", "wl_398", "wl_bad", "wl_403"])

    assert wavelength_columns(df) == ["wl_398", "wl_403", "wl_410"]


def test_advanced_huajiao_regressor_predicts_and_keeps_schema_deployable() -> None:
    df, y = _make_frame()
    model = AdvancedHuajiaoRegressor(
        n_components=4,
        spectral_feature_set="summary",
        regressor_type="ridge",
        ridge_alpha=1.0,
    )

    model.fit(df, y)
    pred = model.predict(df.head(5))

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert model.base_feature_columns_ == list(df.columns)
    forbidden_in_base = sorted(set(FORBIDDEN_TRAINING_COLUMNS) & set(model.base_feature_columns_))
    forbidden_in_engineered = sorted(set(FORBIDDEN_TRAINING_COLUMNS) & set(model.engineered_feature_names_))
    assert forbidden_in_base == []
    assert forbidden_in_engineered == []


def test_advanced_huajiao_regressor_rejects_missing_feature_columns() -> None:
    df, y = _make_frame()
    model = AdvancedHuajiaoRegressor(n_components=3)
    model.fit(df, y)

    bad = df.drop(columns=["wl_400"])

    try:
        model.predict(bad)
    except ValueError as exc:
        assert "missing required feature columns" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing feature columns should fail")
