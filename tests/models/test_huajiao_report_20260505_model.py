from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oil_content_detection.models.huajiao_report_20260505 import (
    Report20260505Config,
    add_report_metadata,
    fit_report_20260505_model,
    wavelength_columns,
)


def _toy_spectra() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "sample_id": ["0001_11", "0002_11", "0003_11", "0004_11", "0007_11", "0008_11"],
            "wl_500": [1.0, 2.2, 2.7, 4.1, 5.3, 6.0],
            "wl_400": [2.0, 2.6, 4.2, 4.6, 6.4, 6.8],
            "wl_600": [3.0, 4.4, 4.9, 6.6, 7.2, 8.8],
            "wl_700": [1.2, 1.5, 2.4, 2.9, 3.8, 4.0],
            "wl_800": [0.8, 1.7, 1.9, 3.2, 3.5, 4.7],
            "pixel_count": [100, 100, 100, 100, 100, 100],
            "valid_pixel_count": [90, 80, 70, 60, 75, 65],
            "coverage_ratio": [0.1, 0.2, 0.3, 0.4, 0.25, 0.35],
            "weight_g": [10.0, 11.0, 12.0, 13.0, 12.5, 13.5],
            "oil_ml_per_100g": [0.5, 0.8, 1.1, 1.4, 1.0, 1.6],
        }
    )
    val = pd.DataFrame(
        {
            "sample_id": ["0005_11", "0006_11"],
            "wl_500": [1.5, 3.5],
            "wl_400": [2.5, 4.5],
            "wl_600": [3.5, 5.5],
            "wl_700": [1.4, 2.6],
            "wl_800": [1.2, 2.7],
            "pixel_count": [100, 100],
            "valid_pixel_count": [85, 65],
            "coverage_ratio": [0.15, 0.35],
            "weight_g": [10.5, 12.5],
            "oil_ml_per_100g": [0.65, 1.25],
        }
    )
    train_md = pd.DataFrame(
        {"sample_dir": [str(Path("/data") / b / sid) for b, sid in zip(["A", "A", "B", "B", "B", "A"], train["sample_id"])]}
    )
    val_md = pd.DataFrame({"sample_dir": [str(Path("/data") / b / sid) for b, sid in zip(["A", "C"], val["sample_id"])]})
    return train, val, train_md, val_md


def test_report_metadata_matches_experiment_feature_recipe() -> None:
    train, _, train_md, _ = _toy_spectra()

    enriched = add_report_metadata(train, train_md)

    assert enriched["batch"].tolist() == ["A", "A", "B", "B", "B", "A"]
    assert enriched["sid_num"].tolist() == [1, 2, 3, 4, 7, 8]
    assert np.allclose(enriched["log_valid_pixel_count"], np.log1p(train["valid_pixel_count"]))
    assert np.allclose(enriched["valid_pixel_ratio"], train["valid_pixel_count"] / train["pixel_count"])


def test_report_model_uses_sorted_wavelengths_union_batch_onehot_and_expected_meta() -> None:
    train, val, train_md, val_md = _toy_spectra()
    config = Report20260505Config(n_components=2, sg_window_length=3)

    bundle = fit_report_20260505_model(train, val, train_md, val_md, config=config)
    pred = bundle.predict(val, val_md)

    assert wavelength_columns(train) == ["wl_400", "wl_500", "wl_600", "wl_700", "wl_800"]
    assert bundle.batch_categories == ["A", "B", "C"]
    assert bundle.meta_columns == ["batch", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "weight_g", "sid_num"]
    assert bundle.feature_names == [
        "pls_1",
        "pls_2",
        "batch=A",
        "batch=B",
        "batch=C",
        "coverage_ratio",
        "log_valid_pixel_count",
        "valid_pixel_ratio",
        "weight_g",
        "sid_num",
    ]
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
