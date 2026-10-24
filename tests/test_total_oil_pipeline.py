from __future__ import annotations

import numpy as np
import pandas as pd

from oil_content_detection.pipelines.total_oil_pipeline import (
    TotalOilExperimentConfig,
    run_total_oil_experiment,
    run_total_oil_experiment_with_predictions,
)


def test_total_oil_experiment_a_b_comparison() -> None:
    rng = np.random.default_rng(42)
    n_samples = 80
    n_wl = 20

    wl_cols = [f"wl_{900 + i * 5}" for i in range(n_wl)]
    X = rng.normal(0.5, 0.1, size=(n_samples, n_wl))

    oil_ml_per_gram = 0.02 + 0.005 * X[:, 2:6].mean(axis=1)

    valid_pixel_count = rng.integers(1000, 6000, size=n_samples)
    coverage_ratio = rng.uniform(0.01, 0.06, size=n_samples)
    weight_g = 0.001 * valid_pixel_count + 20.0 * coverage_ratio

    distill_ml = oil_ml_per_gram * weight_g

    df = pd.DataFrame(X, columns=wl_cols)
    df["oil_ml_per_gram"] = oil_ml_per_gram
    df["weight_g"] = weight_g
    df["distill_ml"] = distill_ml
    df["valid_pixel_count"] = valid_pixel_count
    df["coverage_ratio"] = coverage_ratio

    cfg = TotalOilExperimentConfig(
        spectral_preprocess=(),
        use_ga=False,
        test_size=0.2,
        random_state=0,
        direct_include_shape=True,
    )

    result = run_total_oil_experiment(df, config=cfg)

    assert result.metrics["total_a_val"]["r2"] > 0.95
    assert result.metrics["total_b_val"]["r2"] > 0.95
    assert result.metrics["oil_per_gram_val"]["r2"] > 0.8
    assert result.metrics["weight_val"]["r2"] > 0.8

    result2, train_pred_df, val_pred_df = run_total_oil_experiment_with_predictions(df, config=cfg)
    assert result2.metrics["total_a_val"]["r2"] == result.metrics["total_a_val"]["r2"]
    assert len(train_pred_df) + len(val_pred_df) == len(df)
    for col in [
        "sample_id",
        "actual_oil_ml_total",
        "predicted_oil_ml_total_a",
        "predicted_oil_ml_total_b",
        "absolute_error_a",
        "relative_error_a_percent",
        "absolute_error_b",
        "relative_error_b_percent",
    ]:
        assert col in val_pred_df.columns
