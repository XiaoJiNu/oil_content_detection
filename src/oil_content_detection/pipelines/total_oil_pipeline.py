"""Total oil content prediction pipelines (A+B comparison).

Scheme A: predict oil per gram via PLSR, predict weight via ROI features,
          then multiply to obtain total oil.
Scheme B: predict total oil directly via PLSR on (spectral [+ shape]) features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from oil_content_detection.feature_selection.ga_selector import GAConfig
from oil_content_detection.models.plsr_pipeline import PLSRExperimentConfig, fit_plsr_model
from oil_content_detection.models.weight_regressor import WeightRegressor, WeightRegressorConfig
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline
from oil_content_detection.utils import get_logger, rmse

logger = get_logger(__name__)


def build_shape_feature_matrix(
    df: pd.DataFrame,
    cols: Sequence[str] = ("valid_pixel_count", "coverage_ratio"),
) -> Tuple[np.ndarray, List[str]]:
    """Build a simple shape feature matrix from dataframe columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing shape feature columns: {missing}")

    base = df[list(cols)].to_numpy(dtype=float)
    names = list(cols)

    if "valid_pixel_count" in cols:
        area = df["valid_pixel_count"].to_numpy(dtype=float)
        base = np.column_stack([base, np.sqrt(area), np.log1p(area)])
        names.extend(["sqrt_valid_pixel_count", "log1p_valid_pixel_count"])

    if "coverage_ratio" in cols:
        cov = df["coverage_ratio"].to_numpy(dtype=float)
        base = np.column_stack([base, cov**2])
        names.append("coverage_ratio_sq")

    return base, names


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"r2": float(r2), "rmse": float(rmse_val), "mae": float(mae), "mape": float(mape)}


@dataclass
class TotalOilExperimentConfig:
    spectral_preprocess: Sequence[PreprocessStep | str] = ()
    use_ga: bool = False
    ga_config: Optional[GAConfig] = None
    test_size: float = 0.25
    random_state: int = 2024
    cv_splits: int = 5
    max_components: int = 12

    weight_model_config: WeightRegressorConfig = field(default_factory=WeightRegressorConfig)
    shape_feature_cols: Tuple[str, ...] = ("valid_pixel_count", "coverage_ratio")
    scale_shape: bool = True
    direct_include_shape: bool = True


@dataclass
class TotalOilExperimentResult:
    split_source: str
    n_train: int
    n_test: int
    shape_feature_names: List[str]

    oil_model_selected_wavelengths: List[int]
    oil_model_n_components: int
    oil_model_ga_score: Optional[float]

    direct_model_selected_features: List[int]
    direct_model_n_components: int
    direct_model_ga_score: Optional[float]

    metrics: Dict[str, Dict[str, float]]


def _build_predictions_table(
    df: pd.DataFrame,
    y_total_true: np.ndarray,
    y_total_pred_a: np.ndarray,
    y_total_pred_b: np.ndarray,
    y_oil_true: np.ndarray,
    y_oil_pred: np.ndarray,
    y_weight_true: np.ndarray,
    y_weight_pred: np.ndarray,
) -> pd.DataFrame:
    sample_ids: Union[np.ndarray, List[str]]
    if "sample_id" in df.columns:
        sample_ids = df["sample_id"].to_numpy()
    else:
        sample_ids = df.index.astype(str).to_numpy()

    abs_err_a = np.abs(y_total_true - y_total_pred_a)
    rel_err_a = (y_total_true - y_total_pred_a) / np.maximum(y_total_true, 1e-8) * 100.0
    abs_err_b = np.abs(y_total_true - y_total_pred_b)
    rel_err_b = (y_total_true - y_total_pred_b) / np.maximum(y_total_true, 1e-8) * 100.0

    abs_err_oil = np.abs(y_oil_true - y_oil_pred)
    rel_err_oil = (y_oil_true - y_oil_pred) / np.maximum(y_oil_true, 1e-8) * 100.0

    abs_err_w = np.abs(y_weight_true - y_weight_pred)
    rel_err_w = (y_weight_true - y_weight_pred) / np.maximum(y_weight_true, 1e-8) * 100.0

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "actual_oil_ml_total": y_total_true,
            "predicted_oil_ml_total_a": y_total_pred_a,
            "predicted_oil_ml_total_b": y_total_pred_b,
            "absolute_error_a": abs_err_a,
            "relative_error_a_percent": rel_err_a,
            "absolute_error_b": abs_err_b,
            "relative_error_b_percent": rel_err_b,
            "actual_oil_ml_per_g": y_oil_true,
            "predicted_oil_ml_per_g": y_oil_pred,
            "absolute_error_oil_ml_per_g": abs_err_oil,
            "relative_error_oil_ml_per_g_percent": rel_err_oil,
            "actual_weight_g": y_weight_true,
            "predicted_weight_g": y_weight_pred,
            "absolute_error_weight_g": abs_err_w,
            "relative_error_weight_g_percent": rel_err_w,
        }
    )


def _run_total_oil_core(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: TotalOilExperimentConfig,
) -> Tuple[TotalOilExperimentResult, pd.DataFrame, pd.DataFrame]:
    if val_df is None:
        train_df, val_df = train_test_split(
            train_df,
            test_size=config.test_size,
            random_state=config.random_state,
            shuffle=True,
        )
        split_source = "random_split"
    else:
        split_source = "explicit_train_val"

    wl_cols = [c for c in train_df.columns if c.startswith("wl_")]
    if not wl_cols:
        raise ValueError("No spectral columns found (prefix 'wl_').")
    for c in wl_cols:
        if c not in val_df.columns:
            raise KeyError(f"Validation dataframe missing spectral column: {c}")

    wavelengths = [int(float(c.split("_")[1])) for c in wl_cols]

    X_train_spec = train_df[wl_cols].to_numpy(dtype=float)
    X_val_spec = val_df[wl_cols].to_numpy(dtype=float)

    y_train_oil = train_df["oil_ml_per_gram"].to_numpy(dtype=float)
    y_val_oil = val_df["oil_ml_per_gram"].to_numpy(dtype=float)
    y_train_weight = train_df["weight_g"].to_numpy(dtype=float)
    y_val_weight = val_df["weight_g"].to_numpy(dtype=float)

    if "distill_ml" in train_df.columns and "distill_ml" in val_df.columns:
        y_train_total = train_df["distill_ml"].to_numpy(dtype=float)
        y_val_total = val_df["distill_ml"].to_numpy(dtype=float)
    else:
        y_train_total = y_train_oil * y_train_weight
        y_val_total = y_val_oil * y_val_weight

    # Shape features
    X_train_shape, shape_names = build_shape_feature_matrix(train_df, config.shape_feature_cols)
    X_val_shape, _ = build_shape_feature_matrix(val_df, config.shape_feature_cols)
    if config.scale_shape:
        scaler = StandardScaler()
        X_train_shape = scaler.fit_transform(X_train_shape)
        X_val_shape = scaler.transform(X_val_shape)

    # === Scheme A ===
    oil_cfg = PLSRExperimentConfig(
        preprocess=config.spectral_preprocess,
        use_ga=config.use_ga,
        ga_config=config.ga_config,
        random_state=config.random_state,
        cv_splits=config.cv_splits,
        max_components=config.max_components,
        test_size=config.test_size,
    )
    oil_fit = fit_plsr_model(X_train_spec, y_train_oil, wavelengths=wavelengths, config=oil_cfg)

    X_train_spec_proc = apply_preprocessing_pipeline(X_train_spec, config.spectral_preprocess)
    X_val_spec_proc = apply_preprocessing_pipeline(X_val_spec, config.spectral_preprocess)

    y_train_oil_pred = oil_fit.model.predict(X_train_spec_proc[:, oil_fit.support_mask]).ravel()
    y_val_oil_pred = oil_fit.model.predict(X_val_spec_proc[:, oil_fit.support_mask]).ravel()

    weight_reg = WeightRegressor(config.weight_model_config).fit(X_train_shape, y_train_weight)
    y_train_weight_pred = weight_reg.predict(X_train_shape)
    y_val_weight_pred = weight_reg.predict(X_val_shape)

    y_train_total_pred_a = y_train_oil_pred * y_train_weight_pred
    y_val_total_pred_a = y_val_oil_pred * y_val_weight_pred

    # === Scheme B ===
    X_train_direct = X_train_spec_proc
    X_val_direct = X_val_spec_proc
    if config.direct_include_shape:
        X_train_direct = np.hstack([X_train_direct, X_train_shape])
        X_val_direct = np.hstack([X_val_direct, X_val_shape])

    direct_cfg = PLSRExperimentConfig(
        preprocess=(),
        use_ga=config.use_ga,
        ga_config=config.ga_config,
        random_state=config.random_state,
        cv_splits=config.cv_splits,
        max_components=config.max_components,
        test_size=config.test_size,
    )
    direct_fit = fit_plsr_model(X_train_direct, y_train_total, wavelengths=None, config=direct_cfg)

    y_train_total_pred_b = direct_fit.model.predict(X_train_direct[:, direct_fit.support_mask]).ravel()
    y_val_total_pred_b = direct_fit.model.predict(X_val_direct[:, direct_fit.support_mask]).ravel()

    metrics: Dict[str, Dict[str, float]] = {
        "oil_per_gram_train": _compute_metrics(y_train_oil, y_train_oil_pred),
        "oil_per_gram_val": _compute_metrics(y_val_oil, y_val_oil_pred),
        "weight_train": _compute_metrics(y_train_weight, y_train_weight_pred),
        "weight_val": _compute_metrics(y_val_weight, y_val_weight_pred),
        "total_a_train": _compute_metrics(y_train_total, y_train_total_pred_a),
        "total_a_val": _compute_metrics(y_val_total, y_val_total_pred_a),
        "total_b_train": _compute_metrics(y_train_total, y_train_total_pred_b),
        "total_b_val": _compute_metrics(y_val_total, y_val_total_pred_b),
    }

    logger.info(
        "Total oil comparison done. A(val) R2=%.4f, RMSE=%.6f; B(val) R2=%.4f, RMSE=%.6f",
        metrics["total_a_val"]["r2"],
        metrics["total_a_val"]["rmse"],
        metrics["total_b_val"]["r2"],
        metrics["total_b_val"]["rmse"],
    )

    result = TotalOilExperimentResult(
        split_source=split_source,
        n_train=len(train_df),
        n_test=len(val_df),
        shape_feature_names=shape_names,
        oil_model_selected_wavelengths=oil_fit.selected_wavelengths,
        oil_model_n_components=oil_fit.n_components,
        oil_model_ga_score=oil_fit.ga_score,
        direct_model_selected_features=direct_fit.selected_wavelengths,
        direct_model_n_components=direct_fit.n_components,
        direct_model_ga_score=direct_fit.ga_score,
        metrics=metrics,
    )

    train_predictions = _build_predictions_table(
        train_df,
        y_train_total,
        y_train_total_pred_a,
        y_train_total_pred_b,
        y_train_oil,
        y_train_oil_pred,
        y_train_weight,
        y_train_weight_pred,
    )
    val_predictions = _build_predictions_table(
        val_df,
        y_val_total,
        y_val_total_pred_a,
        y_val_total_pred_b,
        y_val_oil,
        y_val_oil_pred,
        y_val_weight,
        y_val_weight_pred,
    )
    return result, train_predictions, val_predictions


def run_total_oil_experiment(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    config: TotalOilExperimentConfig = TotalOilExperimentConfig(),
) -> TotalOilExperimentResult:
    """Run A+B total oil comparison and return metrics only."""
    result, _, _ = _run_total_oil_core(train_df, val_df, config)
    return result


def run_total_oil_experiment_with_predictions(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    config: TotalOilExperimentConfig = TotalOilExperimentConfig(),
) -> Tuple[TotalOilExperimentResult, pd.DataFrame, pd.DataFrame]:
    """Run A+B comparison and also return per-sample prediction tables.

    Returns:
        (result, train_predictions_df, val_predictions_df)
    """
    return _run_total_oil_core(train_df, val_df, config)


__all__ = [
    "TotalOilExperimentConfig",
    "TotalOilExperimentResult",
    "build_shape_feature_matrix",
    "run_total_oil_experiment",
    "run_total_oil_experiment_with_predictions",
]
