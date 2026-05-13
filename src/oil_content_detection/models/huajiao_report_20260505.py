"""Reproduction utilities for the 2026-05-05 Huajiao oil-content report model.

The report model is the formal split engineering-best model recorded in:
``docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md``.

Recipe:
- SNV + Savitzky-Golay first derivative (window=13, polyorder=2)
- PLSRegression(n_components=10, scale=False)
- batch one-hot + coverage_ratio + log_valid_pixel_count + valid_pixel_ratio + weight_g + sid_num
- StandardScaler + HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=2000)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline

TARGET_COLUMN = "oil_ml_per_100g"
META_COLUMNS = ["batch", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "weight_g", "sid_num"]


@dataclass(frozen=True)
class Report20260505Config:
    """Exact hyperparameters used to reproduce the 2026-05-05 report model."""

    target: str = TARGET_COLUMN
    n_components: int = 10
    sg_window_length: int = 13
    sg_polyorder: int = 2
    sg_deriv: int = 1
    huber_alpha: float = 0.001
    huber_epsilon: float = 1.35
    huber_max_iter: int = 2000
    pls_scale: bool = False

    def preprocess_steps(self) -> list[PreprocessStep]:
        return [
            PreprocessStep("snv"),
            PreprocessStep(
                "sg",
                {
                    "window_length": self.sg_window_length,
                    "polyorder": self.sg_polyorder,
                    "deriv": self.sg_deriv,
                },
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportModelBundle:
    """Serializable bundle for the report model and its feature recipe."""

    config: Report20260505Config
    wavelength_columns: list[str]
    meta_columns: list[str]
    batch_categories: list[str]
    feature_names: list[str]
    pls_model: PLSRegression
    final_model: Pipeline

    def transform(self, spectra_df: pd.DataFrame, metadata_df: pd.DataFrame) -> np.ndarray:
        enriched = add_report_metadata(spectra_df, metadata_df)
        X = enriched[self.wavelength_columns].to_numpy(dtype=float)
        X_proc = apply_preprocessing_pipeline(X, self.config.preprocess_steps())
        X_pls = self.pls_model.transform(X_proc)
        features, _ = append_report_metadata_features(
            X_pls,
            enriched,
            meta_columns=self.meta_columns,
            batch_categories=self.batch_categories,
        )
        return features

    def predict(self, spectra_df: pd.DataFrame, metadata_df: pd.DataFrame) -> np.ndarray:
        features = self.transform(spectra_df, metadata_df)
        return np.asarray(self.final_model.predict(features), dtype=float).reshape(-1)

    def manifest(self) -> dict[str, Any]:
        return {
            "model_family": "huajiao_report_20260505_snv_sg1_pls10_batch_quality_huber",
            "target": self.config.target,
            "unit": "ml/100g",
            "config": self.config.to_dict(),
            "wavelength_columns": self.wavelength_columns,
            "meta_columns": self.meta_columns,
            "batch_categories": self.batch_categories,
            "feature_names": self.feature_names,
            "n_wavelength_features": len(self.wavelength_columns),
            "n_final_features": len(self.feature_names),
            "uses_weight_g": "weight_g" in self.meta_columns,
            "uses_batch": "batch" in self.meta_columns,
            "uses_sid_num": "sid_num" in self.meta_columns,
        }


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    """Return wl_* columns sorted by numeric wavelength suffix."""
    return sorted(
        [col for col in df.columns if col.startswith("wl_") and col[3:].isdigit()],
        key=lambda col: int(col.split("_", maxsplit=1)[1]),
    )


def add_report_metadata(spectra_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add the exact metadata features used in the report search scripts."""
    if len(spectra_df) != len(metadata_df):
        raise ValueError(f"spectra and metadata row counts differ: {len(spectra_df)} != {len(metadata_df)}")
    required = {"sample_id", "pixel_count", "valid_pixel_count", "weight_g"}
    missing = sorted(required - set(spectra_df.columns))
    if missing:
        raise ValueError(f"spectra missing required columns: {missing}")
    if "sample_dir" not in metadata_df.columns:
        raise ValueError("metadata missing required column: sample_dir")

    out = spectra_df.copy()
    out["batch"] = metadata_df["sample_dir"].astype(str).apply(lambda item: Path(item).parent.name).to_numpy()
    sid = out["sample_id"].astype(str).str.extract(r"(\d+)", expand=False)
    if sid.isna().any():
        bad_ids = out.loc[sid.isna(), "sample_id"].astype(str).tolist()
        raise ValueError(f"sample_id values do not contain numeric id: {bad_ids}")
    out["sid_num"] = sid.astype(int).to_numpy()
    out["log_valid_pixel_count"] = np.log1p(out["valid_pixel_count"].to_numpy(dtype=float))
    pixel_count = out["pixel_count"].to_numpy(dtype=float)
    valid_pixel_count = out["valid_pixel_count"].to_numpy(dtype=float)
    out["valid_pixel_ratio"] = np.divide(valid_pixel_count, pixel_count, out=np.zeros_like(valid_pixel_count), where=pixel_count != 0)
    return out


def append_report_metadata_features(
    spectral_features: np.ndarray,
    df: pd.DataFrame,
    *,
    meta_columns: Sequence[str] = META_COLUMNS,
    batch_categories: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Append report metadata features to PLS scores with deterministic ordering."""
    feature_df = pd.DataFrame(np.asarray(spectral_features, dtype=float), index=df.index)
    feature_df.columns = [f"pls_{idx + 1}" for idx in range(feature_df.shape[1])]

    for column in meta_columns:
        if column == "batch":
            categories = list(batch_categories) if batch_categories is not None else sorted(df[column].astype(str).unique())
            for category in categories:
                feature_df[f"batch={category}"] = (df[column].astype(str) == category).astype(float).to_numpy()
        else:
            if column not in df.columns:
                raise ValueError(f"metadata feature column missing after enrichment: {column}")
            feature_df[column] = df[column].to_numpy(dtype=float)
    return feature_df.to_numpy(dtype=float), list(feature_df.columns.astype(str))


def fit_report_20260505_model(
    train_spectra: pd.DataFrame,
    val_spectra: pd.DataFrame,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    *,
    config: Report20260505Config | None = None,
) -> ReportModelBundle:
    """Fit the exact report model on the formal train split."""
    cfg = config or Report20260505Config()
    train = add_report_metadata(train_spectra, train_metadata)
    val = add_report_metadata(val_spectra, val_metadata)
    wl_cols = wavelength_columns(train)
    if not wl_cols:
        raise ValueError("no wl_* wavelength columns found")
    missing_val_wl = sorted(set(wl_cols) - set(val.columns))
    if missing_val_wl:
        raise ValueError(f"validation data missing wavelength columns: {missing_val_wl[:5]}")
    y_train = train[cfg.target].to_numpy(dtype=float)
    X_train = train[wl_cols].to_numpy(dtype=float)
    X_train_proc = apply_preprocessing_pipeline(X_train, cfg.preprocess_steps())
    pls_model = PLSRegression(n_components=cfg.n_components, scale=cfg.pls_scale)
    pls_model.fit(X_train_proc, y_train)
    train_pls = pls_model.transform(X_train_proc)
    # Keep the feature schema identical to the archived report predictions: the
    # batch one-hot space is fixed from the formal train+val split categories.
    # This model is for report reproduction, not a strict train-only deployment
    # estimator.
    batch_categories = sorted(set(train["batch"].astype(str)) | set(val["batch"].astype(str)))
    train_features, feature_names = append_report_metadata_features(
        train_pls,
        train,
        meta_columns=META_COLUMNS,
        batch_categories=batch_categories,
    )
    final_model = make_pipeline(
        StandardScaler(),
        HuberRegressor(alpha=cfg.huber_alpha, epsilon=cfg.huber_epsilon, max_iter=cfg.huber_max_iter),
    )
    final_model.fit(train_features, y_train)
    return ReportModelBundle(
        config=cfg,
        wavelength_columns=wl_cols,
        meta_columns=list(META_COLUMNS),
        batch_categories=batch_categories,
        feature_names=feature_names,
        pls_model=pls_model,
        final_model=final_model,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


__all__ = [
    "META_COLUMNS",
    "TARGET_COLUMN",
    "Report20260505Config",
    "ReportModelBundle",
    "add_report_metadata",
    "append_report_metadata_features",
    "fit_report_20260505_model",
    "regression_metrics",
    "wavelength_columns",
]
