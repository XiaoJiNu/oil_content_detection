#!/usr/bin/env python3
"""Unified optimizer for multi-batch Huajiao oil-content experiments."""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.feature_selection import (  # noqa: E402
    CARSConfig,
    CARSSelector,
    GAConfig,
    GeneticAlgorithmSelector,
)
from oil_content_detection.preprocessing import (  # noqa: E402
    PreprocessStep,
    apply_preprocessing_pipeline,
)
from oil_content_detection.utils import get_logger, rmse, save_model, setup_single_thread  # noqa: E402
from oil_content_detection.utils.io import _convert_to_serializable, save_results_json  # noqa: E402

setup_single_thread()
logger = get_logger(__name__)


DEFAULT_QUALITY_FEATURES = ("coverage_ratio", "valid_pixel_ratio", "log_valid_pixel_count")


class PLSFeatureTransformer(BaseEstimator, TransformerMixin):
    """Use PLSRegression as a feature extractor inside sklearn Pipeline."""

    def __init__(self, n_components: int = 8, scale: bool = False) -> None:
        self.n_components = n_components
        self.scale = scale
        self.model_: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatureTransformer":
        n_components = min(self.n_components, max(1, X.shape[0] - 1), X.shape[1])
        self.model_ = PLSRegression(n_components=n_components, scale=self.scale)
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("PLSFeatureTransformer must be fitted before transform")
        transformed = self.model_.transform(X)
        if isinstance(transformed, tuple):
            return np.asarray(transformed[0], dtype=float)
        return np.asarray(transformed, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Huajiao multi-batch oil-content models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/huajiao_refined"))
    parser.add_argument("--output-root", type=Path, default=Path("results/multisample_optimization"))
    parser.add_argument("--mode", choices=["single", "search"], default="search")
    parser.add_argument("--candidate-set", default="crossbatch_v1")
    parser.add_argument("--preprocess", default="snv+sg1")
    parser.add_argument("--selector", choices=["none", "ga", "cars"], default="none")
    parser.add_argument("--model", choices=["plsr", "svr", "pls_svr"], default="pls_svr")
    parser.add_argument("--aux-feature-set", choices=["none", "quality"], default="none")
    parser.add_argument("--target", default="oil_ml_per_100g")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--plsr-max-components", type=int, default=15)
    parser.add_argument("--svr-grid-level", choices=["compact", "wide"], default="compact")
    parser.add_argument("--ga-population", type=int, default=20)
    parser.add_argument("--ga-generations", type=int, default=20)
    parser.add_argument("--ga-min-features", type=int, default=5)
    parser.add_argument("--ga-max-features", type=int, default=30)
    parser.add_argument("--ga-target-features", type=int, default=18)
    parser.add_argument("--ga-feature-penalty", type=float, default=0.0015)
    parser.add_argument("--cars-n-mc", type=int, default=50)
    parser.add_argument("--cars-sample-ratio", type=float, default=0.8)
    parser.add_argument("--cars-min-features", type=int, default=2)
    parser.add_argument("--cars-pls-components", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_preprocess_spec(spec: str) -> list[PreprocessStep]:
    spec = spec.strip().lower()
    if not spec or spec == "raw":
        return []
    steps: list[PreprocessStep] = []
    for token in spec.split("+"):
        token = token.strip()
        if token in {"", "raw"}:
            continue
        if token == "snv":
            steps.append(PreprocessStep("snv"))
        elif token == "msc":
            steps.append(PreprocessStep("msc"))
        elif token == "normalize":
            steps.append(PreprocessStep("normalize"))
        elif token == "detrend":
            steps.append(PreprocessStep("detrend"))
        elif token == "sg":
            steps.append(PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 0}))
        elif token == "sg1":
            steps.append(PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1}))
        else:
            raise ValueError(f"Unknown preprocess token: {token}")
    return steps


def make_candidate_name(candidate: Dict[str, Any]) -> str:
    parts = [
        candidate["preprocess"].replace("+", "_") if candidate["preprocess"] != "raw" else "raw",
        candidate["selector"],
        candidate["model"],
    ]
    if candidate.get("aux_feature_set", "none") != "none":
        parts.append(candidate["aux_feature_set"])
    text = "_".join(parts)
    return re.sub(r"[^0-9A-Za-z_]+", "_", text)


def candidate_set_crossbatch_v1() -> list[dict[str, Any]]:
    return [
        {"preprocess": "raw", "selector": "ga", "model": "plsr", "aux_feature_set": "none"},
        {"preprocess": "snv+sg1", "selector": "none", "model": "svr", "aux_feature_set": "none"},
        {"preprocess": "snv+sg1", "selector": "none", "model": "pls_svr", "aux_feature_set": "none"},
        {"preprocess": "raw+sg", "selector": "none", "model": "pls_svr", "aux_feature_set": "none"},
        {"preprocess": "snv+sg1", "selector": "cars", "model": "plsr", "aux_feature_set": "none"},
        {"preprocess": "msc+sg1", "selector": "cars", "model": "plsr", "aux_feature_set": "none"},
        {"preprocess": "snv+sg1", "selector": "none", "model": "svr", "aux_feature_set": "quality"},
        {"preprocess": "snv+sg1", "selector": "none", "model": "pls_svr", "aux_feature_set": "quality"},
        {"preprocess": "raw+sg", "selector": "none", "model": "pls_svr", "aux_feature_set": "quality"},
        {"preprocess": "msc+sg1", "selector": "none", "model": "pls_svr", "aux_feature_set": "quality"},
    ]


def get_candidate_set(name: str) -> list[dict[str, Any]]:
    if name == "crossbatch_v1":
        return candidate_set_crossbatch_v1()
    raise ValueError(f"Unknown candidate set: {name}")


def numeric_wavelength_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("wl_") and c[3:].isdigit()]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def load_split(split_dir: Path, target: str) -> dict[str, Any]:
    spectra_path = split_dir / "huajiao_spectra.parquet"
    metadata_path = split_dir / "huajiao_metadata.parquet"
    spectra = pd.read_parquet(spectra_path)
    metadata = pd.read_parquet(metadata_path)

    feature_cols = numeric_wavelength_columns(spectra)
    wavelengths = [int(col.split("_")[1]) for col in feature_cols]
    base_cols = [c for c in ["sample_id", "sample_id_raw", "weight_g", "distill_ml", "pixel_count", "valid_pixel_count", "coverage_ratio", target] if c in spectra.columns]
    data = spectra[base_cols + feature_cols].copy()

    meta_cols = [c for c in ["sample_id", "sample_dir", "hdr_path", "roi_visualization"] if c in metadata.columns]
    if "sample_id" in meta_cols:
        data = data.merge(metadata[meta_cols], on="sample_id", how="left")

    if "sample_dir" in data.columns:
        data["batch"] = data["sample_dir"].astype(str).apply(lambda s: Path(s).parent.name if s and s != "nan" else "unknown")
    else:
        data["batch"] = "unknown"

    if "pixel_count" in data.columns and "valid_pixel_count" in data.columns:
        valid = np.maximum(data["pixel_count"].to_numpy(dtype=float), 1.0)
        data["valid_pixel_ratio"] = data["valid_pixel_count"].to_numpy(dtype=float) / valid
        data["removed_pixel_ratio"] = 1.0 - data["valid_pixel_ratio"]
        data["log_valid_pixel_count"] = np.log1p(data["valid_pixel_count"].to_numpy(dtype=float))
        data["log_pixel_count"] = np.log1p(data["pixel_count"].to_numpy(dtype=float))

    return {
        "df": data,
        "X_spectral": data[feature_cols].to_numpy(dtype=float),
        "y": data[target].to_numpy(dtype=float),
        "feature_cols": feature_cols,
        "wavelengths": wavelengths,
        "groups": data["batch"].astype(str).tolist(),
    }


def apply_preprocess_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
    steps: Sequence[PreprocessStep],
) -> tuple[np.ndarray, np.ndarray]:
    if not steps:
        return X_train.copy(), X_val.copy()

    # MSC should reuse the training reference; other transforms are stateless sample-wise.
    train = np.asarray(X_train, dtype=float)
    val = np.asarray(X_val, dtype=float)
    for step in steps:
        name = step.name.lower()
        params = step.params or {}
        if name == "msc":
            ref = np.nanmean(train, axis=0, keepdims=True)

            def transform_msc(X: np.ndarray) -> np.ndarray:
                corrected = np.empty_like(X, dtype=float)
                for i, spectrum in enumerate(X):
                    A = np.hstack([ref.T, np.ones((ref.shape[1], 1))])
                    coeffs, *_ = np.linalg.lstsq(A, spectrum, rcond=None)
                    slope = float(coeffs[0])
                    intercept = float(coeffs[1])
                    corrected[i] = (spectrum - intercept) / max(abs(slope), 1e-8)
                return corrected

            train = transform_msc(train)
            val = transform_msc(val)
        else:
            train = apply_preprocessing_pipeline(train, [step])
            val = apply_preprocessing_pipeline(val, [step])
    return train, val


def get_aux_columns(feature_set: str) -> list[str]:
    if feature_set == "none":
        return []
    if feature_set == "quality":
        return list(DEFAULT_QUALITY_FEATURES)
    raise ValueError(f"Unknown feature set: {feature_set}")


def prepare_design_matrices(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    preprocess_steps: Sequence[PreprocessStep],
    aux_feature_set: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_train_spec, X_val_spec = apply_preprocess_train_val(
        train_data["X_spectral"],
        val_data["X_spectral"],
        preprocess_steps,
    )
    aux_cols = get_aux_columns(aux_feature_set)
    if not aux_cols:
        return X_train_spec, X_val_spec, aux_cols
    X_train_aux = train_data["df"][aux_cols].to_numpy(dtype=float)
    X_val_aux = val_data["df"][aux_cols].to_numpy(dtype=float)
    return (
        np.hstack([X_train_spec, X_train_aux]),
        np.hstack([X_val_spec, X_val_aux]),
        aux_cols,
    )


def make_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Sequence[str]],
    cv_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        splitter = GroupKFold(cv_splits)
        return list(splitter.split(X, y, np.asarray(groups)))
    splitter = KFold(cv_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def tune_plsr_components(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Sequence[str]],
    max_components: int,
    cv_splits: int,
    seed: int,
) -> tuple[int, float]:
    max_allowed = min(max_components, max(1, X.shape[0] - 1), X.shape[1])
    cv = make_cv_splits(X, y, groups, cv_splits, seed)
    best_n = 1
    best_score = float("-inf")
    for n_components in range(1, max_allowed + 1):
        model = PLSRegression(n_components=n_components, scale=False)
        scores = cross_val_predict(model, X, y, cv=cv)
        current_rmse = -rmse(y, scores)
        if current_rmse > best_score:
            best_score = current_rmse
            best_n = n_components
    return best_n, -best_score


def build_selector(
    selector_name: str,
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    if selector_name == "none":
        return None
    if selector_name == "ga":
        return {
            "selector": "ga",
            "config": GAConfig(
                population_size=args.ga_population,
                generations=args.ga_generations,
                min_features=args.ga_min_features,
                max_features=args.ga_max_features,
                target_features=args.ga_target_features,
                random_state=args.seed,
                cv_splits=args.cv_splits,
                patience=6,
                verbose=args.verbose,
                score_metric="neg_rmse",
                feature_penalty=args.ga_feature_penalty,
                max_pls_components=min(args.plsr_max_components, 10),
            ),
        }
    if selector_name == "cars":
        return {
            "selector": "cars",
            "config": CARSConfig(
                n_mc=args.cars_n_mc,
                sample_ratio=args.cars_sample_ratio,
                cv_splits=args.cv_splits,
                pls_n_components=args.cars_pls_components,
                min_features=args.cars_min_features,
                random_state=args.seed,
                verbose=args.verbose,
                store_coef_history=True,
            ),
        }
    raise ValueError(f"Unknown selector: {selector_name}")


def apply_selector(
    selector_spec: Optional[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    groups: Optional[Sequence[str]],
) -> tuple[np.ndarray, list[str], dict[str, Any], Optional[list[dict[str, Any]]]]:
    if selector_spec is None:
        support = np.ones(X_train.shape[1], dtype=bool)
        return support, list(feature_names), {"selector": "none", "selected_count": int(support.sum())}, None

    name = selector_spec["selector"]
    if name == "ga":
        selector = GeneticAlgorithmSelector(selector_spec["config"])
        selector.fit(X_train, y_train, groups=groups)
        support = selector.get_support()
        details = {
            "selector": "ga",
            "ga_config": selector_spec["config"].__dict__,
            "ga_best_score": selector.best_score(),
            "selected_count": int(support.sum()),
        }
        history = selector.history()
    elif name == "cars":
        selector = CARSSelector(selector_spec["config"])
        selector.fit(X_train, y_train, groups=groups)
        support = selector.get_support()
        details = {
            "selector": "cars",
            "cars_config": selector_spec["config"].__dict__,
            "cars_best_rmsecv": selector.best_rmsecv(),
            "cars_best_r2cv": selector.best_r2cv(),
            "cars_best_iteration": selector.best_iteration(),
            "selected_count": int(support.sum()),
        }
        history = selector.history()
    else:
        raise ValueError(f"Unknown selector spec: {name}")

    selected_names = [feature_names[i] for i in np.where(support)[0]]
    return support, selected_names, details, history


def svr_param_grid(level: str) -> dict[str, list[Any]]:
    if level == "wide":
        return {
            "svr__C": [0.1, 0.3, 3.0, 10.0, 30.0, 100.0],
            "svr__epsilon": [0.05, 0.1, 0.2, 0.3],
            "svr__gamma": ["scale", 1e-6, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        }
    return {
        "svr__C": [1.0, 3.0, 10.0, 30.0],
        "svr__epsilon": [0.05, 0.1, 0.2],
        "svr__gamma": ["scale", 1e-3, 1e-2],
    }


def fit_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Optional[Sequence[str]],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    cv = make_cv_splits(X_train, y_train, groups, args.cv_splits, args.seed)
    if model_name == "plsr":
        n_components, cv_rmse = tune_plsr_components(
            X_train,
            y_train,
            groups=groups,
            max_components=args.plsr_max_components,
            cv_splits=args.cv_splits,
            seed=args.seed,
        )
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_train, y_train)
        return model, {"model": "plsr", "plsr_n_components": n_components, "plsr_cv_rmse": cv_rmse}

    if model_name == "svr":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=svr_param_grid(args.svr_grid_level),
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=None,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, {
            "model": "svr",
            "svr_search": "grid",
            "svr_best_params": search.best_params_,
            "svr_best_score_neg_rmse": float(search.best_score_),
        }

    if model_name == "pls_svr":
        max_pls = min(args.plsr_max_components, max(1, X_train.shape[0] - 1), X_train.shape[1])
        pls_components = sorted({n for n in [4, 6, 8, 10, 12, 15, 20] if 1 <= n <= max_pls})
        pipeline = Pipeline(
            [
                ("pls", PLSFeatureTransformer(scale=False)),
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        param_grid = svr_param_grid(args.svr_grid_level)
        param_grid["pls__n_components"] = pls_components
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=None,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, {
            "model": "pls_svr",
            "pls_svr_search": "grid",
            "pls_svr_best_params": search.best_params_,
            "pls_svr_best_score_neg_rmse": float(search.best_score_),
        }

    raise ValueError(f"Unknown model: {model_name}")


def prediction_frame(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    split_name: Optional[str] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].values,
            "sample_id_raw": df["sample_id_raw"].values if "sample_id_raw" in df.columns else df["sample_id"].values,
            f"{target}_true": y_true,
            f"{target}_pred": y_pred,
        }
    )
    out["error"] = out[f"{target}_pred"] - out[f"{target}_true"]
    out["abs_error"] = out["error"].abs()
    safe_true = np.where(np.abs(out[f"{target}_true"].to_numpy(dtype=float)) < 1e-8, np.nan, out[f"{target}_true"].to_numpy(dtype=float))
    out["relative_error_percent"] = out["error"].to_numpy(dtype=float) / safe_true * 100.0
    out["abs_relative_error_percent"] = np.abs(out["relative_error_percent"])
    for col in ["weight_g", "distill_ml", "pixel_count", "valid_pixel_count", "coverage_ratio", "sample_dir", "batch"]:
        if col in df.columns:
            out[col] = df[col].values
    if split_name is not None:
        out["split"] = split_name
    return out


def evaluate_model(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_groups: Optional[Sequence[str]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    y_train_pred = np.asarray(estimator.predict(X_train)).ravel()
    y_val_pred = np.asarray(estimator.predict(X_val)).ravel()
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    cv = make_cv_splits(X_train, y_train, train_groups, args.cv_splits, args.seed)
    y_oof = np.asarray(cross_val_predict(clone(estimator), X_train, y_train, cv=cv, n_jobs=None)).ravel()
    cv_metrics = compute_metrics(y_train, y_oof)
    cv_metrics["n_splits"] = args.cv_splits
    if train_groups is not None:
        cv_metrics["n_groups"] = int(len(set(train_groups)))
    return {"train": train_metrics, "val": val_metrics, "train_groupkfold_cv": cv_metrics}, y_train_pred, y_val_pred


def save_history(history: Optional[list[dict[str, Any]]], output_dir: Path, selector_name: str) -> None:
    if not history:
        return
    filename = "ga_history.json" if selector_name == "ga" else "cars_history.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(_convert_to_serializable(history), f, indent=2)


def run_candidate(candidate: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    logger.info("Running candidate: %s", candidate)
    train_data = load_split(args.data_dir / "train", args.target)
    val_data = load_split(args.data_dir / "val", args.target)
    preprocess_steps = parse_preprocess_spec(candidate["preprocess"])
    X_train_full, X_val_full, aux_cols = prepare_design_matrices(
        train_data,
        val_data,
        preprocess_steps=preprocess_steps,
        aux_feature_set=candidate.get("aux_feature_set", "none"),
    )

    spectral_feature_names = train_data["feature_cols"]
    feature_names = spectral_feature_names + aux_cols

    selector_spec = build_selector(candidate["selector"], args)
    support, selected_names, selector_details, selector_history = apply_selector(
        selector_spec,
        X_train_full,
        train_data["y"],
        feature_names,
        groups=train_data["groups"],
    )
    X_train = X_train_full[:, support]
    X_val = X_val_full[:, support]
    estimator, model_details = fit_model(
        candidate["model"],
        X_train,
        train_data["y"],
        groups=train_data["groups"],
        args=args,
    )
    metrics_bundle, y_train_pred, y_val_pred = evaluate_model(
        estimator,
        X_train,
        train_data["y"],
        X_val,
        val_data["y"],
        train_groups=train_data["groups"],
        args=args,
    )

    train_pred_df = prediction_frame(train_data["df"], train_data["y"], y_train_pred, args.target)
    val_pred_df = prediction_frame(val_data["df"], val_data["y"], y_val_pred, args.target)
    all_pred_df = pd.concat(
        [
            prediction_frame(train_data["df"], train_data["y"], y_train_pred, args.target, split_name="train"),
            prediction_frame(val_data["df"], val_data["y"], y_val_pred, args.target, split_name="val"),
        ],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_pred_df.to_csv(output_dir / "train_predictions.csv", index=False)
    val_pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    all_pred_df.to_csv(output_dir / "predictions_all.csv", index=False)
    train_pred_df.sort_values("abs_error", ascending=False).to_csv(output_dir / "train_predictions_sorted_by_abs_error.csv", index=False)
    val_pred_df.sort_values("abs_error", ascending=False).to_csv(output_dir / "val_predictions_sorted_by_abs_error.csv", index=False)
    save_history(selector_history, output_dir, selector_details["selector"])
    save_model(estimator, output_dir / ("svr_model.pkl" if "svr" in candidate["model"] else "plsr_model.pkl"))

    selected_wavelengths = [int(name.split("_")[1]) for name in selected_names if name.startswith("wl_")]
    result_payload = {
        "target": args.target,
        "wl_count": len(train_data["feature_cols"]),
        "preprocess": [{"name": step.name, "params": step.params} for step in preprocess_steps],
        "selected_wavelength_count": len(selected_wavelengths),
        "selected_wavelengths": selected_wavelengths,
        "selected_aux_features": [name for name in selected_names if not name.startswith("wl_")],
        "selector": selector_details,
        "model": model_details,
        "metrics": metrics_bundle,
        "train_spectra": str(args.data_dir / "train" / "huajiao_spectra.parquet"),
        "val_spectra": str(args.data_dir / "val" / "huajiao_spectra.parquet"),
        "train_metadata": str(args.data_dir / "train" / "huajiao_metadata.parquet"),
        "val_metadata": str(args.data_dir / "val" / "huajiao_metadata.parquet"),
        "candidate": candidate,
    }
    save_results_json(result_payload, output_dir / "results.json", include_timestamp=False)
    with open(output_dir / "selected_wavelengths.json", "w") as f:
        json.dump(
            {
                "count": len(selected_wavelengths),
                "selected_wavelengths": selected_wavelengths,
                "selected_aux_features": [name for name in selected_names if not name.startswith("wl_")],
            },
            f,
            indent=2,
        )
    return result_payload


def summarize_results(results: list[dict[str, Any]], output_root: Path) -> None:
    rows = []
    for result in results:
        row = {
            "candidate_name": make_candidate_name(result["candidate"]),
            "preprocess": result["candidate"]["preprocess"],
            "selector": result["candidate"]["selector"],
            "model": result["candidate"]["model"],
            "aux_feature_set": result["candidate"].get("aux_feature_set", "none"),
            "train_r2": result["metrics"]["train"]["r2"],
            "val_r2": result["metrics"]["val"]["r2"],
            "val_rmse": result["metrics"]["val"]["rmse"],
            "val_mae": result["metrics"]["val"]["mae"],
            "groupcv_r2": result["metrics"]["train_groupkfold_cv"]["r2"],
            "groupcv_rmse": result["metrics"]["train_groupkfold_cv"]["rmse"],
            "selected_wavelength_count": result["selected_wavelength_count"],
            "selected_aux_feature_count": len(result.get("selected_aux_features", [])),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["groupcv_r2", "val_r2"], ascending=[False, False])
    summary.to_csv(output_root / "candidate_summary.csv", index=False)
    with open(output_root / "candidate_summary.json", "w") as f:
        json.dump(_convert_to_serializable(rows), f, indent=2)


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        candidate = {
            "preprocess": args.preprocess,
            "selector": args.selector,
            "model": args.model,
            "aux_feature_set": args.aux_feature_set,
        }
        output_dir = args.output_root / make_candidate_name(candidate)
        run_candidate(candidate, args, output_dir)
        return

    candidates = get_candidate_set(args.candidate_set)
    results = []
    for candidate in candidates:
        candidate_args = deepcopy(args)
        output_dir = args.output_root / make_candidate_name(candidate)
        result = run_candidate(candidate, candidate_args, output_dir)
        results.append(result)
    summarize_results(results, args.output_root)


if __name__ == "__main__":
    main()
