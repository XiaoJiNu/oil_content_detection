#!/usr/bin/env python3
"""Optimize SVR-based models for cross-batch generalization with nested GroupKFold.

This script searches preprocessing + model hyperparameters using batch-level
GroupKFold splits on the *training set* to avoid optimistic leakage when the
deployment target is a new batch.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Ensure local src is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.preprocessing import (  # noqa: E402
    PLSScoreTransformer,
    PreprocessStep,
    SpectralPreprocessor,
)
from oil_content_detection.utils import (  # noqa: E402
    get_logger,
    save_model,
    save_results_json,
    save_wavelengths,
    setup_single_thread,
)

setup_single_thread()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested GroupKFold optimization for huajiao oil content (SVR-based)"
    )
    parser.add_argument(
        "--train-spectra",
        type=Path,
        default=Path("data/processed/huajiao_refined/train/huajiao_spectra.parquet"),
    )
    parser.add_argument(
        "--val-spectra",
        type=Path,
        default=Path("data/processed/huajiao_refined/val/huajiao_spectra.parquet"),
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("data/processed/huajiao_refined/train/huajiao_metadata.parquet"),
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=Path("data/processed/huajiao_refined/val/huajiao_metadata.parquet"),
    )
    parser.add_argument("--target", type=str, default="oil_ml_per_100g")

    parser.add_argument("--sg-window", type=int, default=13)
    parser.add_argument("--sg-polyorder", type=int, default=2)

    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/huajiao_refined_oil_content_20260105_groupcv_opt"),
    )
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def _load_df(path: Path) -> pd.DataFrame:
    if path.exists():
        return (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_csv(path)
        )
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"File not found: {path} (or {csv_path})")


def _wl_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("wl_")]
    if not cols:
        raise ValueError("No wl_ columns found in spectra table")
    return cols


def _wavelengths_from_cols(cols: Sequence[str]) -> List[int]:
    out: List[int] = []
    for c in cols:
        try:
            out.append(int(c.split("_", 1)[1]))
        except Exception:
            out.append(-1)
    return out


def _batch_from_sample_dir(sample_dir: str) -> str:
    try:
        return Path(sample_dir).parent.name
    except Exception:
        return "UNKNOWN"


def _attach_batch(df: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if (
        meta is None
        or "sample_id" not in meta.columns
        or "sample_dir" not in meta.columns
    ):
        return df
    sub = meta[["sample_id", "sample_dir"]].copy()
    sub["sample_id"] = sub["sample_id"].astype(str)
    sub["sample_dir"] = sub["sample_dir"].astype(str)
    sub["batch"] = sub["sample_dir"].map(_batch_from_sample_dir)
    return df.merge(sub, on="sample_id", how="left")


def _build_prediction_table(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target: str,
) -> pd.DataFrame:
    true_col = f"{target}_true"
    pred_col = f"{target}_pred"
    out = pd.DataFrame(
        {
            "sample_id": (
                df["sample_id"].astype(str).to_numpy()
                if "sample_id" in df.columns
                else np.arange(len(y_true))
            ),
            "sample_id_raw": (
                df["sample_id_raw"].astype(str).to_numpy()
                if "sample_id_raw" in df.columns
                else ""
            ),
            true_col: y_true,
            pred_col: y_pred,
        }
    )
    out["error"] = out[pred_col] - out[true_col]
    out["abs_error"] = out["error"].abs()
    denom = out[true_col].abs()
    out["relative_error_percent"] = np.where(
        denom > 0, out["error"] / denom * 100.0, np.nan
    )
    out["abs_relative_error_percent"] = np.where(
        denom > 0, out["abs_error"] / denom * 100.0, np.nan
    )

    for col in [
        "weight_g",
        "distill_ml",
        "pixel_count",
        "valid_pixel_count",
        "coverage_ratio",
    ]:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].to_numpy()
    return out


def _candidate_preprocess_configs(
    args: argparse.Namespace,
) -> List[Tuple[str, List[PreprocessStep]]]:
    sg = PreprocessStep(
        name="sg",
        params={
            "window_length": int(args.sg_window),
            "polyorder": int(args.sg_polyorder),
            "deriv": 0,
        },
    )
    sg1 = PreprocessStep(
        name="sg",
        params={
            "window_length": int(args.sg_window),
            "polyorder": int(args.sg_polyorder),
            "deriv": 1,
        },
    )
    return [
        ("raw", []),
        ("raw+sg", [sg]),
        ("snv", [PreprocessStep(name="snv")]),
        ("msc", [PreprocessStep(name="msc")]),
        ("snv+sg1", [PreprocessStep(name="snv"), sg1]),
        ("msc+sg1", [PreprocessStep(name="msc"), sg1]),
    ]


def _svr_search_space() -> Dict[str, Any]:
    return {
        "svr__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        "svr__gamma": ["scale", 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "svr__epsilon": [0.05, 0.1, 0.2, 0.3],
    }


def _pls_svr_search_space() -> Dict[str, Any]:
    space = dict(_svr_search_space())
    space["pls__n_components"] = [5, 8, 10, 12, 15, 20]
    return space


def _nested_groupcv_eval(
    pipeline: Pipeline,
    param_distributions: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    outer_splits: int,
    inner_splits: int,
    n_iter: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray]:
    outer = GroupKFold(n_splits=int(outer_splits))
    oof_pred = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        g_tr = groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        unique_groups = np.unique(g_tr)
        inner_n = int(min(max(2, inner_splits), unique_groups.size))
        inner_cv = GroupKFold(n_splits=inner_n)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=int(n_iter),
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            random_state=int(seed),
            n_jobs=None,
            refit=True,
        )
        search.fit(X_tr, y_tr, groups=g_tr)

        best = search.best_estimator_
        pred = best.predict(X_te).ravel()
        oof_pred[test_idx] = pred

        fold_rows.append(
            {
                "fold": int(fold),
                "inner_splits": int(inner_n),
                "test_n": int(len(y_te)),
                "test_groups": int(np.unique(groups[test_idx]).size),
                "rmse": float(root_mean_squared_error(y_te, pred)),
                "mae": float(mean_absolute_error(y_te, pred)),
                "r2": float(r2_score(y_te, pred)),
                "best_params": search.best_params_,
                "best_score_neg_rmse": float(search.best_score_),
            }
        )

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF predictions contain NaN; check GroupKFold splitting")

    summary = {
        "r2": float(r2_score(y, oof_pred)),
        "rmse": float(root_mean_squared_error(y, oof_pred)),
        "mae": float(mean_absolute_error(y, oof_pred)),
    }
    return summary, fold_rows, oof_pred


def main() -> None:
    args = parse_args()

    train_df = _load_df(args.train_spectra)
    val_df = _load_df(args.val_spectra)
    train_meta = _load_df(args.train_metadata)
    val_meta = _load_df(args.val_metadata) if args.val_metadata else None

    wl_cols = _wl_cols(train_df)
    if args.target not in train_df.columns or args.target not in val_df.columns:
        raise KeyError(f"Target column missing: {args.target}")

    # Align batch groups to train_df order.
    train_groups_df = train_df[["sample_id"]].copy()
    train_groups_df["sample_id"] = train_groups_df["sample_id"].astype(str)
    meta_sub = train_meta[["sample_id", "sample_dir"]].copy()
    meta_sub["sample_id"] = meta_sub["sample_id"].astype(str)
    meta_sub["sample_dir"] = meta_sub["sample_dir"].astype(str)
    train_groups_df = train_groups_df.merge(meta_sub, on="sample_id", how="left")
    if train_groups_df["sample_dir"].isna().any():
        missing = train_groups_df.loc[
            train_groups_df["sample_dir"].isna(), "sample_id"
        ].tolist()[:10]
        raise ValueError(
            f"Missing sample_dir for some training samples, e.g. {missing}"
        )

    groups = train_groups_df["sample_dir"].map(_batch_from_sample_dir).to_numpy()

    X_train = train_df[wl_cols].to_numpy(dtype=float)
    y_train = train_df[args.target].to_numpy(dtype=float)
    X_val = val_df[wl_cols].to_numpy(dtype=float)
    y_val = val_df[args.target].to_numpy(dtype=float)

    wavelengths = _wavelengths_from_cols(wl_cols)

    candidates: List[Dict[str, Any]] = []
    preprocess_candidates = _candidate_preprocess_configs(args)

    for prep_name, steps in preprocess_candidates:
        # 1) Direct SVR
        direct = Pipeline(
            steps=[
                ("preprocess", SpectralPreprocessor(steps)),
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        logger.info("Evaluating candidate: %s | direct_svr", prep_name)
        summary, folds, _ = _nested_groupcv_eval(
            direct,
            _svr_search_space(),
            X_train,
            y_train,
            groups,
            outer_splits=int(args.outer_splits),
            inner_splits=int(args.inner_splits),
            n_iter=int(args.n_iter),
            seed=int(args.seed),
        )
        candidates.append(
            {
                "candidate": f"{prep_name} | direct_svr",
                "preprocess": [asdict(s) for s in steps],
                "model": "direct_svr",
                "groupcv_summary": summary,
                "groupcv_folds": folds,
            }
        )

        # 2) PLS -> SVR
        pls_svr = Pipeline(
            steps=[
                ("preprocess", SpectralPreprocessor(steps)),
                ("scaler", StandardScaler()),
                ("pls", PLSScoreTransformer(n_components=10, scale=False)),
                ("svr_scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        logger.info("Evaluating candidate: %s | pls_svr", prep_name)
        summary, folds, _ = _nested_groupcv_eval(
            pls_svr,
            _pls_svr_search_space(),
            X_train,
            y_train,
            groups,
            outer_splits=int(args.outer_splits),
            inner_splits=int(args.inner_splits),
            n_iter=int(args.n_iter),
            seed=int(args.seed),
        )
        candidates.append(
            {
                "candidate": f"{prep_name} | pls_svr",
                "preprocess": [asdict(s) for s in steps],
                "model": "pls_svr",
                "groupcv_summary": summary,
                "groupcv_folds": folds,
            }
        )

    # Pick best by groupcv RMSE (lower is better).
    best = min(candidates, key=lambda row: float(row["groupcv_summary"]["rmse"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save candidate summary table.
    summary_rows = []
    for row in candidates:
        summary_rows.append(
            {
                "candidate": row["candidate"],
                "groupcv_r2": float(row["groupcv_summary"]["r2"]),
                "groupcv_rmse": float(row["groupcv_summary"]["rmse"]),
                "groupcv_mae": float(row["groupcv_summary"]["mae"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("groupcv_rmse", ascending=True)
    summary_df.to_csv(output_dir / "groupcv_candidates.csv", index=False)

    # Refit best candidate on full train set with GroupKFold search, then evaluate fixed val.
    best_steps = [PreprocessStep(**d) for d in best["preprocess"]]
    if best["model"] == "pls_svr":
        best_pipeline = Pipeline(
            steps=[
                ("preprocess", SpectralPreprocessor(best_steps)),
                ("scaler", StandardScaler()),
                ("pls", PLSScoreTransformer(n_components=10, scale=False)),
                ("svr_scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        space = _pls_svr_search_space()
    else:
        best_pipeline = Pipeline(
            steps=[
                ("preprocess", SpectralPreprocessor(best_steps)),
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        space = _svr_search_space()

    unique_groups = np.unique(groups)
    inner_n = int(min(max(2, int(args.inner_splits)), unique_groups.size))
    inner_cv = GroupKFold(n_splits=inner_n)
    final_search = RandomizedSearchCV(
        best_pipeline,
        param_distributions=space,
        n_iter=int(args.n_iter),
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=int(args.seed),
        n_jobs=None,
        refit=True,
    )
    final_search.fit(X_train, y_train, groups=groups)
    final_model = final_search.best_estimator_

    train_pred = final_model.predict(X_train).ravel()
    val_pred = final_model.predict(X_val).ravel()

    metrics = {
        "train": {
            "n": int(len(y_train)),
            "r2": float(r2_score(y_train, train_pred)),
            "rmse": float(root_mean_squared_error(y_train, train_pred)),
            "mae": float(mean_absolute_error(y_train, train_pred)),
        },
        "val": {
            "n": int(len(y_val)),
            "r2": float(r2_score(y_val, val_pred)),
            "rmse": float(root_mean_squared_error(y_val, val_pred)),
            "mae": float(mean_absolute_error(y_val, val_pred)),
        },
    }

    # Save fixed-split predictions aligned with existing analysis workflow.
    train_pred_df = _build_prediction_table(
        train_df, y_train, train_pred, target=args.target
    )
    val_pred_df = _build_prediction_table(val_df, y_val, val_pred, target=args.target)
    train_pred_df = _attach_batch(train_pred_df, train_meta)
    val_pred_df = _attach_batch(val_pred_df, val_meta)

    train_pred_df.to_csv(output_dir / "train_predictions.csv", index=False)
    val_pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    pd.concat(
        [train_pred_df.assign(split="train"), val_pred_df.assign(split="val")],
        ignore_index=True,
    ).to_csv(output_dir / "predictions_all.csv", index=False)

    # Save wavelengths list (no feature selection here, keep full list).
    save_wavelengths(
        [int(x) for x in wavelengths], output_dir / "selected_wavelengths.json"
    )

    payload = {
        "target": str(args.target),
        "train_spectra": str(args.train_spectra),
        "val_spectra": str(args.val_spectra),
        "train_metadata": str(args.train_metadata),
        "val_metadata": str(args.val_metadata) if args.val_metadata else None,
        "wl_count": int(len(wl_cols)),
        "batch_group_count": int(np.unique(groups).size),
        "outer_splits": int(args.outer_splits),
        "inner_splits": int(args.inner_splits),
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "candidates": candidates,
        "best_candidate": best["candidate"],
        "best_candidate_groupcv": best["groupcv_summary"],
        "final_best_params": final_search.best_params_,
        "final_best_score_neg_rmse": float(final_search.best_score_),
        "metrics": metrics,
    }
    save_results_json(payload, output_dir / "results.json", include_timestamp=False)

    if not args.no_save_model:
        save_model(final_model, output_dir / "svr_groupcv_model.pkl")

    print("=== GroupCV Optimization Result ===")
    print(f"Output: {output_dir}")
    print(f"Best candidate: {best['candidate']}")
    print(
        "Best(GroupCV): "
        f"R²={best['groupcv_summary']['r2']:.4f}, "
        f"RMSE={best['groupcv_summary']['rmse']:.4f}, "
        f"MAE={best['groupcv_summary']['mae']:.4f}"
    )
    print(
        "Fixed Val: "
        f"R²={metrics['val']['r2']:.4f}, "
        f"RMSE={metrics['val']['rmse']:.4f}, "
        f"MAE={metrics['val']['mae']:.4f}"
    )


if __name__ == "__main__":
    main()
