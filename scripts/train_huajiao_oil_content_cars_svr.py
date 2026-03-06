#!/usr/bin/env python3
"""Train huajiao oil-content model using CARS/GA selection and SVR/PLSR.

This script extends the training pipeline to support:
1. Feature Selection: CARS (new), GA, or None
2. Model: SVR (RBF kernel), PLSR
3. Validation: GroupKFold (by batch) or KFold
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Ensure local src is importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.feature_selection import (
    CARSConfig, CARSSelector, 
    GAConfig, GeneticAlgorithmSelector
)
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline
from oil_content_detection.utils import get_logger, save_model, setup_single_thread

setup_single_thread()
logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Huajiao Oil Content (CARS + SVR/PLSR)")
    
    # Data paths
    parser.add_argument("--train-spectra", type=Path, default=Path("data/processed/huajiao_refined/train/huajiao_spectra.parquet"))
    parser.add_argument("--val-spectra", type=Path, default=Path("data/processed/huajiao_refined/val/huajiao_spectra.parquet"))
    parser.add_argument("--train-meta", type=Path, default=None, help="Path to metadata (for groups)")
    
    # Targets & Preprocessing
    parser.add_argument("--target", type=str, default="oil_ml_per_100g")
    parser.add_argument("--preprocess", nargs="*", default=["snv", "sg1"], 
                        help="Steps: raw, snv, msc, sg, sg1. Recommended: snv sg1")
    
    # Feature Selection
    parser.add_argument("--selector", type=str, choices=["cars", "ga", "none"], default="cars")
    parser.add_argument("--cars-mc", type=int, default=50, help="CARS Monte Carlo runs")
    parser.add_argument("--cars-pls-comp", type=int, default=8, help="CARS PLS components")
    parser.add_argument("--ga-gen", type=int, default=20, help="GA generations")
    parser.add_argument("--ga-pop", type=int, default=20, help="GA population")
    
    # Model
    parser.add_argument("--model", type=str, choices=["plsr", "svr"], default="svr")
    parser.add_argument("--groupkfold", action="store_true", help="Use GroupKFold (by batch) for CV")
    
    # Evaluation
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("results/huajiao_cars_svr_experiment"))
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()

def _load_data(spectra_path: Path, meta_path: Optional[Path] = None) -> pd.DataFrame:
    if not spectra_path.exists():
        raise FileNotFoundError(f"{spectra_path}")
    df = pd.read_parquet(spectra_path)
    
    # Try to load metadata if needed for groups
    if meta_path is None:
        # Assumption: metadata is in same folder named huajiao_metadata.parquet
        possible_meta = spectra_path.parent / "huajiao_metadata.parquet"
        if possible_meta.exists():
            meta_path = possible_meta
    
    if meta_path and meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "sample_id" in df.columns and "sample_id" in meta.columns:
            # Drop overlapping columns from meta except sample_id
            cols_to_use = ["sample_id"] + [c for c in meta.columns if c not in df.columns]
            df = pd.merge(df, meta[cols_to_use], on="sample_id", how="left")
            
    return df

def _get_groups(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "sample_dir" in df.columns:
        # parent of sample dir is batch
        return df["sample_dir"].apply(lambda x: Path(x).parent.name).to_numpy()
    return None

def _parse_preprocess_steps(names: List[str]) -> List[PreprocessStep]:
    steps = []
    for name in names:
        if name == "raw": continue
        if name == "snv": steps.append(PreprocessStep("snv"))
        elif name == "msc": steps.append(PreprocessStep("msc"))
        elif name == "sg": steps.append(PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 0}))
        elif name == "sg1": steps.append(PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1}))
    return steps

def optimize_svr(X, y, groups, cv_splits, seed, verbose=False):
    # SVR usually requires scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])
    
    # Parameter grid
    param_dist = {
        'svr__C': np.logspace(-1, 4, 20),
        'svr__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 10)),
        'svr__epsilon': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    }
    
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=50, 
        scoring='neg_root_mean_squared_error',
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, -search.best_score_

def optimize_plsr(X, y, groups, cv_splits, seed, max_comp=15, verbose=False):
    best_rmse = float("inf")
    best_n = 1
    
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    
    limit = min(max_comp, X.shape[1], len(y) - 1)
    if limit < 1: limit = 1
    
    for n in range(1, limit + 1):
        model = PLSRegression(n_components=n, scale=False)
        scores = []
        # split returns train_idx, test_idx
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx]).ravel()
            scores.append(root_mean_squared_error(y[test_idx], pred))
        
        mean_rmse = np.mean(scores)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_n = n
            
    final_model = PLSRegression(n_components=best_n, scale=False)
    final_model.fit(X, y)
    return final_model, {"n_components": best_n}, best_rmse

def _convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    return obj

def main():
    args = parse_args()
    
    # 1. Load Data
    train_df = _load_data(args.train_spectra, args.train_meta)
    val_df = _load_data(args.val_spectra)
    
    wl_cols = [c for c in train_df.columns if c.startswith("wl_")]
    X_train = train_df[wl_cols].values
    y_train = train_df[args.target].values
    X_val = val_df[wl_cols].values
    y_val = val_df[args.target].values
    
    # Groups
    groups = _get_groups(train_df) if args.groupkfold else None
    if args.groupkfold and groups is None:
        logger.warning("GroupKFold requested but no group info found. Falling back to KFold.")
    
    # 2. Preprocessing
    steps = _parse_preprocess_steps(args.preprocess)
    X_train = apply_preprocessing_pipeline(X_train, steps)
    X_val = apply_preprocessing_pipeline(X_val, steps)
    
    # Remove low variance features to prevent scaling issues
    std_train = np.std(X_train, axis=0)
    # Use a safe threshold
    valid_feats = std_train > 1e-8
    if (~valid_feats).any():
        dropped = (~valid_feats).sum()
        logger.warning(f"Dropping {dropped} low-variance features (std < 1e-8).")
        X_train = X_train[:, valid_feats]
        X_val = X_val[:, valid_feats]
        wl_cols = [c for i, c in enumerate(wl_cols) if valid_feats[i]]
    
    # 3. Feature Selection
    support = np.ones(X_train.shape[1], dtype=bool)
    sel_history = None
    
    if args.selector == "cars":
        logger.info("Running CARS selection...")
        cfg = CARSConfig(n_mc=args.cars_mc, pls_n_components=args.cars_pls_comp, 
                         n_splits=args.cv_splits, random_state=args.seed, verbose=args.verbose)
        selector = CARSSelector(cfg)
        selector.fit(X_train, y_train)
        support = selector.get_support()
        sel_history = selector._history
        logger.info(f"CARS selected {support.sum()} features.")
        
    elif args.selector == "ga":
        logger.info("Running GA selection...")
        cfg = GAConfig(
            generations=args.ga_gen,
            population_size=args.ga_pop,
            random_state=args.seed, 
            cv_splits=args.cv_splits, 
            verbose=args.verbose
        )
        selector = GeneticAlgorithmSelector(cfg)
        selector.fit(X_train, y_train)
        support = selector.get_support()
        sel_history = selector.history()
        logger.info(f"GA selected {support.sum()} features.")
        
    X_train_sel = X_train[:, support]
    X_val_sel = X_val[:, support]
    
    # 4. Model Optimization & Training
    logger.info(f"Training {args.model.upper()} model...")
    
    if args.model == "svr":
        model, params, best_cv_score = optimize_svr(X_train_sel, y_train, groups, args.cv_splits, args.seed, args.verbose)
    else:
        model, params, best_cv_score = optimize_plsr(X_train_sel, y_train, groups, args.cv_splits, args.seed, verbose=args.verbose)
        
    logger.info(f"Best CV RMSE: {best_cv_score:.4f} with params {params}")
    
    # 5. Evaluation
    train_pred = model.predict(X_train_sel).ravel()
    val_pred = model.predict(X_val_sel).ravel()
    
    metrics = {
        "train": {
            "r2": r2_score(y_train, train_pred),
            "rmse": root_mean_squared_error(y_train, train_pred),
            "mae": mean_absolute_error(y_train, train_pred)
        },
        "val": {
            "r2": r2_score(y_val, val_pred),
            "rmse": root_mean_squared_error(y_val, val_pred),
            "mae": mean_absolute_error(y_val, val_pred)
        }
    }
    
    # 6. Save Results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_preds(df, y_t, y_p, name):
        out = df[["sample_id"]].copy() if "sample_id" in df else pd.DataFrame()
        out["true"] = y_t
        out["pred"] = y_p
        out["error"] = y_p - y_t
        out["abs_error"] = np.abs(out["error"])
        out.to_csv(args.output_dir / f"{name}_predictions.csv", index=False)
        return out
        
    tr_out = save_preds(train_df, y_train, train_pred, "train")
    va_out = save_preds(val_df, y_val, val_pred, "val")
    pd.concat([tr_out.assign(split="train"), va_out.assign(split="val")]).to_csv(args.output_dir / "predictions_all.csv", index=False)
    
    save_model(model, args.output_dir / f"{args.model}_model.pkl")
    
    info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "params": params,
        "metrics": metrics,
        "selected_features": [int(x) for x in np.where(support)[0]],
        "wavelengths": [int(wl_cols[i].split('_')[1]) for i in np.where(support)[0]]
    }
    
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(info, f, indent=2)
        
    if sel_history:
         with open(args.output_dir / f"{args.selector}_history.json", "w") as f:
             json.dump(_convert_to_serializable(sel_history), f, indent=2)

    print(f"=== Finished ===")
    print(f"Val R2: {metrics['val']['r2']:.4f}")
    print(f"Val RMSE: {metrics['val']['rmse']:.4f}")

if __name__ == "__main__":
    main()