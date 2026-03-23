#!/usr/bin/env python3
"""
Train and evaluate oil content prediction model using refined Huajiao dataset.
Dataset: data/processed/huajiao_refined (train/val splits)
Target: oil_ml_per_100g (Oil Content)
Method: GA Feature Selection + PLSR
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

# Ensure the repository's src directory is importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.utils import setup_single_thread, get_logger, rmse
from oil_content_detection.feature_selection.ga_selector import GAConfig, GeneticAlgorithmSelector
from oil_content_detection.utils.io import _convert_to_serializable
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline

setup_single_thread()
logger = get_logger(__name__)

def load_split_data(data_dir: Path):
    """Load spectra and metadata for a split (train or val)."""
    spectra_path = data_dir / "huajiao_spectra.parquet"
    metadata_path = data_dir / "huajiao_metadata.parquet"
    
    if not spectra_path.exists():
        raise FileNotFoundError(f"Spectra file not found: {spectra_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    logger.info(f"Loading data from {data_dir}...")
    
    # Try reading parquet, if fails (due to env issues), we might need another way or fix env
    # Assuming the environment is usable for now.
    spectra_df = pd.read_parquet(spectra_path)
    metadata_df = pd.read_parquet(metadata_path)
    
    # Merge on sample_id to ensure alignment, though they should be aligned by construction
    # We only need 'sample_id' and 'oil_ml_per_100g' from metadata/spectra
    # Check where the target variable is. Usually in metadata or both.
    # In `huajiao_dataset.py`, `feature_row` has `oil_ml_per_100g`.
    
    if "oil_ml_per_100g" in spectra_df.columns:
        target_col = "oil_ml_per_100g"
        merged = spectra_df
    elif "oil_ml_per_100g" in metadata_df.columns:
        target_col = "oil_ml_per_100g"
        merged = pd.merge(spectra_df, metadata_df[["sample_id", "oil_ml_per_100g"]], on="sample_id")
    else:
        raise ValueError("Target 'oil_ml_per_100g' not found in spectra or metadata.")
        
    # Extract features (wavelengths)
    # Looking for columns starting with 'wl_' and not containing other characters if we only want primary stat
    # Or just all 'wl_' columns?
    # Usually we train on the primary statistic (e.g. trimmed_mean).
    # Let's filter for columns that look like 'wl_\d+$' (integers)
    
    feature_cols = [c for c in merged.columns if c.startswith("wl_") and c[3:].isdigit()]
    
    if not feature_cols:
         # Maybe float wavelengths?
         feature_cols = [c for c in merged.columns if c.startswith("wl_")]
         # Filter out aux stats like _std, _median if they exist and we only want primary
         # If the primary stat is 'trimmed_mean' and it was saved as 'wl_XXX', then 'wl_XXX_std' would be separate.
         # Let's prefer 'wl_XXX' format.
         primary_cols = [c for c in feature_cols if "_" not in c[3:]]
         if primary_cols:
             feature_cols = primary_cols
    
    # Sort features by wavelength to be sure
    try:
        feature_cols = sorted(feature_cols, key=lambda x: float(x.split('_')[1]))
    except:
        feature_cols = sorted(feature_cols)
        
    X = merged[feature_cols].values
    y = merged[target_col].values
    sample_ids = merged["sample_id"].values
    
    logger.info(f"Loaded {len(merged)} samples. Features: {len(feature_cols)}. Target: {target_col}")
    
    return X, y, sample_ids, feature_cols

def build_preprocess_steps(args: argparse.Namespace) -> list[PreprocessStep]:
    """Build preprocessing pipeline steps from CLI args."""
    if args.preprocess == "raw":
        return []
    if args.preprocess == "sg1":
        return [
            PreprocessStep(
                name="savgol",
                params={"window_length": args.sg_window_length, "polyorder": args.sg_polyorder, "deriv": 1},
            )
        ]
    raise ValueError(f"Unknown preprocess mode: {args.preprocess}")


def tune_plsr_components(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_components: int,
    cv_splits: int,
    seed: int,
) -> tuple[int, float]:
    """Select PLSR n_components by CV RMSE on training data."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] != len(y):
        raise ValueError("X and y must have same number of samples")

    max_allowed = min(max_components, max(1, X.shape[0] - 1), X.shape[1])
    if max_allowed < 1:
        return 1, float("nan")

    cv = KFold(cv_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))

    best_n = 1
    best_score = float("-inf")  # maximizing neg RMSE
    for n_components in range(1, max_allowed + 1):
        model = PLSRegression(n_components=n_components, scale=False)
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                scoring="neg_root_mean_squared_error",
                cv=splits,
                n_jobs=None,
                error_score=np.nan,
            )
        except Exception:
            continue
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            continue
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_n = n_components

    if not np.isfinite(best_score):
        return 1, float("nan")
    best_rmse = -best_score
    return best_n, best_rmse


def train_and_evaluate(args):
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    X_train, y_train, train_ids, feature_names = load_split_data(train_dir)
    X_val, y_val, val_ids, val_feature_names = load_split_data(val_dir)
    
    # Ensure features match
    if feature_names != val_feature_names:
        raise ValueError("Feature mismatch between train and val sets.")
        
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    
    # 1.1 Spectral preprocessing (optional)
    steps = build_preprocess_steps(args)
    if steps:
        logger.info("Applying preprocessing pipeline: %s", steps)
        X_train = apply_preprocessing_pipeline(X_train, steps)
        X_val = apply_preprocessing_pipeline(X_val, steps)

    # 2. GA Feature Selection
    logger.info("Starting GA Feature Selection...")
    ga_config = GAConfig(
        population_size=args.ga_population,
        generations=args.ga_generations,
        min_features=args.min_features,
        max_features=args.max_features,
        target_features=args.target_features,
        random_state=args.seed,
        cv_splits=args.cv_splits,  # CV on training set
        verbose=args.verbose
    )
    
    selector = GeneticAlgorithmSelector(ga_config)
    selector.fit(X_train, y_train)
    
    selected_indices = selector.selected_indices()
    selected_features = [feature_names[i] for i in selected_indices]
    selected_wavelengths = [float(f.split('_')[1]) for f in selected_features]
    
    logger.info(f"GA Selected {len(selected_features)} features.")
    logger.info(f"Best CV Score (R² minus penalty): {selector.best_score():.4f}")
    
    # 3. Train PLSR on selected features
    X_train_sel = X_train[:, selected_indices]
    X_val_sel = X_val[:, selected_indices]
    
    # 3.1 Tune n_components with CV on training set
    n_components, cv_rmse = tune_plsr_components(
        X_train_sel,
        y_train,
        max_components=args.plsr_max_components,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )
    logger.info("Selected PLSR n_components=%d (CV RMSE=%.4f)", n_components, cv_rmse)
    
    model = PLSRegression(n_components=n_components, scale=False) # Data should be centered by PLSR, scaling?
    # It's better to verify if we need scaling. Usually spectra are not scaled to unit variance, just centered. 
    # run_best_method.py sets scale=False.
    
    model.fit(X_train_sel, y_train)
    
    # 4. Evaluate
    y_train_pred = model.predict(X_train_sel).ravel()
    y_val_pred = model.predict(X_val_sel).ravel()
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = rmse(y_train, y_train_pred)
    
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = rmse(y_val, y_val_pred)
    
    logger.info("=== Results ===")
    logger.info(f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    logger.info(f"Val   R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    logger.info(f"Selected Wavelengths: {sorted(selected_wavelengths)}")
    
    # 5. Save Results
    results = {
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "val_r2": val_r2,
        "val_rmse": val_rmse,
        "n_components": n_components,
        "plsr_cv_rmse": cv_rmse,
        "selected_wavelengths": sorted(selected_wavelengths),
        "preprocess": _convert_to_serializable([{"name": s.name, "params": s.params} for s in steps]),
        "ga_params": {
            "generations": args.ga_generations,
            "population": args.ga_population,
            "seed": args.seed
        }
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Save predictions
    pred_df = pd.DataFrame({
        "sample_id": val_ids,
        "actual": y_val,
        "predicted": y_val_pred,
        "error": y_val_pred - y_val
    })
    pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Huajiao Oil Content Model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/huajiao_refined"), help="Data directory containing train/val subdirs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/huajiao_refined_oil_content"), help="Output directory")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--preprocess", type=str, default="sg1", choices=["raw", "sg1"], help="Spectral preprocessing mode")
    parser.add_argument("--sg-window-length", type=int, default=13, help="Savitzky-Golay window length (odd)")
    parser.add_argument("--sg-polyorder", type=int, default=2, help="Savitzky-Golay polyorder")
    parser.add_argument("--ga-generations", type=int, default=20)
    parser.add_argument("--ga-population", type=int, default=20)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--target-features", type=int, default=15)
    parser.add_argument("--cv-splits", type=int, default=5, help="CV splits used for GA and PLSR tuning")
    parser.add_argument("--plsr-max-components", type=int, default=15, help="Maximum PLSR components to try during tuning")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
