#!/usr/bin/env python3
"""Compare total oil prediction schemes A (two-stage) and B (direct).

Example:
  python scripts/compare_total_oil_methods.py \
    --train data/processed/huajiao/train/huajiao_spectra.parquet \
    --val data/processed/huajiao/val/huajiao_spectra.parquet \
    --use-ga \
    --output-dir results/total_oil_comparison
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running as a script.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.feature_selection.ga_selector import GAConfig
from oil_content_detection.pipelines.total_oil_pipeline import (
    TotalOilExperimentConfig,
    run_total_oil_experiment_with_predictions,
)
from oil_content_detection.preprocessing import PreprocessStep
from oil_content_detection.utils import save_results_json, setup_single_thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare total oil prediction schemes A and B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/huajiao/train/huajiao_spectra.parquet"),
        help="Training spectra parquet",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/processed/huajiao/val/huajiao_spectra.parquet"),
        help="Validation spectra parquet",
    )
    parser.add_argument(
        "--use-ga",
        action="store_true",
        help="Enable GA feature selection for PLSR",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    setup_single_thread()
    args = parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)

    preprocess = (PreprocessStep("savgol", {"window_length": 9, "polyorder": 2, "deriv": 1}),)
    ga_cfg = None
    if args.use_ga:
        ga_cfg = GAConfig(
            generations=15,
            population_size=16,
            min_features=6,
            max_features=12,
            target_features=10,
            mutation_rate=0.08,
            crossover_rate=0.85,
            elite_count=2,
            patience=5,
            cv_splits=5,
            verbose=True,
            random_state=args.seed,
        )

    cfg = TotalOilExperimentConfig(
        spectral_preprocess=preprocess,
        use_ga=args.use_ga,
        ga_config=ga_cfg,
        random_state=args.seed,
        direct_include_shape=True,
    )

    result, train_pred_df, val_pred_df = run_total_oil_experiment_with_predictions(
        train_df, val_df=val_df, config=cfg
    )

    print("=== 总含油量对照结果（验证集）===")
    print(f"A(两阶段) R²={result.metrics['total_a_val']['r2']:.4f} "
          f"RMSE={result.metrics['total_a_val']['rmse']:.6f} "
          f"MAE={result.metrics['total_a_val']['mae']:.6f}")
    print(f"B(直接)   R²={result.metrics['total_b_val']['r2']:.4f} "
          f"RMSE={result.metrics['total_b_val']['rmse']:.6f} "
          f"MAE={result.metrics['total_b_val']['mae']:.6f}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_results_json(result, args.output_dir / "results.json", include_timestamp=True)

        train_csv = args.output_dir / "train_predictions_total_oil.csv"
        val_csv = args.output_dir / "val_predictions_total_oil.csv"
        train_pred_df.to_csv(train_csv, index=False, float_format="%.6f")
        val_pred_df.to_csv(val_csv, index=False, float_format="%.6f")

        print(f"结果已保存: {out_path}")
        print(f"训练集逐样本预测: {train_csv}")
        print(f"验证集逐样本预测: {val_csv}")


if __name__ == "__main__":
    main()
