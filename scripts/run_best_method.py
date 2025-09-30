#!/usr/bin/env python3
"""Execute the GA + PLSR pipeline on the simulated spectral dataset."""
import argparse
import sys
from pathlib import Path

# Ensure the repository's src directory is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.utils import setup_single_thread

setup_single_thread()

from oil_content_detection.models.plsr_best import RunConfig, run_and_print


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GA + PLSR pipeline for oil content detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/set_II/mean_spectra.csv"),
        help="Path to input CSV with spectral data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (model, wavelengths, history)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=34 / 102,
        help="Proportion of dataset for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=10,
        help="Number of GA generations",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=12,
        help="GA population size",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=10,
        help="Minimum number of features to select",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=22,
        help="Maximum number of features to select",
    )
    parser.add_argument(
        "--target-features",
        type=int,
        default=18,
        help="Target number of features (soft constraint)",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save the trained model file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print GA progress during training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = RunConfig(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.seed,
        ga_generations=args.ga_generations,
        ga_population=args.ga_population,
        ga_min_features=args.min_features,
        ga_max_features=args.max_features,
        target_features=args.target_features,
        output_dir=args.output_dir,
        save_model_file=not args.no_save_model,
        verbose=args.verbose,
    )

    run_and_print(config)
