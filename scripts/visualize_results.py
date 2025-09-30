#!/usr/bin/env python3
"""Visualize experiment results from GA + PLSR pipeline."""
import argparse
import sys
from pathlib import Path

# Ensure the repository's src directory is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.visualization.plots import (
    plot_all_results,
    plot_ga_history,
    plot_prediction_results,
    plot_spectral_selection,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize GA + PLSR experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing experiment results (with ga_history.json, etc.)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/set_II/mean_spectra.csv"),
        help="Path to original CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["all", "ga", "spectral", "prediction"],
        default="all",
        help="Type of plot to generate",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)

    if not args.data.exists():
        print(f"Error: Data file {args.data} does not exist")
        sys.exit(1)

    output_dir = args.output_dir or (args.results_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing results from: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    if args.plot_type == "all":
        plot_all_results(args.results_dir, args.data, output_dir)

    elif args.plot_type == "ga":
        history_path = args.results_dir / "ga_history.json"
        if not history_path.exists():
            print(f"Error: {history_path} not found")
            sys.exit(1)
        plot_ga_history(history_path, output_dir / "ga_history.png")

    elif args.plot_type == "spectral":
        wavelengths_path = args.results_dir / "selected_wavelengths.json"
        if not wavelengths_path.exists():
            print(f"Error: {wavelengths_path} not found")
            sys.exit(1)
        plot_spectral_selection(args.data, wavelengths_path, output_dir / "spectral_selection.png")

    elif args.plot_type == "prediction":
        model_path = args.results_dir / "plsr_model.pkl"
        support_path = args.results_dir / "feature_support.npy"
        if not model_path.exists() or not support_path.exists():
            print(f"Error: Model or support file not found in {args.results_dir}")
            sys.exit(1)
        plot_prediction_results(args.data, model_path, support_path, output_dir / "prediction_results.png")

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()