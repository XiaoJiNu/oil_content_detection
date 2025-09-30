#!/usr/bin/env python3
"""Visualize spatial oil content distribution for individual camellia seeds."""
import argparse
import sys
from pathlib import Path

# Ensure the repository's src directory is importable when running directly.
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from oil_content_detection.visualization.spatial_distribution import (
    create_summary_grid,
    visualize_all_seeds,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize oil content spatial distribution in camellia seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing trained model and support mask",
    )
    parser.add_argument(
        "--cube-data",
        type=Path,
        default=Path("data/processed/set_II/simulated_set_II_cube.npz"),
        help="Path to NPZ file containing hyperspectral cubes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save visualizations (default: results_dir/oil_distributions)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "summary", "single"],
        default="summary",
        help="Visualization mode: all (all seeds), summary (grid), single (specific samples)",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Sample indices to visualize (for 'single' mode)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=12,
        help="Number of samples in summary grid",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for colorbar",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for colorbar",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.results_dir.exists():
        print(f"Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)

    if not args.cube_data.exists():
        print(f"Error: Cube data file {args.cube_data} does not exist")
        sys.exit(1)

    model_path = args.results_dir / "plsr_model.pkl"
    support_path = args.results_dir / "feature_support.npy"

    if not model_path.exists() or not support_path.exists():
        print(f"Error: Model or support file not found in {args.results_dir}")
        print(f"Expected files: plsr_model.pkl, feature_support.npy")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        output_dir = args.results_dir / "oil_distributions"
    else:
        output_dir = args.output_dir

    print(f"Results directory: {args.results_dir}")
    print(f"Cube data: {args.cube_data}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print()

    # Execute based on mode
    if args.mode == "summary":
        print(f"Creating summary grid with {args.n_samples} samples...")
        output_path = output_dir / "oil_distribution_summary.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_summary_grid(
            cube_data_path=args.cube_data,
            model_path=model_path,
            support_path=support_path,
            output_path=output_path,
            n_samples=args.n_samples,
            seed=args.seed,
        )

        print(f"\n✓ Summary grid saved to: {output_path}")

    elif args.mode == "all":
        print(f"Visualizing all seeds...")
        output_dir.mkdir(parents=True, exist_ok=True)

        visualize_all_seeds(
            cube_data_path=args.cube_data,
            model_path=model_path,
            support_path=support_path,
            output_dir=output_dir,
            sample_indices=None,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        print(f"\n✓ All visualizations saved to: {output_dir}")

    elif args.mode == "single":
        if args.sample_indices is None:
            print("Error: --sample-indices required for 'single' mode")
            sys.exit(1)

        print(f"Visualizing samples: {args.sample_indices}")
        output_dir.mkdir(parents=True, exist_ok=True)

        visualize_all_seeds(
            cube_data_path=args.cube_data,
            model_path=model_path,
            support_path=support_path,
            output_dir=output_dir,
            sample_indices=args.sample_indices,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        print(f"\n✓ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()