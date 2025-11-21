#!/usr/bin/env python3
"""Generate a simulated hyperspectral dataset resembling Spectral Set II.

The script creates:
- a compressed NPZ file with calibrated hyperspectral cubes and ROI masks;
- a CSV file containing ROI-average spectra with oil content labels.

The resulting files live under ``data/processed/set_II``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    num_samples: int = 102
    height: int = 24
    width: int = 24
    wavelength_start: int = 900
    wavelength_end: int = 1700
    wavelength_step: int = 5
    noise_level: float = 0.015
    background_level: float = 0.08
    seed: int = 2024

    @property
    def wavelengths(self) -> np.ndarray:
        return np.arange(self.wavelength_start, self.wavelength_end + 1, self.wavelength_step)

    @property
    def num_wavelengths(self) -> int:
        return self.wavelengths.size


def make_output_dirs(output_root: Path) -> None:
    (output_root / "set_II").mkdir(parents=True, exist_ok=True)


def gaussian_profile(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def simulate_roi_mask(rng: "np.random.Generator", height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cy = rng.uniform(height * 0.4, height * 0.6)
    cx = rng.uniform(width * 0.4, width * 0.6)
    ay = rng.uniform(height * 0.35, height * 0.45)
    ax = rng.uniform(width * 0.35, width * 0.45)
    angle = rng.uniform(-0.4, 0.4)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_rot = (xx - cx) * cos_a + (yy - cy) * sin_a
    y_rot = (yy - cy) * cos_a - (xx - cx) * sin_a
    mask = (x_rot / ax) ** 2 + (y_rot / ay) ** 2 <= 1.0
    return mask


def build_base_spectrum(rng: "np.random.Generator", wavelengths: np.ndarray) -> np.ndarray:
    baseline = 0.35 + rng.uniform(-0.03, 0.03)
    spec = np.full_like(wavelengths, baseline, dtype=np.float32)

    # Add several absorption/emission features with smooth transitions.
    peak_centers = rng.uniform(960, 1650, size=4)
    peak_widths = rng.uniform(35, 110, size=4)
    peak_amplitudes = rng.uniform(0.08, 0.22, size=4)

    for center, width, amplitude in zip(peak_centers, peak_widths, peak_amplitudes):
        sign = rng.choice([-1.0, 1.2])  # allow both troughs and peaks
        spec += sign * amplitude * gaussian_profile(wavelengths, center, width)

    # Add a gentle slope to mimic scattering behaviour.
    slope = rng.uniform(-1.5e-4, 1.5e-4)
    spec += slope * (wavelengths - wavelengths.mean())

    return np.clip(spec, 0.05, 0.95)


def derive_oil_content(mean_spectrum: np.ndarray, wavelengths: np.ndarray, rng: "np.random.Generator") -> float:
    # Use absorption around 1200-1500 nm to drive oil content.
    idx_band = (wavelengths >= 1180) & (wavelengths <= 1530)
    absorption_index = 1.0 - mean_spectrum[idx_band].mean()
    baseline = 22.0 + 24.0 * absorption_index
    noisy = baseline + rng.normal(0.0, 1.2)
    return float(np.clip(noisy, 19.0, 45.5))


def simulate_sample(
    rng: "np.random.Generator",
    cfg: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mask = simulate_roi_mask(rng, cfg.height, cfg.width)
    base_spectrum = build_base_spectrum(rng, cfg.wavelengths)

    cube = np.empty((cfg.height, cfg.width, cfg.num_wavelengths), dtype=np.float32)

    roi_noise = rng.normal(0.0, cfg.noise_level, size=(cfg.height, cfg.width, cfg.num_wavelengths))
    shading = 1.0 + rng.normal(0.0, 0.02, size=(cfg.height, cfg.width, 1))
    cube[:] = (base_spectrum * shading + roi_noise)

    # background pixels lowered reflectance and higher noise.
    background = cfg.background_level + rng.normal(0.0, cfg.noise_level * 1.5, size=(cfg.height, cfg.width, cfg.num_wavelengths))
    cube[~mask] = background[~mask]

    cube = np.clip(cube, 0.01, 0.97)
    roi_mean = cube[mask].mean(axis=0)
    oil_content = derive_oil_content(roi_mean, cfg.wavelengths, rng)
    return cube, mask, oil_content, roi_mean


def run_simulation(cfg: SimulationConfig, output_root: Path) -> None:
    rng = np.random.default_rng(cfg.seed)

    cubes = np.empty((cfg.num_samples, cfg.height, cfg.width, cfg.num_wavelengths), dtype=np.float32)
    masks = np.empty((cfg.num_samples, cfg.height, cfg.width), dtype=bool)
    oil_contents = np.empty(cfg.num_samples, dtype=np.float32)
    roi_means = np.empty((cfg.num_samples, cfg.num_wavelengths), dtype=np.float32)
    sample_ids = []

    for idx in range(cfg.num_samples):
        cube, mask, oil_content, roi_mean = simulate_sample(rng, cfg)
        cubes[idx] = cube
        masks[idx] = mask
        oil_contents[idx] = oil_content
        roi_means[idx] = roi_mean
        sample_ids.append(f"sample_{idx:03d}")

    out_dir = output_root / "set_II"
    npz_path = out_dir / "simulated_set_II_cube.npz"
    csv_path = out_dir / "mean_spectra.csv"

    np.savez_compressed(
        npz_path,
        cubes=cubes,
        roi_masks=masks,
        wavelengths=cfg.wavelengths.astype(np.int32),
        sample_ids=np.array(sample_ids),
        oil_content=oil_contents,
    )

    df = pd.DataFrame(roi_means, columns=[f"wl_{wl}" for wl in cfg.wavelengths])
    df.insert(0, "sample_id", sample_ids)
    df["oil_content"] = oil_contents
    df.to_csv(csv_path, index=False)

    print(f"Saved cubes to {npz_path}")
    print(f"Saved ROI mean spectra to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated hyperspectral dataset for Spectral Set II")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Root directory for processed outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SimulationConfig().seed,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(seed=args.seed)
    make_output_dirs(args.output)
    run_simulation(cfg, args.output)


if __name__ == "__main__":
    main()
