"""构建花椒高光谱数据集（HDR/DAT + 理化标签）的管线。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import trim_mean

from oil_content_detection.acquisition import load_envi_cube, nearest_wavelength_index
from oil_content_detection.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LabelConfig:
    sample_id_cols: tuple[str, ...] = ("高光谱图件编号", "图件编号", "编号", "样本编号")
    distill_volume_cols: tuple[str, ...] = ("蒸馏量（初）ml", "蒸馏量初ml", "蒸馏量", "蒸馏量_ml")
    weight_cols: tuple[str, ...] = ("重量", "重量(g)", "重量g", "样品重量")
    sheet_name: Optional[str] = None


@dataclass
class HuajiaoROIConfig:
    nir_target_nm: float = 800.0
    red_target_nm: float = 650.0
    ratio_quantile: float = 0.90
    ratio_floor: float = 1.05
    intensity_quantile: float = 0.15
    closing_size: int = 3
    opening_size: int = 3
    min_area: int = 64
    clip_low: float = 0.01
    clip_high: float = 0.99


@dataclass
class AggregationConfig:
    trim_fraction: float = 0.10
    primary_stat: str = "trimmed_mean"
    include_stats: tuple[str, ...] = ("mean", "median", "trimmed_mean", "std")


def normalize_sample_id(raw: str) -> str:
    cleaned = str(raw).strip()
    cleaned = cleaned.replace("-", "_").replace("—", "_").replace("－", "_")
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns found: {candidates}")


def load_huajiao_labels(excel_path: Path, config: LabelConfig | None = None) -> pd.DataFrame:
    config = config or LabelConfig()
    try:
        df = pd.read_excel(excel_path, engine="xlrd", sheet_name=config.sheet_name)
    except ImportError as exc:
        raise ImportError("xlrd is required to read .xls files; please install the requirement") from exc

    id_col = _first_existing_column(df, config.sample_id_cols)
    distill_col = _first_existing_column(df, config.distill_volume_cols)
    weight_col = _first_existing_column(df, config.weight_cols)

    result = pd.DataFrame(
        {
            "sample_id_raw": df[id_col],
            "distill_ml": pd.to_numeric(df[distill_col], errors="coerce"),
            "weight_g": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )

    result["sample_id"] = result["sample_id_raw"].apply(normalize_sample_id)
    result["weight_g"].replace(0, np.nan, inplace=True)
    result.dropna(subset=["distill_ml", "weight_g"], inplace=True)
    result["oil_ml_per_gram"] = result["distill_ml"] / result["weight_g"]
    result["oil_ml_per_100g"] = result["oil_ml_per_gram"] * 100

    logger.info(
        "Loaded %d label rows from %s (sheet=%s)",
        len(result),
        excel_path,
        config.sheet_name or "default",
    )
    return result.reset_index(drop=True)


def discover_huajiao_cubes(raw_root: Path) -> Dict[str, Path]:
    """发现原始数据目录下的 HDR 文件。"""
    mapping: Dict[str, Path] = {}
    for hdr_path in raw_root.glob("**/REFLECTANCE_*.hdr"):
        stem = hdr_path.stem
        match = re.search(r"(\d{3,}[_-]\d+)", stem)
        if not match:
            match = re.search(r"(\d{3,}[_-]\d+)", hdr_path.parent.name)
        if not match:
            logger.debug("Skip HDR without recognizable id: %s", hdr_path)
            continue
        sample_id = normalize_sample_id(match.group(1))
        if sample_id in mapping:
            logger.warning("Duplicate HDR for %s: keeping %s, skipping %s", sample_id, mapping[sample_id], hdr_path)
            continue
        mapping[sample_id] = hdr_path
    logger.info("Discovered %d HDR files under %s", len(mapping), raw_root)
    return mapping


def create_huajiao_mask(
    cube: np.ndarray,
    wavelengths: List[float] | None,
    config: HuajiaoROIConfig,
) -> tuple[np.ndarray, Dict[str, float]]:
    """基于 NIR/Red 比值与强度分位的掩膜。"""
    bands = cube.shape[2]
    wl = wavelengths if wavelengths is not None else list(range(bands))
    nir_idx = nearest_wavelength_index(wl, config.nir_target_nm)
    red_idx = nearest_wavelength_index(wl, config.red_target_nm)

    red_band = cube[:, :, red_idx]
    nir_band = cube[:, :, nir_idx]
    ratio = nir_band / np.maximum(red_band, 1e-6)
    intensity = np.nanmean(cube, axis=2)

    ratio_valid = ratio[np.isfinite(ratio)]
    intensity_valid = intensity[np.isfinite(intensity)]
    if ratio_valid.size == 0 or intensity_valid.size == 0:
        logger.warning("No valid ratio/intensity values for ROI creation")
        empty = np.zeros_like(intensity, dtype=bool)
        return empty, {
            "ratio_threshold": float("nan"),
            "intensity_threshold": float("nan"),
            "nir_band_index": int(nir_idx),
            "red_band_index": int(red_idx),
        }

    ratio_threshold = max(config.ratio_floor, float(np.quantile(ratio_valid, config.ratio_quantile)))
    intensity_threshold = float(np.quantile(intensity_valid, config.intensity_quantile))

    mask = (
        (ratio >= ratio_threshold)
        & (intensity >= intensity_threshold)
        & np.isfinite(ratio)
        & np.isfinite(intensity)
    )

    structure = np.ones((config.closing_size, config.closing_size), dtype=bool)
    if config.closing_size > 1:
        mask = ndimage.binary_closing(mask, structure=structure)
    if config.opening_size > 1:
        mask = ndimage.binary_opening(mask, structure=structure)

    if config.min_area > 0:
        labels, n_labels = ndimage.label(mask)
        if n_labels > 0:
            counts = ndimage.sum(mask, labels, index=range(1, n_labels + 1))
            keep_labels = [i + 1 for i, c in enumerate(counts) if c >= config.min_area]
            mask = np.isin(labels, keep_labels)

    info = {
        "ratio_threshold": ratio_threshold,
        "intensity_threshold": intensity_threshold,
        "nir_band_index": int(nir_idx),
        "red_band_index": int(red_idx),
    }
    return mask, info


def clean_mask_extremes(
    cube: np.ndarray, mask: np.ndarray, config: HuajiaoROIConfig
) -> tuple[np.ndarray, Dict[str, float]]:
    """按全波段平均反射率分位剔除极亮/极暗像素。"""
    if mask.sum() == 0:
        return mask, {"low": np.nan, "high": np.nan, "removed_ratio": 0.0}

    pixels = cube[mask]  # (n_pixels, bands)
    mean_reflectance = np.nanmean(pixels, axis=1)
    low = float(np.nanquantile(mean_reflectance, config.clip_low))
    high = float(np.nanquantile(mean_reflectance, config.clip_high))

    keep = (mean_reflectance >= low) & (mean_reflectance <= high) & np.isfinite(mean_reflectance)
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    cleaned_flat = cleaned_mask.reshape(-1)
    mask_indices = np.flatnonzero(mask)
    cleaned_flat[mask_indices[keep]] = True

    removed_ratio = 1.0 - (cleaned_mask.sum() / mask.sum())
    info = {"low": low, "high": high, "removed_ratio": removed_ratio}
    return cleaned_mask, info


def _aggregate_spectra(
    cube: np.ndarray,
    mask: np.ndarray,
    trim_fraction: float,
) -> Dict[str, np.ndarray]:
    if mask.sum() == 0:
        raise ValueError("No pixels available for aggregation")
    pixels = cube[mask]  # (n_pixels, bands)
    stats: Dict[str, np.ndarray] = {
        "mean": np.nanmean(pixels, axis=0),
        "median": np.nanmedian(pixels, axis=0),
        "std": np.nanstd(pixels, axis=0),
    }
    if trim_fraction > 0:
        stats["trimmed_mean"] = trim_mean(pixels, proportiontocut=trim_fraction, axis=0)
    else:
        stats["trimmed_mean"] = stats["mean"]
    return stats


def _column_name(wavelength: float, stat: str, primary_stat: str) -> str:
    base = f"wl_{int(round(wavelength))}"
    if stat == primary_stat:
        return base
    return f"{base}_{stat}"


def build_huajiao_dataset(
    raw_root: Path,
    excel_path: Path,
    output_dir: Path = Path("data/processed/huajiao"),
    label_config: LabelConfig | None = None,
    roi_config: HuajiaoROIConfig | None = None,
    agg_config: AggregationConfig | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原始 HDR/DAT + 标签构建特征表与元数据表。"""
    label_cfg = label_config or LabelConfig()
    roi_cfg = roi_config or HuajiaoROIConfig()
    agg_cfg = agg_config or AggregationConfig()

    labels_df = load_huajiao_labels(excel_path, label_cfg)
    cube_map = discover_huajiao_cubes(raw_root)

    spectra_rows: List[Dict[str, float]] = []
    meta_rows: List[Dict[str, float]] = []

    for _, row in labels_df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in cube_map:
            logger.warning("No HDR/DAT found for sample_id=%s", sample_id)
            continue
        hdr_path = cube_map[sample_id]
        cube, header = load_envi_cube(hdr_path)
        wavelengths = header.wavelengths or list(range(cube.shape[2]))

        mask, mask_info = create_huajiao_mask(cube, wavelengths, roi_cfg)
        cleaned_mask, clean_info = clean_mask_extremes(cube, mask, roi_cfg)
        if cleaned_mask.sum() == 0:
            logger.warning("Sample %s has zero valid pixels after cleaning", sample_id)
            continue

        stats = _aggregate_spectra(cube, cleaned_mask, trim_fraction=agg_cfg.trim_fraction)

        feature_row: Dict[str, float] = {
            "sample_id": sample_id,
            "distill_ml": float(row["distill_ml"]),
            "weight_g": float(row["weight_g"]),
            "oil_ml_per_gram": float(row["oil_ml_per_gram"]),
            "oil_ml_per_100g": float(row["oil_ml_per_100g"]),
            "pixel_count": int(mask.sum()),
            "valid_pixel_count": int(cleaned_mask.sum()),
            "coverage_ratio": cleaned_mask.sum() / float(cube.shape[0] * cube.shape[1]),
        }

        for stat_name in agg_cfg.include_stats:
            if stat_name not in stats:
                continue
            values = stats[stat_name]
            for idx, wl in enumerate(wavelengths):
                col = _column_name(wl, stat_name, agg_cfg.primary_stat)
                feature_row[col] = float(values[idx])

        spectra_rows.append(feature_row)

        meta_rows.append(
            {
                "sample_id": sample_id,
                "sample_id_raw": row["sample_id_raw"],
                "hdr_path": str(hdr_path),
                "dat_path": str(header.dat_path) if header.dat_path else str(hdr_path.with_suffix(".dat")),
                "distill_ml": float(row["distill_ml"]),
                "weight_g": float(row["weight_g"]),
                "oil_ml_per_gram": float(row["oil_ml_per_gram"]),
                "oil_ml_per_100g": float(row["oil_ml_per_100g"]),
                "pixel_count": int(mask.sum()),
                "valid_pixel_count": int(cleaned_mask.sum()),
                "coverage_ratio": cleaned_mask.sum() / float(cube.shape[0] * cube.shape[1]),
                "ratio_threshold": mask_info["ratio_threshold"],
                "intensity_threshold": mask_info["intensity_threshold"],
                "clip_low": clean_info["low"],
                "clip_high": clean_info["high"],
                "wavelength_count": len(wavelengths),
                "nir_band_index": mask_info["nir_band_index"],
                "red_band_index": mask_info["red_band_index"],
            }
        )

    spectra_df = pd.DataFrame(spectra_rows)
    metadata_df = pd.DataFrame(meta_rows)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_with_fallback(metadata_df, output_dir / "huajiao_metadata.parquet")
        _save_with_fallback(spectra_df, output_dir / "huajiao_spectra.parquet")

    logger.info("Built dataset: %d samples with spectra, %d metadata rows", len(spectra_df), len(metadata_df))
    return spectra_df, metadata_df


def _save_with_fallback(df: pd.DataFrame, path: Path) -> None:
    """保存为 Parquet；若依赖缺失则降级 CSV。"""
    try:
        df.to_parquet(path, index=False)
        logger.info("Saved %s", path)
    except Exception as exc:  # Parquet engine missing or other issues
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning("Parquet save failed (%s); fallback to CSV: %s", exc, csv_path)
