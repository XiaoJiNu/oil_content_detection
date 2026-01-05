"""构建花椒高光谱数据集（HDR/DAT + 理化标签）的管线。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import trim_mean

from oil_content_detection.acquisition import load_envi_cube, nearest_wavelength_index
from oil_content_detection.preprocessing.hsv_tuner import apply_hsv_mask
from oil_content_detection.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LabelConfig:
    sample_id_cols: tuple[str, ...] = ("高光谱图件编号", "图件编号", "编号", "样本编号")
    distill_volume_cols: tuple[str, ...] = ("蒸馏量（初）ml", "蒸馏量(初)ml", "蒸馏量初ml", "蒸馏量", "蒸馏量_ml")
    weight_cols: tuple[str, ...] = ("重量", "重量(g)", "重量（g）", "重量g", "样品重量")
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
    # --- spectral background filtering (inside ROI) ---
    spectral_bg_filter_enabled: bool = False
    spectral_bg_filter_method: str = "ratio_median"  # {"ratio_median", "cosine_margin"}
    spectral_bg_filter_min_bg_pixels: int = 1024
    spectral_bg_filter_eps: float = 1e-6
    spectral_bg_filter_sample_size: int = 20000
    spectral_bg_filter_seed: int = 2026
    spectral_bg_filter_cosine_margin: float = 0.03
    spectral_bg_filter_max_removed_ratio: float = 0.20
    spectral_bg_filter_chunk_size: int = 50000
    spectral_bg_filter_apply_min_removed_ratio: float = 0.0
    use_hsv: bool = True  # If True, try HSV-based mask first
    hsv_lower: tuple[int, int, int] = (30, 70, 30)  # Defaults from hsv_tuner.py
    hsv_upper: tuple[int, int, int] = (65, 255, 255)
    hsv_blur_ksize: int = 5
    hsv_closing_size: int = 5
    hsv_opening_size: int = 3
    hsv_min_area: int = 256
    hsv_border_crop: float = 0.02  # remove borders before masking
    hsv_min_area_ratio: float = 0.01  # fallback to spectral if HSV mask too small (<1% pixels)
    hsv_max_width: Optional[int] = 400  # resize for HSV, None to disable


@dataclass
class AggregationConfig:
    trim_fraction: float = 0.10
    primary_stat: str = "trimmed_mean"
    include_stats: tuple[str, ...] = ("mean", "median", "trimmed_mean", "std")


@dataclass
class SampleInfo:
    sample_id_raw: str
    sample_id: str
    sample_dir: Path
    distill_ml: float
    weight_g: float
    oil_ml_per_gram: float
    oil_ml_per_100g: float
    hdr_path: Path


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
    suffix = excel_path.suffix.lower()
    engine = "openpyxl" if suffix in {".xlsx", ".xlsm"} else "xlrd"
    try:
        df = pd.read_excel(excel_path, engine=engine, sheet_name=config.sheet_name)
    except ImportError as exc:
        raise ImportError(f"{engine} is required to read {excel_path}; please install it") from exc
    except Exception as exc:
        df = pd.read_excel(excel_path, sheet_name=config.sheet_name)

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


def filter_mask_background(
    cube: np.ndarray,
    mask: np.ndarray,
    manual_mask: np.ndarray,
    wavelengths: Sequence[float],
    config: HuajiaoROIConfig,
) -> tuple[np.ndarray, Dict[str, float]]:
    """在手工 ROI 内估计背景并从当前 mask 中剔除。

    支持两种方法：
    - ratio_median: 使用 ``nir/red`` 比值的中位数阈值（轻量）。
    - cosine_margin: 基于全光谱均值的余弦相似度差（更稳健，且对亮度变化更不敏感）。
    """
    if not config.spectral_bg_filter_enabled:
        return mask, {}

    if mask.sum() == 0:
        return mask, {"spectral_bg_filter_skipped": 1.0, "spectral_bg_filter_skip_reason_code": 1.0}

    manual_mask = manual_mask.astype(bool)
    if manual_mask.shape != mask.shape:
        raise ValueError("manual_mask and mask must have the same shape")

    bg_mask = manual_mask & (~mask)
    if bg_mask.sum() < config.spectral_bg_filter_min_bg_pixels:
        return mask, {
            "spectral_bg_filter_skipped": 1.0,
            "spectral_bg_filter_skip_reason_code": 2.0,
            "spectral_bg_filter_bg_pixel_count": float(bg_mask.sum()),
        }

    method = str(config.spectral_bg_filter_method or "ratio_median").strip().lower()
    if method not in {"ratio_median", "cosine_margin"}:
        raise ValueError(f"Unknown spectral_bg_filter_method: {config.spectral_bg_filter_method}")

    # Prepare output info base.
    info: Dict[str, float] = {
        "spectral_bg_filter_enabled": 1.0,
        "spectral_bg_filter_bg_pixel_count": float(bg_mask.sum()),
    }

    if method == "ratio_median":
        nir_idx = nearest_wavelength_index(list(wavelengths), config.nir_target_nm)
        red_idx = nearest_wavelength_index(list(wavelengths), config.red_target_nm)
        info["spectral_bg_filter_method_ratio_median"] = 1.0
        info["spectral_bg_filter_nir_band_index"] = float(nir_idx)
        info["spectral_bg_filter_red_band_index"] = float(red_idx)

        nir_band = cube[:, :, nir_idx].astype(float)
        red_band = cube[:, :, red_idx].astype(float)

        ratio_pepper_full = nir_band[mask] / np.maximum(red_band[mask], config.spectral_bg_filter_eps)
        ratio_bg_full = nir_band[bg_mask] / np.maximum(red_band[bg_mask], config.spectral_bg_filter_eps)
        ratio_pepper = ratio_pepper_full[np.isfinite(ratio_pepper_full)]
        ratio_bg = ratio_bg_full[np.isfinite(ratio_bg_full)]

        if ratio_pepper.size == 0 or ratio_bg.size == 0:
            return mask, {"spectral_bg_filter_skipped": 1.0, "spectral_bg_filter_skip_reason_code": 3.0}

        pepper_median = float(np.median(ratio_pepper))
        bg_median = float(np.median(ratio_bg))
        threshold = (pepper_median + bg_median) / 2.0

        # Decide which side is pepper by comparing medians.
        keep_high = bg_median < pepper_median
        if keep_high:
            keep_flags = np.isfinite(ratio_pepper_full) & (ratio_pepper_full >= threshold)
        else:
            keep_flags = np.isfinite(ratio_pepper_full) & (ratio_pepper_full <= threshold)

        keep_flags = np.asarray(keep_flags, dtype=bool)
        kept = int(keep_flags.sum())
        total = int(mask.sum())
        removed_ratio = 1.0 - (kept / float(max(total, 1)))

        filtered_mask = np.zeros_like(mask, dtype=bool)
        flat = filtered_mask.reshape(-1)
        mask_indices = np.flatnonzero(mask)
        flat[mask_indices[keep_flags]] = True

        info.update(
            {
                "spectral_bg_filter_threshold": float(threshold),
                "spectral_bg_filter_pepper_median": float(pepper_median),
                "spectral_bg_filter_bg_median": float(bg_median),
                "spectral_bg_filter_keep_high": 1.0 if keep_high else 0.0,
                "spectral_bg_filter_removed_ratio": float(removed_ratio),
            }
        )

        if removed_ratio < float(config.spectral_bg_filter_apply_min_removed_ratio):
            info["spectral_bg_filter_skipped"] = 1.0
            info["spectral_bg_filter_skip_reason_code"] = 7.0
            return mask, info
        return filtered_mask, info

    # cosine_margin method
    info["spectral_bg_filter_method_cosine_margin"] = 1.0
    info["spectral_bg_filter_cosine_margin"] = float(config.spectral_bg_filter_cosine_margin)

    rng = np.random.default_rng(config.spectral_bg_filter_seed)
    mask_indices = np.flatnonzero(mask)
    bg_indices = np.flatnonzero(bg_mask)

    pepper_sample_n = int(min(config.spectral_bg_filter_sample_size, mask_indices.size))
    bg_sample_n = int(min(config.spectral_bg_filter_sample_size, bg_indices.size))
    if pepper_sample_n < 3 or bg_sample_n < 3:
        return mask, {"spectral_bg_filter_skipped": 1.0, "spectral_bg_filter_skip_reason_code": 4.0}

    pepper_sample_idx = rng.choice(mask_indices, size=pepper_sample_n, replace=False)
    bg_sample_idx = rng.choice(bg_indices, size=bg_sample_n, replace=False)

    flat_cube = cube.reshape(-1, cube.shape[2])
    pepper_sample = flat_cube[pepper_sample_idx].astype(np.float32, copy=False)
    bg_sample = flat_cube[bg_sample_idx].astype(np.float32, copy=False)

    pepper_mean = np.nanmean(pepper_sample, axis=0)
    bg_mean = np.nanmean(bg_sample, axis=0)
    pep_norm = float(np.linalg.norm(pepper_mean))
    bg_norm = float(np.linalg.norm(bg_mean))
    if pep_norm <= 0 or bg_norm <= 0:
        return mask, {"spectral_bg_filter_skipped": 1.0, "spectral_bg_filter_skip_reason_code": 5.0}

    pepper_mean_unit = (pepper_mean / pep_norm).astype(np.float32, copy=False)
    bg_mean_unit = (bg_mean / bg_norm).astype(np.float32, copy=False)

    margin = float(config.spectral_bg_filter_cosine_margin)
    chunk = int(max(1024, config.spectral_bg_filter_chunk_size))
    keep_flags = np.empty(mask_indices.size, dtype=bool)

    for start in range(0, mask_indices.size, chunk):
        end = min(start + chunk, mask_indices.size)
        idx = mask_indices[start:end]
        pix = flat_cube[idx].astype(np.float32, copy=False)
        norms = np.linalg.norm(pix, axis=1)
        norms = np.maximum(norms, float(config.spectral_bg_filter_eps))
        sim_pep = (pix @ pepper_mean_unit) / norms
        sim_bg = (pix @ bg_mean_unit) / norms
        keep_flags[start:end] = (sim_bg - sim_pep) <= margin

    kept = int(keep_flags.sum())
    total = int(mask_indices.size)
    removed_ratio = 1.0 - (kept / float(max(total, 1)))
    info["spectral_bg_filter_removed_ratio"] = float(removed_ratio)

    if removed_ratio < float(config.spectral_bg_filter_apply_min_removed_ratio):
        info["spectral_bg_filter_skipped"] = 1.0
        info["spectral_bg_filter_skip_reason_code"] = 7.0
        return mask, info

    if removed_ratio > float(config.spectral_bg_filter_max_removed_ratio):
        info["spectral_bg_filter_skipped"] = 1.0
        info["spectral_bg_filter_skip_reason_code"] = 6.0
        return mask, info

    filtered_mask = np.zeros_like(mask, dtype=bool)
    filtered_mask.reshape(-1)[mask_indices[keep_flags]] = True
    return filtered_mask, info


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


def _find_sample_image(sample_dir: Path, sample_id_raw: str, sample_id: str) -> Optional[Path]:
    """Find a representative image for ROI overlay."""
    candidates = [
        sample_dir / f"{sample_id_raw}.png",
        sample_dir / f"{sample_id}.png",
        sample_dir / f"{sample_id_raw}.jpg",
        sample_dir / f"{sample_id}.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    for folder in (sample_dir, sample_dir / "capture"):
        pngs = sorted(folder.glob("*.png"))
        if pngs:
            return pngs[0]
    return None


def create_hsv_mask(
    image_path: Path,
    target_shape: tuple[int, int],
    cfg: HuajiaoROIConfig,
) -> Optional[np.ndarray]:
    """Create ROI mask from RGB image using HSV thresholds (delegates to apply_hsv_mask)."""
    try:
        mask = apply_hsv_mask(
            str(image_path),
            lower_hsv=tuple(cfg.hsv_lower),
            upper_hsv=tuple(cfg.hsv_upper),
            max_width=cfg.hsv_max_width,
            border_crop=cfg.hsv_border_crop,
            blur_ksize=cfg.hsv_blur_ksize,
            closing_size=cfg.hsv_closing_size,
            opening_size=cfg.hsv_opening_size,
            keep_largest=True,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("HSV masking failed for %s: %s", image_path, exc)
        return None

    mask = mask > 0
    # Remove tiny masks.
    if mask.sum() < cfg.hsv_min_area:
        logger.warning("HSV mask too small (%d pixels) for %s", mask.sum(), image_path)
        return None

    # Resize to cube shape.
    target_h, target_w = target_shape
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        try:
            import cv2
        except Exception:
            logger.warning("OpenCV unavailable for resizing HSV mask")
            return None
        mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0

    return mask


def _save_roi_visualization(
    cube: np.ndarray,
    mask: np.ndarray,
    sample_dir: Path,
    sample_id_raw: str,
    sample_id: str,
    roi_dir: Optional[Path],
) -> Optional[Path]:
    """Save ROI overlay image to roi_dir; returns saved path or None."""
    if roi_dir is None:
        return None

    roi_dir.mkdir(parents=True, exist_ok=True)
    image_path = _find_sample_image(sample_dir, sample_id_raw, sample_id)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - visualization fallback
        logger.warning("Matplotlib unavailable for ROI visualization: %s", exc)
        return None

    overlay_base: np.ndarray
    try:
        if image_path and image_path.exists():
            try:
                from PIL import Image
            except Exception:
                Image = None  # type: ignore

            if "Image" in locals() and Image is not None:
                img = Image.open(image_path).convert("RGB")
                img_arr = np.asarray(img, dtype=float) / 255.0
                mask_arr = mask
                if mask.shape[0] != img_arr.shape[0] or mask.shape[1] != img_arr.shape[1]:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((img_arr.shape[1], img_arr.shape[0]), resample=Image.NEAREST)
                    mask_arr = np.asarray(mask_img) > 0
                overlay_base = img_arr
                mask_use = mask_arr
            else:
                raise ImportError("Pillow not available; fallback to cube intensity")
        else:
            raise FileNotFoundError("Sample image not found; fallback to cube intensity")
    except Exception as exc:
        logger.debug("ROI overlay using cube intensity due to: %s", exc)
        intensity = np.nanmean(cube, axis=2)
        intensity = np.nan_to_num(intensity, nan=0.0)
        if intensity.max() > intensity.min():
            norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        else:
            norm = intensity
        overlay_base = np.stack([norm] * 3, axis=-1)
        mask_use = mask

    overlay = overlay_base.copy()
    alpha = 0.6
    overlay[mask_use] = overlay[mask_use] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha

    out_path = roi_dir / f"{sample_id}_roi.png"
    plt.imsave(out_path, np.clip(overlay, 0.0, 1.0))
    return out_path


def _save_hsv_debug(
    image_path: Optional[Path],
    mask: np.ndarray,
    roi_dir: Optional[Path],
    sample_id: str,
) -> tuple[Optional[Path], Optional[Path]]:
    """Save HSV mask and overlay for debugging."""
    if roi_dir is None or image_path is None:
        return None, None
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception:
        return None, None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None, None

    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_uint8)
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
        mask_uint8 = np.asarray(mask_img)

    roi_dir.mkdir(parents=True, exist_ok=True)
    mask_path = roi_dir / f"{sample_id}_hsv_mask.png"
    mask_img.save(mask_path)

    base = np.asarray(img, dtype=float) / 255.0
    overlay = base.copy()
    overlay[mask_uint8 > 0] = overlay[mask_uint8 > 0] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6
    overlay_path = roi_dir / f"{sample_id}_hsv_overlay.png"
    plt.imsave(overlay_path, np.clip(overlay, 0.0, 1.0))
    return mask_path, overlay_path


def _process_sample(
    sample: SampleInfo,
    roi_cfg: HuajiaoROIConfig,
    agg_cfg: AggregationConfig,
    roi_visualization_dir: Optional[Path],
    roi_mask_dir: Optional[Path] = None,
    manual_mask_dir: Optional[Path] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    cube, header = load_envi_cube(sample.hdr_path)
    wavelengths = header.wavelengths or list(range(cube.shape[2]))

    # Prefer HSV-based mask from RGB image; fallback to spectral ratio if unavailable.
    mask_info: Dict[str, float] = {}
    mask: Optional[np.ndarray] = None
    hsv_mask_path: Optional[Path] = None
    hsv_overlay_path: Optional[Path] = None
    image_path = _find_sample_image(sample.sample_dir, sample.sample_id_raw, sample.sample_id)

    # If pre-generated mask exists, load and resize.
    if roi_mask_dir:
        pre_mask_path = roi_mask_dir / f"{sample.sample_id}_mask.png"
        if pre_mask_path.exists():
            try:
                from PIL import Image
            except Exception:
                Image = None  # type: ignore
            if "Image" in locals() and Image is not None:
                mask_img = Image.open(pre_mask_path).convert("L")
                if mask_img.size != (cube.shape[1], cube.shape[0]):
                    mask_img = mask_img.resize((cube.shape[1], cube.shape[0]), resample=Image.NEAREST)
                mask = (np.asarray(mask_img) > 0).astype(bool)
                mask_info["mask_source"] = "pre_generated"

    manual_mask: Optional[np.ndarray] = None
    if manual_mask_dir:
        manual_mask_path = manual_mask_dir / f"{sample.sample_id}_mask.png"
        if manual_mask_path.exists():
            try:
                from PIL import Image
            except Exception:
                Image = None  # type: ignore
            if "Image" in locals() and Image is not None:
                mm_img = Image.open(manual_mask_path).convert("L")
                if mm_img.size != (cube.shape[1], cube.shape[0]):
                    mm_img = mm_img.resize((cube.shape[1], cube.shape[0]), resample=Image.NEAREST)
                manual_mask = (np.asarray(mm_img) > 0).astype(bool)

    if roi_cfg.use_hsv and mask is None:
        if image_path:
            mask = create_hsv_mask(image_path, (cube.shape[0], cube.shape[1]), roi_cfg)
            if mask is not None:
                mask_info["mask_source"] = "hsv"
                mask_info["hsv_lower"] = str(roi_cfg.hsv_lower)
                mask_info["hsv_upper"] = str(roi_cfg.hsv_upper)
                area_ratio = mask.sum() / float(mask.size)
                hsv_mask_path, hsv_overlay_path = _save_hsv_debug(image_path, mask, roi_visualization_dir, sample.sample_id)
                # Fallback to spectral if mask too small
                if mask.sum() < roi_cfg.hsv_min_area or area_ratio < roi_cfg.hsv_min_area_ratio:
                    logger.warning(
                        "HSV mask too small for %s (pixels=%d, ratio=%.4f), falling back to spectral mask",
                        sample.sample_id,
                        mask.sum(),
                        area_ratio,
                    )
                    mask = None

    if mask is None:
        mask, mask_info = create_huajiao_mask(cube, wavelengths, roi_cfg)
        mask_info["mask_source"] = "spectral"

    if roi_cfg.spectral_bg_filter_enabled and manual_mask is not None:
        mask, bg_info = filter_mask_background(cube, mask, manual_mask, wavelengths, roi_cfg)
        mask_info.update(bg_info)

    cleaned_mask, clean_info = clean_mask_extremes(cube, mask, roi_cfg)
    if cleaned_mask.sum() == 0:
        raise ValueError(f"Sample {sample.sample_id} has zero valid pixels after cleaning")

    stats = _aggregate_spectra(cube, cleaned_mask, trim_fraction=agg_cfg.trim_fraction)

    feature_row: Dict[str, float] = {
        "sample_id": sample.sample_id,
        "sample_id_raw": sample.sample_id_raw,
        "distill_ml": float(sample.distill_ml),
        "weight_g": float(sample.weight_g),
        "oil_ml_per_gram": float(sample.oil_ml_per_gram),
        "oil_ml_per_100g": float(sample.oil_ml_per_100g),
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

    roi_path = _save_roi_visualization(
        cube,
        cleaned_mask,
        sample.sample_dir,
        sample.sample_id_raw,
        sample.sample_id,
        roi_visualization_dir,
    )

    ratio_threshold = mask_info.get("ratio_threshold", np.nan)
    intensity_threshold = mask_info.get("intensity_threshold", np.nan)
    nir_band_index = mask_info.get("nir_band_index", np.nan)
    red_band_index = mask_info.get("red_band_index", np.nan)

    meta_row = {
        "sample_id": sample.sample_id,
        "sample_id_raw": sample.sample_id_raw,
        "sample_dir": str(sample.sample_dir),
        "hdr_path": str(sample.hdr_path),
        "dat_path": str(header.dat_path) if header.dat_path else str(sample.hdr_path.with_suffix(".dat")),
        "distill_ml": float(sample.distill_ml),
        "weight_g": float(sample.weight_g),
        "oil_ml_per_gram": float(sample.oil_ml_per_gram),
        "oil_ml_per_100g": float(sample.oil_ml_per_100g),
        "pixel_count": int(mask.sum()),
        "valid_pixel_count": int(cleaned_mask.sum()),
        "coverage_ratio": cleaned_mask.sum() / float(cube.shape[0] * cube.shape[1]),
        "ratio_threshold": ratio_threshold,
        "intensity_threshold": intensity_threshold,
        "clip_low": clean_info["low"],
        "clip_high": clean_info["high"],
        "wavelength_count": len(wavelengths),
        "nir_band_index": nir_band_index,
        "red_band_index": red_band_index,
        "roi_visualization": str(roi_path) if roi_path else "",
        "hsv_mask_path": str(hsv_mask_path) if hsv_mask_path else "",
        "hsv_overlay_path": str(hsv_overlay_path) if hsv_overlay_path else "",
    }

    for key, value in mask_info.items():
        if key.startswith("spectral_bg_filter_"):
            meta_row[key] = float(value)
    return feature_row, meta_row


def _build_dataset_from_samples(
    samples: Sequence[SampleInfo],
    output_dir: Path,
    roi_config: HuajiaoROIConfig,
    agg_config: AggregationConfig,
    save: bool = True,
    roi_visualization_dir: Optional[Path] = None,
    roi_mask_dir: Optional[Path] = None,
    manual_mask_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra_rows: List[Dict[str, float]] = []
    meta_rows: List[Dict[str, float]] = []

    for sample in samples:
        try:
            feature_row, meta_row = _process_sample(
                sample,
                roi_config,
                agg_config,
                roi_visualization_dir,
                roi_mask_dir,
                manual_mask_dir,
            )
        except Exception as exc:
            logger.warning("Skipping sample %s due to error: %s", sample.sample_id, exc)
            continue
        spectra_rows.append(feature_row)
        meta_rows.append(meta_row)

    spectra_df = pd.DataFrame(spectra_rows)
    metadata_df = pd.DataFrame(meta_rows)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_with_fallback(metadata_df, output_dir / "huajiao_metadata.parquet")
        _save_with_fallback(spectra_df, output_dir / "huajiao_spectra.parquet")

    logger.info("Built dataset: %d samples with spectra, %d metadata rows", len(spectra_df), len(metadata_df))
    return spectra_df, metadata_df


def build_huajiao_dataset(
    raw_root: Path,
    excel_path: Path,
    output_dir: Path = Path("data/processed/huajiao"),
    label_config: LabelConfig | None = None,
    roi_config: HuajiaoROIConfig | None = None,
    agg_config: AggregationConfig | None = None,
    save: bool = True,
    roi_visualization_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原始 HDR/DAT + 标签构建特征表与元数据表。"""
    label_cfg = label_config or LabelConfig()
    roi_cfg = roi_config or HuajiaoROIConfig()
    agg_cfg = agg_config or AggregationConfig()

    labels_df = load_huajiao_labels(excel_path, label_cfg)
    cube_map = discover_huajiao_cubes(raw_root)

    samples: List[SampleInfo] = []
    for _, row in labels_df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in cube_map:
            logger.warning("No HDR/DAT found for sample_id=%s", sample_id)
            continue
        hdr_path = cube_map[sample_id]
        sample_dir = hdr_path.parent.parent
        samples.append(
            SampleInfo(
                sample_id_raw=row["sample_id_raw"],
                sample_id=sample_id,
                sample_dir=sample_dir,
                distill_ml=float(row["distill_ml"]),
                weight_g=float(row["weight_g"]),
                oil_ml_per_gram=float(row["oil_ml_per_gram"]),
                oil_ml_per_100g=float(row["oil_ml_per_100g"]),
                hdr_path=hdr_path,
            )
        )

    return _build_dataset_from_samples(
        samples,
        output_dir,
        roi_cfg,
        agg_cfg,
        save=save,
        roi_visualization_dir=roi_visualization_dir,
    )


def build_huajiao_dataset_from_split(
    split_path: Path,
    output_dir: Path,
    roi_config: HuajiaoROIConfig | None = None,
    agg_config: AggregationConfig | None = None,
    save: bool = True,
    roi_visualization_dir: Optional[Path] = None,
    roi_mask_dir: Optional[Path] = None,
    manual_mask_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于 train/val txt 清单构建特征表。"""
    roi_cfg = roi_config or HuajiaoROIConfig()
    agg_cfg = agg_config or AggregationConfig()

    samples: List[SampleInfo] = []
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    lines = split_path.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            logger.warning("Skip malformed line in %s: %s", split_path, line)
            continue

        sample_id_raw = parts[0]
        try:
            weight_g = float(parts[1])
            distill_ml = float(parts[2])
        except ValueError:
            logger.warning("Skip line with non-numeric weight/distill: %s", line)
            continue

        sample_dir = Path(" ".join(parts[4:])) if len(parts) > 5 else Path(parts[4])
        sample_id = normalize_sample_id(sample_id_raw)
        hdr_path = sample_dir / "capture" / f"REFLECTANCE_{sample_id}.hdr"
        if not hdr_path.exists():
            logger.warning("HDR not found for %s at %s", sample_id, hdr_path)
            continue
        if weight_g <= 0:
            logger.warning("Weight <= 0 for %s, skip", sample_id)
            continue
        oil_ml_per_gram = distill_ml / weight_g
        oil_ml_per_100g = oil_ml_per_gram * 100

        samples.append(
            SampleInfo(
                sample_id_raw=sample_id_raw,
                sample_id=sample_id,
                sample_dir=sample_dir,
                distill_ml=distill_ml,
                weight_g=weight_g,
                oil_ml_per_gram=oil_ml_per_gram,
                oil_ml_per_100g=oil_ml_per_100g,
                hdr_path=hdr_path,
            )
        )

    return _build_dataset_from_samples(
        samples,
        output_dir,
        roi_cfg,
        agg_cfg,
        save=save,
        roi_visualization_dir=roi_visualization_dir,
        roi_mask_dir=roi_mask_dir,
        manual_mask_dir=manual_mask_dir,
    )


def _save_with_fallback(df: pd.DataFrame, path: Path) -> None:
    """保存为 Parquet；若依赖缺失则降级 CSV。"""
    try:
        df.to_parquet(path, index=False)
        logger.info("Saved %s", path)
    except Exception as exc:  # Parquet engine missing or other issues
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning("Parquet save failed (%s); fallback to CSV: %s", exc, csv_path)
