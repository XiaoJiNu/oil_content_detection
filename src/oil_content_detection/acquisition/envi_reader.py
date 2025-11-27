"""Lightweight ENVI ``.hdr``/``.dat`` reader for hyperspectral cubes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from oil_content_detection.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EnviHeader:
    """Parsed ENVI header metadata."""

    samples: int
    lines: int
    bands: int
    data_type: int
    interleave: str
    byte_order: int
    header_offset: int = 0
    wavelengths: Optional[List[float]] = None
    fwhm: Optional[List[float]] = None
    dat_path: Optional[Path] = None


_DATA_TYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def _parse_scalar(text: str, key: str, cast_type) -> Optional[int]:
    match = re.search(rf"{key}\s*=\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    try:
        return cast_type(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {key}: {value}") from exc


def _parse_list(text: str, key: str) -> Optional[List[float]]:
    match = re.search(rf"{key}\s*=\s*{{([^}}]+)}}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = match.group(1)
    items = re.split(r"[,\s]+", raw.strip())
    values: List[float] = []
    for item in items:
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            logger.debug("Skipping non-numeric wavelength entry '%s' for key %s", item, key)
            continue
    return values if values else None


def parse_envi_header(hdr_path: Path) -> EnviHeader:
    """Parse an ENVI ``.hdr`` file."""
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")

    samples = _parse_scalar(text, "samples", int)
    lines = _parse_scalar(text, "lines", int)
    bands = _parse_scalar(text, "bands", int)
    data_type = _parse_scalar(text, "data type", int)
    interleave = _parse_scalar(text, "interleave", str)
    byte_order = _parse_scalar(text, "byte order", int) or 0
    header_offset = _parse_scalar(text, "header offset", int) or 0
    wavelengths = _parse_list(text, "wavelength")
    fwhm = _parse_list(text, "fwhm")

    if samples is None or lines is None or bands is None or data_type is None or interleave is None:
        raise ValueError(f"Missing required fields in header: {hdr_path}")

    interleave_norm = str(interleave).strip().lower()
    if interleave_norm not in {"bsq", "bil", "bip"}:
        raise ValueError(f"Unsupported interleave '{interleave}' in {hdr_path}")

    if data_type not in _DATA_TYPE_MAP:
        raise ValueError(f"Unsupported ENVI data type '{data_type}' in {hdr_path}")

    dat_path = hdr_path.with_suffix(".dat")
    if not dat_path.exists():
        logger.warning("Associated .dat file not found for %s", hdr_path)

    header = EnviHeader(
        samples=samples,
        lines=lines,
        bands=bands,
        data_type=data_type,
        interleave=interleave_norm,
        byte_order=byte_order,
        header_offset=header_offset,
        wavelengths=wavelengths,
        fwhm=fwhm,
        dat_path=dat_path if dat_path.exists() else None,
    )
    logger.debug("Parsed ENVI header: %s", header)
    return header


def _dtype_with_byte_order(data_type: int, byte_order: int) -> np.dtype:
    dtype = np.dtype(_DATA_TYPE_MAP[data_type])
    if byte_order == 0:
        return dtype.newbyteorder("<")
    return dtype.newbyteorder(">")


def load_envi_cube(hdr_path: Path, *, memmap: bool = True) -> tuple[np.ndarray, EnviHeader]:
    """Load ENVI cube to (lines, samples, bands) numpy array."""
    header = parse_envi_header(hdr_path)
    dat_path = header.dat_path or hdr_path.with_suffix(".dat")
    if not dat_path.exists():
        raise FileNotFoundError(f".dat file not found for header {hdr_path}")

    dtype = _dtype_with_byte_order(header.data_type, header.byte_order)
    shape_raw: tuple[int, ...]

    if header.interleave == "bsq":
        shape_raw = (header.bands, header.lines, header.samples)
    elif header.interleave == "bil":
        shape_raw = (header.lines, header.bands, header.samples)
    else:  # bip
        shape_raw = (header.lines, header.samples, header.bands)

    if memmap:
        cube_raw = np.memmap(
            dat_path,
            dtype=dtype,
            mode="r",
            offset=header.header_offset,
            shape=shape_raw,
        )
    else:
        data = np.fromfile(dat_path, dtype=dtype, offset=header.header_offset)
        expected = int(np.prod(shape_raw))
        if data.size != expected:
            raise ValueError(f"Unexpected data size in {dat_path}: got {data.size}, expected {expected}")
        cube_raw = data.reshape(shape_raw)

    if header.interleave == "bsq":
        cube = np.transpose(cube_raw, (1, 2, 0))
    elif header.interleave == "bil":
        cube = np.transpose(cube_raw, (0, 2, 1))
    else:
        cube = cube_raw

    logger.info("Loaded ENVI cube %s with shape %s", hdr_path.name, cube.shape)
    return np.asarray(cube), header


def nearest_wavelength_index(wavelengths: List[float], target_nm: float) -> int:
    """Find the index of the wavelength closest to target."""
    arr = np.asarray(wavelengths, dtype=float)
    idx = int(np.abs(arr - target_nm).argmin())
    return idx


__all__ = ["EnviHeader", "parse_envi_header", "load_envi_cube", "nearest_wavelength_index"]
