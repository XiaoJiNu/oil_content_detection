"""Upload storage helpers for the Huajiao Web application."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from .settings import HuajiaoWebSettings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
HDR_EXTENSIONS = {".hdr"}
SPECTRAL_DATA_EXTENSIONS = {".dat", ".raw"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class UploadJob:
    job_id: str
    upload_dir: Path
    output_dir: Path
    debug_dir: Path
    input_output_dir: Path


def create_upload_job(settings: HuajiaoWebSettings) -> UploadJob:
    job_id = uuid.uuid4().hex[:12]
    upload_dir = settings.upload_root / job_id
    output_dir = settings.output_root / job_id
    debug_dir = output_dir / "debug"
    input_output_dir = output_dir / "input"
    for path in [upload_dir, output_dir, debug_dir, input_output_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return UploadJob(
        job_id=job_id,
        upload_dir=upload_dir,
        output_dir=output_dir,
        debug_dir=debug_dir,
        input_output_dir=input_output_dir,
    )


def sanitize_filename(filename: str | None, fallback: str) -> str:
    raw = Path(filename or fallback).name.strip() or fallback
    safe = _SAFE_NAME_RE.sub("_", raw)
    safe = safe.strip("._")
    return safe or fallback


def validate_extension(
    filename: str, allowed_extensions: Iterable[str], role: str
) -> None:
    suffix = Path(filename).suffix.lower()
    allowed = {ext.lower() for ext in allowed_extensions}
    if suffix not in allowed:
        raise ValueError(
            f"Unsupported {role} file extension '{suffix}'. Allowed: {sorted(allowed)}"
        )


async def save_upload_file(
    upload_file: UploadFile,
    *,
    destination_dir: Path,
    allowed_extensions: Iterable[str],
    role: str,
    max_bytes: int,
    fallback_name: str,
) -> Path:
    """Save one uploaded file with extension and size checks."""

    filename = sanitize_filename(upload_file.filename, fallback=fallback_name)
    validate_extension(filename, allowed_extensions, role)
    destination = destination_dir / filename
    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise ValueError(f"{role} file exceeds max_upload_mb limit.")
            handle.write(chunk)
    if total == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(f"{role} file is empty.")
    return destination


def ensure_dat_sibling(hdr_path: Path, uploaded_data_path: Path) -> Path:
    """Ensure the current ENVI reader can find a .dat file next to the .hdr."""

    expected_dat_path = hdr_path.with_suffix(".dat")
    if uploaded_data_path.resolve() == expected_dat_path.resolve():
        return expected_dat_path
    shutil.copyfile(uploaded_data_path, expected_dat_path)
    return expected_dat_path


def copy_input_preview(image_path: Path, job: UploadJob) -> Path:
    destination = job.input_output_dir / image_path.name
    shutil.copyfile(image_path, destination)
    return destination


def output_url(settings: HuajiaoWebSettings, path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    try:
        relative = file_path.resolve().relative_to(settings.output_root.resolve())
    except ValueError:
        return None
    return "/outputs/" + relative.as_posix()


__all__ = [
    "HDR_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "SPECTRAL_DATA_EXTENSIONS",
    "UploadJob",
    "copy_input_preview",
    "create_upload_job",
    "ensure_dat_sibling",
    "output_url",
    "sanitize_filename",
    "save_upload_file",
]
