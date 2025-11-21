"""I/O utilities for saving and loading results."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from oil_content_detection.utils.logging import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: Path) -> None:
    """Save a trained model to disk using pickle.

    Args:
        model: Model object to save
        path: Path to save the model
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> Any:
    """Load a trained model from disk.

    Args:
        path: Path to the saved model

    Returns:
        Loaded model object
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def save_results_json(results: Any, path: Path, include_timestamp: bool = True) -> Path:
    """Save experiment results to JSON file.

    Args:
        results: Results object (typically a dataclass)
        path: Path to save the JSON file
        include_timestamp: If True, append timestamp to filename

    Returns:
        Path to the saved file
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict if needed
    if hasattr(results, "__dataclass_fields__"):
        results_dict = asdict(results)
    elif isinstance(results, dict):
        results_dict = results
    else:
        results_dict = {"results": str(results)}

    # Convert numpy types to native Python types
    results_dict = _convert_to_serializable(results_dict)

    # Add metadata
    results_dict["saved_at"] = datetime.now().isoformat()

    # Optionally add timestamp to filename
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        suffix = path.suffix
        path = path.parent / f"{stem}_{timestamp}{suffix}"

    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {path}")
    return path


def save_wavelengths(wavelengths: list[int], path: Path) -> None:
    """Save selected wavelengths to JSON file.

    Args:
        wavelengths: List of selected wavelength values
        path: Path to save the file
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "wavelengths": wavelengths,
        "count": len(wavelengths),
        "saved_at": datetime.now().isoformat(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Wavelengths saved to {path}")


def load_wavelengths(path: Path) -> list[int]:
    """Load selected wavelengths from JSON file.

    Args:
        path: Path to the wavelength file

    Returns:
        List of wavelength values
    """
    with open(path, "r") as f:
        data = json.load(f)

    wavelengths = data["wavelengths"]
    logger.info(f"Loaded {len(wavelengths)} wavelengths from {path}")
    return wavelengths


__all__ = [
    "save_model",
    "load_model",
    "save_results_json",
    "save_wavelengths",
    "load_wavelengths",
]