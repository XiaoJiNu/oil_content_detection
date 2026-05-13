"""光谱与图像预处理模块。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SPECTRAL_EXPORTS = {
    "PreprocessStep",
    "apply_preprocessing_pipeline",
    "detrend",
    "msc",
    "normalize",
    "savgol",
    "snv",
}

_SKLEARN_EXPORTS = {
    "PLSScoreTransformer",
    "SpectralPreprocessor",
}

__all__ = sorted(_SPECTRAL_EXPORTS | _SKLEARN_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SPECTRAL_EXPORTS:
        spectral = import_module(".spectral", __name__)
        return getattr(spectral, name)
    if name in _SKLEARN_EXPORTS:
        sklearn_mod = import_module(".sklearn", __name__)
        return getattr(sklearn_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _SPECTRAL_EXPORTS | _SKLEARN_EXPORTS)

