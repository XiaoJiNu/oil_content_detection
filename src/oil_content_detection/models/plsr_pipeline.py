"""PLSR 训练/评估管线，支持预处理和可选 GA 波段筛选。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from oil_content_detection.feature_selection.ga_selector import GAConfig, GeneticAlgorithmSelector
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline
from oil_content_detection.utils import get_logger, rmse, setup_single_thread

setup_single_thread()

logger = get_logger(__name__)


@dataclass
class PLSRExperimentConfig:
    preprocess: Sequence[PreprocessStep | str] = ()
    use_ga: bool = False
    ga_config: Optional[GAConfig] = None
    test_size: float = 0.25
    random_state: int = 2024
    cv_splits: int = 5
    max_components: int = 12


@dataclass
class PLSRExperimentResult:
    preprocess_steps: List[str]
    use_ga: bool
    selected_wavelengths: List[int]
    support_mask: np.ndarray
    n_components: int
    train_r2: float
    test_r2: float
    rmsec: float
    rmsep: float
    rmsecv: float
    r2cv: float
    ga_score: Optional[float]


@dataclass
class PLSRFitResult:
    """Result of fitting a PLSR model on a given train set."""

    preprocess_steps: List[str]
    use_ga: bool
    selected_wavelengths: List[int]
    support_mask: np.ndarray
    n_components: int
    ga_score: Optional[float]
    model: PLSRegression


def _select_support(
    X_train: np.ndarray,
    y_train: np.ndarray,
    wavelengths: Optional[Sequence[int]],
    cfg: PLSRExperimentConfig,
) -> tuple[np.ndarray, List[int], Optional[float]]:
    if not cfg.use_ga:
        support = np.ones(X_train.shape[1], dtype=bool)
        if wavelengths is None:
            selected_wls = list(range(X_train.shape[1]))
        else:
            selected_wls = list(wavelengths)
        return support, selected_wls, None

    ga_cfg = cfg.ga_config or GAConfig()
    if ga_cfg.random_state is None:
        ga_cfg.random_state = cfg.random_state
    selector = GeneticAlgorithmSelector(ga_cfg)
    selector.fit(X_train, y_train)
    support = selector.get_support()
    if wavelengths is None:
        selected_wavelengths = list(selector.selected_indices())
    else:
        selected_wavelengths = [int(wavelengths[idx]) for idx in selector.selected_indices()]
    return support, selected_wavelengths, selector.best_score()


def fit_plsr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    wavelengths: Optional[Sequence[int]] = None,
    config: PLSRExperimentConfig = PLSRExperimentConfig(),
) -> PLSRFitResult:
    """Fit a PLSR model on the provided training split.

    This is a lower-level API compared to ``run_plsr_experiment``; it does not
    perform its own train/test split.
    """
    X_proc = apply_preprocessing_pipeline(X_train, config.preprocess)

    support, selected_wavelengths, ga_score = _select_support(X_proc, y_train, wavelengths, config)
    n_components = min(config.max_components, max(1, support.sum() // 2))

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_proc[:, support], y_train)

    return PLSRFitResult(
        preprocess_steps=[s.name if isinstance(s, PreprocessStep) else str(s) for s in config.preprocess],
        use_ga=config.use_ga,
        selected_wavelengths=selected_wavelengths,
        support_mask=support,
        n_components=n_components,
        ga_score=ga_score,
        model=model,
    )


def run_plsr_experiment(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: Optional[Sequence[int]] = None,
    config: PLSRExperimentConfig = PLSRExperimentConfig(),
) -> PLSRExperimentResult:
    """训练并评估单条 PLSR 方案。"""
    X_proc = apply_preprocessing_pipeline(X, config.preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    support, selected_wavelengths, ga_score = _select_support(X_train, y_train, wavelengths, config)
    n_components = min(config.max_components, max(1, support.sum() // 2))

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train[:, support], y_train)

    y_train_pred = model.predict(X_train[:, support]).ravel()
    y_test_pred = model.predict(X_test[:, support]).ravel()

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmsec = rmse(y_train, y_train_pred)
    rmsep = rmse(y_test, y_test_pred)

    cv = KFold(config.cv_splits, shuffle=True, random_state=config.random_state)
    y_cv_pred = cross_val_predict(model, X_proc[:, support], y, cv=cv)
    rmsecv = rmse(y, y_cv_pred)
    r2cv = r2_score(y, y_cv_pred)

    return PLSRExperimentResult(
        preprocess_steps=[s.name if isinstance(s, PreprocessStep) else str(s) for s in config.preprocess],
        use_ga=config.use_ga,
        selected_wavelengths=selected_wavelengths,
        support_mask=support,
        n_components=n_components,
        train_r2=train_r2,
        test_r2=test_r2,
        rmsec=rmsec,
        rmsep=rmsep,
        rmsecv=rmsecv,
        r2cv=r2cv,
        ga_score=ga_score,
    )


__all__ = [
    "PLSRExperimentConfig",
    "PLSRFitResult",
    "fit_plsr_model",
    "PLSRExperimentResult",
    "run_plsr_experiment",
]
