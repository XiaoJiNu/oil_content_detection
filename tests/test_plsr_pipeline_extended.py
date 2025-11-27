import numpy as np

from oil_content_detection.models.plsr_pipeline import (
    PLSRExperimentConfig,
    run_plsr_experiment,
)
from oil_content_detection.preprocessing import PreprocessStep
from oil_content_detection.feature_selection.ga_selector import GAConfig


def _make_mock_data(n_samples: int = 60, n_features: int = 40):
    rng = np.random.default_rng(42)
    wavelengths = np.arange(900, 900 + n_features)
    X = rng.normal(0.5, 0.1, size=(n_samples, n_features))
    # Embed a signal on a subset of wavelengths
    y = 15 + 8 * X[:, 5:10].mean(axis=1) + 3 * X[:, 20:25].mean(axis=1) + rng.normal(0, 0.8, size=n_samples)
    return X, y, wavelengths


def test_plsr_pipeline_full_band_with_preprocessing():
    X, y, wavelengths = _make_mock_data()
    cfg = PLSRExperimentConfig(
        preprocess=(PreprocessStep("snv"), PreprocessStep("savgol", {"window_length": 7, "polyorder": 2})),
        use_ga=False,
        test_size=0.2,
        random_state=2024,
        cv_splits=4,
    )

    result = run_plsr_experiment(X, y, wavelengths=wavelengths, config=cfg)

    assert not result.use_ga
    assert result.support_mask.sum() == X.shape[1]
    assert result.n_components >= 1
    assert result.train_r2 > 0
    assert result.test_r2 > -1  # allow slight negative but not extreme
    assert result.rmsec > 0 and result.rmsep > 0 and result.rmsecv > 0
    assert result.r2cv > -1


def test_plsr_pipeline_with_ga_selection():
    X, y, wavelengths = _make_mock_data()
    cfg = PLSRExperimentConfig(
        preprocess=(),
        use_ga=True,
        ga_config=GAConfig(generations=3, population_size=6, min_features=6, max_features=15, random_state=1),
        test_size=0.25,
        random_state=123,
        cv_splits=3,
        max_components=8,
    )

    result = run_plsr_experiment(X, y, wavelengths=wavelengths, config=cfg)

    assert result.use_ga
    assert 6 <= result.support_mask.sum() <= 15
    assert len(result.selected_wavelengths) == result.support_mask.sum()
    assert result.ga_score is not None
    assert result.n_components <= cfg.max_components
    assert result.train_r2 > -1
