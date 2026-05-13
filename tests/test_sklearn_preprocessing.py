"""Tests for scikit-learn compatible spectral preprocessing."""

import numpy as np

from oil_content_detection.preprocessing import (
    PLSScoreTransformer,
    PreprocessStep,
    SpectralPreprocessor,
)
from oil_content_detection.preprocessing.spectral import msc, snv


def _msc_with_ref(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float).reshape(1, -1)
    corrected = np.empty_like(X)
    ref_col = ref.reshape(-1, 1)
    ref_vec = ref.reshape(-1)
    for i, spectrum in enumerate(X):
        coeffs, *_ = np.linalg.lstsq(ref_col, spectrum, rcond=None)
        slope = float(coeffs[0]) if np.ndim(coeffs) > 0 else float(coeffs)
        intercept = float(np.mean(spectrum - slope * ref_vec))
        corrected[i] = (spectrum - intercept) / max(slope, 1e-8)
    return corrected


def test_spectral_preprocessor_snv_matches_functional_api():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 20))

    prep = SpectralPreprocessor([PreprocessStep(name="snv")])
    out = prep.fit_transform(X)
    np.testing.assert_allclose(out, snv(X))


def test_spectral_preprocessor_msc_uses_training_reference():
    rng = np.random.default_rng(1)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(12, 15))
    X_test = rng.normal(loc=1.0, scale=2.0, size=(5, 15))  # different distribution

    prep = SpectralPreprocessor([PreprocessStep(name="msc")])
    out_train = prep.fit_transform(X_train)

    # On train split, the sklearn transformer matches the original functional API.
    np.testing.assert_allclose(out_train, msc(X_train))

    # On test split, it must use the fitted training reference spectrum.
    ref = np.mean(X_train, axis=0, keepdims=True)
    expected_test = _msc_with_ref(X_test, ref)
    out_test = prep.transform(X_test)
    np.testing.assert_allclose(out_test, expected_test)


def test_pls_score_transformer_returns_x_scores_only():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 30))
    y = rng.normal(size=(20,))

    pls = PLSScoreTransformer(n_components=5, scale=False)
    scores = pls.fit_transform(X, y)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (20, 5)
