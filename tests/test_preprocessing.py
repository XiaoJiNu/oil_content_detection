import numpy as np

from oil_content_detection.preprocessing import (
    PreprocessStep,
    apply_preprocessing_pipeline,
    detrend,
    msc,
    normalize,
    savgol,
    snv,
)


def test_snv_zero_mean_unit_std():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    out = snv(X)
    np.testing.assert_allclose(out[0].mean(), 0.0)
    np.testing.assert_allclose(out[0].std(), 1.0)
    # constant row should remain zeros after normalization guard
    np.testing.assert_allclose(out[1], 0.0)


def test_msc_reduces_baseline_shift():
    base = np.array([0.4, 0.5, 0.6])
    spectrum_a = base + 0.05  # baseline shift
    spectrum_b = base - 0.05
    X = np.vstack([spectrum_a, spectrum_b])
    corrected = msc(X)
    # After MSC, both spectra should be closer to shared mean
    mean_diff = np.abs(corrected[0] - corrected[1]).mean()
    assert mean_diff < 0.05


def test_savgol_derivative_shape_and_non_nan():
    X = np.tile(np.linspace(0, 1, 25), (4, 1))
    out = savgol(X, window_length=7, polyorder=2, deriv=1)
    assert out.shape == X.shape
    assert np.isfinite(out).all()


def test_normalize_and_detrend():
    X = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    normed = normalize(X)
    np.testing.assert_allclose(normed[0].max(), 1.0)
    detrended = detrend(X)
    assert detrended.shape == X.shape
    assert np.abs(detrended[0].mean()) < 1e-6


def test_preprocessing_pipeline_sequential():
    X = np.tile(np.linspace(0, 1, 11), (3, 1))
    steps = [
        "snv",
        PreprocessStep(name="savgol", params={"window_length": 5, "polyorder": 2, "deriv": 1}),
        "normalize",
    ]
    out = apply_preprocessing_pipeline(X, steps)
    assert out.shape == X.shape
    assert np.isfinite(out).all()
    # After normalization, each row max should be 1
    np.testing.assert_allclose(out.max(axis=1), 1.0)
