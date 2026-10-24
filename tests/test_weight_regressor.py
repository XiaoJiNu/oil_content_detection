import numpy as np

from oil_content_detection.models.weight_regressor import WeightRegressor, WeightRegressorConfig


def test_weight_regressor_linear_fit_predict() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(60, 3))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.1 * X[:, 2] + 3.0

    reg = WeightRegressor(WeightRegressorConfig(model_type="linear"))
    reg.fit(X, y)
    pred = reg.predict(X)

    assert np.mean(np.abs(pred - y)) < 1e-6

