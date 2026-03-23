"""CARS wavelength selector for hyperspectral regression tasks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


@dataclass
class CARSConfig:
    n_mc: int = 50
    sample_ratio: float = 0.8
    cv_splits: int = 5
    pls_n_components: int = 8
    min_features: int = 2
    pls_scale: bool = False
    random_state: Optional[int] = None
    verbose: bool = False
    store_coef_history: bool = False


class CARSSelector:
    """Competitive Adaptive Reweighted Sampling selector."""

    def __init__(self, config: CARSConfig | None = None) -> None:
        self.config = config or CARSConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self._best_mask: Optional[np.ndarray] = None
        self._best_rmsecv: float = float("inf")
        self._best_r2cv: float = float("-inf")
        self._best_iteration: int = -1
        self._history: List[dict] = []

    def _build_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence[str]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is not None:
            groups_arr = np.asarray(groups)
            cv = GroupKFold(self.config.cv_splits)
            return list(cv.split(X, y, groups_arr))
        cv = KFold(self.config.cv_splits, shuffle=True, random_state=self.config.random_state)
        return list(cv.split(X, y))

    def _sample_rows(self, n_samples: int) -> np.ndarray:
        keep_n = max(self.config.cv_splits + 1, int(round(n_samples * self.config.sample_ratio)))
        keep_n = min(keep_n, n_samples)
        return self._rng.choice(n_samples, size=keep_n, replace=False)

    @staticmethod
    def _edf_keep_count(initial_count: int, iteration: int, n_iterations: int, min_features: int) -> int:
        if n_iterations <= 1:
            return max(min_features, initial_count)
        if initial_count <= min_features:
            return min_features
        k = math.log(initial_count / float(min_features)) / float(n_iterations - 1)
        count = int(round(initial_count * math.exp(-k * iteration)))
        return max(min_features, min(initial_count, count))

    def _fit_pls(self, X: np.ndarray, y: np.ndarray) -> PLSRegression:
        n_components = min(
            self.config.pls_n_components,
            max(1, X.shape[0] - 1),
            X.shape[1],
        )
        model = PLSRegression(n_components=n_components, scale=self.config.pls_scale)
        model.fit(X, y)
        return model

    def _evaluate_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        groups: Optional[Sequence[str]] = None,
    ) -> tuple[float, float]:
        if feature_indices.size == 0:
            return float("inf"), float("-inf")
        X_sel = X[:, feature_indices]
        model = self._fit_pls(X_sel, y)
        cv_splits = self._build_cv(X_sel, y, groups)
        rmse_scores = cross_val_score(
            model,
            X_sel,
            y,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=None,
        )
        r2_scores = cross_val_score(
            model,
            X_sel,
            y,
            scoring="r2",
            cv=cv_splits,
            n_jobs=None,
        )
        rmse_scores = np.asarray(rmse_scores, dtype=float)
        r2_scores = np.asarray(r2_scores, dtype=float)
        rmse_scores = rmse_scores[np.isfinite(rmse_scores)]
        r2_scores = r2_scores[np.isfinite(r2_scores)]
        if rmse_scores.size == 0 or r2_scores.size == 0:
            return float("inf"), float("-inf")
        return float(-rmse_scores.mean()), float(np.clip(r2_scores, -1e3, 1.0).mean())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence[str]] = None,
    ) -> "CARSSelector":
        n_samples, n_features = X.shape
        current_features = np.arange(n_features)
        self._history = []

        for iteration in range(self.config.n_mc):
            if current_features.size <= self.config.min_features:
                feature_indices = current_features.copy()
                rmsecv, r2cv = self._evaluate_subset(X, y, feature_indices, groups)
                self._record(iteration, feature_indices, rmsecv, r2cv, None)
                self._maybe_update_best(iteration, n_features, feature_indices, rmsecv, r2cv)
                continue

            sampled_rows = self._sample_rows(n_samples)
            sampled_X = X[np.ix_(sampled_rows, current_features)]
            sampled_y = y[sampled_rows]
            model = self._fit_pls(sampled_X, sampled_y)
            coef_abs = np.abs(model.coef_.ravel())
            ranked = np.argsort(coef_abs)[::-1]
            keep_count = self._edf_keep_count(
                initial_count=n_features,
                iteration=iteration,
                n_iterations=self.config.n_mc,
                min_features=self.config.min_features,
            )
            keep_count = min(keep_count, current_features.size)
            kept_ranked = ranked[:keep_count]
            feature_indices = np.sort(current_features[kept_ranked])
            rmsecv, r2cv = self._evaluate_subset(X, y, feature_indices, groups)
            coef_payload = coef_abs.tolist() if self.config.store_coef_history else None
            self._record(iteration, feature_indices, rmsecv, r2cv, coef_payload)
            self._maybe_update_best(iteration, n_features, feature_indices, rmsecv, r2cv)
            current_features = feature_indices

        if self._best_mask is None:
            raise RuntimeError("CARSSelector failed to produce a feature subset")
        return self

    def _record(
        self,
        iteration: int,
        feature_indices: np.ndarray,
        rmsecv: float,
        r2cv: float,
        coef_abs: Optional[list[float]],
    ) -> None:
        entry = {
            "iteration": iteration,
            "n_features": int(feature_indices.size),
            "rmsecv": rmsecv,
            "r2cv": r2cv,
            "selected_indices": feature_indices.tolist(),
        }
        if coef_abs is not None:
            entry["coef_abs"] = coef_abs
        self._history.append(entry)
        if self.config.verbose:
            print(
                f"CARS iter {iteration:3d}: n_features={feature_indices.size:3d}, "
                f"rmsecv={rmsecv:.4f}, r2cv={r2cv:.4f}"
            )

    def _maybe_update_best(
        self,
        iteration: int,
        total_features: int,
        feature_indices: np.ndarray,
        rmsecv: float,
        r2cv: float,
    ) -> None:
        if rmsecv < self._best_rmsecv:
            mask = np.zeros(total_features, dtype=bool)
            mask[feature_indices] = True
            self._best_mask = mask
            self._best_rmsecv = rmsecv
            self._best_r2cv = r2cv
            self._best_iteration = iteration

    def get_support(self) -> np.ndarray:
        if self._best_mask is None:
            raise RuntimeError("CARSSelector.fit must be called before get_support")
        return self._best_mask

    def selected_indices(self) -> np.ndarray:
        return np.where(self.get_support())[0]

    def selected_count(self) -> int:
        return int(self.get_support().sum())

    def best_rmsecv(self) -> float:
        return self._best_rmsecv

    def best_r2cv(self) -> float:
        return self._best_r2cv

    def best_iteration(self) -> int:
        return self._best_iteration

    def history(self) -> List[dict]:
        return self._history


def select_wavelengths_cars(
    X: np.ndarray,
    y: np.ndarray,
    config: CARSConfig | None = None,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, CARSSelector]:
    selector = CARSSelector(config)
    selector.fit(X, y, groups=groups)
    return selector.get_support(), selector


__all__ = ["CARSConfig", "CARSSelector", "select_wavelengths_cars"]
