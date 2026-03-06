"""Competitive Adaptive Reweighted Sampling (CARS) for wavelength selection.

This module implements the CARS algorithm for selecting key wavelengths 
in hyperspectral data analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score


@dataclass
class CARSConfig:
    n_mc: int = 50          # Number of Monte Carlo sampling runs
    n_splits: int = 5       # Number of CV splits for evaluation
    pls_n_components: int = 8  # Number of PLS components (can be optimized or fixed)
    sample_ratio: float = 0.8  # Ratio of samples used in each MC run
    random_state: Optional[int] = None
    verbose: bool = False


class CARSSelector:
    """Feature selector using Competitive Adaptive Reweighted Sampling (CARS)."""

    def __init__(self, config: CARSConfig | None = None) -> None:
        self.config = config or CARSConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self._best_mask: Optional[np.ndarray] = None
        self._best_score: float = float("inf") # RMSECV, lower is better
        self._history: List[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARSSelector":
        """Run CARS algorithm to select features."""
        n_samples, n_features = X.shape
        n_iter = self.config.n_mc
        
        # EDF parameters
        # We target ending with min 2 variables
        target_final_vars = 2
        
        # k = ln(n_features / 2) / (n_iter - 1)
        # a = 1 if we start with ratio=1 at iter 0
        if n_iter > 1:
            k = math.log(n_features / target_final_vars) / (n_iter - 1)
        else:
            k = 0
        a = 1.0
        
        current_vars = np.arange(n_features)
        
        RMSECV_list = []
        subset_list = []
        
        for i in range(n_iter):
            # --- 1. Model Sampling (Monte Carlo) ---
            n_train = int(self.config.sample_ratio * n_samples)
            rand_idx = self._rng.choice(n_samples, size=n_train, replace=False)
            
            # Subset for calibration
            X_cal = X[rand_idx][:, current_vars]
            y_cal = y[rand_idx]
            
            # --- 2. Train PLS & Get Weights ---
            # Using scale=True to ensure weights are comparable if features have diff scales
            # (though in spectra usually scales are similar, but variance differs)
            n_comp = min(self.config.pls_n_components, len(current_vars), n_train - 1)
            if n_comp < 1: n_comp = 1
            
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_cal, y_cal)
            
            # Use absolute regression coefficients as importance
            # Note: coef_ is (n_features, n_targets). y is (n_samples,).
            weights = np.abs(pls.coef_).flatten()
            
            # --- 3. Variable Selection (EDF + ARS) ---
            
            # Calculate ratio of variables to keep using EDF
            ratio = a * math.exp(-k * i)
            n_keep = int(round(ratio * n_features))
            # Ensure boundaries
            n_keep = max(target_final_vars, min(n_keep, len(current_vars)))
            
            # Normalize weights for ARS (probabilistic selection)
            w_sum = weights.sum()
            if w_sum == 0:
                probs = np.ones_like(weights) / len(weights)
            else:
                probs = weights / w_sum
            
            # ARS: Select n_keep variables based on weights
            # We use choice with replace=False to get exactly n_keep unique variables
            # prioritized by their weight.
            # If we used replace=True (strictly following "Sampling"), we'd need to handle
            # the unique set size possibly being < n_keep.
            # "Competitive" often implies the stronger ones survive.
            # Using replace=False with probabilities is a good approximation of "Survival of fittest".
            
            sel_relative_indices = self._rng.choice(
                len(current_vars), 
                size=n_keep, 
                replace=False, 
                p=probs
            )
            
            current_vars = current_vars[sel_relative_indices]
            current_vars.sort()
            
            # --- 4. Cross Validation Evaluation ---
            # Evaluate the CURRENT subset
            
            n_comp_cv = min(self.config.pls_n_components, len(current_vars))
            pls_cv = PLSRegression(n_components=n_comp_cv, scale=False)
            
            # KFold CV
            cv = KFold(self.config.n_splits, shuffle=True, random_state=self.config.random_state)
            
            # scoring='neg_root_mean_squared_error' returns negative values (higher is better)
            # so we negate it to get RMSE (lower is better)
            scores = cross_val_score(
                pls_cv, 
                X[:, current_vars], 
                y, 
                cv=cv, 
                scoring='neg_root_mean_squared_error'
            )
            rmsecv = -np.mean(scores)
            
            RMSECV_list.append(rmsecv)
            subset_list.append(current_vars.copy())
            
            if self.config.verbose and (i == 0 or (i + 1) % 5 == 0):
                print(f"[CARS] Iter {i+1}/{n_iter}: n_feat={len(current_vars)}, RMSECV={rmsecv:.4f}")
        
        # --- 5. Select Best Subset ---
        # Find iteration with minimum RMSECV
        best_run_idx = np.argmin(RMSECV_list)
        self._best_mask = np.zeros(n_features, dtype=bool)
        self._best_mask[subset_list[best_run_idx]] = True
        self._best_score = RMSECV_list[best_run_idx]
        
        self._history = [
            {"iter": i, "n_features": len(subset_list[i]), "rmsecv": RMSECV_list[i]}
            for i in range(len(RMSECV_list))
        ]
        
        if self.config.verbose:
            print(f"[CARS] Best subset found at iter {best_run_idx+1}: "
                  f"n_feat={self._best_mask.sum()}, RMSECV={self._best_score:.4f}")

        return self

    def get_support(self) -> np.ndarray:
        if self._best_mask is None:
            raise RuntimeError("CARSSelector.fit must be called before get_support")
        return self._best_mask

    def selected_indices(self) -> np.ndarray:
        return np.where(self.get_support())[0]