"""Genetic algorithm-based wavelength selector for PLSR models.

This module implements a lightweight genetic algorithm tailored for
hyperspectral wavelength selection, inspired by the approach used in
《基于高光谱成像的油茶籽含油率检测方法》.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score


@dataclass
class GAConfig:
    population_size: int = 12
    generations: int = 10
    crossover_rate: float = 0.85
    mutation_rate: float = 0.04
    elite_count: int = 1
    min_features: int = 8
    max_features: int = 35
    target_features: int = 18
    random_state: Optional[int] = None
    cv_splits: int = 3
    patience: int = 4


class GeneticAlgorithmSelector:
    """Feature selector that searches wavelength subsets with a GA."""

    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config or GAConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self._best_mask: Optional[np.ndarray] = None
        self._best_score: float = float("-inf")

    @staticmethod
    def _ensure_bounds(mask: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
        idx = np.where(mask)[0]
        if idx.size < cfg.min_features:
            add_idx = rng.choice(np.where(~mask)[0], size=cfg.min_features - idx.size, replace=False)
            mask[add_idx] = True
        elif idx.size > cfg.max_features:
            drop_idx = rng.choice(idx, size=idx.size - cfg.max_features, replace=False)
            mask[drop_idx] = False
        return mask

    @staticmethod
    def _pls_components(feature_count: int) -> int:
        # limit components to maintain numerical stability
        if feature_count <= 2:
            return 1
        return min(10, feature_count // 2)

    def _evaluate(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        selected = np.where(mask)[0]
        if selected.size == 0:
            return float("-inf")
        n_components = self._pls_components(selected.size)
        model = PLSRegression(n_components=n_components, scale=False)
        cv = KFold(self.config.cv_splits, shuffle=True, random_state=self.config.random_state)
        scores = cross_val_score(model, X[:, selected], y, scoring="r2", cv=cv, n_jobs=None)
        mean_score = scores.mean()
        # soft penalty to bias towards target feature count
        penalty = 0.004 * abs(selected.size - self.config.target_features)
        return mean_score - penalty

    def _initial_population(self, n_features: int) -> np.ndarray:
        pop = np.zeros((self.config.population_size, n_features), dtype=bool)
        for i in range(self.config.population_size):
            active = self._rng.integers(self.config.min_features, self.config.max_features + 1)
            idx = self._rng.choice(n_features, size=active, replace=False)
            pop[i, idx] = True
        return pop

    def _select_parents(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # tournament selection
        def pick_one() -> np.ndarray:
            contenders = self._rng.choice(population.shape[0], size=3, replace=False)
            best_idx = contenders[np.argmax(fitness[contenders])]
            return population[best_idx]

        return pick_one(), pick_one()

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._rng.random() > self.config.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        cut = self._rng.integers(1, parent_a.size - 1)
        child_a = np.concatenate([parent_a[:cut], parent_b[cut:]])
        child_b = np.concatenate([parent_b[:cut], parent_a[cut:]])
        return child_a, child_b

    def _mutate(self, mask: np.ndarray) -> np.ndarray:
        mutation_flags = self._rng.random(mask.size) < self.config.mutation_rate
        mask ^= mutation_flags
        return mask

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeneticAlgorithmSelector":
        n_features = X.shape[1]
        population = self._initial_population(n_features)
        fitness = np.array([self._evaluate(X, y, ind) for ind in population])

        best_idx = np.argmax(fitness)
        self._best_mask = population[best_idx].copy()
        self._best_score = fitness[best_idx]

        no_improve = 0
        for gen in range(self.config.generations):
            elite_indices = np.argsort(fitness)[-self.config.elite_count :]
            elites = population[elite_indices]

            new_population: List[np.ndarray] = [elites[i].copy() for i in range(elites.shape[0])]
            while len(new_population) < self.config.population_size:
                parent_a, parent_b = self._select_parents(population, fitness)
                child_a, child_b = self._crossover(parent_a, parent_b)
                child_a = self._mutate(child_a)
                child_b = self._mutate(child_b)
                child_a = self._ensure_bounds(child_a, self.config, self._rng)
                child_b = self._ensure_bounds(child_b, self.config, self._rng)
                new_population.extend([child_a, child_b])

            population = np.array(new_population[: self.config.population_size], dtype=bool)
            fitness = np.array([self._evaluate(X, y, ind) for ind in population])

            gen_best_idx = np.argmax(fitness)
            gen_best_score = fitness[gen_best_idx]
            if gen_best_score > self._best_score:
                self._best_score = gen_best_score
                self._best_mask = population[gen_best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            if self.config.patience and no_improve >= self.config.patience:
                break

        return self

    def get_support(self) -> np.ndarray:
        if self._best_mask is None:
            raise RuntimeError("GeneticAlgorithmSelector.fit must be called before get_support")
        return self._best_mask

    def selected_indices(self) -> np.ndarray:
        return np.where(self.get_support())[0]

    def selected_count(self) -> int:
        return self.get_support().sum()

    def best_score(self) -> float:
        return self._best_score


def select_wavelengths(
    X: np.ndarray,
    y: np.ndarray,
    config: GAConfig | None = None,
) -> Tuple[np.ndarray, GeneticAlgorithmSelector]:
    selector = GeneticAlgorithmSelector(config)
    selector.fit(X, y)
    return selector.get_support(), selector


__all__ = ["GAConfig", "GeneticAlgorithmSelector", "select_wavelengths"]
