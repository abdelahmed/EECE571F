from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .bins import BinManager
from .tracker import EMAHardnessTracker


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("cannot normalize values with non-positive total")
    return [value / total for value in values]


def _sample_index(probabilities: list[float], rng: Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probabilities):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probabilities) - 1


@dataclass
class BaselineBinSampler:
    bins: BinManager
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def probabilities(self) -> list[float]:
        return list(self.bins.baseline_masses)

    def sample_bin(self) -> int:
        return _sample_index(self.probabilities(), self.rng)


@dataclass
class AdaptiveBinSampler:
    bins: BinManager
    tracker: EMAHardnessTracker
    alpha: float = 1.0
    epsilon: float = 1e-8
    rho: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        self.rng = Random(self.seed)

    def probabilities(self) -> list[float]:
        p0 = self.bins.baseline_masses
        scores = [(self.epsilon + hardness) ** self.alpha for hardness in self.tracker.values]
        raw = _normalize([base * score for base, score in zip(p0, scores)])
        return [(1.0 - self.rho) * q + self.rho * base for q, base in zip(raw, p0)]

    def importance_weights(self) -> list[float]:
        q = self.probabilities()
        p0 = self.bins.baseline_masses
        return [base / adapted for base, adapted in zip(p0, q)]

    def sample_bin(self) -> int:
        return _sample_index(self.probabilities(), self.rng)
