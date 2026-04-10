from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EMAHardnessTracker:
    num_bins: int
    beta: float = 0.95
    initial_value: float = 1.0
    values: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.values = [float(self.initial_value) for _ in range(self.num_bins)]

    def update(self, bin_to_observations: dict[int, list[float]]) -> None:
        for bin_idx, observations in bin_to_observations.items():
            if not observations:
                continue
            mean_value = sum(observations) / len(observations)
            self.values[bin_idx] = self.beta * self.values[bin_idx] + (1.0 - self.beta) * mean_value

    def reset(self) -> None:
        self.values = [float(self.initial_value) for _ in range(self.num_bins)]
