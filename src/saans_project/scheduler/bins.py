from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right


@dataclass
class BinManager:
    num_bins: int
    t_min: float = 0.0
    t_max: float = 1.0
    baseline_masses: list[float] | None = None
    edges: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min")

        width = (self.t_max - self.t_min) / self.num_bins
        self.edges = [self.t_min + i * width for i in range(self.num_bins + 1)]

        if self.baseline_masses is None:
            self.baseline_masses = [1.0 / self.num_bins] * self.num_bins
        else:
            if len(self.baseline_masses) != self.num_bins:
                raise ValueError("baseline_masses must match num_bins")
            total = sum(self.baseline_masses)
            if total <= 0:
                raise ValueError("baseline_masses must sum to a positive value")
            self.baseline_masses = [mass / total for mass in self.baseline_masses]

    def bin_index(self, t: float) -> int:
        clipped = min(max(t, self.t_min), self.t_max - 1e-12)
        idx = bisect_right(self.edges, clipped) - 1
        return max(0, min(idx, self.num_bins - 1))

    def interval(self, idx: int) -> tuple[float, float]:
        if not 0 <= idx < self.num_bins:
            raise IndexError("bin index out of range")
        return self.edges[idx], self.edges[idx + 1]
