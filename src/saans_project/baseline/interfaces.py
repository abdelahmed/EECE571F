from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MoleculeBatch:
    coord_targets: list[float]
    feat_targets: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.coord_targets)


@dataclass
class NoisyBatch:
    coord_inputs: list[float]
    feat_inputs: list[float]
    timesteps: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutputs:
    pred_coord: list[float]
    pred_feat: list[float]


@dataclass
class LossBreakdown:
    total: float
    coord: float
    feat: float
    per_sample_total: list[float]
    per_sample_coord: list[float]
    per_sample_feat: list[float]


@runtime_checkable
class DiffusionBaselineAdapter(Protocol):
    def name(self) -> str:
        ...

    def sample_timesteps(self, batch_size: int) -> list[float]:
        ...

    def corrupt_batch(self, clean_batch: MoleculeBatch, timesteps: list[float]) -> NoisyBatch:
        ...

    def forward(self, noisy_batch: NoisyBatch) -> ModelOutputs:
        ...

    def loss_fn(self, outputs: ModelOutputs, clean_batch: MoleculeBatch) -> LossBreakdown:
        ...
