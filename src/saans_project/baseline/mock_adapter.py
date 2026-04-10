from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .interfaces import LossBreakdown, ModelOutputs, MoleculeBatch, NoisyBatch


@dataclass
class MockEquivariantDiffusionAdapter:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def name(self) -> str:
        return "mock_equivariant_diffusion"

    def make_synthetic_batch(self, batch_size: int = 4) -> MoleculeBatch:
        coord_targets = [0.5 + 0.2 * i for i in range(batch_size)]
        feat_targets = [1.0 + 0.1 * i for i in range(batch_size)]
        return MoleculeBatch(coord_targets=coord_targets, feat_targets=feat_targets)

    def sample_timesteps(self, batch_size: int) -> list[float]:
        return [self.rng.random() for _ in range(batch_size)]

    def corrupt_batch(self, clean_batch: MoleculeBatch, timesteps: list[float]) -> NoisyBatch:
        coord_inputs = [coord + 0.5 * t for coord, t in zip(clean_batch.coord_targets, timesteps)]
        feat_inputs = [feat + 0.25 * t for feat, t in zip(clean_batch.feat_targets, timesteps)]
        return NoisyBatch(coord_inputs=coord_inputs, feat_inputs=feat_inputs, timesteps=timesteps)

    def forward(self, noisy_batch: NoisyBatch) -> ModelOutputs:
        pred_coord = [coord - 0.4 * t for coord, t in zip(noisy_batch.coord_inputs, noisy_batch.timesteps)]
        pred_feat = [feat - 0.2 * t for feat, t in zip(noisy_batch.feat_inputs, noisy_batch.timesteps)]
        return ModelOutputs(pred_coord=pred_coord, pred_feat=pred_feat)

    def loss_fn(self, outputs: ModelOutputs, clean_batch: MoleculeBatch) -> LossBreakdown:
        per_coord = [(pred - target) ** 2 for pred, target in zip(outputs.pred_coord, clean_batch.coord_targets)]
        per_feat = [(pred - target) ** 2 for pred, target in zip(outputs.pred_feat, clean_batch.feat_targets)]
        per_total = [c + f for c, f in zip(per_coord, per_feat)]
        n = max(len(per_total), 1)
        return LossBreakdown(
            total=sum(per_total) / n,
            coord=sum(per_coord) / n,
            feat=sum(per_feat) / n,
            per_sample_total=per_total,
            per_sample_coord=per_coord,
            per_sample_feat=per_feat,
        )
