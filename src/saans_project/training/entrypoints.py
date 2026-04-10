from __future__ import annotations

from dataclasses import dataclass

from saans_project.baseline import DiffusionBaselineAdapter, LossBreakdown, MoleculeBatch
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker, combined_hardness, coord_only_hardness


@dataclass
class TrainingStepResult:
    timesteps: list[float]
    bin_indices: list[int]
    sample_weights: list[float]
    losses: LossBreakdown
    weighted_total: float


@dataclass
class EvaluationResult:
    timesteps: list[float]
    losses: LossBreakdown


def _midpoint(interval: tuple[float, float]) -> float:
    lo, hi = interval
    return 0.5 * (lo + hi)


def run_training_step(
    adapter: DiffusionBaselineAdapter,
    clean_batch: MoleculeBatch,
    bin_manager: BinManager,
    adaptive_sampler: AdaptiveBinSampler | None = None,
    tracker: EMAHardnessTracker | None = None,
    hardness_type: str = "coord_only",
    lambda_coord: float = 1.0,
    lambda_feat: float = 1.0,
) -> TrainingStepResult:
    if adaptive_sampler is None:
        timesteps = adapter.sample_timesteps(clean_batch.batch_size)
        bin_indices = [bin_manager.bin_index(t) for t in timesteps]
        sample_weights = [1.0] * clean_batch.batch_size
    else:
        timestep_probs = adaptive_sampler.importance_weights()
        bin_indices = [adaptive_sampler.sample_bin() for _ in range(clean_batch.batch_size)]
        timesteps = [_midpoint(bin_manager.interval(bin_idx)) for bin_idx in bin_indices]
        sample_weights = [timestep_probs[bin_idx] for bin_idx in bin_indices]

    noisy_batch = adapter.corrupt_batch(clean_batch, timesteps)
    outputs = adapter.forward(noisy_batch)
    losses = adapter.loss_fn(outputs, clean_batch)

    weighted_total = sum(w * loss for w, loss in zip(sample_weights, losses.per_sample_total)) / max(clean_batch.batch_size, 1)

    if tracker is not None:
        observations: dict[int, list[float]] = {}
        for idx, (coord_loss, feat_loss) in enumerate(zip(losses.per_sample_coord, losses.per_sample_feat)):
            bin_idx = bin_indices[idx]
            if hardness_type == "coord_plus_feat":
                hardness = combined_hardness(coord_loss, feat_loss, lambda_coord=lambda_coord, lambda_feat=lambda_feat)
            else:
                hardness = coord_only_hardness(coord_loss)
            observations.setdefault(bin_idx, []).append(hardness)
        tracker.update(observations)

    return TrainingStepResult(
        timesteps=timesteps,
        bin_indices=bin_indices,
        sample_weights=sample_weights,
        losses=losses,
        weighted_total=weighted_total,
    )


def run_evaluation_stub(
    adapter: DiffusionBaselineAdapter,
    clean_batch: MoleculeBatch,
) -> EvaluationResult:
    timesteps = adapter.sample_timesteps(clean_batch.batch_size)
    noisy_batch = adapter.corrupt_batch(clean_batch, timesteps)
    outputs = adapter.forward(noisy_batch)
    losses = adapter.loss_fn(outputs, clean_batch)
    return EvaluationResult(timesteps=timesteps, losses=losses)
