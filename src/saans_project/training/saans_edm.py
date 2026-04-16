from __future__ import annotations

from dataclasses import dataclass

import torch

from saans_project.baseline.edm_qm9 import EDMQM9Runtime
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker, combined_hardness, coord_only_hardness


@dataclass
class SAANSEDMStepResult:
    timestep_normalized: list[float]
    bin_indices: list[int]
    importance_weights: list[float]
    adaptive_probabilities: list[float]
    per_sample_coord: list[float]
    per_sample_feat: list[float]
    weighted_loss: float


def compute_saans_edm_step(
    runtime: EDMQM9Runtime,
    data: dict,
    bin_manager: BinManager,
    tracker: EMAHardnessTracker,
    adaptive_sampler: AdaptiveBinSampler,
) -> SAANSEDMStepResult:
    if runtime.model is None:
        runtime.build_model()

    model = runtime.model
    x, h, node_mask, edge_mask, context = runtime.unpack_batch(data)
    x, h, _ = model.normalize(x, h, node_mask)
    xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
    bs = x.size(0)

    bin_indices = [adaptive_sampler.sample_bin() for _ in range(bs)]
    t_values = [0.5 * sum(bin_manager.interval(idx)) for idx in bin_indices]
    t = torch.tensor(t_values, device=x.device, dtype=x.dtype).view(bs, 1)
    s = torch.clamp(t - 1.0 / float(model.T), min=0.0)

    gamma_s = model.inflate_batch_array(model.gamma(s), x)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    eps = model.sample_combined_position_feature_noise(bs, x.size(1), node_mask)
    z_t = alpha_t * xh + sigma_t * eps
    net_out = model.phi(z_t, t, node_mask, edge_mask, context)

    coord_dims = model.n_dims
    coord_error = ((eps[:, :, :coord_dims] - net_out[:, :, :coord_dims]) ** 2).reshape(bs, -1).mean(dim=1)
    feat_error = ((eps[:, :, coord_dims:] - net_out[:, :, coord_dims:]) ** 2).reshape(bs, -1).mean(dim=1)
    total_error = 0.5 * (coord_error + feat_error)

    adaptive_probabilities = adaptive_sampler.probabilities()
    importance_weights_by_bin = adaptive_sampler.importance_weights()
    sample_weights = torch.tensor([importance_weights_by_bin[idx] for idx in bin_indices], device=x.device, dtype=x.dtype)
    weighted_loss = torch.mean(sample_weights * total_error)

    observations: dict[int, list[float]] = {}
    for idx, coord_val, feat_val in zip(bin_indices, coord_error.detach().cpu().tolist(), feat_error.detach().cpu().tolist()):
        if adaptive_sampler.alpha > 0 and runtime.config.scheduler.hardness_type == "coord_plus_feat":
            hardness = combined_hardness(coord_val, feat_val, runtime.config.scheduler.lambda_coord, runtime.config.scheduler.lambda_feat)
        else:
            hardness = coord_only_hardness(coord_val)
        observations.setdefault(idx, []).append(hardness)
    tracker.update(observations)

    return SAANSEDMStepResult(
        timestep_normalized=t_values,
        bin_indices=bin_indices,
        importance_weights=[float(x) for x in sample_weights.detach().cpu().tolist()],
        adaptive_probabilities=[float(x) for x in adaptive_probabilities],
        per_sample_coord=[float(x) for x in coord_error.detach().cpu().tolist()],
        per_sample_feat=[float(x) for x in feat_error.detach().cpu().tolist()],
        weighted_loss=float(weighted_loss.item()),
    )


def train_saans_batch_step(
    runtime: EDMQM9Runtime,
    data: dict,
    bin_manager: BinManager,
    tracker: EMAHardnessTracker,
    adaptive_sampler: AdaptiveBinSampler,
    use_importance_weights: bool = True,
) -> SAANSEDMStepResult:
    if runtime.model is None or runtime.optim is None:
        runtime.build_model()

    runtime.model.train()
    runtime.optim.zero_grad()
    result = compute_saans_edm_step(runtime, data, bin_manager, tracker, adaptive_sampler)

    # Recompute differentiable weighted loss for the optimizer step.
    model = runtime.model
    x, h, node_mask, edge_mask, context = runtime.unpack_batch(data)
    x, h, _ = model.normalize(x, h, node_mask)
    xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
    bs = x.size(0)

    t = torch.tensor(result.timestep_normalized, device=x.device, dtype=x.dtype).view(bs, 1)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)
    eps = model.sample_combined_position_feature_noise(bs, x.size(1), node_mask)
    z_t = alpha_t * xh + sigma_t * eps
    net_out = model.phi(z_t, t, node_mask, edge_mask, context)

    coord_dims = model.n_dims
    coord_error = ((eps[:, :, :coord_dims] - net_out[:, :, :coord_dims]) ** 2).reshape(bs, -1).mean(dim=1)
    feat_error = ((eps[:, :, coord_dims:] - net_out[:, :, coord_dims:]) ** 2).reshape(bs, -1).mean(dim=1)
    total_error = 0.5 * (coord_error + feat_error)
    if use_importance_weights:
        sample_weights = torch.tensor(result.importance_weights, device=x.device, dtype=x.dtype)
    else:
        sample_weights = torch.ones_like(total_error)
    weighted_loss = torch.mean(sample_weights * total_error)
    weighted_loss.backward()
    runtime.optim.step()
    return result
