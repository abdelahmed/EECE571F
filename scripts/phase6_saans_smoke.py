from pathlib import Path

import torch

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker
from saans_project.training import compute_saans_edm_step


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_saans_smoke.toml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    runtime = EDMQM9Runtime(cfg, project_root=project_root, device=device).prepare()

    bins = BinManager(num_bins=cfg.scheduler.num_bins)
    tracker = EMAHardnessTracker(num_bins=cfg.scheduler.num_bins, beta=cfg.scheduler.ema_beta)
    sampler = AdaptiveBinSampler(
        bins=bins,
        tracker=tracker,
        alpha=cfg.scheduler.alpha,
        rho=cfg.scheduler.baseline_mix_rho,
    )

    batch = runtime.first_batch("train")
    result = compute_saans_edm_step(runtime, batch, bins, tracker, sampler)
    print("Weighted SAANS loss:", round(result.weighted_loss, 6))
    print("Sampled bins:", result.bin_indices)
    print("Importance weights:", [round(x, 6) for x in result.importance_weights])


if __name__ == "__main__":
    main()
