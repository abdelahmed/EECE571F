from __future__ import annotations

from pathlib import Path

import torch

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.experiments import ShortRunResult, ensure_artifact_dir, save_result
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker
from saans_project.training import train_saans_batch_step


TRAIN_STEPS = 3
EVAL_BATCHES = 2


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

    train_metrics: list[float] = []
    tracker_snapshots: list[list[float]] = []
    for step, batch in enumerate(runtime.dataloaders["train"]):
        result = train_saans_batch_step(runtime, batch, bins, tracker, sampler)
        train_metrics.append(result.weighted_loss)
        tracker_snapshots.append(list(tracker.values))
        if step + 1 >= TRAIN_STEPS:
            break

    eval_metrics: list[float] = []
    for step, batch in enumerate(runtime.dataloaders["valid"]):
        eval_metrics.append(runtime.compute_batch_nll(batch))
        if step + 1 >= EVAL_BATCHES:
            break

    result = ShortRunResult(
        mode="saans_short",
        train_steps=len(train_metrics),
        eval_batches=len(eval_metrics),
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        summary={
            "train_last": train_metrics[-1] if train_metrics else 0.0,
            "eval_mean": sum(eval_metrics) / len(eval_metrics) if eval_metrics else 0.0,
        },
        extra={"tracker_snapshots": tracker_snapshots},
    )

    out_dir = ensure_artifact_dir(project_root)
    out_path = out_dir / "saans_short_run.json"
    save_result(result, out_path)
    print(f"Saved SAANS short-run artifact to {out_path}")
    print(result.to_dict())


if __name__ == "__main__":
    main()
