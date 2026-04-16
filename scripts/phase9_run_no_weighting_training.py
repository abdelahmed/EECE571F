from __future__ import annotations

from pathlib import Path
import json

import torch

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker
from saans_project.training import train_saans_batch_step


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_saans_smoke.toml")

    cfg.exp_name = "kaggle_saans_no_weighting"
    cfg.batch_size = 32
    cfg.num_workers = 2
    cfg.test_epochs = 1
    cfg.n_report_steps = 50
    cfg.visualize_every_batch = int(1e8)
    cfg.no_wandb = True
    cfg.save_model = True

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

    out_dir = project_root / "artifacts" / "phase9"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    tracker_history: list[list[float]] = []
    max_steps = 1000

    for step, batch in enumerate(runtime.dataloaders["train"]):
        result = train_saans_batch_step(runtime, batch, bins, tracker, sampler, use_importance_weights=False)
        unweighted_loss = sum(result.per_sample_coord[i] * 0.5 + result.per_sample_feat[i] * 0.5 for i in range(len(result.per_sample_coord))) / len(result.per_sample_coord)
        train_losses.append(float(unweighted_loss))
        tracker_history.append(list(tracker.values))

        if step % 50 == 0:
            print(
                f"No-weight step {step}: unweighted_loss={unweighted_loss:.4f}, "
                f"first_probs={[round(x, 4) for x in result.adaptive_probabilities[:4]]}"
            )

        if step + 1 >= max_steps:
            break

    eval_metrics: list[float] = []
    for i, batch in enumerate(runtime.dataloaders["valid"]):
        eval_metrics.append(runtime.compute_batch_nll(batch))
        if i >= 49:
            break

    summary = {
        "mode": "kaggle_saans_no_weighting",
        "train_steps": len(train_losses),
        "train_last": train_losses[-1] if train_losses else None,
        "train_mean_last_50": sum(train_losses[-50:]) / min(50, len(train_losses)) if train_losses else None,
        "eval_mean": sum(eval_metrics) / len(eval_metrics) if eval_metrics else None,
        "eval_batches": len(eval_metrics),
    }

    with open(out_dir / "kaggle_saans_no_weighting_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "train_losses": train_losses,
                "tracker_history": tracker_history,
            },
            handle,
            indent=2,
        )

    print("Saved no-weighting summary to:", out_dir / "kaggle_saans_no_weighting_summary.json")
    print(summary)


if __name__ == "__main__":
    main()