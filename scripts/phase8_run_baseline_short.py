from __future__ import annotations

from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.experiments import ShortRunResult, ensure_artifact_dir, save_result


TRAIN_STEPS = 3
EVAL_BATCHES = 2


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_smoke.toml")
    runtime = EDMQM9Runtime(cfg, project_root=project_root, device="cpu").prepare()

    train_metrics: list[float] = []
    for step, batch in enumerate(runtime.dataloaders["train"]):
        train_metrics.append(runtime.train_batch_step(batch))
        if step + 1 >= TRAIN_STEPS:
            break

    eval_metrics: list[float] = []
    for step, batch in enumerate(runtime.dataloaders["valid"]):
        eval_metrics.append(runtime.compute_batch_nll(batch))
        if step + 1 >= EVAL_BATCHES:
            break

    result = ShortRunResult(
        mode="baseline_short",
        train_steps=len(train_metrics),
        eval_batches=len(eval_metrics),
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        summary={
            "train_last": train_metrics[-1] if train_metrics else 0.0,
            "eval_mean": sum(eval_metrics) / len(eval_metrics) if eval_metrics else 0.0,
        },
    )

    out_dir = ensure_artifact_dir(project_root)
    out_path = out_dir / "baseline_short_run.json"
    save_result(result, out_path)
    print(f"Saved baseline short-run artifact to {out_path}")
    print(result.to_dict())


if __name__ == "__main__":
    main()
