from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_smoke.toml")
    runtime = EDMQM9Runtime(cfg, project_root=project_root, device="cpu").prepare()
    batch = runtime.first_batch("train")
    train_nll = runtime.train_batch_step(batch)
    eval_nll = runtime.compute_batch_nll(batch)
    print("Baseline train-step NLL:", round(train_nll, 6))
    print("Baseline eval NLL:", round(eval_nll, 6))


if __name__ == "__main__":
    main()
