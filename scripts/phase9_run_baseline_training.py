from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    edm_root = project_root / "external" / "e3_diffusion_for_molecules"
    out_dir = project_root / "artifacts" / "phase9"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_name = "kaggle_baseline_fixed"
    command = [
        sys.executable,
        "main_qm9.py",
        "--exp_name", exp_name,
        "--n_epochs", "10",
        "--batch_size", "16",
        "--num_workers", "2",
        "--test_epochs", "1",
        "--n_stability_samples", "100",
        "--no_wandb",
    ]

    print("Running baseline command:")
    print(" ".join(command))
    subprocess.run(command, cwd=edm_root, check=True)

    summary = {
        "mode": "kaggle_baseline_training",
        "exp_name": exp_name,
        "output_dir": str(edm_root / "outputs" / exp_name),
    }

    with open(out_dir / "kaggle_baseline_training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved baseline training summary to:", out_dir / "kaggle_baseline_training_summary.json")
    print(summary)


if __name__ == "__main__":
    main()