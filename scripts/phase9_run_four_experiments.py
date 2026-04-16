from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import os


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    scripts = [
        "phase9_run_baseline_training.py",
        "phase9_run_saans_training.py",
        "phase9_run_no_weighting_training.py",
        "phase9_run_alpha_ablation_training.py",
        "phase9_compare_runs.py",
    ]

    for script in scripts:
        print(f"=== Running {script} ===")
        subprocess.run([sys.executable, str(project_root / "scripts" / script)], check=True, env=env)


if __name__ == "__main__":
    main()