from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "smoke_demo.py",
        "phase2_demo.py",
        "phase3_prepare_qm9.py",
        "phase4_baseline_smoke.py",
        "phase5_timestep_diagnostics.py",
        "phase6_saans_smoke.py",
        "phase7_toy_study.py",
        "phase8_experiment_matrix.py",
        "phase8_run_baseline_short.py",
        "phase8_run_saans_short.py",
        "phase8_compare_short_runs.py",
    ]

    for script in scripts:
        print(f"=== Running {script} ===")
        subprocess.run(
            [sys.executable, str(project_root / "scripts" / script)],
            check=True,
            env={**dict(), **{"PYTHONPATH": str(project_root / 'src')}},
        )


if __name__ == "__main__":
    main()