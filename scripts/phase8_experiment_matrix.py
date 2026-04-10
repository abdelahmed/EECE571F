from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    commands = [
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase3_prepare_qm9.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase4_baseline_smoke.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase5_timestep_diagnostics.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase6_saans_smoke.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase7_toy_study.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase8_run_baseline_short.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase8_run_saans_short.py",
        f"cd {project_root} && source .venv/bin/activate && PYTHONPATH=src python scripts/phase8_compare_short_runs.py",
    ]
    print("Planned experiment / validation command matrix:")
    for command in commands:
        print(command)


if __name__ == "__main__":
    main()
