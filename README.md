# SAANS Project Scaffold

This repository contains the implementation scaffold for the SAANS project:

> **Symmetry-Safe Adaptive Noise Scheduling for E(3)-Equivariant Diffusion Training**

At the moment, this repository is progressing through **Phase 3+** of the execution plan recorded in `PROJECT_PHASE_PLAN.md`.

## What exists now

- markdown phase plan
- Python package scaffold under `src/saans_project/`
- scheduler core modules
- baseline integration interfaces and a mock baseline wrapper
- vendored public EDM baseline under `external/e3_diffusion_for_molecules/`
- EDM QM9 runtime wrapper and smoke scripts
- timestep diagnostics aggregation utilities
- preliminary SAANS-on-EDM smoke step implementation
- training/evaluation entrypoint stubs
- smoke demo script
- autonomous progress report and packaged smoke command flow
- unit tests for key scheduler invariants

## Quick start

```bash
PYTHONPATH=src python3 scripts/smoke_demo.py
PYTHONPATH=src python3 scripts/phase2_demo.py
PYTHONPATH=src python3 scripts/phase3_prepare_qm9.py
PYTHONPATH=src python3 scripts/phase4_baseline_smoke.py
PYTHONPATH=src python3 scripts/phase5_timestep_diagnostics.py
PYTHONPATH=src python3 scripts/phase6_saans_smoke.py
PYTHONPATH=src python3 scripts/phase7_toy_study.py
PYTHONPATH=src python3 scripts/phase8_experiment_matrix.py
PYTHONPATH=src python3 scripts/phase8_run_baseline_short.py
PYTHONPATH=src python3 scripts/phase8_run_saans_short.py
PYTHONPATH=src python3 scripts/phase8_compare_short_runs.py
PYTHONPATH=src python3 scripts/run_all_smokes.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Or simply:

```bash
make all-smokes
make test
```

## Next major milestone

After the current scaffold work, the remaining heavy work is long-running experiment execution and deeper method iteration.
