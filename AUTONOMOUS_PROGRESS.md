# Autonomous Progress Report

This file records what has been completed autonomously in the repository and what remains compute-bound.

## What has been completed

### Phase 1
- created project scaffold
- added scheduler core modules
- added smoke demo and basic scheduler tests

### Phase 2
- added baseline integration interfaces
- added a mock baseline adapter
- added training/evaluation stubs for the abstract interface

### Phase 3
- vendored the public EDM molecular diffusion baseline
- added an EDM QM9 runtime wrapper under `src/saans_project/baseline/edm_qm9.py`
- added patching logic for known legacy issues:
  - outdated Figshare URLs
  - NumPy deprecations (`np.int`, `np.bool`)
- added QM9 preparation script

### Phase 4
- added baseline smoke script that builds the EDM model and runs a train/eval batch step

### Phase 5
- added timestep diagnostic aggregation utilities
- added a diagnostic script that inspects timestep-indexed loss behavior on EDM batches

### Phase 6
- added a preliminary SAANS-on-EDM step implementation
- added a smoke script that samples timestep bins, computes importance weights, and updates tracker state

### Phase 7
- added a toy study script showing adaptive probabilities and a variance proxy
- expanded test coverage across scheduler, baseline, runtime, and smoke behaviors

### Phase 8
- added experiment config files
- added an experiment command matrix script to drive the current validation pipeline
- added short-run experiment scripts for:
  - baseline EDM on QM9
  - SAANS-on-EDM on QM9
  - baseline-vs-SAANS comparison reporting

### Phase 8 artifacts currently produced
- `artifacts/phase8/baseline_short_run.json`
- `artifacts/phase8/saans_short_run.json`
- `artifacts/phase8/comparison_report.md`

Current short-run comparison snapshot:
- baseline eval mean: `2.920835`
- saans eval mean: `2.630423`
- delta (saans - baseline): `-0.290412`

These are smoke-test numbers only, not publication-grade results.

### Phase 9
- documented current status in this file
- updated the phase plan to reflect the actual implementation state
- added a single-command smoke runner

## Verified runnable commands

From the repository root:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/smoke_demo.py
PYTHONPATH=src python scripts/phase2_demo.py
PYTHONPATH=src python scripts/phase3_prepare_qm9.py
PYTHONPATH=src python scripts/phase4_baseline_smoke.py
PYTHONPATH=src python scripts/phase5_timestep_diagnostics.py
PYTHONPATH=src python scripts/phase6_saans_smoke.py
PYTHONPATH=src python scripts/phase7_toy_study.py
PYTHONPATH=src python scripts/phase8_experiment_matrix.py
PYTHONPATH=src python scripts/phase8_run_baseline_short.py
PYTHONPATH=src python scripts/phase8_run_saans_short.py
PYTHONPATH=src python scripts/phase8_compare_short_runs.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

Or use:

```bash
make all-smokes
make test
```

At the current checkpoint:
- all smoke scripts run successfully
- full unit test suite passes (`15` tests)

## What still remains

These remaining tasks are not blocked by missing repository structure. They are mostly blocked by compute/time:

1. run long baseline training instead of smoke batches
2. run real SAANS training over many steps/epochs
3. collect statistically meaningful multi-seed results
4. execute ablations over scheduler parameters and hardness variants
5. generate final plots/tables from longer experiments
6. optionally test on a harder dataset

## Honest status summary

The repository is now in a state where the remaining work is primarily:

- experiment execution,
- debugging under longer training,
- and empirical validation.

In other words, the project is no longer blocked by missing engineering scaffolding.