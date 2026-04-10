# SAANS Project Phase Plan

This file is the execution-order source of truth for completing the project.

## Rule of execution

Work strictly in phase order. Do not begin a later phase unless the current phase has met its exit criteria or an explicit blocking note has been recorded.

---

## Phase 1 — Workspace bootstrap and scheduler core scaffold

### Goal
Create the actual code repository structure for the project so implementation can proceed in an orderly way.

### Deliverables
- project metadata files (`README.md`, `pyproject.toml`, `.gitignore`, `Makefile`)
- config directory with baseline and adaptive scheduler configs
- Python package scaffold under `src/saans_project/`
- initial scheduler core modules:
  - timestep bin manager
  - EMA hardness tracker
  - hardness builders
  - adaptive sampler
- simple smoke demo script
- unit tests for scheduler invariants

### Exit criteria
- repository has a runnable package scaffold
- smoke demo runs successfully
- unit tests pass

---

## Phase 2 — Baseline integration interface

### Goal
Prepare a stable adapter layer so the project can plug into a real E(3)-equivariant molecular diffusion baseline.

### Deliverables
- baseline adapter interface
- placeholder or real baseline wrapper
- training/evaluation entrypoint stubs
- clear config contract for baseline-vs-SAANS runs

### Exit criteria
- one consistent API exists for `forward`, `loss_fn`, and timestep handling
- code structure does not depend on hardcoded baseline internals everywhere

---

## Phase 3 — Data acquisition and preprocessing pipeline

### Goal
Obtain and prepare QM9 for experiments.

### Deliverables
- documented data path
- preprocessing or baseline-provided dataset hookup
- verified train/valid/test split availability

### Exit criteria
- QM9 can be loaded reproducibly
- a single batch can be inspected successfully from the training pipeline

---

## Phase 4 — Baseline reproduction

### Goal
Run the unmodified baseline and verify that the project has a trustworthy starting point.

### Deliverables
- baseline training command
- baseline evaluation command
- logged metrics and checkpoints

### Exit criteria
- baseline trains end-to-end
- evaluation runs without code changes to the method modules

---

## Phase 5 — Timestep diagnostics and instrumentation

### Goal
Measure how the baseline behaves across timesteps before changing training allocation.

### Deliverables
- timestep histogram logging
- per-bin loss summaries
- optional gradient norm / variance summaries
- plots or machine-readable logs

### Exit criteria
- different timestep regions can be inspected quantitatively
- logged outputs are reusable for later analysis

---

## Phase 6 — SAANS scheduler implementation

### Goal
Integrate adaptive timestep sampling with objective-preserving reweighting.

### Deliverables
- baseline sampler
- adaptive sampler
- importance weights
- EMA hardness updates
- scheduler config support in training loop

### Exit criteria
- training loop runs with and without SAANS
- alpha=0 and rho=1 recover baseline behavior

---

## Phase 7 — Correctness validation and toy study

### Goal
Verify the adaptive method is mathematically and numerically sane before expensive runs.

### Deliverables
- estimator-equivalence tests
- support-condition tests
- recovery tests
- toy experiment or synthetic sanity study

### Exit criteria
- tests pass
- toy run demonstrates expected adaptive behavior

---

## Phase 8 — Main QM9 experiments

### Goal
Run the actual course-project comparisons.

### Deliverables
- baseline vs SAANS
- adaptive-without-weighting ablation
- hardness-signal ablations
- logged diagnostics and tables

### Exit criteria
- main comparison runs are complete and reproducible
- metrics and plots are ready for reporting

---

## Phase 9 — Extension and packaging

### Goal
Strengthen the project and package it for final submission.

### Deliverables
- optional harder benchmark or extension study
- cleaned code
- final report figures/tables
- reproducibility notes

### Exit criteria
- code and artifacts are organized for final delivery

---

## Current phase

**Completed phase:** Phase 1 — Workspace bootstrap and scheduler core scaffold

**Completed phase:** Phase 2 — Baseline integration interface

**Completed phase:** Phase 3 — Data acquisition and preprocessing pipeline

**Completed phase:** Phase 4 — Baseline reproduction (smoke-test level in this environment)

**Completed phase:** Phase 5 — Timestep diagnostics and instrumentation

**Completed phase:** Phase 6 — SAANS scheduler integration (smoke-test level)

**Completed phase:** Phase 7 — Correctness validation and toy study

**Completed phase:** Phase 8 — Experiment configs and execution matrix scaffolding

**Active phase:** Phase 9 — Packaging, status reporting, and handoff

## Important note on completion semantics

The codebase now contains runnable implementations or smoke-test versions for all phases through Phase 8.

What is still not fully "complete" in the research sense is:

- long-running multi-seed QM9 experiments,
- full ablation sweeps,
- optional harder-benchmark runs,
- publication-grade result validation.

Those tasks depend on substantial wall-clock compute rather than missing software structure.
