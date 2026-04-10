from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class SchedulerConfig:
    enabled: bool
    num_bins: int
    alpha: float
    ema_beta: float
    baseline_mix_rho: float
    hardness_type: str
    lambda_coord: float = 1.0
    lambda_feat: float = 1.0


@dataclass
class RunConfig:
    dataset: str
    mode: str
    scheduler: SchedulerConfig


def load_run_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    project = raw.get("project", {})
    scheduler = raw.get("scheduler", {})

    return RunConfig(
        dataset=project.get("dataset", "qm9"),
        mode=project.get("mode", "baseline"),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler.get("enabled", False)),
            num_bins=int(scheduler.get("num_bins", 32)),
            alpha=float(scheduler.get("alpha", 0.0)),
            ema_beta=float(scheduler.get("ema_beta", 0.95)),
            baseline_mix_rho=float(scheduler.get("baseline_mix_rho", 1.0)),
            hardness_type=str(scheduler.get("hardness_type", "coord_only")),
            lambda_coord=float(scheduler.get("lambda_coord", 1.0)),
            lambda_feat=float(scheduler.get("lambda_feat", 1.0)),
        ),
    )
