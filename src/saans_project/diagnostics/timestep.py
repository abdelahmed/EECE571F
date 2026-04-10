from __future__ import annotations

from dataclasses import dataclass

from saans_project.scheduler import BinManager


@dataclass
class TimestepBinSummary:
    counts: list[int]
    mean_loss_t: list[float]
    mean_error: list[float]
    mean_nll: list[float]


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_timestep_records(records: list[dict], bin_manager: BinManager) -> TimestepBinSummary:
    per_bin_loss: list[list[float]] = [[] for _ in range(bin_manager.num_bins)]
    per_bin_error: list[list[float]] = [[] for _ in range(bin_manager.num_bins)]
    per_bin_nll: list[list[float]] = [[] for _ in range(bin_manager.num_bins)]

    for record in records:
        for t, loss_t, error, nll in zip(
            record["timestep_normalized"],
            record["loss_t"],
            record["error"],
            record["nll"],
        ):
            idx = bin_manager.bin_index(float(t))
            per_bin_loss[idx].append(float(loss_t))
            per_bin_error[idx].append(float(error))
            per_bin_nll[idx].append(float(nll))

    return TimestepBinSummary(
        counts=[len(x) for x in per_bin_loss],
        mean_loss_t=[_safe_mean(x) for x in per_bin_loss],
        mean_error=[_safe_mean(x) for x in per_bin_error],
        mean_nll=[_safe_mean(x) for x in per_bin_nll],
    )
