from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.diagnostics import aggregate_timestep_records
from saans_project.scheduler import BinManager


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_diagnostics.toml")
    runtime = EDMQM9Runtime(cfg, project_root=project_root, device="cpu").prepare()
    loader = runtime.dataloaders["train"]

    records = []
    for batch_idx, batch in enumerate(loader):
        records.append(runtime.instrument_batch(batch).__dict__)
        if batch_idx >= 1:
            break

    summary = aggregate_timestep_records(records, BinManager(num_bins=cfg.scheduler.num_bins))
    print("Counts:", summary.counts)
    print("Mean loss_t:", [round(x, 6) for x in summary.mean_loss_t])
    print("Mean nll:", [round(x, 6) for x in summary.mean_nll])


if __name__ == "__main__":
    main()
