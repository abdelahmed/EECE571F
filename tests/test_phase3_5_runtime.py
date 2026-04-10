import unittest
from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config
from saans_project.diagnostics import aggregate_timestep_records
from saans_project.scheduler import BinManager


class TestPhase3To5Runtime(unittest.TestCase):
    def test_load_edm_config(self) -> None:
        cfg = load_edm_qm9_config("configs/edm_qm9_smoke.toml")
        self.assertEqual(cfg.dataset, "qm9")
        self.assertEqual(cfg.batch_size, 4)

    def test_patch_vendored_repo(self) -> None:
        runtime = EDMQM9Runtime(load_edm_qm9_config("configs/edm_qm9_smoke.toml"), project_root=Path.cwd())
        changed = runtime.patch_vendored_repo()
        self.assertIsInstance(changed, list)

    def test_aggregate_timestep_records(self) -> None:
        records = [
            {"timestep_normalized": [0.1, 0.2], "loss_t": [1.0, 2.0], "error": [1.5, 2.5], "nll": [3.0, 4.0]},
            {"timestep_normalized": [0.8], "loss_t": [0.5], "error": [0.7], "nll": [1.0]},
        ]
        summary = aggregate_timestep_records(records, BinManager(num_bins=4))
        self.assertEqual(sum(summary.counts), 3)
        self.assertEqual(len(summary.mean_nll), 4)


if __name__ == "__main__":
    unittest.main()
