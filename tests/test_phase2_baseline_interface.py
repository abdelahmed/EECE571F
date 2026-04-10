import unittest

from saans_project.baseline import MockEquivariantDiffusionAdapter
from saans_project.config import load_run_config
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker
from saans_project.training import run_evaluation_stub, run_training_step


class TestPhase2BaselineInterface(unittest.TestCase):
    def test_config_contract(self) -> None:
        baseline_cfg = load_run_config("configs/qm9_baseline.toml")
        saans_cfg = load_run_config("configs/qm9_saans.toml")
        self.assertFalse(baseline_cfg.scheduler.enabled)
        self.assertTrue(saans_cfg.scheduler.enabled)
        self.assertEqual(baseline_cfg.dataset, "qm9")

    def test_mock_training_step_without_adaptive_sampler(self) -> None:
        adapter = MockEquivariantDiffusionAdapter(seed=1)
        batch = adapter.make_synthetic_batch(batch_size=3)
        result = run_training_step(adapter, batch, BinManager(num_bins=4))
        self.assertEqual(len(result.timesteps), 3)
        self.assertEqual(len(result.bin_indices), 3)
        self.assertEqual(result.sample_weights, [1.0, 1.0, 1.0])
        self.assertGreaterEqual(result.weighted_total, 0.0)

    def test_mock_training_step_with_adaptive_sampler(self) -> None:
        adapter = MockEquivariantDiffusionAdapter(seed=2)
        batch = adapter.make_synthetic_batch(batch_size=4)
        bins = BinManager(num_bins=4)
        tracker = EMAHardnessTracker(num_bins=4)
        sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=1.0, rho=0.1)
        result = run_training_step(
            adapter,
            batch,
            bins,
            adaptive_sampler=sampler,
            tracker=tracker,
            hardness_type="coord_plus_feat",
            lambda_coord=1.0,
            lambda_feat=1.0,
        )
        self.assertEqual(len(result.sample_weights), 4)
        self.assertTrue(all(weight > 0 for weight in result.sample_weights))
        self.assertGreaterEqual(result.weighted_total, 0.0)

    def test_evaluation_stub(self) -> None:
        adapter = MockEquivariantDiffusionAdapter(seed=3)
        batch = adapter.make_synthetic_batch(batch_size=2)
        result = run_evaluation_stub(adapter, batch)
        self.assertEqual(len(result.timesteps), 2)
        self.assertGreaterEqual(result.losses.total, 0.0)


if __name__ == "__main__":
    unittest.main()
