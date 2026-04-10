import unittest

from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker


class TestSchedulerBasics(unittest.TestCase):
    def test_uniform_baseline_masses(self) -> None:
        bins = BinManager(num_bins=4)
        self.assertEqual(len(bins.baseline_masses), 4)
        self.assertAlmostEqual(sum(bins.baseline_masses), 1.0)

    def test_alpha_zero_recovers_baseline(self) -> None:
        bins = BinManager(num_bins=4)
        tracker = EMAHardnessTracker(num_bins=4)
        sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=0.0, rho=0.1)
        self.assertEqual(sampler.probabilities(), bins.baseline_masses)

    def test_rho_one_recovers_baseline(self) -> None:
        bins = BinManager(num_bins=4)
        tracker = EMAHardnessTracker(num_bins=4)
        sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=2.0, rho=1.0)
        self.assertEqual(sampler.probabilities(), bins.baseline_masses)

    def test_support_and_weight_shapes(self) -> None:
        bins = BinManager(num_bins=4)
        tracker = EMAHardnessTracker(num_bins=4)
        sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=1.0, rho=0.2)
        probs = sampler.probabilities()
        weights = sampler.importance_weights()
        self.assertEqual(len(probs), 4)
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(p > 0 for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_ema_update_changes_state(self) -> None:
        tracker = EMAHardnessTracker(num_bins=2, beta=0.5, initial_value=1.0)
        tracker.update({1: [3.0]})
        self.assertAlmostEqual(tracker.values[0], 1.0)
        self.assertAlmostEqual(tracker.values[1], 2.0)


if __name__ == "__main__":
    unittest.main()
