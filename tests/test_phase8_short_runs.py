import unittest

from saans_project.experiments import ShortRunResult


class TestPhase8ShortRuns(unittest.TestCase):
    def test_short_run_result_to_dict(self) -> None:
        result = ShortRunResult(
            mode="baseline_short",
            train_steps=2,
            eval_batches=1,
            train_metrics=[1.0, 0.9],
            eval_metrics=[0.8],
            summary={"train_last": 0.9},
        )
        as_dict = result.to_dict()
        self.assertEqual(as_dict["mode"], "baseline_short")
        self.assertEqual(as_dict["train_steps"], 2)


if __name__ == "__main__":
    unittest.main()
