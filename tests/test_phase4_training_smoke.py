import unittest
from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config


class TestPhase4TrainingSmoke(unittest.TestCase):
    def test_runtime_has_prepared_paths(self) -> None:
        runtime = EDMQM9Runtime(load_edm_qm9_config("configs/edm_qm9_smoke.toml"), project_root=Path.cwd())
        files = runtime.prepared_qm9_files
        self.assertEqual(set(files.keys()), {"train", "valid", "test"})


if __name__ == "__main__":
    unittest.main()