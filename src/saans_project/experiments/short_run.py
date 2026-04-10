from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class ShortRunResult:
    mode: str
    train_steps: int
    eval_batches: int
    train_metrics: list[float]
    eval_metrics: list[float]
    summary: dict[str, float | int | str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_artifact_dir(project_root: str | Path) -> Path:
    path = Path(project_root) / "artifacts" / "phase8"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_result(result: ShortRunResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)


def load_result(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
