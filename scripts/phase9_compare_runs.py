from __future__ import annotations

from pathlib import Path
import json


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "artifacts" / "phase9"

    baseline = _load(out_dir / "kaggle_baseline_training_summary.json")
    saans = _load(out_dir / "kaggle_saans_training_summary.json")
    no_weight = _load(out_dir / "kaggle_saans_no_weighting_summary.json")
    alpha = _load(out_dir / "kaggle_saans_alpha05_summary.json")

    rows = [
        ("baseline", baseline.get("output_dir", "see output dir"), None),
        ("saans", saans["summary"].get("eval_mean"), saans["summary"].get("train_last")),
        ("no_weighting", no_weight["summary"].get("eval_mean"), no_weight["summary"].get("train_last")),
        ("alpha_0.5", alpha["summary"].get("eval_mean"), alpha["summary"].get("train_last")),
    ]

    report_lines = [
        "# Phase 9 experiment report",
        "",
        "This report summarizes the four Kaggle runs.",
        "",
        "| Run | Eval summary | Train summary |",
        "|---|---:|---:|",
    ]

    for name, eval_value, train_value in rows:
        report_lines.append(f"| {name} | {eval_value} | {train_value} |")

    report_lines += [
        "",
        "Notes:",
        "- The baseline run uses the upstream full training path.",
        "- The SAANS-related runs use the project-side multi-step training scripts.",
        "- These results are experiment outputs, not final polished paper numbers.",
    ]

    report_path = out_dir / "phase9_comparison_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()