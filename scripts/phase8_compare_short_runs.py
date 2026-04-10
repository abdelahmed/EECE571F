from __future__ import annotations

from pathlib import Path

from saans_project.experiments import ensure_artifact_dir, load_result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = ensure_artifact_dir(project_root)
    baseline = load_result(out_dir / "baseline_short_run.json")
    saans = load_result(out_dir / "saans_short_run.json")

    baseline_eval = baseline["summary"]["eval_mean"]
    saans_eval = saans["summary"]["eval_mean"]
    delta = saans_eval - baseline_eval

    report = out_dir / "comparison_report.md"
    lines = [
        "# Short-run comparison report",
        "",
        f"- baseline eval mean: {baseline_eval:.6f}",
        f"- saans eval mean: {saans_eval:.6f}",
        f"- delta (saans - baseline): {delta:.6f}",
        "",
        "These numbers are short-run smoke outputs only and are not publication-grade results.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
