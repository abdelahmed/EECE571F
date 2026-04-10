from saans_project.baseline import MockEquivariantDiffusionAdapter
from saans_project.config import load_run_config
from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker
from saans_project.training import run_evaluation_stub, run_training_step


def main() -> None:
    adapter = MockEquivariantDiffusionAdapter(seed=7)
    clean_batch = adapter.make_synthetic_batch(batch_size=4)

    baseline_cfg = load_run_config("configs/qm9_baseline.toml")
    saans_cfg = load_run_config("configs/qm9_saans.toml")

    bin_manager = BinManager(num_bins=saans_cfg.scheduler.num_bins)
    tracker = EMAHardnessTracker(num_bins=saans_cfg.scheduler.num_bins, beta=saans_cfg.scheduler.ema_beta)
    adaptive_sampler = AdaptiveBinSampler(
        bins=bin_manager,
        tracker=tracker,
        alpha=saans_cfg.scheduler.alpha,
        rho=saans_cfg.scheduler.baseline_mix_rho,
    )

    baseline_step = run_training_step(adapter, clean_batch, bin_manager)
    adaptive_step = run_training_step(
        adapter,
        clean_batch,
        bin_manager,
        adaptive_sampler=adaptive_sampler,
        tracker=tracker,
        hardness_type=saans_cfg.scheduler.hardness_type,
        lambda_coord=saans_cfg.scheduler.lambda_coord,
        lambda_feat=saans_cfg.scheduler.lambda_feat,
    )
    evaluation = run_evaluation_stub(adapter, clean_batch)

    print("Baseline mode:", baseline_cfg.mode)
    print("Adaptive mode:", saans_cfg.mode)
    print("Adapter:", adapter.name())
    print("Baseline weighted total:", round(baseline_step.weighted_total, 6))
    print("Adaptive weighted total:", round(adaptive_step.weighted_total, 6))
    print("Adaptive tracker preview:", [round(v, 4) for v in tracker.values[:4]])
    print("Eval total loss:", round(evaluation.losses.total, 6))


if __name__ == "__main__":
    main()
