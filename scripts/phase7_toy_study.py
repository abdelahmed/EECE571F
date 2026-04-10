from statistics import variance

from saans_project.scheduler import AdaptiveBinSampler, BinManager, EMAHardnessTracker


def main() -> None:
    bins = BinManager(num_bins=4)
    tracker = EMAHardnessTracker(num_bins=4, beta=0.9, initial_value=1.0)
    tracker.update({0: [0.8], 1: [1.1], 2: [2.5], 3: [0.7]})
    sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=1.0, rho=0.1, seed=0)

    baseline_probs = bins.baseline_masses
    adaptive_probs = sampler.probabilities()
    weights = sampler.importance_weights()

    synthetic_losses = [0.8, 1.1, 2.5, 0.7]
    baseline_estimates = synthetic_losses
    adaptive_weighted = [w * synthetic_losses[i] for i, w in enumerate(weights)]

    print("Baseline probs:", [round(x, 4) for x in baseline_probs])
    print("Adaptive probs:", [round(x, 4) for x in adaptive_probs])
    print("Importance weights:", [round(x, 4) for x in weights])
    print("Baseline variance proxy:", round(variance(baseline_estimates), 6))
    print("Adaptive weighted variance proxy:", round(variance(adaptive_weighted), 6))


if __name__ == "__main__":
    main()
