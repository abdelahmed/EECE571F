from saans_project.scheduler import (
    AdaptiveBinSampler,
    BinManager,
    EMAHardnessTracker,
    combined_hardness,
)


def main() -> None:
    bins = BinManager(num_bins=4)
    tracker = EMAHardnessTracker(num_bins=4, beta=0.9, initial_value=1.0)

    batch_observations = {
        0: [combined_hardness(0.8, 0.2)],
        1: [combined_hardness(1.2, 0.3)],
        2: [combined_hardness(2.0, 0.8)],
        3: [combined_hardness(0.6, 0.1)],
    }
    tracker.update(batch_observations)

    sampler = AdaptiveBinSampler(bins=bins, tracker=tracker, alpha=1.0, rho=0.1)

    print("Baseline masses:", [round(x, 4) for x in bins.baseline_masses])
    print("Tracker values:", [round(x, 4) for x in tracker.values])
    print("Adaptive probabilities:", [round(x, 4) for x in sampler.probabilities()])
    print("Importance weights:", [round(x, 4) for x in sampler.importance_weights()])
    print("Sampled bin:", sampler.sample_bin())


if __name__ == "__main__":
    main()
