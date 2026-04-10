# SAANS 5-Minute Presentation Notes

Use this as a speaking guide while recording.

## Slide 1 — Title
"This project is called Symmetry-Safe Adaptive Noise Scheduling for E(3)-Equivariant Diffusion Training. The main question is whether we can train equivariant molecular diffusion models more efficiently by adaptively reallocating training effort across timesteps, while still preserving the original objective."

## Slide 2 — Motivation and Problem
"Diffusion models train across many noise levels or timesteps. Most implementations use a fixed schedule, even though different timesteps may have very different difficulty and optimization value. In 3D molecular diffusion, this is especially interesting because the model must learn geometric denoising while respecting E(3) symmetry. If fixed schedules are inefficient, we may be wasting compute on easy regions and undersampling harder ones."

## Slide 3 — Related Work and Gap
"There are already strong equivariant diffusion baselines for 3D molecule generation, and separate diffusion literature has shown that timestep weighting and scheduling matter for optimization. Our project sits at the intersection. We are not proposing a new molecular backbone. Instead, we are asking whether adaptive, symmetry-safe timestep allocation can improve training in equivariant molecular diffusion."

## Slide 4 — Main Idea
"The main idea is to partition timesteps into bins, estimate which bins are currently hard using detached loss-based statistics, and then sample harder bins more often. To keep the objective aligned with the baseline, we importance-weight the loss by the ratio between the baseline and adaptive bin probabilities. A key design point is that the hardness signal should be symmetry-safe, meaning it should not depend on arbitrary rotation or translation of the molecule."

## Slide 5 — Draft Main Figure
"This draft figure illustrates the intended behavior. The blue bars represent the fixed baseline schedule, while the orange bars represent an adaptive schedule that shifts more probability mass to harder timestep regions. In the final version, we want figures showing how the adaptive distribution evolves over training, along with per-bin hardness and loss diagnostics."

## Slide 6 — Experiments and Current Status
"Our main experiments are planned on QM9. We will compare the baseline EDM scheduler against the full SAANS method, as well as ablations like adaptive sampling without importance weighting and different hardness definitions. The current implementation scaffold is already in place: we have baseline integration, a working QM9 pipeline, diagnostic utilities, and short-run smoke tests for both baseline and SAANS. The next major step is to run longer GPU experiments and measure whether adaptive timestep allocation improves optimization efficiency and generation quality."

## Slide 7 — Takeaway
"To summarize, this project studies adaptive timestep allocation for E(3)-equivariant molecular diffusion. The contribution is a training-time scheduler, not a new backbone. The implementation groundwork is in place, and the next phase is empirical validation through longer controlled experiments."