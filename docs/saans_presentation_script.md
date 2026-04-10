# SAANS Presentation Script

Presenter: **Abdelrahman Ahmed**

Target length: **about 4 minutes 30 seconds to 5 minutes**

Use this as a word-for-word script while recording your presentation.

---

## Slide 1 — Title
**Target time: 20–25 seconds**

Hello everyone, my name is Abdelrahman Ahmed, and my project is called **Symmetry-Safe Adaptive Noise Scheduling for E(3)-Equivariant Diffusion Training**, or **SAANS**.

The main question in this project is whether we can train equivariant molecular diffusion models more efficiently by adaptively reallocating training effort across diffusion timesteps, while still preserving the original learning objective.

---

## Slide 2 — Motivation and Problem
**Target time: 45–55 seconds**

Diffusion models are trained by corrupting data at many different noise levels, or timesteps, and learning how to denoise from each of them.

In most implementations, the training schedule over timesteps is fixed. However, different timestep regions can have very different difficulty, variance, and optimization value.

For example, low-noise regions may focus more on local structure, high-noise regions may focus more on global coarse structure, and middle regions may often be the hardest to learn.

This becomes especially interesting in 3D molecular diffusion, because the model is not just denoising arbitrary data. It also needs to preserve geometric symmetry, meaning the learning process should behave consistently under rotations and translations.

So the core problem is: can we spend more training effort on harder timestep regions without changing the baseline objective?

If the answer is yes, then we can learn something useful about training efficiency in equivariant molecular diffusion.

---

## Slide 3 — Related Work and Project Gap
**Target time: 35–45 seconds**

There are two main areas of related work behind this project.

First, there are already strong **E(3)-equivariant molecular diffusion baselines** for 3D molecule generation. So this project does not try to invent a new molecule generator architecture.

Second, there is broader diffusion literature showing that timestep weighting and scheduling can matter a lot for optimization efficiency and variance.

Our project sits at the intersection of these two areas.

The specific gap we are targeting is **symmetry-safe adaptive timestep allocation for E(3)-equivariant molecular diffusion**.

So the contribution is mainly a **training-time scheduling mechanism**, not a new backbone network.

---

## Slide 4 — Main Idea: SAANS
**Target time: 75–90 seconds**

The main idea of SAANS is to adaptively change how training timesteps are sampled.

First, we partition the timestep range into bins.

Second, for each bin, we track a detached hardness estimate. This gives us a moving summary of which timestep regions currently seem more difficult for the model.

Third, we build an adaptive timestep distribution that increases the sampling probability of harder bins.

Mathematically, the adaptive bin probability is proportional to the baseline bin mass multiplied by a hardness-based score.

But an important issue is that if we simply oversample hard bins, we may accidentally change the objective we are optimizing.

So to keep the method principled, we use **importance weighting**. That means each sampled bin is weighted by the ratio between the baseline probability and the adaptive probability.

Another key design point is that the hardness signal should be **symmetry-safe**. Since this is a geometric 3D setting, the hardness estimate should not depend on arbitrary rotations or translations of the molecule.

So the full loop is: sample a timestep bin, compute loss, update hardness estimates, and then adapt the timestep distribution over training.

---

## Slide 5 — Draft Main Figure
**Target time: 35–45 seconds**

This slide shows the draft figure idea for the project.

The blue bars represent the fixed baseline timestep schedule, while the orange bars represent an adaptive schedule that shifts more probability mass toward bins that appear harder.

This is only an illustrative draft, but it captures the main intuition of the method.

In the final version of the project, we want figures like this that show how the adaptive distribution evolves over training, along with supporting diagnostics such as hardness trajectories, bin occupancy, and per-bin loss variance.

So the purpose of this figure is to visually explain the core idea: training effort is being reallocated, not the model architecture itself.

---

## Slide 6 — Planned Experiments and Current Status
**Target time: 55–65 seconds**

Our main experiments are planned on the **QM9** molecular dataset.

We want to compare the baseline EDM scheduler against the full SAANS method, as well as ablations such as adaptive sampling without importance weighting and different hardness definitions.

The main questions we want these experiments to answer are:

which timestep regions are hardest, whether adaptive allocation improves optimization efficiency, whether importance weighting is necessary, and whether symmetry-safe hardness design matters.

In terms of current progress, the implementation groundwork is already in place.

We have integrated a public EDM baseline, built a working QM9 data pipeline, added timestep diagnostics, and implemented smoke-test versions of both baseline and SAANS-style runs.

So the next major step is to move from engineering validation to longer controlled GPU experiments.

---

## Slide 7 — Takeaway and Closing
**Target time: 25–35 seconds**

To summarize, this project studies whether adaptive timestep allocation can improve training for E(3)-equivariant molecular diffusion.

The main contribution is a symmetry-safe, objective-preserving scheduler rather than a new diffusion backbone.

The implementation scaffold and short-run validation are already in place, and the next phase is empirical evaluation through longer experiments.

Thank you.

---

## If you are running short on time
Use this shorter closing sentence for Slide 6:

"So overall, the implementation is in place, and the main remaining step is to run longer experiments to measure whether SAANS improves training efficiency and generation quality."

## If you need a final single-sentence summary
Use this:

"In short, SAANS asks whether we can train equivariant molecular diffusion models more efficiently by adaptively focusing on harder timestep regions while preserving the baseline objective."