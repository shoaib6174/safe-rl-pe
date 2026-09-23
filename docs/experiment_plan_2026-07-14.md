# Experiment Plan — Structural ∀/∃ Asymmetry

**Date**: 2026-07-14  
**Session**: S113 continuation  
**Goal**: Run four targeted experiments to strengthen the RA-L / ICRA 2027 paper before returning to manuscript writing.

---

## PBT v3 Status

- **State**: RUNNING on niro-1 (RTX 5090)
- **PIDs**: 10679 (seed 0), 10744 (seed 1), 10810 (seed 2)
- **Elapsed**: ~5 hours
- **Progress**:
  - Seed 0: ~4,700 episodes
  - Seed 1: ~12,000 episodes
  - Seed 2: ~11,250 episodes
- **Result so far**: **Pursuer lock-in across all seeds**. Seed 1 tournament at episode 12,000 reports `P_WR=1.000`, `E_WR=0.000`, `Balance=-1.000` for every agent.
- **Decision**: Do **not** hinge the paper on PBT v3. Treat it as additional negative evidence if it completes.

---

## Experiment 1 — Exact / Tabular ∀/∃ Analysis

**Purpose**: Isolate the structural ∀/∃ asymmetry from deep RL approximation artifacts.

**Sub-experiment 1a — Gridworld best-response dynamics**:
- Small gridworld PE game with exact value iteration.
- Compare Nash equilibrium vs Pop-FSP-style asymmetric best response vs symmetric best response vs EMA-like single opponent.
- **Result**: All exact best-response regimes converge to the same Nash value (CR ≈ 0.44), confirming the bias is not in the game itself.

**Sub-experiment 1b — Pure-strategy commitment asymmetry**:
- Construct a 2×2 zero-sum matrix game whose mixed Nash is balanced.
- Compute pure-strategy commitment values:
  - Pursuer commits first: `min_p max_e M[p,e]`
  - Evader commits first: `max_e min_p M[p,e]`
- **Result**: Nash = 0.500; pursuer-first = 0.600; evader-first = 0.400.
- This shows that when agents use deterministic policies and one side can specialize/commit first, the ∃ player (pursuer) obtains a higher value.

**Deliverables**:
- `scripts/experiment_1_tabular_forall_exists.py` — gridworld analysis.
- `scripts/experiment_1b_matrix_forall_exists.py` — matrix-game commitment analysis.
- `paper/figures/experiment_1_tabular_regimes.{pdf,png}`
- `paper/figures/experiment_1b_matrix_commitment.{pdf,png}`
- `results/experiment_1_tabular/summary.md`
- `results/experiment_1_tabular/summary_matrix.md`

**Compute**: Local CPU only. Runs in seconds.
**Person-time**: 2–3 days.
**Status**: COMPLETE.

---

## Experiment 2 — Opponent-Sampling Ablation

**Purpose**: Test whether changing how the evader samples opponents can mitigate the pursuer bias.

**Conditions** (3 seeds each, same base hyperparameters):
1. **Uniform** — baseline (already exists).
2. **Recency-biased** — baseline (already exists).
3. **Evader worst-case** — evader samples the pursuer with lowest historical CR against it.
4. **Evader full-population critic** — evader's critic receives features from all pursuer policies.
5. **Pursuer best-case** — pursuer samples the evader with highest historical CR (may accelerate lock-in).

**Deliverables**:
- CR trajectories per condition.
- Balance-rate table.
- Recommendation on whether any sampling strategy fixes the asymmetry.

**Compute**: GPU training, ~5–7 days per condition.
**Person-time**: 2–3 days setup.
**Start**: After Experiment 3 is launched.

---

## Experiment 3 — Larger Symmetric Point-Mass Cohort

**Purpose**: Estimate the true balance rate under symmetric dynamics with enough seeds for statistical confidence.

**Setup**:
- Same symmetric point-mass environment as before.
- Run **25–50 seeds** for 5,000 episodes each.
- Use the existing Pop-FSP pipeline with no code changes.

**Deliverables**:
- Updated Figure 3 with larger sample.
- Reported balance rate with a simple confidence interval.
- Test whether 2/9 was a small-sample fluke.

**Compute**: GPU training, ~3–5 days if seeds run in parallel.
**Person-time**: 1 day setup.
**Start**: As soon as niro-1/niro-2 capacity is available.

---

## Experiment 4 — Asymmetric Training Favoring the Evader

**Purpose**: Test whether explicitly biasing training resources toward the evader can compensate for the ∀/∃ asymmetry.

**Variants** (3 seeds each):
1. **Larger evader population** — 32 evaders vs 8 pursuers.
2. **Lower evader learning rate** — evader LR = 1/5 of pursuer LR.
3. **Evader robustness objective** — evader trains against the top-k strongest pursuers from the population.

**Deliverables**:
- CR trajectories and final balance verdict per variant.
- Discussion of whether resource asymmetry can cancel structural asymmetry.

**Compute**: GPU training, ~5–7 days per variant.
**Person-time**: 2–3 days setup.
**Start**: After Experiment 2 results inform design.

---

## Execution Order

1. **Experiment 1** — start immediately (local CPU, no GPU needed).
2. **Experiment 3** — launch as soon as GPU is free (low code risk).
3. **Experiment 2** — begin after Experiment 3 is running.
4. **Experiment 4** — begin after Experiment 2 yields interpretable results.

PBT v3 continues untouched and will be analyzed only when it finishes.

---

## Time Estimate

| Experiment | Person-time | Compute time | Parallelizable |
|---|---|---|---|
| 1. Tabular analysis | 2–3 days | minutes | N/A |
| 3. Symmetric cohort | 1 day | 3–5 days | Yes (many seeds) |
| 2. Sampling ablation | 2–3 days | 5–7 days per condition | Limited |
| 4. Asymmetric training | 2–3 days | 5–7 days per variant | Limited |

- **Total wall time** if resources are available: **2–3 weeks**.
- **Total person-time**: **1.5–2 weeks**.
- **Return to paper writing**: after Experiments 1 and 3 complete; Experiments 2 and 4 can run in parallel with writing.

---

## Tracking

- This plan is linked from `docs/worklogs/2026-07-14_S113.md` and `memory/training-runs.md`.
- Future `status?` checks should report: which experiment is in progress, PBT v3 episode count, and next scheduled launch.

