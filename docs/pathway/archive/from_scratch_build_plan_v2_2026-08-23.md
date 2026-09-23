# [ARCHIVED v2 — superseded by v3 on 2026-09-03 (D86). Steps 0–5 of this plan were executed in S119; Step 6 gates were pre-registered against it.]

# From-Scratch Build Plan — Safe Adversarial Co-Adaptation (Pathway B v2)

**Status**: Plan v2 (S118, 2026-08-23). Supersedes the phase ordering of `pathway_B_safe_exploitability_bounded_PE.md` (v1): spine is now **safety-first** — convergence is characterized, never load-bearing.
**Spine claim**: hard safety guarantees hold *throughout* adversarial co-adaptation (lock-in, oscillation, collapse — not just at equilibrium), under partial observability, on nonholonomic robots.
**Method rule**: **every step ships with (a) an automated verify and (b) a visual gate** — a rendered artifact (GIF/MP4/figure) inspected against an explicit checklist before the step is closed. No visual sign-off, no progress.

## Stated assumptions
1. New package `pe2/` at repo root. Old code is read-only reference. Proven *parameters* carry over from `docs/reference/lessons_learned.md` §3–§4.
2. RL libraries are not rewritten: Gymnasium + Stable-Baselines3 (SAC primary — D1; PPO for comparison).
3. Seeded everything from the first commit (L1.1). Determinism CI from Step 0.
4. Training on niro-1 (RTX 5090). Steps 0–6 CPU-cheap; GPU from Step 7.
5. Visual artifacts deterministic, stored in `results/visual_gates/step_##/`, sign-off recorded in the session worklog.

## Renderer-first principle
Renderer built in Step 1 alongside the env; overlays accumulate (FOV cone Step 2, belief Step 4, CBF margins Step 6, interventions Step 7).

## Steps (summary of v2 as executed)
- **Step 0** Foundations: `pe2/` skeleton, `rng.py` seed contract, frozen configs, determinism CI. *(Done S119, D65–D66)*
- **Step 1** Env (unicycle, 10×10, 2 obstacles, dt 0.05, v 1.0, ω 2.84, 1.2× evader) + renderer; gate `step_01_random_rollouts.gif`. *(Done S119)*
- **Step 2** Sensing (FOV 90°/8 m, LOS, combined masking) + obs (own pose mandatory); gate `step_02_occlusion.gif`. *(Done S119)*
- **Step 3** Scripted agents + viability gate + seeded battery N=500; gate `step_03_scripted_duels.gif`. *(Done S119)*
- **Step 4** Belief: last-seen baseline (BiMDN deferred, D67); gate `step_04_belief.gif`. *(Done S119)*
- **Step 5** Eval + exploitability harness before any RL; BR sanity gate. *(Done S119 — gate PASSED by BC clone; RL-from-scratch and RL fine-tune refuted, D68–D82; D83 gate refinements; D84 PPO-GAE retry)*
- **Step 6** DCBF γ=0.2 shield, obstacle/boundary only, never inter-agent on pursuer; 3-tier fallback; gates G1–G4 pre-registered. *(In progress at S119 end; D85)*
- **Step 7** Single-agent learning under the shield, 3 arms (shield from step 0 / at deploy / + margin shaping); exit: arm (a) within 10pp, intervention < 15%.
- **Step 8** Self-play co-adaptation under the shield on the proven chassis; headline = violations ≈ 0 through every regime.
- **Step 9** Convergence characterization (exploitability governor; optional).
- **Step 10** Cohort N≥9 + ablations + regenerable figures.

## Hard rules (v2)
Seed everything; frozen+seeded eval only; CR≈0.5 ≠ equilibrium; ≥4 seeds; baselines see truth; best-checkpoint + Q-monitoring; inter-agent CBF never on pursuer; no populations / no differentiable QP / no blind tuning; no step closes without its visual gate; never kill runs for disk.
