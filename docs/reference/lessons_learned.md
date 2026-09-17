# Lessons Learned — Reference for the System Paper (Pathway B)

**Purpose**: Consolidated, citation-backed lessons from ~118 sessions (S01–S117) of this project, written as the design input for the new system paper: *safe, exploitability-bounded self-play for pursuit-evasion on nonholonomic ground robots under partial observability*.

**How to read**: Each lesson states (a) the rule, (b) the evidence (decision IDs D## from `docs/decisions.md`, session worklogs S##, or docs), (c) the consequence for Pathway B. Companion document: `docs/reference/codebase_reference.md` (what code implements each piece).

---

## 1. Evaluation & Measurement — the most expensive lessons

### L1.1 Seed everything, always, from day one — resets AND trainer RNG
- **Evidence**: Repro env resets use `np.random.default_rng()` (OS entropy) — repeat N=50 evals of the same checkpoint ranged 0.24–0.42 CR (D57; `pursuit_evasion_env.py:74` in repro). Two band-edge seeds flipped classification between N=300 draws. Worse: `training/train.py` never seeds numpy/torch/random from `config["seed"]` (only `train_pbt.py` does), and `vectorized_env.py:84` resets unseeded — so even *training runs* are not reproducible in the repro codebase; only evaluations were made deterministic post-hoc.
- **Rule**: Seed (a) env resets, (b) the eval init set shared across policies (paired design), (c) all trainer RNGs (numpy/torch/random) at every entry point. N=500 standard; N=1000 for band-edge cases (CR within ~0.07 of the 0.45/0.55 balance band).
- **Pathway B consequence**: The new env must have a seedable `reset(seed=...)` and every trainer must seed all RNGs at startup — hard requirements, tested in CI. This is cheap at design time and was the single largest reproducibility debt in the project.

### L1.2 Frozen-opponent evaluation is the only trustworthy CR
- **Evidence**: Training-log CR is a ~2-episode window mean under exploration noise (σ=0.1) against a *live, co-adapting* opponent. Seed 9/16 logs read 0.500 while frozen-opponent CRs were 0.262/0.563; mechanism decomposition showed opponent non-stationarity dominates over measurement protocol (D55, D57; worklog S117).
- **Rule**: Report only frozen final-vs-final (and vs-baselines) CR. Treat training-log CR as a health indicator, never a result.
- **Pathway B consequence**: Build the frozen-opponent + seeded eval harness *before* the first training run, not after.

### L1.3 CR ≈ 0.5 is not equilibrium — exploitability must be measured
- **Evidence**: SP17b seeds 48/49 converged to CR≈0.51–0.52, but best-response (BR) exploitability testing showed the evader side was exploitable (BR evader gained +0.08 to +0.10 over baseline > ε=0.05; "L3" verdict, D34–D35). The project reframed to "Framing C" because of this.
- **Rule**: Apparent balance (CR near 0.5) is a *snapshot statistic*; equilibrium claims require BR exploitability ≤ ε on both roles, plus oscillation analysis over the final window.
- **Pathway B consequence**: The paper's convergence claim is "bounded measured exploitability," and the BR-exploitability harness (`scripts/analyze_exploitability.py`, `training/ne_tools.py`) is a first-class deliverable.

### L1.4 Multi-seed cohorts are mandatory; single seeds are anecdotes
- **Evidence**: Same config, different seeds: SP11e 85% vs SP11i 30% CR (D10). CUDA non-determinism dominates run-to-run variance. Balanced outcomes are rare: 4/34 in the symmetric cohort (D58).
- **Rule**: ≥4 seeds for any claim; cohort-scale (9+) for rate claims; Wilson CIs + binomial tests for all balance rates (D56, D25).
- **Pathway B consequence**: Every experiment in the plan specifies its seed count and statistical test up front.

### L1.5 Baselines must see the truth
- **Evidence**: A greedy pursuer evaluated under the same partial observability as the learned policy showed 96% "escape rate" — it was blind, not the evader good. Full-obs greedy baseline gave 16% (D13).
- **Rule**: Scripted baselines (greedy pursuer/evader) always run with full observability; only learned policies are sensor-limited.

### L1.6 Checkpoint best, not last
- **Evidence**: Peak-to-final regressions of −23pp and −60pp observed (D14). Late-training regression and NaN endings are common (D43).
- **Rule**: Keep best-checkpoint-by-eval, not final weights; log health metrics to catch NaN/Q-explosion early.

---

## 2. Self-Play Dynamics — the failure taxonomy

### L2.1 The default outcome is pursuer lock-in
- **Evidence**: 26/34 symmetric-cohort seeds pursuer lock-in/biased, 4/34 evader, 4/34 balanced (D58). PBT with an explicit balance fitness still went 0/3, all agents P_WR=1.000 (§6 of `docs/paper_data_summary.md`).
- **Rule**: Assume lock-in is the attractor; any method must demonstrate escape from it, not just delay it.

### L2.2 Populations bias toward the ∃-player (the ∀/∃ asymmetry)
- **Evidence**: Pop-FSP: pursuer trains against *some* sampled evader (∃), evader against *all* sampled pursuers (∀) → systematic gradient imbalance. 0/17 across 5 configs; game modifications (arena, speed ratio 1.43–1.67x) shift the equilibrium but cannot balance it (D37–D39; `docs/pop_fsp_structural_asymmetry_analysis.md`). Exact tabular analysis confirms: game is balanced at Nash (CR=0.436), but pure-strategy commitment favors the ∃ player (0.60/0.40) (Exp 1, §6b of paper_data_summary).
- **Rule**: Population-based opponent sampling is structurally unsafe in asymmetric zero-sum games unless the quantifier asymmetry is deliberately compensated.
- **Pathway B consequence**: NFSP-style *single-opponent* average-policy training (or PSRO with explicit asymmetric design) is the candidate class; naive populations are ruled out by our own published result.

### L2.3 Single-opponent smoothing avoids ∀/∃ but is seed-fragile
- **Evidence**: EMA opponent: best result in the project (robustness 0.698, seed 0) but 1/7 seeds balanced (Wilson CI [0.026, 0.513]). Asymmetric-LR variant (A-EMA) 0/3 (D42, D44). Shared policy with role conditioning 0/3 — critics on different value scales, shared actor follows the dominant critic (D45). Transfer learning (warm-start from EMA) 0/3 — frozen EMA opponent was random at transition (S106).
- **Rule**: These are not "tuning failures" — each has a diagnosed mechanism. Do not retry them unchanged.

### L2.4 NaN/Q-explosion is intrinsic to long coupled training
- **Evidence**: NaN losses at the end of EMA seed 4 and all GI vanilla 20K runs, *after* the AMP/FP16 cause was fixed with Q-clipping (D43).
- **Rule**: Long self-play runs need Q-value monitoring + clipping + health-gated checkpointing as standard, not as debugging afterthoughts.

### L2.5 "Converged" runs can sit at the wrong equilibrium
- **Evidence**: Pop-FSP v6: oscillation around CR≈0.72 — stable, persistent, and wrong. Population diversity delayed lock-in ~3× but could not move the equilibrium point (D38).
- **Rule**: Distinguish three properties — stability (variance), balance (mean), exploitability (BR gap). A method can have the first and fail the other two.

---

## 3. Self-Play Stabilization — what actually worked (Pathway A)

| Mechanism | Evidence | Status |
|---|---|---|
| SAC over PPO for self-play | SAC 30–40% where PPO stalls; RecurrentPPO 0–2% across 6 variants (D1) | **Adopt** |
| Fixed ent_coef=0.03 (never auto) | Auto entropy collapsed 24%→6% (D2, D3) | **Adopt** |
| Freeze→unfreeze curriculum | 10%→42% with frozen evader; 1:1 co-evolution and adaptive boost failed (D9) | **Adopt** |
| Forced role switch every 2M steps | SP17b (2M): 2/2 seeds converged; 3M: 1 collapsed, 1 oscillated (D32, D23) | **Adopt** |
| Alternate-freeze threshold 0.6–0.65 | 0.8 unreachable under full PO: 0 switches in 10 runs (D19, D21) | **Adopt** |
| Visibility reward (w≈0.1–0.2) under PO | 0.57 avg CR with vs 0.10 without (D20) | **Adopt (pursuer)** |
| Search/staleness reward w=0.0001 | Search is a critical missing skill; w≥0.0003 teaches patrol-not-hunt (D24, D4) | **Adopt** |
| Collapse rollback + PFSP sampling | Safety nets for SR drops and pool bias (D11) | **Adopt** |
| EWC | Identical results with/without when no role switches occur (D22) | **Drop unless switching** |
| Masking curriculum (full→partial anneal) | +10–15pp over baseline at 5–8M steps (D15) | **Adopt** |
| Cold-start under partial obs | Bootstrapped to 53%; full-obs cold-starts all failed (D12) | **Adopt** |

- **Evidence base**: SP13–SP17b campaign (S59–S90), summarized in `docs/SP13_SP14_full_report.md`, `docs/SP16_converged_runs_report.md`, D1–D36.
- **Pathway B consequence**: This is the *training-dynamics chassis*. The new system inherits it whole; the research contribution lives elsewhere (safety + exploitability-bounded convergence).

---

## 4. Environment & Game Design

- **Speed ratio**: 1.2× evader speed makes a viable PO game; equal speed is degenerate (hide-and-nothing); 1.05× insufficient (D5). *Note the tension*: PE game balance and self-play training balance are different axes — the symmetric point-mass cohort showed symmetric dynamics alone give only 4/34 (D58).
- **Sensing**: FOV 90° + 8m range is the proven primary config (SP13a peaked 93–97% CR; D18). 3m omnidirectional abandoned — 2.8% arena coverage is unlearnable (8 runs × 13.2M steps, D17). Combined masking (radius + LOS) — LOS-only too restrictive, radius-only sees through obstacles (D6).
- **Obstacles**: 2 obstacles in 10×10; 4 too cluttered, 0 removes occlusion play (D7).
- **Phase length**: micro_phase_steps=2048; long frozen phases are fundamentally wrong — rapid alternation prevents catastrophic forgetting (D8).
- **Own pose in observations**: without self-localization, belief learning is mathematically impossible (BiMDN in-FOV RMSE 8.76m → 0.81m after fix; `docs/training_insights.md`). Partial observability is about the *opponent*, never self-state.
- **Out-of-FOV belief**: "never seen" targets are unpredictable in principle (~89% of lost windows carry zero information); separate "just lost" vs "never seen" metrics or the evaluation is meaningless (training_insights).
- **Pathway B consequence**: The new nonholonomic + occlusion env reuses these proven parameters as defaults; deviations must be justified by an experiment, not taste.

---

## 5. Safety / CBF

**Source**: `docs/phases/phase2_5_decision_report.md` (S27–S35, 2026-02-22); code inventory in `docs/reference/codebase_reference.md` §3.

### L5.1 End-to-end differentiable safety layers fail — exploration-safety conflict
- **Evidence**: BarrierNet (differentiable QP in the policy) after 324K steps: **2% capture, 88% QP intervention**. The QP controls ~85% of executed actions, so the policy gets almost no feedback on *its own* actions; the `log π(u_nom)` gradient approximation breaks when QP corrections are large. Root cause is structural, not tuning (phase2_5 report §4.2).
- **Rule**: Do not put a QP inside the policy's gradient path for this task class. (BarrierNet also currently untestable: `qpth` missing from venv/requirements.)

### L5.2 The working architecture: train with obstacles + post-hoc DCBF γ=0.2
- **Evidence**: 14-config sweep — DCBF γ=0.2 gives **100% capture, 3.57% violations, 6.6% intervention**, strictly dominating no-filter (99%/6.49%) and continuous-time CBF (95%/7.72%, 45.9% intervention). CT-CBF overshoots constraint boundaries at dt=0.05; the discrete formulation `h(x_{k+1}) ≥ (1−γ)h(x_k)` fixes it. PPO trained *with* obstacle features learns implicit avoidance (6.07% violations unfiltered).
- **Rule**: DCBF γ=0.2 post-hoc is the default shield. Continuous-time CBF is invalid at our dt.

### L5.3 Safety during adversarial co-adaptation is UNTESTED — that is the paper's gap
- **Evidence**: every Phase 2.5 result is single-agent vs a *random* evader. Nothing is known about shield behavior when both agents learn, the opponent exploits the shield, or lock-in dynamics drive repeated boundary interactions.
- **Pathway B consequence**: the shield-integration study (shield-during-training vs at-deploy, intervention rate as headline metric) is genuinely novel ground, not a re-run of Phase 2.5. Fallback if shield blocks learning: the proven post-hoc architecture (claim narrows to deployment-time guarantees).
- **Design constraint (restated)**: obstacle/boundary/wall constraints only — the inter-robot collision CBF (`vcp_cbf.py:190`) must never constrain the pursuer relative to the evader; capture requires contact.

## 5b. Reward & Curriculum Failure Modes

- **Evader curriculum-collapse is a one-way trap** (`docs/research_evader_l2_collapse.md`): pursuer skill transfers across curriculum levels, evader tactics don't; one pursuer phase annihilated the fragile L2 evader (SR_E 0.31→0.00 in a single phase); 40+ rollbacks never recovered it (rollback restores weights but not the opponent pool, now saturated with strong pursuers). **Rule**: rollback must be bilateral + pool-aware; curriculum gates must require *balance*, not dominance.
- **Visibility reward has no gradient toward obstacles** (`docs/research_pbrs_obstacle_seeking.md`): flat −1 plateau in open space, +1 spike behind obstacles — 8 consecutive runs (H–O) collapsed at obstacle levels because nothing points the evader at cover. Fix adopted: potential-based shaping (PBRS) toward nearest obstacle + prep phase + survival bonus. **Rule**: every dense reward must be checked for plateaus on the relevant state region before launching runs.

---

## 6. Engineering & Process

- **Reproduction audits pay off**: 7 implementation bugs found in the GI audit (actor obs selection, target-Q action swap, reward routing, BiMDN warm-up/architecture, belief normalization, capture counting) — each silently corrupted results (§1 of paper_data_summary).
- **Capture counting**: `if ep_reward > 0` undercounts when step penalties make rewards negative — count captures from the termination flag, never the reward sign (GI fix #7).
- **Citation hygiene**: the master bib contained placeholder metadata that propagated into the paper — never copy bib entries without verification (D59).
- **Never kill runs for disk**; no-resume trainers + disk management destroyed promising runs (D16).
- **GPU is mandatory for SAC + Dict-obs evaluation/viz**: CPU ~1.8 steps/s vs GPU 60× faster (D27).
- **niro-1 (RTX 5090) is the training machine**; niro-2 offline since 2026-07-14 (S110).

---


### L6.x Freeze a round before launching it (S8 incident, 2026-09-17)
- **Evidence**: S8's arms were queued to start from the working tree after a long reference measurement; an unrelated commit (S9's `n_physical_obstacles`) landed on the branch in between. The whole-branch reviewer caught it before the arms started; the change was numerically inert, but the round would not have run the pre-registered code. A second trap: from a git worktree, `python scripts/x.py` still imports the editable-installed `pe2` from the MAIN tree unless `PYTHONPATH` puts the worktree first.
- **Rule**: launch every round from a detached `git worktree` at the pre-registered commit, with `results/` and `venv/` symlinked and `PYTHONPATH=<worktree>` exported; verify `pe2.__file__` before launch; record the worktree commit in provenance.

## 7. Design Constraints for Pathway B (synthesis)

The plan in `docs/pathway/` must satisfy, by construction:

1. **Seedable everything** (L1.1) — env resets, eval init sets, opponent sampling.
2. **Frozen + seeded eval harness before training** (L1.2), with BR exploitability as the equilibrium criterion (L1.3) and ≥4-seed minimums (L1.4).
3. **Single-opponent convergence mechanism** (NFSP or asymmetric-aware PSRO) — naive populations are ruled out by our own ∀/∃ result (L2.2); the mechanism must target *measured exploitability*, acknowledging that CR-balance and low exploitability are different (L1.3, L2.5).
4. **Proven training chassis**: SAC, ent 0.03, freeze curriculum, forced 2M switching, visibility + search rewards, rollback+PFSP (§3).
5. **Proven env defaults**: FOV 90°/8m, combined masking, 2 obstacles, own-pose in obs (§4) — extended to nonholonomic dynamics + occlusion.
6. **Safety filter on both agents for obstacles/bounds only — never inter-agent** (capture requires contact); shield-intervention rate is a headline metric (§5).
7. **Health monitoring + best-checkpointing + Q-clipping** as infrastructure, not afterthoughts (L2.4, L1.6).
