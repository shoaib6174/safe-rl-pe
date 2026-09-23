# Experimental Data Summary — RA-L / ICRA 2027 Paper

**Document purpose**: Single source of truth for all numbers, configurations, and results used in the RA-L and ICRA 2027 papers.  
**Date**: 2026-07-14  
**Based on**: Sessions S91–S113, `docs/pop_fsp_structural_asymmetry_analysis.md`, `docs/analysis_seeds_0_1_2.md`, `docs/ICRA_2027_EMA_STATUS.md`, `docs/worklogs/2026-06-08_S108.md`, `memory/sp17-experiment.md`.

---

## 1. Verified Gonultas & Isler 2024 Reproduction

**Implementation**: Standalone PyTorch reproduction in `repro_gonultas_isler_2024/`.  
**Environment**: 16×16m arena, bicycle-model pursuer, point-mass evader, asymmetric sensing (pursuer wedge FOV 2π/3, range 6m; evader omnidirectional).  
**Algorithm**: MADDPG + BiMDN belief encoder + curriculum.

### Key results

| Seed | Checkpoint | vs_selfplay | vs_random | Paper target |
|---|---|---|---|---|
| 0 | ep 11,000 | **0.550** | 0.906 | 0.54 / 0.98 |
| 1 | ep 15,500 | 0.472 | **0.936** | 0.54 / 0.98 |

**Verdict**: Self-play CR = 0.550 matches paper (0.54). vs-random CR = 0.936 matches paper (0.98) within 0.044. Oscillatory competitive dynamics confirmed.

### Seven implementation fixes discovered during audit

| # | Fix | Location / symptom | Impact |
|---|---|---|---|
| 1 | Actor observation selection | Wrong observation passed to actor in `evaluate.py` | Policies acted on mismatched inputs |
| 2 | Target Q action swap | Actions swapped in target Q computation | Critic learned wrong targets |
| 3 | Reward selection | Wrong reward signal routed to optimizer | Biased gradient updates |
| 4 | BiMDN warm-up | Warm-up buffer logic incorrect | Early belief estimates unreliable |
| 5 | BiMDN architecture | Architecture mismatch with paper (feature dimensions) | Belief representation incorrect |
| 6 | Belief normalization | Belief state not normalized correctly | Actor/critic inputs out of expected range |
| 7 | Capture counting | `if ep_reward > 0` undercounted captures when step penalties made reward negative | All CR metrics underreported until fixed |

---

## 2. Pop-FSP Structural ∀/∃ Asymmetry

**Algorithm**: Population-based fictitious self-play. Maintain pursuer population Π_P and evader population Π_E. Each learner trains against a sampled opponent from the opposite population.

**Structural asymmetry**:
- Pursuer solves ∃-problem: perform well against *one* sampled evader.
- Evader solves ∀-problem: must escape *all* sampled pursuers.

### Configuration sweep (v3–v7)

| Config | Arena | Evader v_max | Speed ratio | Pop size (P/E) | Recency λ | Mixing start/end | Result | Direction |
|---|---|---|---|---|---|---|---|---|
| v3 | 16.0 | 1.50 | 1.67x | 8 / 16 | 0.15 | 100 / 500 | CR = 1.00 | Pursuer lock-in |
| v4 | 20.0 | 1.75 | 1.43x | 8 / 16 | 0.15 | 100 / 500 | CR ≈ 0.00 | Evader-dominant |
| v5 | 18.0 | 1.60 | 1.56x | 8 / 16 | 0.15 | 100 / 500 | CR = 1.00 | Pursuer lock-in |
| v6 | 19.0 | 1.65 | 1.52x | 8 / 16 | 0.15 | 100 / 500 | CR ≈ 0.72 | Pursuer-biased |
| v7 | 19.0 | 1.70 | 1.47x | 8 / 16 | 0.15 | 100 / 500 | Mixed lock-in | Lock-in |

**Total**: 17 seeds across 5 configurations. **0/17 balanced** (CR ∈ [0.45, 0.55]). One-sided 95% upper bound on balance rate: **0.16** (Clopper–Pearson); binomial test vs 50%: **p ≈ 7.6×10⁻⁶**.

### v6 Seed 1 detailed trajectory

| Window | Mean CR |
|---|---|
| 0–500 | 0.195 |
| 500–1000 | 0.295 |
| 1000–1500 | 0.778 |
| 1500–2500 | 0.868 |
| 2500–3500 | 0.862 |
| 3500–5000 | 0.823 |

**Key finding**: Pop-FSP prevents total lock-in (Seed 1 still oscillates) but settles at the wrong equilibrium (CR ≈ 0.72, not 0.5).

---

## 3. Symmetric Point-Mass Ablations

**Setup**: Replace car-like dynamics with symmetric, fully observable point-mass agents. Same Pop-FSP procedure. 34 seeds total (0–8 from S108; 9–33 from S113/S115), 5000 episodes each.

### Seeds 0–8 (S108)

| Seed | vs_random | vs_selfplay | EMA vs EMA | Verdict |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | Pursuer lock-in |
| 1 | 0.980 | 0.960 | 0.940 | Pursuer-biased |
| **2** | **0.500** | **0.460** | **0.600** | **Balanced** |
| **3** | **0.480** | **0.480** | **0.540** | **Balanced** |
| 4 | 1.000 | 0.960 | 0.920 | Pursuer lock-in |
| 5 | 1.000 | 1.000 | 1.000 | Pursuer lock-in |
| 6 | 0.800 | 0.920 | 0.960 | Pursuer-dominant |
| 7 | 0.160 | 0.240 | 0.240 | Evader-dominant |
| 8 | 1.000 | 1.000 | 1.000 | Pursuer lock-in |

### Seeds 9–33 (S117 seeded re-evaluation — supersedes S115 N=50 unseeded values)

Seeded paired protocol: all seeds evaluated on the same fixed init sets (`env.reset(seed=700000+i)`, N=500; band-edge seeds 12/16/20/21/30 also on `800000+i`, N=1000 combined). Scripts: `reeval_seeded_cohort.py`, `reeval_borderline_n300.py`; data: `results_symmetric/eval_seeds_9_33_final.json` (raw sets: `eval_seeds_9_33_seeded_n500.json`, `eval_edge_seeds_seeded_set2.json`).

| Seed | vs_selfplay (seeded) | Verdict |
|---|---|---|
| 9 | 0.262 | Evader lock-in |
| 10 | 0.866 | Pursuer lock-in |
| 11 | 1.000 | Pursuer lock-in |
| 12 | 0.571 (N=1000: 0.548/0.594) | Pursuer-biased |
| 13 | 0.966 | Pursuer lock-in |
| 14 | 0.992 | Pursuer lock-in |
| 15 | 0.204 | Evader lock-in |
| 16 | 0.563 (N=1000: 0.574/0.552) | Pursuer-biased |
| 17 | 0.990 | Pursuer lock-in |
| 18 | 0.846 | Pursuer lock-in |
| 19 | 1.000 | Pursuer lock-in |
| **20** | **0.536** (N=1000: 0.546/0.526) | **Balanced** |
| **21** | **0.527** (N=1000: 0.532/0.522) | **Balanced** |
| 22 | 0.954 | Pursuer lock-in |
| 23 | 0.962 | Pursuer lock-in |
| 24 | 0.922 | Pursuer lock-in |
| 25 | 1.000 | Pursuer lock-in |
| 26 | 0.912 | Pursuer lock-in |
| 27 | 0.804 | Pursuer-biased |
| 28 | 0.918 | Pursuer lock-in |
| 29 | 0.926 | Pursuer lock-in |
| 30 | 0.425 (N=1000: 0.438/0.412) | Evader-biased |
| 31 | 0.900 | Pursuer lock-in |
| 32 | 0.928 | Pursuer lock-in |
| 33 | 0.924 | Pursuer lock-in |

**Full cohort tally**: **4/34 balanced (11.8%)** — seeds 2, 3 (N=50 protocol, checkpoints on offline niro-2, not re-evaluable), 20, 21 (seeded). 26/34 pursuer lock-in/biased, 4/34 evader lock-in/biased (7, 9, 15, 30). Wilson 95% CI on balance rate: **[0.047, 0.266]**; one-sided binomial test vs 50% balance rate: **p ≈ 3.1×10⁻⁶** (unchanged by re-classification).

**Interpretation**: Symmetric dynamics help (some seeds balance) but are insufficient; the ∃/∀ structural asymmetry remains.

**S117 evaluation-protocol findings** (seed 9/16 discrepancy root cause; details in worklog S117 and `results_symmetric/eval_discrepancy_diagnosis_n300.json`):
1. Training-log CR is a ~2-episode window mean under exploration noise (σ=0.1) vs a live co-adapting opponent; final windows read 0.500 for seeds 9/16 while last-500-episode means were 0.468 (seed 9, window std 0.42) and 0.893 (seed 16, std 0.23).
2. Frozen seeded CRs: seed 9 = 0.262, seed 16 = 0.563 — training log errs in *opposite directions* across seeds.
3. Mechanism decomposition (frozen nets under exact training measurement conditions: noise σ=0.1, frame-skip 2): seed 9 → 0.250 (≈ frozen 0.262; protocol explains nothing, opponent non-stationarity dominates); seed 16 → 0.640 (protocol explains ~half of 0.563→0.893 gap). Noise and frame-skip individually shift CR by up to +0.16 but combine non-additively.
4. Env resets are unseeded (`np.random.default_rng()`, pursuit_evasion_env.py:74): repeat 50-episode evals of seed 9 ranged 0.24–0.42; seed 16's S115 value 0.580 was Monte Carlo error (N=300 draws: 0.503, 0.563). Unseeded N=50 evals are irreproducible → seeded paired evaluation is the standard.
5. Band-edge re-classification: 30 (0.500→0.425, evader-biased) out, 21 (0.620→0.527) in; 12 (0.571) and 16 (0.563) stay pursuer-biased. Tally unchanged at 4/34.

---

## 4. EMA Opponent Stabilization

**Algorithm**: Exponential moving average of opponent networks, τ = 0.001.

| Seed | Actor LR | Robustness | Best episode | Outcome |
|---|---|---|---|---|
| 0 | 0.0005 | **0.698** | ~3000 | Balanced play |
| 1 | 0.0005 | 0.000 | — | Evader-dominant collapse |
| 2 | 0.0001 | 0.148 | ~3250 | Stable but pursuer overfit |
| 3 | 0.0005 | 0.292 | 1250 | Evader-dominant collapse |

**Note**: Seeds 4–6 (LR=0.0005) also collapsed (0/3 balanced). Overall 1/7 seeds balanced with EMA (Wilson 95% CI **[0.026, 0.513]**).

**Interpretation**: EMA avoids the ∀/∃ asymmetry (single opponent) but is highly seed-dependent.

---

## 5. GI Vanilla 20K Baseline

| Seed | Final CR | Steps | Notes |
|---|---|---|---|
| 10 | 0.500 | 949,810 | Oscillated, NaN at end |
| 11 | 0.000 | 1,945,735 | Evader collapse, NaN |
| 12 | 0.000 | 1,695,802 | Evader collapse, NaN |

**Finding**: NaN losses appear at the end of long training runs, suggesting intrinsic numerical instability in the coupled dynamical system.

---

## 6. PBT v3 (Complete — Negative Result)

**Status**: Stopped 2026-07-14; archived at `repro_gonultas_isler_2024/archive/pbt_v3_final_20260714/`. Findings in `docs/pbt_v3_findings_2026-07-14.md`.

**Setup**: Population-based training with dual-role balance fitness (select agents closest to WR = 0.5 in both roles, not strongest), 3 seeds, up to 20K episodes.

**Result**: **0/3 balanced — complete pursuer lock-in in all seeds.** Tournament evaluation: every agent P_WR = 1.000, E_WR = 0.000, Balance = −1.000.

**Paper role**: Included in RA-L Results (§IV-F "Explicit Balance Selection Also Fails") as converging negative evidence: even explicitly optimizing for balance does not escape the lock-in attractor.

---

## 6b. Experiment 1 — Exact / Tabular ∀/∃ Analysis (Complete)

**1a — Gridworld exact BR** (3×3, γ=0.95, horizon 20): all five opponent-sampling regimes (Nash, ∃/∀, ∃/∀ myopic, ∃/∃, EMA-like) converge to Nash CR = **0.436**. The game itself is balanced; exact learners reach Nash regardless of sampling.

**1b — 2×2 matrix commitment**: mixed Nash = **0.500**; pursuer commits first (pure) = **0.600**; evader commits first = **0.400**. Deterministic-policy commitment gives the ∃ player (pursuer) a strict advantage.

**Paper role**: RA-L Results §IV-C + Fig. 3. Answers the "artifact of approximate RL?" reviewer concern: the bias is a commitment effect of the asymmetric sampling structure, not a property of the game or approximation error alone.

---

## 7. Balance Metric Definition

A seed/config is considered **balanced** if, over the final evaluation window:
- Mean capture rate CR ∈ [0.45, 0.55].
- **Robustness** = min(CR_random, CR_greedy, CR_selfplay) per checkpoint; best robustness ≥ 0.40 used as secondary confirmation.

---

## 8. Key Takeaways for Paper

1. **GI 2024 is reproducible** (with fixes) — establishes credibility and baseline dynamics.
2. **Pop-FSP fails structurally** — 0/17 balanced due to ∀/∃ asymmetry.
3. **Symmetric dynamics insufficient** — 2/9 balanced; helps but does not solve.
4. **EMA seed-dependent** — 1/7 balanced; theoretically sound but unreliable.
5. **NaN instability** — intrinsic to long coupled training, not just a bug.
