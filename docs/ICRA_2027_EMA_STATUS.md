# ICRA 2027: EMA Opponent Stabilization — Project Status

**Last Updated:** 2026-05-22 (S99)
**Paper Title:** *Exponential Moving Average Opponents Stabilize Competitive Training in Partially Observable Pursuit-Evasion Games*
**Target Venue:** ICRA 2027 (deadline: ~September 2026)

---

## Current Status

### Completed

| Phase | Description | Status |
|-------|-------------|--------|
| Reproduction | Gonultas & Isler 2024 MADDPG+BiMDN reproduced | **DONE** — Self-play CR=0.550 matches paper |
| Phase 1 | EMA core implementation + critical fixes | **DONE** — Commit `75bcd72` |
| Phase 1a | Post-implementation critical analysis | **DONE** — 10 issues identified and fixed |
| Phase 1b | Evaluation pipeline + launch scripts | **DONE** — Sweep script, EMA eval modes, niro-2 launcher |
| Phase 2 | 5K smoke test launch | **IN PROGRESS** — Both runs live on niro-2 |

### What's Running Now

**niro-2 (completed 2026-05-22 ~21:00 UTC):**
- **EMA τ=0.001** (seed 3, 5000 episodes): **COMPLETED** — ep 4999/5000
- **Configuration:** Actor LR=0.0005, Q-clipping [-1500,1500], actor Q-clipping, NaN recovery
- **Final status:** CR=0.000, PA=1452 (96.8%), EA=-884 (59.0%)
- **Outcome:** Oscillation degenerated into permanent evader-dominant equilibrium after ep ~1888
- **Evaluation:** `evaluate_checkpoint_sweep.py` running (PID 92870), 20 checkpoints
- **Key finding:** LR=0.0005 initially healthy but still collapsed to evader-dominant trap

### Completed Seeds

**Seed 0 (2026-05-18):**
- **Vanilla:** Pursuer-dominant collapse. Robustness=0.008, CR_selfplay=1.0
- **EMA:** **Robustness=0.698** — best result so far. CR_selfplay=0.698, CR_random=0.792, CR_greedy=0.99
- **Verdict:** EMA prevents collapse and enables balanced play

**Seed 1 (2026-05-19):**
- **Vanilla:** NaN collapse at ep 2382 (AMP FP16 overflow, pre-Q-clipping)
- **EMA:** Evader-dominant collapse (CR=0.000), but losses stable (no NaN)
- **Verdict:** Seed-dependent; Q-clipping + NaN recovery needed (applied for seed 2)

---

## Revised Plan (EMA-First)

**Original plan:** Population-Based Fictitious Self-Play (Pop-FSP)
**Revised plan:** EMA opponent stabilization as primary method

**Why EMA over Pop-FSP:**
- Lower complexity (1 network vs K networks)
- Fewer hyperparameters (1 vs 3+)
- Stronger theoretical grounding (opponent smoothing)
- Less overlap with prior work (AlphaStar/PSRO already explored populations)
- Pop-FSP reserved as fallback if EMA fails

---

## Phase Details

### Phase 1: EMA Implementation ✅ COMPLETE

**Deliverable:** Working EMA code with fixes

**Key changes:**
- `agents/maddpg.py`: `update()` accepts `opponent_actor: nn.Module`
- `training/train.py`: EMA network creation, updates, checkpoint save
- `config_*.yaml`: `ema_tau: 0.001` (half-life ≈ 3.5 episodes)
- `evaluation/evaluate.py`: 7 EMA evaluation matchups
- `evaluate_checkpoint_sweep.py`: Per-checkpoint robustness + oscillation metrics

**Critical fixes applied:**
1. Default `ema_tau` = 0.0 (backward compatible)
2. EMA weights saved in checkpoints
3. Tau lowered from 0.005 → 0.001 (mathematical analysis: 0.005 half-life = 0.69 episodes, too fast)
4. EMA status logging
5. Unit tests for EMA correctness

---

### Phase 2: 5K Smoke Test ✅ COMPLETE (3/3 seeds)

**Deliverable:** Oscillation comparison: vanilla vs EMA across 3 seeds

**Results:**

| Seed | Vanilla Robustness | EMA Robustness | Vanilla Outcome | EMA Outcome |
|------|-------------------|----------------|-----------------|-------------|
| 0 | 0.008 | **0.698** | Pursuer-dominant collapse | Balanced play |
| 1 | NaN collapse | 0.000 | AMP overflow (pre-fix) | Evader-dominant collapse |
| 2 | 0.000 (ep 164 collapse) | **0.148** | Evader-dominant collapse (PA→Q_MAX) | Stable but pursuer overfit to weak evader |
| 3 | — | **0.292** | Not launched (vanilla proven to collapse) | Evader-dominant collapse (ep 1888→5000) |

**Critical finding:** EMA performance is **highly seed-dependent and sensitive to hyperparameters.**

| Seed | Actor LR | EMA Robustness | Best Episode | Outcome |
|------|----------|---------------|--------------|---------|
| 0 | 0.0005 | **0.698** | ~3000 | Balanced play |
| 1 | 0.0005 | 0.000 | — | Evader-dominant collapse |
| 2 | 0.0001 | 0.148 | ~3250 | Pursuer overfit |
| 3 | 0.0005 | **0.292** | 1250 | Evader-dominant collapse |

**Revised hypothesis:** LR alone does not explain the variance. Seed 0 achieved 0.698, but seeds 1 and 3 with the same LR collapsed evader-dominant. The initialization seed determines whether EMA finds a balanced equilibrium or falls into a dominant-strategy trap.

**Analysis docs:**
- `docs/analysis_seeds_0_1_2.md` — Seeds 0-2 analysis
- `docs/worklogs/2026-05-22_S99.md` — Seed 3 results

**Key findings:**
- **EMA seed 0:** Robustness=0.698 (exceeds >0.50 target) — outlier result
- **Seeds 1/3 (LR=0.0005):** Both collapsed evader-dominant — seed-dependent failure
- **Seed 2 (LR=0.0001):** Collapsed pursuer-dominant — LR too low
- **Seed 3 oscillation amplitude:** 0.370 (high — genuine cycling occurred before collapse)
- **Vanilla collapses** even with fixes — EMA is essential for stability but not sufficient for convergence

**Go/No-Go Decision (post seed 3):**
- Seed 3 robustness = 0.292 (below 0.40 threshold) → **NO-GO**
- LR=0.0005 is not the sole determinant — seed 0 is an outlier
- **Revised path:** Investigate τ ablation (faster/slower EMA) or Pop-FSP fallback

**Metrics to compare:**
| Metric | How Computed |
|--------|-------------|
| Oscillation amplitude | `std(CR_selfplay)` over last 20 checkpoints |
| Single-checkpoint robustness | `min(CR_random, CR_greedy, CR_selfplay)` per checkpoint |
| Best robustness | Max robustness across all checkpoints |

---

### Phase 3: Full EMA Experiments (Conditional)

**Trigger:** Phase 2 GO decision

**Deliverable:** 20K-episode results, 5 seeds

**Experiment matrix:**
| Method | Seeds | Episodes | Purpose |
|--------|-------|----------|---------|
| Vanilla SP | 3 | 20K | Baseline |
| EMA (τ=0.001) | 5 | 20K | Primary |
| EMA (τ=0.0005) | 3 | 20K | Ablate: slower |
| EMA (τ=0.005) | 3 | 20K | Ablate: faster |

**Success criteria:**
- **Minimum:** Single-checkpoint robustness >0.40 (vanilla is ~0.25)
- **Target:** Oscillation amplitude reduced by >50%, robustness >0.50
- **Stretch:** Single checkpoint matches ALL G&I metrics simultaneously

---

### Phase 4: Oscillation Diagnostics (Parallel)

**Deliverable:** FFT analysis, phase portraits, sensitivity analysis

**Analysis on existing data:**
1. **FFT:** Frequency spectrum of CR time series
2. **Phase portrait:** CR(t) vs CR(t-Δt)
3. **Policy PCA:** Weight trajectory dimensionality
4. **Sensitivity:** Correlation of oscillation with LR, batch size, noise, τ

---

### Phase 5: Pop-FSP (Fallback)

**Trigger:** EMA fails to achieve robustness >0.40

**Approach:** Simplified Pop-FSP (K=4 or 8, actor-only, uniform sampling)

---

### Phase 6: Paper Writing

**Target structure (8 pages ICRA):**
```
I.   Introduction (1 page)
II.  Related Work (0.75 page)
III. Preliminaries (0.5 page)
IV.  Method: EMA + Oscillation Analysis (1.5 pages)
V.   Experiments (2.5 pages)
VI.  Discussion (0.5 page)
VII. Conclusion (0.25 page)
```

**Key figures:**
1. Time series: vanilla vs EMA CR (3 seeds)
2. FFT/spectral analysis
3. Robustness score vs checkpoint
4. τ ablation
5. Phase portrait of policy weights

---

## Open Questions

1. **Should we anneal τ during training?** (Start low for strong stabilization, increase for fine-tuning)
2. **Should we include Pop-FSP in the paper regardless of EMA success?** (Recommended: yes, as secondary comparison)
3. **How to handle BiMDN non-stationarity?** (EMA stabilizes actor but not belief network)

---

## Immediate Next Steps

- [x] Implement EMA core
- [x] Critical analysis + fixes
- [x] Launch 5K comparison on niro-2
- [ ] **WAIT:** 5K runs complete (~2-3 hours)
- [ ] Evaluate both runs with `evaluate_checkpoint_sweep.py`
- [ ] Compare oscillation amplitude and robustness
- [ ] Make Go/No-Go decision
- [ ] If GO: launch full 20K experiments (5 seeds)

---

## Key Files

| File | Purpose |
|------|---------|
| `repro_gonultas_isler_2024/training/train.py` | EMA training loop |
| `repro_gonultas_isler_2024/agents/maddpg.py` | Refactored update() |
| `repro_gonultas_isler_2024/evaluation/evaluate.py` | EMA evaluation modes |
| `repro_gonultas_isler_2024/evaluate_checkpoint_sweep.py` | Batch evaluation + metrics |
| `launch_ema_5k_comparison.sh` | niro-2 launch script |
| `docs/worklogs/2026-05-18_S97.md` | Full session worklog |

---

## References

- **Original Pop-FSP plan:** `docs/superpowers/plans/2026-05-18-stabilized-self-play-icra.md`
- **Approved EMA plan:** `.claude/plans/wild-baking-hickey.md`
- **G&I reproduction spec:** `docs/superpowers/specs/2026-05-08-repro-gonultas-isler-design.md`
