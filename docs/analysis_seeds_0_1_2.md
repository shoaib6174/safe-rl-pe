# EMA 5K Smoke Test Analysis: Seeds 0-3

**Date:** 2026-05-22 (updated with seed 3)
**Author:** Session 98-99

---

## Executive Summary

| Seed | Vanilla Robustness | EMA Robustness | EMA Outcome |
|------|-------------------|----------------|-------------|
| 0 | 0.008 | **0.698** | Balanced play, strong generalization |
| 1 | NaN collapse | 0.000 | Evader-dominant collapse (pre-actor-Q-clip) |
| 2 | 0.000 (ep 164) | 0.148 | Stable but pursuer overfit to weak evader |
| 3 | Not run | **0.292** | Evader-dominant collapse (ep 1888→5000) |

**Key Finding:** EMA performance is **highly seed-dependent.** Seed 0 (0.698) is an outlier. Seeds 1 and 3 with the same LR=0.0005 both collapsed evader-dominant. LR is not the sole determinant.

---

## Detailed Results

### Seed 0 (2026-05-18) — The Success Story

**Configuration:** Actor LR=0.0005, EMA τ=0.001, no actor Q-clipping, no NaN recovery

**Vanilla:**
- Collapsed pursuer-dominant (CR=1.0 throughout)
- Robustness: 0.008
- Cannot beat greedy evader (CR_greedy=0.008)

**EMA τ=0.001:**
- **Robustness: 0.698** (exceeds 0.50 target)
- CR breakdown: selfplay=0.698, random=0.792, greedy=0.99
- Oscillation amplitude: 0.30
- Training showed oscillating CR (0→0.5→1.0), indicating genuine strategy exploration
- **Verdict:** EMA enabled balanced, generalizable policies

---

### Seed 1 (2026-05-19) — The Learning Experience

**Configuration:** Actor LR=0.0005, EMA τ=0.001, NO actor Q-clipping

**Vanilla:**
- NaN collapse at ep 2382 (AMP FP16 overflow)
- 1,310 NaN lines in log
- Pre-Q-clipping era — expected failure

**EMA τ=0.001:**
- Evader-dominant collapse: CR=0.000 from ep 2000
- BUT losses stable (no NaN): PC=0.7-30, PA=500, EC=91-171, EA=-1440
- Actor losses extreme: PA=500, EA=-1450 (near clip boundary)
- **Root cause:** No actor Q-clipping → Q-values for actor exploded despite target Q-clipping
- **Lesson:** Target Q-clipping alone is insufficient; actor Q-clipping is mandatory

---

### Seed 2 (2026-05-20) — The Overfitting Trap

**Configuration:** Actor LR=0.0001, EMA τ=0.001, actor Q-clipping, Q_MIN=-1500, NaN recovery

**Vanilla:**
- Collapsed at ep 164 (PA=1478, 98.5% of Q_MAX boundary)
- CR=0.000 (evader-dominant)
- Even with ALL fixes, vanilla collapses — confirms EMA is essential

**EMA τ=0.001:**
- Completed all 5000 episodes successfully
- No NaN, losses stable throughout
- **Best robustness: 0.148** at ep 3250 (far below 0.50 target)
- CR breakdown: selfplay=0.982, random=0.148, greedy=0.216
- Final oscillation amplitude: 0.016 (very low — no oscillation, stuck)

---

### Seed 3 (2026-05-22) — The Oscillation Trap

**Configuration:** Actor LR=0.0005, EMA τ=0.001, Q-clipping [-1500,1500], actor Q-clipping, NaN recovery

**Training trajectory:**
- Early training (ep 0-1500): Healthy oscillation, CR cycling 0→1→0.5
- Mid training (ep 1500-1888): CR stabilizes around 0.5, losses moderate
- Late training (ep 1888-5000): Evader-dominant trap — CR=0.000 for 3100+ consecutive episodes
- PA climbed to 1394 (93%) by ep 2064, peaked at 1467 (98%) around ep 2498
- Final: CR=0.000, PA=1452 (97%), EA=-884 (59%)

**Evaluation results:**
- **Best robustness: 0.292** at episode 1250
  - cr_random=0.292, cr_greedy=0.964, cr_selfplay=0.938
- Final robustness (ep 5000): 0.014
- Oscillation amplitude: **0.370** (high — confirms genuine cycling)

**Key insight:** The best checkpoint (ep 1250) was BEFORE the prolonged evader-dominant phase began (~ep 1888). EMA enabled oscillation but could not prevent the system from eventually falling into a permanent evader-dominant trap.

**Per-checkpoint robustness trajectory (selected):**

| Episode | Robustness |
|---------|------------|
| 250 | 0.014 |
| 500 | 0.146 |
| 750 | 0.172 |
| 1000 | 0.194 |
| **1250** | **0.292** ← best |
| 1500 | (not evaluated) |
| 5000 | 0.014 |

---

## Critical Insight: Seed-Dependence

| | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|--|--------|--------|--------|--------|
| Actor LR | 0.0005 | 0.0005 | 0.0001 | 0.0005 |
| Robustness | **0.698** | 0.000 | 0.148 | 0.292 |
| Best Episode | ~3000 | — | ~3250 | 1250 |
| CR_selfplay | 0.698 (balanced) | 0.000 | 0.982 (dominant) | 0.938 (dominant) |
| Oscillation amp | 0.30 | — | 0.016 | **0.370** |
| Outcome | Success | Collapse | Overfit | Oscillate then collapse |

**Revised hypothesis:** LR alone does not explain the variance. Seed 0 achieved 0.698, but seeds 1 and 3 with the same LR (0.0005) collapsed evader-dominant. The initialization seed determines whether EMA finds a balanced equilibrium or falls into a dominant-strategy trap.

**Evidence:**
1. Seed 0: Oscillation converges to balanced equilibrium (CR=0.698)
2. Seed 1: Early evader-dominant collapse (pre-Q-clip artifact)
3. Seed 2: Low LR → no oscillation → pursuer overfits to stagnant EMA opponent
4. Seed 3: Healthy oscillation (amp=0.370) but eventually degenerates into evader-dominant trap

---

## What Worked vs What Didn't

### Fixes That Prevent Collapse (✅)
- **Q-clipping on target Q:** Prevents NaN in critic
- **Actor Q-clipping:** Prevents runaway actor losses
- **Q_MIN=-1500:** Matches actual discounted return range
- **NaN recovery:** Clears Adam buffers, allows continuation
- **EMA opponent:** Prevents high-frequency oscillatory divergence

### What Did Not Ensure Convergence (❌)
- **Actor LR=0.0001:** Too conservative — evader cannot adapt (seed 2)
- **Actor LR=0.0005:** Better but still seed-dependent (seeds 0 vs 1 vs 3)
- **EMA τ=0.001:** Allows oscillation but does not guarantee balanced convergence
- **Seed-dependent initialization:** Determines which basin of attraction the system falls into

---

## Theoretical Interpretation

**Vanilla self-play** = high-frequency oscillation → immediate collapse
**EMA + LR=0.0001** = damped oscillation → convergence to weak equilibrium (pursuer overfits)
**EMA + LR=0.0005 (seed 3)** = sustained oscillation → temporary exploration → eventual collapse to evader-dominant
**EMA + LR=0.0005 (seed 0)** = sustained oscillation → convergence to balanced equilibrium

The EMA τ and actor LR form a coupled dynamical system:
- τ controls opponent change rate
- LR controls agent adaptation rate
- Initialization seed determines basin of attraction
- For balanced play, all three must align

---

## Recommendations

### Immediate: τ Ablation
- Test τ=0.005 (faster EMA, more oscillation) with LR=0.0005
- Test τ=0.0005 (slower EMA, less oscillation) with LR=0.0005
- Hypothesis: faster EMA may prevent the system from settling into a dominant trap

### Alternative: Pop-FSP
- Population-Based Fictitious Self-Play as fallback
- K=4 or 8 opponents, uniform sampling
- May provide stronger regularization against dominant strategies

### Paper Angle (Revised)
The seed-dependency is a fundamental finding:
- EMA prevents high-frequency collapse (stability benefit)
- But convergence to balanced equilibrium is seed-dependent
- The coupled dynamical system (τ, LR, initialization) has multiple attractors
- Only some attractors correspond to balanced play

---

## Open Questions

1. Why does seed 0 achieve 0.698 but seeds 1 and 3 collapse? What initialization property matters?
2. Would τ=0.005 (faster EMA) prevent the evader-dominant trap seen in seed 3?
3. Does the BiMDN belief network also overfit when the actor overfits?
4. Can we detect which basin of attraction a run is in, and intervene early?

---

## Files Referenced

- `docs/ICRA_2027_EMA_STATUS.md` — Project status
- `docs/worklogs/2026-05-20_S98.md` — Session 98 worklog
- `docs/worklogs/2026-05-22_S99.md` — Session 99 worklog (seed 3 results)
- `repro_gonultas_isler_2024/evaluate_checkpoint_sweep.py` — Evaluation pipeline
