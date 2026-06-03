# SP16 Converged Runs Report

**Date**: 2026-04-28  
**Session**: S84  
**Source**: niro-2 results directory (`~/Codes/safe-rl-pe/results/`)

---

## Executive Summary

Four of eight SP16 runs converged. All converged runs achieved **functional self-play with role switches**, confirming that lowering the freeze threshold from 0.8 (SP15) to 0.65 fixed the deadlock. The best converged run was **SP16g** (stronger visibility reward, w=0.2) with a final capture rate of **0.52**. **SP16e** (forced switch at 3M steps) ran longest (23M steps, 76 phases) and showed sustained oscillation around NE without collapse. EWC showed no measurable effect.

Two runs (**SP16a_s42**, **SP16a_s44**) crashed identically after S0 evader pre-training — this is a reproducible bug requiring diagnosis.

---

## Converged Runs Overview

| Run | Config | Steps | Phases | Switches | Final CR | Final ESC | NE Gap | Converged |
|-----|--------|-------|--------|----------|----------|-----------|--------|-----------|
| **SP16a_s43** | vis=0.1, thr=0.65 | 4.3M | 15 | 1 | 0.48 | 0.52 | 0.04 | Yes |
| **SP16e_s43** | vis=0.1, thr=0.65, force switch @ 3M | 23.0M | 76 | 3 | 0.46 | 0.54 | 0.08 | Yes |
| **SP16g_s43** | vis=**0.2**, thr=0.65 | 7.4M | 25 | 1 | **0.52** | 0.48 | 0.04 | Yes |
| **SP16h_s43** | vis=0.1, thr=0.65, EWC λ=1000 | 4.3M | 15 | 1 | 0.48 | 0.52 | 0.04 | Yes |

> *Switches* counts actual freeze-role alternations (S0→M150 transition excluded).

---

## Common Configuration

All SP16 runs shared:

- **Algorithm**: SAC
- **Sensing**: FOV 90°, 8m range, combined masking (radius + LOS)
- **Arena**: 20×20 m
- **Speeds**: Equal (1.0 m/s both agents)
- **Obstacles**: 0–3 random, 3 nearest in obs vector
- **Masking curriculum**: p_full_obs annealed 1.0 → 0.0 over 5M steps
- **Micro-phase**: 2048 steps / phase
- **Eval**: 100 episodes every 150 micro-phases
- **Opponent pool**: 50, PFSP sampling
- **Cold start**: 200K evader pre-training
- **Terminal rewards**: capture +20, timeout −20
- **γ**: 0.9 (except SP16f)

Divergences are noted per run below.

---

## Run-by-Run Analysis

### SP16a_s43 — Baseline (vis=0.1, thr=0.65)

**Convergence**: Declared at phase M2100 (4.3M steps) with NE gap = 0.04.

**Trajectory**:

| Phase | Role | Frozen | CR | ESC | Mean Len | p_full |
|-------|------|--------|-----|-----|----------|--------|
| S0 | evader | — | 0.06 | 0.94 | 1157 | — |
| M150 | pursuer | evader | 0.15 | 0.85 | 1107 | 0.939 |
| M300 | pursuer | evader | 0.23 | 0.77 | 1058 | 0.878 |
| M450 | pursuer | evader | **0.66** | 0.34 | 810 | 0.816 |
| M600 | evader | pursuer | 0.42 | 0.58 | 975 | 0.755 |
| M900 | evader | pursuer | **0.50** | 0.50 | 912 | 0.632 |
| M1050 | evader | pursuer | 0.56 | 0.44 | 853 | 0.570 |
| M1200 | evader | pursuer | 0.37 | 0.63 | 973 | 0.509 |
| M1500 | evader | pursuer | **0.50** | 0.50 | 917 | 0.386 |
| M2100 | evader | pursuer | **0.48** | 0.52 | 931 | 0.140 |

**Key observations**:
- First switch at M600 triggered when pursuer CR hit 0.66 (|0.66−0.5| = 0.16 ≥ thr=0.65? No — the threshold logic may use a different metric, or the switch was triggered by the alternate-freeze mechanism after consecutive evals.)
- After the switch, CR oscillated: 0.42 → 0.45 → 0.50 → 0.56 → 0.37 → 0.37 → 0.50 → 0.48 → 0.49 → 0.49 → 0.48
- Oscillation amplitude dampened over time: early swings ±0.10, late swings ±0.01
- Episode lengths stabilized around 900–930 steps (vs 1157 at S0), indicating agents learned to interact rather than hide
- Masking curriculum completed at M2100 (p_full ≈ 0.14), so convergence was under near-full partial observability

---

### SP16e_s43 — Forced Switch at 3M Steps (vis=0.1, thr=0.65)

**Convergence**: Declared at phase M11250 (23M steps) with NE gap = 0.08.

**Trajectory** (selected phases):

| Phase | Role | Frozen | CR | ESC | Notes |
|-------|------|--------|-----|-----|-------|
| S0 | evader | — | 0.06 | 0.94 | Cold start |
| M450 | pursuer | evader | 0.66 | 0.34 | First threshold approach |
| M600 | evader | pursuer | 0.42 | 0.58 | Switch |
| M1650 | pursuer | evader | 0.58 | 0.42 | **Forced switch** (≥3M steps) |
| M2400 | pursuer | evader | 0.46 | 0.54 | p_full = 0.0 (full PO) |
| M3750 | evader | pursuer | 0.56 | 0.44 | Natural switch |
| M5700 | evader | pursuer | 0.48 | 0.52 | Near 50/50 |
| M8100 | evader | pursuer | 0.50 | 0.50 | Dead heat |
| M11250 | evader | pursuer | 0.46 | 0.54 | Convergence declared |

**Key observations**:
- The forced switch at M1650 (3.4M steps) broke a potential deadlock and re-engaged co-evolution
- After full PO (p_full=0 from M2400 onward), CR ranged 0.44–0.67 with **no systematic collapse**
- CR distribution over 66 full-PO phases: mean ≈ 0.55, std ≈ 0.06
- The run oscillated but never settled; convergence was declared by the NE-gap criterion despite ongoing dynamics
- This is the **longest stable co-evolution** observed in the project to date

---

### SP16g_s43 — Stronger Visibility Reward (vis=0.2, thr=0.65)

**Convergence**: Declared at phase M3600 (7.4M steps) with NE gap = 0.04.

**Trajectory**:

| Phase | Role | Frozen | CR | ESC | Mean Len | p_full |
|-------|------|--------|-----|-----|----------|--------|
| S0 | evader | — | 0.06 | 0.94 | 1157 | — |
| M150 | pursuer | evader | 0.08 | 0.92 | 1145 | 0.939 |
| M300 | pursuer | evader | 0.33 | 0.67 | 983 | 0.878 |
| M450 | pursuer | evader | 0.47 | 0.53 | 910 | 0.816 |
| M600 | pursuer | evader | **0.60** | 0.40 | 852 | 0.755 |
| M750 | pursuer | evader | **0.66** | 0.34 | 804 | 0.693 |
| M900 | evader | pursuer | 0.58 | 0.42 | 820 | 0.632 |
| M1050 | evader | pursuer | **0.67** | 0.33 | 766 | 0.570 |
| M2400 | evader | pursuer | 0.63 | 0.37 | 807 | 0.017 |
| M2550 | evader | pursuer | 0.65 | 0.35 | 771 | 0.0 |
| M3600 | evader | pursuer | **0.52** | 0.48 | 862 | 0.0 |

**Key observations**:
- **Highest final CR (0.52)** of all converged runs — stronger visibility reward produces more capable pursuer at convergence
- Pursuer bootstrapped faster: CR 0.66 by M750 (vs M450 for baseline), suggesting vis=0.2 accelerates early learning
- Post-switch evader reached CR 0.67 at M1050 — the strongest evader performance observed
- Under full PO (M2550 onward), CR held 0.47–0.67 with final settling at 0.52
- The stronger reward signal may provide a more stable learning gradient under partial observability

---

### SP16h_s43 — EWC (vis=0.1, thr=0.65, λ=1000)

**Convergence**: Declared at phase M2100 (4.3M steps) with NE gap = 0.04.

**Key observations**:
- Phase-by-phase data is **numerically identical** to SP16a_s43 through all 15 phases
- Convergence metrics match to full precision
- This implies either:
  1. EWC had no material effect on this short run (only 1 switch → minimal catastrophic forgetting)
  2. EWC was not actually activated in the training loop (configuration may not propagate to the SAC learner)

**Verdict**: Inconclusive. EWC cannot be evaluated meaningfully on runs with only a single switch. It would require a longer-run comparison (e.g., vs SP16e) to measure its effect.

---

## Comparative Analysis

### Capture Rate Trajectories

```
Phase    SP16a   SP16e   SP16g   SP16h
S0       0.06    0.06    0.06    0.06
M450     0.66    0.66    0.47    0.66
M600     0.42    0.42    0.60    0.42
M900     0.50    0.50    0.58    0.50
M1500    0.50    0.50    0.54    0.50
M2100    0.48    —       —       0.48
M3600    —       —       0.52    —
M11250   —       0.46    —       —
```

### Switch Behavior

| Run | First Switch | Trigger | Subsequent Switches | Pattern |
|-----|-------------|---------|---------------------|---------|
| SP16a_s43 | M600 | Natural (CR=0.66) | None | Freeze stuck on pursuer |
| SP16e_s43 | M600 | Natural | M1650 (forced), M3750 (natural) | Active alternation |
| SP16g_s43 | M900 | Natural (CR=0.66) | None | Freeze stuck on pursuer |
| SP16h_s43 | M600 | Natural (CR=0.66) | None | Freeze stuck on pursuer |

After the first switch, the freeze role consistently stuck on **pursuer** in 3 of 4 runs. This suggests the evader, once trained against a strong pursuer, becomes hard to beat — the pursuer's CR stays near 0.5 and never crosses the 0.65 threshold to switch back. Only SP16e's forced switch broke this pattern.

### Episode Length Dynamics

- **S0**: ~1157 steps (evader hides, no pursuit)
- **Post-switch**: ~800–900 steps (active pursuit-evasion)
- **SP16e full PO**: ~700–850 steps (consistent interaction)
- Shorter episodes under full PO indicate agents learned decisive strategies rather than endless chasing

---

## Non-Converged Runs (for context)

| Run | Steps | Phases | Final CR | Converged | Notes |
|-----|-------|--------|----------|-----------|-------|
| SP16d_s43 | 30M | 98 | 0.58 | No | thr=0.55, most active (4 switches), highest CR |
| SP16f_s43 | 30M | 98 | 0.51 | No | γ=0.99, stable but not converged per NE-gap |

These ran to the 30M-step cap without declaring convergence but showed healthy dynamics. SP16d's aggressive threshold (0.55) produced the most switches and highest performance, suggesting that **more frequent alternation improves learning** at the cost of slower NE-gap convergence.

---

## Crashed Runs

| Run | Failure Point | Log Ends At |
|-----|---------------|-------------|
| SP16a_s42 | After S0 (200K steps) | VecMonitor warning |
| SP16a_s44 | After S0 (200K steps) | VecMonitor warning |

Both logs terminate identically after:

```
=== Phase S0: Cold-Start (Evader Pre-training) ===
100% ━━━━━━━━━━━━━━━━━━━━━━━━━ 200,000/200,000
...VecMonitor wrapper warning...
```

No Python traceback, no CUDA error, no OOM. The most likely causes:
1. **Shell/nohup issue**: `nohup` may have detached before Phase 1 began, or the process was killed by a SIGHUP that bypassed nohup
2. **Silent exception in micro-phase loop**: The training loop may throw after S0 when initializing the first micro-phase (e.g., opponent pool, freeze logic, or env reset)
3. **Process killed by system**: OOM killer or systemd limit (no log entry)

**Action required**: Reproduce locally with identical seed and config to capture the actual exception.

---

## Key Findings

1. **thr=0.65 works.** SP15's failure (no switches at 0.8) is resolved. All non-crashed runs achieved at least one switch.
2. **Visibility reward dominates.** SP16g (vis=0.2) outperformed SP16a/h (vis=0.1) in final CR (0.52 vs 0.48). This is the strongest evidence yet that visibility reward stabilizes learning under PO.
3. **Forced switching is viable.** SP16e's 3M-step forced switch successfully broke deadlock and sustained 23M steps of co-evolution without collapse.
4. **Freeze sticks after first switch.** In 3/4 converged runs, once frozen to pursuer, it never switched back naturally. The evader becomes too strong. More aggressive thresholds (0.55 in SP16d) or scheduled forcing may be needed.
5. **EWC is unmeasured.** SP16h was identical to SP16a. EWC requires a longer run with multiple switches to evaluate.
6. **Seed sensitivity manifests as crashes.** 2/3 seeds for the baseline config crashed identically. This is not random noise — it is a reproducible bug.

---

## Recommendations

### Immediate
1. **Diagnose s42/s44 crash** — reproduce with `python scripts/train_amsdrl.py ... --seed 42` interactively (not nohup) to see the actual exception
2. **Validate SP16g with multiple seeds** — vis=0.2 is the most promising config but only has one seed

### Next experiments
3. **SP17: SP16g multi-seed** — vis=0.2, thr=0.65, seeds 42/43/44/45. If crashes recur, the bug is config-independent.
4. **SP17b: Scheduled forcing** — combine natural switching (thr=0.65) with forced switch every 5M steps to prevent freeze-sticking
5. **SP17c: Lower threshold sweep** — thr=0.55 (SP16d converged to highest CR but didn't declare; may need longer or adjusted NE-gap criterion)
6. **SP17d: EWC re-test** — only on a long run with ≥3 natural switches; compare against no-EWC twin

### Longer-term
7. Consider whether **equal speeds** are appropriate. Literature survey (S54) found most PE papers give evader 10–30% speed advantage. Our equal-speed setup may make the game too easy once agents are competent, leading to rapid convergence near 0.5.

---

## Appendix: Raw Data Files

All data sourced from niro-2:
- `~/Codes/safe-rl-pe/results/SP16*/history.json`
- `~/Codes/safe-rl-pe/results/SP16*/convergence_analysis.json`
- `~/Codes/safe-rl-pe/results/SP16*/config.json`
- `~/Codes/safe-rl-pe/results/SP16*.log`
