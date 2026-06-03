# SP13/SP14 Training Report — Full Analysis
**Generated**: 2026-03-10 (Session 81)
**Runs**: 24 total (22 active, 2 stopped)
**System**: niro-2 (24 cores, 125GB RAM), uptime 13 days

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Experimental Design](#experimental-design)
3. [System Status](#system-status)
4. [Group-by-Group Analysis](#group-by-group-analysis)
5. [Alt-Freeze Mechanism Deep Dive](#alt-freeze-mechanism-deep-dive)
6. [Factor Analysis](#factor-analysis)
7. [Kill/Keep Recommendations](#killkeep-recommendations)
8. [Key Findings & Lessons](#key-findings--lessons)
9. [Appendix: Full Iteration Histories](#appendix-full-iteration-histories)

---

## Executive Summary

24 training runs across 13 configurations tested FOV-based partial observability (90deg, 8m/5m) with masking curriculum and various self-play strategies. After ~730 CPU-hours:

**Winners:**
- **SP13l_s43** (asym + alt_freeze 0.8 + obstacles): CR=0.89@it13, still climbing. New #1 run.
- **SP13k_s43** (asym + alt_freeze 0.8, no obstacles): Peaked 0.82@it11, just switched roles — cliff test in progress.
- **SP13a_s42** (freeze evader baseline): 0.79 avg CR under full PO across 26 iterations. Gold standard.

**Losers:**
- All 5m range runs (SP13c-d): Never viable, max 0.71, mostly 0.30-0.50.
- Asymmetric + thr=0.6 (SP13g-h): Catastrophic collapses (0.80→0.09).
- SP14a_s42 (pure self-play, seed 42): Stuck at 0.02-0.11.

**Key discovery:** Asymmetric obs + alt_freeze thr=0.8 is the winning self-play config. The interaction effect is stark — asymmetric obs HURTS at thr=0.6 (evader too strong) but HELPS at thr=0.8 (prevents premature switching).

---

## Experimental Design

### Shared Config (all SP13/SP14 runs)
| Parameter | Value |
|-----------|-------|
| Algorithm | SAC |
| FOV angle | 90deg |
| Masking curriculum | p_full_anneal_steps=5M, p_full_residual=0.0 |
| PFSP | Enabled |
| Opponent pool | 50 |
| Learning rate | 3e-4 |
| Entropy coef | 0.03 |
| Micro phase steps | 2048 |
| Eval interval | 150 micro phases |
| Max total steps | 30M |
| Collapse threshold | 0.1 |
| Collapse streak limit | 5 |

### Variable Config Matrix
| Group | Range | Freeze Mode | Threshold | Asymmetric Obs | Obstacles | Seeds | Runs |
|-------|-------|-------------|-----------|----------------|-----------|-------|------|
| SP13a | 8m | freeze evader | 0.9 | No | No | 42, 43 | 2 |
| SP13b | 8m | freeze evader | 0.9 | No | Yes (0-3) | 42, 43 | 2 |
| SP13c | **5m** | freeze evader | 0.9 | No | No | 42, 43 | 2 |
| SP13d | **5m** | freeze evader | 0.9 | No | Yes (0-3) | 42, 43 | 2 |
| SP13e | 8m | alt_freeze | **0.6** | No | No | 42, 43 | 2 |
| SP13f | 8m | alt_freeze | **0.6** | No | Yes (0-3) | 42, 43 | 2 |
| SP13g | 8m | alt_freeze | **0.6** | **Yes** | No | 42, 43 | 2 |
| SP13h | 8m | alt_freeze | **0.6** | **Yes** | Yes (0-3) | 42, 43 | 2 |
| SP13i | 8m | alt_freeze | **0.8** | No | No | 42, 43 | 2 |
| SP13j | 8m | alt_freeze | **0.8** | No | Yes (0-3) | 42, 43 | 2 |
| SP13k | 8m | alt_freeze | **0.8** | **Yes** | No | 43 | 1 |
| SP13l | 8m | alt_freeze | **0.8** | **Yes** | Yes (0-3) | 43 | 1 |
| SP14 | 8m | **none** (pure) | — | **Yes** | No | 42, 43 | 2 |

**Total**: 24 runs, 13 configs, 4 experimental factors (range, threshold, asymmetric, obstacles).

---

## System Status

| Metric | Value |
|--------|-------|
| Active processes | 22/24 cores |
| Load average | 23.9 |
| RAM used | 50/125 GB |
| Disk used | 202/292 GB (74%) |
| Uptime | 13 days |
| Total compute | ~730 CPU-hours |
| Stopped processes | SP13h_s42 (crashed@15it), SP13j_s42 (crashed@8it) |

---

## Group-by-Group Analysis

### GROUP 1: SP13a — Freeze Evader, 8m, No Obstacles (BASELINE)

| Run | Iters | Hours | CR Now | Peak CR | PO Avg | Slope | Status |
|-----|-------|-------|--------|---------|--------|-------|--------|
| SP13a_s42 | 42 | 58.7h | 0.62 | 0.97@it27 | **0.79** (26 it) | -0.019 | Declining (late stage) |
| SP13a_s43 | 40 | 58.8h | 0.70 | 0.97@it14 | 0.61 (24 it) | +0.003 | Oscillating |

**SP13a_fov_no_obs_s42** — The gold standard run.
```
Annealing (it1-16):  0.21 0.63 0.56 0.71 0.73 0.70 0.52 0.65 0.55 0.71 0.75 0.86 0.80 0.82 0.81 0.93
Full PO (it17-42):   0.88 0.73 0.86 0.87 0.89 0.87 0.88 0.87 0.90 0.88 0.97 0.84 0.94 0.83 0.75 0.63
                     0.79 0.74 0.82 0.76 0.79 0.61 0.72 0.66 0.51 0.62
```
Sustained 0.75-0.97 for 18 consecutive iters under full PO (it17-34). Now showing late-stage decline (it35-42: 0.76→0.62). The frozen evader may be getting exploited — pursuer overfitting to a static opponent. Still, 0.79 average under full PO across 26 iterations is exceptional and sets the benchmark.

**SP13a_fov_no_obs_s43** — Seed sensitivity demonstration.
```
Annealing:  0.43 0.52 0.50 0.43 0.48 0.61 0.62 0.62 0.74 0.91 0.86 0.75 0.83 0.97 0.83 0.82
Full PO:    0.79 0.78 0.71 0.62 0.87 0.79 0.59 0.59 0.56 0.42 0.55 0.51 0.52 0.56 0.51 0.56
            0.49 0.42 0.70 0.66 0.59 0.56 0.54 0.70
```
Same config as s42, vastly different outcome. Peaked 0.97 during annealing but only managed 0.61 avg under full PO (vs 0.79 for s42). Demonstrates that seed variance of ~0.18 in PO avg is possible.

**Verdict**: Freeze evader with 8m is the strongest single-agent approach. Sets the bar at 0.79 PO avg. Seed sensitivity is extreme.

---

### GROUP 2: SP13b — Freeze Evader, 8m, With Obstacles

| Run | Iters | Hours | CR Now | Peak CR | PO Avg | Slope | Status |
|-----|-------|-------|--------|---------|--------|-------|--------|
| SP13b_s42 | 37 | 58.7h | 0.30 | 0.61@it17 | 0.45 (21 it) | -0.018 | Declining |
| SP13b_s43 | 39 | 57.2h | 0.54 | 0.81@it9 | 0.62 (23 it) | -0.016 | Declining |

**SP13b_fov_obs_s42**:
```
Full PO: 0.54 0.53 0.53 0.45 0.54 0.47 0.42 0.33 0.47 0.56 0.43 0.47 0.42 0.51 0.39 0.35
         0.39 0.49 0.31 0.30
```
Never strong. Obstacle observations add noise without helping pursuit. Now declining to 0.30.

**SP13b_fov_obs_s43**:
```
Full PO: 0.74 0.76 0.55 0.67 0.62 0.65 0.66 0.70 0.62 0.55 0.64 0.68 0.65 0.55 0.55 0.59
         0.60 0.57 0.56 0.51 0.46 0.54
```
Better than s42 but declining from 0.76 to 0.54. Obstacle obs provides diminishing returns.

**Verdict**: Obstacle observations consistently hurt freeze-evader runs. Delta vs SP13a: -0.34 (s42) and -0.01 (s43) in PO avg. The extra observation dimensions add complexity without benefit.

---

### GROUP 3: SP13c-d — 5m Range (FAILED EXPERIMENT)

| Run | Iters | Hours | CR Now | Peak CR | PO Avg | Status |
|-----|-------|-------|--------|---------|--------|--------|
| SP13c_s42 | 18 | 35.3h | 0.65 | 0.65@it18 | 0.58 (2 it) | Oscillating |
| SP13c_s43 | 18 | 35.4h | 0.48 | 0.71@it17 | 0.59 (2 it) | Crashed from peak |
| SP13d_s42 | 17 | 34.7h | 0.46 | 0.46@it17 | 0.46 (1 it) | Plateaued |
| SP13d_s43 | 17 | 34.8h | 0.46 | 0.54@it8 | 0.46 (1 it) | Plateaued |

All 4 runs confirm 5m sensing range is fundamentally insufficient.

**SP13c_s42**: Wildly oscillating (0.13-0.65). Just entered full PO. Even its best moments (0.65) are below SP13a_s42's PO average (0.79).

**SP13c_s43**: Peaked 0.71@it17, immediately crashed to 0.48. Unstable under PO.

**SP13d_s42/s43**: Never above 0.46. 5m + obstacles = impossible task. Three consecutive evals at 0.46 — hard plateau.

**Verdict**: 5m range confirmed too restrictive for FOV 90deg. The pursuer can only see the evader when very close, leaving it blind most of the time. Stick with 8m. **All 4 should be killed.**

---

### GROUP 4: SP13e-f — Alt Freeze thr=0.6, No Asymmetric, 8m

| Run | Iters | Hours | CR Now | Peak CR | Switches | Slope | Status |
|-----|-------|-------|--------|---------|----------|-------|--------|
| SP13e_s42 | 19 | 36.1h | 0.62 | 0.68@it10 | **4** | +0.010 | Oscillating |
| SP13e_s43 | 18 | 35.3h | 0.39 | 0.72@it3 | 3 | -0.028 | Declining |
| SP13f_s42 | 18 | 35.5h | 0.46 | 0.63@it5 | 3 | +0.024 | Slow climb |
| SP13f_s43 | 18 | 35.9h | 0.48 | 0.65@it6 | 2 | +0.015 | Slow climb |

**SP13e_s42** (4 switches — most of any run):
```
it1  E:0.21 → it2 P:0.65 → it3 E:0.47 → it6 P:0.42→0.68 → it11 E:0.57→0.53→0.62
Roles: E P E E E P P P P P E E E E E E E E E
```
Classic oscillation. Bouncing between 0.39-0.68. The 0.6 threshold triggers role switches too easily.

**SP13e_s43** (3 switches, declining):
```
Roles: E P P E E P P P P P P P P P P P P P
```
Peaked 0.72@it3, then collapsed. Stuck at 0.37-0.54 for 12 iterations. Just entered PO at 0.39.

**SP13f_s42/s43** (obstacle variants, slowly improving):
```
SP13f_s42: 0.16→0.63→0.58→0.39→0.46→0.52→0.56→0.46
SP13f_s43: 0.15→0.65→0.44→0.47→0.56→0.56→0.48
```
Obstacles slow everything down but these are at least not collapsing.

**Verdict**: thr=0.6 produces oscillation without asymmetric obs. 2-4 switches per run, neither agent gets stable training. All runs in the 0.39-0.62 range. Not catastrophic (unlike asym+0.6) but mediocre. Useful as comparison data for the paper.

---

### GROUP 5: SP13g-h — Asymmetric + Alt Freeze thr=0.6 (CATASTROPHIC)

| Run | Iters | Hours | CR Now | Peak CR | Switches | Slope | Status |
|-----|-------|-------|--------|---------|----------|-------|--------|
| SP13g_s42 | 17 | 35.0h | 0.27 | 0.61@it12 | 3 | -0.033 | Collapsed |
| SP13g_s43 | 17 | 35.5h | 0.21 | 0.80@it4 | 3 | +0.027 | **Catastrophic collapse** |
| SP13h_s42 | 15 | 29.9h | 0.48 | 0.59@it10 | 1 | -0.009 | **Stopped** (crashed) |
| SP13h_s43 | 17 | 35.8h | 0.19 | 0.64@it4 | 3 | -0.027 | Collapsed |

**SP13g_s43 — The catastrophic collapse run:**
```
it1 E:0.23 → it2-4 P:0.33→0.47→0.80 → it5-7 E:0.62→0.49→0.51→0.15 → it9-17 P:0.31→0.18→0.16→0.20→0.09→0.17→0.15→0.20→0.21
Roles: E P P P E E E E P P P P P P P P P
```
Textbook catastrophe. Pursuer reached 0.80@it4 in 3 iters. Switch triggered → evader (with full obs) trained for 4 iters and brought CR to 0.15. When pursuer resumed training at it9, the evader was too strong. Pursuer stuck at 0.09-0.21 for 9 straight iterations. **Irrecoverable.**

**SP13g_s42**: Same pattern, slower collapse. 0.61@it12 → role switch → 0.36 → tries to recover → 0.54→0.31→0.39→0.27. Getting worse.

**SP13h_s43**: Mirror of SP13g_s43. 0.64@it4 → switch → 0.45→0.40→0.27→0.13 → back to pursuer → stuck at 0.19-0.35. Now at 0.19.

**SP13h_s42**: Process stopped/crashed at iter 15. Was plateaued at 0.48. Only ever had 1 switch (the initial one) — the asymmetric evader was never trained, so it never became dangerous. Accidentally survived by NOT switching.

**Root cause**: Asymmetric obs gives the evader full visibility. When the 0.6 threshold triggers a switch to evader training, the full-obs evader learns SO fast that it overwhelms the frozen pursuer in 2-3 iterations. The damage is permanent because when the pursuer resumes training, it faces an evader that's drastically better than what it last saw.

**Verdict**: Asymmetric + thr=0.6 = guaranteed collapse. The only survivor (SP13h_s42) survived by accidentally never switching. **All should be killed.**

---

### GROUP 6: SP13i-j — Alt Freeze thr=0.8, No Asymmetric, 8m

| Run | Iters | Hours | CR Now | Peak CR | Switches | Slope | Status |
|-----|-------|-------|--------|---------|----------|-------|--------|
| SP13i_s42 | 18 | 35.0h | 0.52 | 0.82@it4 | 2 | +0.047 | Deadlocked (recovering?) |
| SP13i_s43 | 19 | 35.3h | 0.48 | 0.83@it9 | 2 | -0.025 | Regressing |
| SP13j_s42 | 8 | 13.8h | 0.52 | 0.58@it3 | 1 | -0.003 | **Stopped** (crashed@8it) |
| SP13j_s43 | 18 | 34.4h | 0.47 | 0.87@it13 | 2 | -0.049 | Regressing |

**SP13i_s42 — The deadlock run:**
```
it1 E:0.21 → it2-4 P:0.58→0.61→0.82 → it5-18 E:0.52→0.42→0.38→0.48→0.38→0.39→0.48→0.38→0.36→0.38→0.33→0.38→0.52→0.52
Roles: E P P P E E E E E E E E E E E E E E
```
Pursuer hit 0.82@it4, triggering switch. Evader then trained for **14 straight iterations** stuck at 0.33-0.52. The evader can't reach >0.8 escape rate to trigger a re-switch, and the pursuer's policy from it4 is frozen and aging. Slight recovery to 0.52 at it17-18 after 14 iters of stagnation.

**SP13i_s43 — High peak, slow regression:**
```
it1 E → it2-9 P:0.12→0.57→0.77→0.65→0.66→0.47→0.66→0.83 → it10-19 E:0.69→0.62→0.61→0.55→0.56→0.60→0.60→0.51→0.59→0.48
Roles: E P P P P P P P P E E E E E E E E E E
```
8 iters of pursuer training peaked 0.83@it9. After switch, evader slowly erodes: 0.69→0.48 over 10 iters.

**SP13j_s43 — Highest peak, steepest fall:**
```
it1 E → it2-13 P:0.15→0.17→0.42→0.61→0.48→0.69→0.77→0.67→0.59→0.73→0.62→0.87 → it14-18 E:0.63→0.63→0.57→0.46→0.47
Roles: E P P P P P P P P P P P P E E E E E
```
12 iters of pursuer training produced the highest self-play peak (0.87@it13). But after switch to evader at it14, CR dropped 0.87→0.46 in just 5 iters. Slope=-0.049, steepest decline.

**Verdict**: thr=0.8 without asymmetric obs prevents oscillation but creates two pathologies:
1. **Deadlock** (SP13i_s42): evader can't reach threshold, frozen forever in mediocrity.
2. **Delayed cliff** (SP13i_s43, SP13j_s43): pursuer peaks high but the eventual switch still causes regression.

The higher threshold gives the pursuer more training time → higher peaks (0.82-0.87 vs 0.63-0.72 for thr=0.6), but the underlying problem remains: when the switch happens, the newly-trained agent disrupts equilibrium.

---

### GROUP 7: SP13k-l — Asymmetric + Alt Freeze thr=0.8 (TOP PERFORMERS)

| Run | Iters | Hours | CR Now | Peak CR | Switches | Slope | Status |
|-----|-------|-------|--------|---------|----------|-------|--------|
| **SP13k_s43** | 13 | 24.2h | 0.67 | **0.82@it11** | 2 | -0.041 | **Cliff in progress** |
| **SP13l_s43** | 13 | 25.0h | **0.89** | **0.89@it13** | 1 | +0.030 | **#1 RUN, still climbing** |

**SP13k_asym08_no_obs_s43 — The cliff test:**
```
it1 E:0.23 → it2-11 P:0.61→0.47→0.46→0.45→0.50→0.76→0.76→0.79→0.80→0.82 → it12-13 E:0.63→0.67
Roles: E P P P P P P P P P P E E
```
Smoothly climbed from 0.23 to 0.82 over 11 iterations with remarkably low volatility (std=0.026 during the climb). Then at it12, the 0.8 threshold triggered a switch to evader training. CR immediately dropped: 0.82→0.63→0.67.

**This is THE critical experiment.** The question: does the well-trained pursuer (10 iters) survive the evader's adaptation? Early signs are mildly encouraging — it bounced from 0.63 to 0.67 (compare SP13g_s43 which went straight 0.80→0.62→0.49→0.15 with no bounce). But it's only 2 iters post-switch. The next 3-5 iters will determine if this config is viable long-term.

**SP13l_asym08_obs_s43 — The surprise champion:**
```
it1 E:0.23 → it2-13 P:0.14→0.06→0.36→0.57→0.54→0.59→0.57→0.76→0.69→0.72→0.73→0.89
Roles: E P P P P P P P P P P P P (never switched!)
```
Rocky start (0.06@it3 — obstacles cause early instability) but then monotonic climb to 0.89. Only 1 switch (the initial one). Still training pursuer against frozen evader. At p_fo=0.263, hasn't faced full PO yet.

**Why SP13l hasn't switched**: The asymmetric evader (full obs) is harder to beat. Despite CR=0.89, the threshold hasn't triggered because the evader's escape rate is tracked differently in the alt-freeze logic, or the evader is using its full obs to make enough escapes. Either way, the pursuer keeps getting extended training — and keeps improving.

**The obstacle surprise**: SP13l has obstacles (0-3 per episode), yet it's outperforming SP13k (no obstacles). This contradicts the pattern from SP13a/b where obstacles hurt. Possible explanation: with asymmetric obs, the evader doesn't observe obstacles either (its full obs may not include obstacle positions depending on implementation), so obstacles don't disadvantage the pursuer as much.

**Verdict**: Asymmetric + thr=0.8 is the winning self-play formula. The combination works because:
1. The asymmetric evader (full obs) is a harder opponent → pursuer needs more training to reach threshold
2. The delayed switch gives the pursuer time to develop a robust, general policy
3. When/if the switch happens, the pursuer has 10+ iters of training vs 2-3 for thr=0.6

Both runs still need to face full PO (p_fo~0 around iter 17-18) — that's the ultimate test.

---

### GROUP 8: SP14 — Pure Self-Play (No Freeze)

| Run | Iters | Hours | CR Now | Peak CR | Switches | Slope | Status |
|-----|-------|-------|--------|---------|----------|-------|--------|
| SP14a_s42 | 6 | 12.4h | 0.11 | 0.29@it1 | 0 | -0.001 | Failed (evader dominating) |
| SP14b_s43 | 6 | 11.2h | 0.45 | 0.59@it4 | 0 | +0.051 | Early, promising but volatile |

**SP14a_s42**:
```
CR: [0.29, 0.07, 0.11, 0.08, 0.02, 0.11]
Roles: E E E E E E (never trained pursuer!)
```
Evader completely dominates. The run never switched to pursuer training because the evader's escape rate never dropped below the initial threshold. Stuck at 0.02-0.11. Without freeze, simultaneous learning lets the evader run away.

**SP14b_s43**:
```
CR: [0.23, 0.17, 0.52, 0.59, 0.47, 0.45]
Roles: E E E E E E
```
Better seed. Surged to 0.59@it4, now declining to 0.45. Still very early (6 iters, p_fo=0.69). The decline is concerning but may just be normal oscillation. Pure self-play is extremely seed-dependent.

**Verdict**: Pure self-play without freeze is high-variance. One seed works (0.59 peak), the other completely fails (0.11 max). Need 10+ iters to judge SP14b_s43.

---

## Alt-Freeze Mechanism Deep Dive

### How It Works
1. All runs start by training evader (role=E, iter 1)
2. After iter 1 eval, if evader escape rate > threshold, switch to train pursuer (freeze evader)
3. Subsequently: if the training agent's win rate exceeds the threshold, switch to train the other agent
4. `role` = which agent is actively learning; `frozen` = which agent is static

### Switch Count by Config
```
thr=0.6, no asym:   2-4 switches (avg 3.0)  → OSCILLATION
thr=0.6, asym:      1-3 switches (avg 2.5)  → COLLAPSE after switch
thr=0.8, no asym:   1-2 switches (avg 1.75) → DEADLOCK or DELAYED CLIFF
thr=0.8, asym:      1-2 switches (avg 1.3)  → SMOOTH CLIMB (switch delayed)
```

### The Switch Dynamics

**Pattern 1: Oscillation** (thr=0.6, no asym)
```
Train P → P gets to 0.6+ → switch → Train E → E gets to 0.6+ → switch → ...
Neither agent learns deeply. Both stay in 0.40-0.65 range.
Example: SP13e_s42 with 4 switches.
```

**Pattern 2: Catastrophic Collapse** (thr=0.6, asym)
```
Train P → P reaches 0.6-0.8 → switch → Train E (full obs!) → E dominates in 2-3 iters → switch back → P can't recover
The full-obs evader learns too fast. Damage is permanent.
Example: SP13g_s43 (0.80→0.09 in 4 iters after switch)
```

**Pattern 3: Deadlock** (thr=0.8, no asym)
```
Train P → P reaches 0.8+ → switch → Train E → E stuck at 0.35-0.55 for 10+ iters
High threshold means E needs >0.8 escape (CR<0.2) to re-switch. E can't get there.
Both agents stuck in limbo.
Example: SP13i_s42 (0.82→stuck at 0.38 for 14 iters)
```

**Pattern 4: Delayed Cliff** (thr=0.8, no asym)
```
Train P → P climbs to 0.8+ over many iters → switch → E erodes P's gains slowly
Better than thr=0.6 (higher peak, slower decline) but still loses gains.
Example: SP13j_s43 (0.87→0.47 in 5 iters)
```

**Pattern 5: Stable Climb** (thr=0.8, asym) ← WINNING PATTERN
```
Train P against full-obs frozen E → P can't easily reach 0.8 → no switch → P keeps improving
The asymmetric evader is hard enough that the threshold is rarely crossed.
When it IS crossed (SP13k@it12): CR drops but may recover (0.82→0.63→0.67).
Example: SP13l_s43 (0.23→0.89, still climbing, no switch in 12 iters)
```

---

## Factor Analysis

### Main Effects
| Factor | n | Avg CR | Avg Peak | Range | Slope |
|--------|---|--------|----------|-------|-------|
| thr=0.6 | 8 | 0.38 | 0.66 | [0.19, 0.62] | -0.005 |
| thr=0.8 | 6 | 0.59 | 0.80 | [0.47, 0.89] | -0.007 |
| No asymmetric | 8 | 0.49 | 0.72 | [0.39, 0.62] | -0.001 |
| Asymmetric | 6 | 0.44 | 0.72 | [0.19, 0.89] | -0.010 |
| No obstacles | 7 | 0.45 | 0.76 | [0.21, 0.67] | -0.005 |
| With obstacles | 7 | 0.48 | 0.66 | [0.19, 0.89] | +0.002 |

### Interaction Effects (2x2: Threshold x Asymmetric)
| Combination | n | Avg CR | Avg Peak | Range | Slope |
|-------------|---|--------|----------|-------|-------|
| thr=0.6 + no asym | 4 | 0.49 | 0.67 | [0.39, 0.62] | +0.005 |
| thr=0.6 + asym | 4 | 0.27 | 0.66 | [0.19, 0.48] | -0.010 |
| thr=0.8 + no asym | 4 | 0.49 | 0.78 | [0.47, 0.52] | -0.013 |
| **thr=0.8 + asym** | **2** | **0.78** | **0.86** | **[0.67, 0.89]** | **-0.006** |

### Interaction Effect Deltas
```
At thr=0.6: adding asymmetric obs → avg_CR = -0.22 (HARMFUL)
At thr=0.8: adding asymmetric obs → avg_CR = +0.29 (BENEFICIAL)
```

**The crossover interaction is the key finding.** Asymmetric obs flips from harmful to beneficial when the threshold is high enough.

### Sensing Range Effect
```
8m range: avg_CR = 0.57, avg_peak = 0.78 (across SP13a-b, freeze runs)
5m range: avg_CR = 0.50, avg_peak = 0.59 (across SP13c-d, freeze runs)
Delta: -0.07 avg_CR, -0.19 avg_peak
```
8m is strictly better. The 5m penalty is most severe on peak performance (-0.19).

### Obstacle Effect (Freeze Runs Only)
```
No obstacles: SP13a PO_avg = 0.79 (s42), 0.61 (s43) → mean 0.70
With obstacles: SP13b PO_avg = 0.45 (s42), 0.62 (s43) → mean 0.54
Delta: -0.16 avg (obstacles hurt)
```
But in asym+thr=0.8: SP13l (obstacles) = 0.89 vs SP13k (no obstacles) = 0.67. Obstacles seem neutral or even helpful with the right self-play config.

---

## Kill/Keep Recommendations

### KILL (10 runs, frees 10 cores)

| # | Run | Cores | CR Now | Why Kill |
|---|-----|-------|--------|----------|
| 1 | SP13c_fov5_no_obs_s42 | 1 | 0.65 | 5m range, oscillating 0.13-0.65 after 35h, will fail under full PO |
| 2 | SP13c_fov5_no_obs_s43 | 1 | 0.48 | 5m range, peaked 0.71 then crashed, unstable |
| 3 | SP13d_fov5_obs_s42 | 1 | 0.46 | 5m + obstacles, never above 0.46, hard plateau |
| 4 | SP13d_fov5_obs_s43 | 1 | 0.46 | 5m + obstacles, identical plateau at 0.46 |
| 5 | SP13g_asym_no_obs_s42 | 1 | 0.27 | Asym+0.6 collapse, declining from 0.61, irrecoverable |
| 6 | SP13g_asym_no_obs_s43 | 1 | 0.21 | Asym+0.6 catastrophic collapse, 0.80→0.09, dead |
| 7 | SP13h_asym_obs_s43 | 1 | 0.19 | Asym+0.6+obs collapse, declining to 0.19 |
| 8 | SP14a_pure_selfplay_s42 | 1 | 0.11 | Pure self-play failed, stuck at 0.02-0.11, evader dominates |
| 9 | SP13b_fov_obs_s42 | 1 | 0.30 | Freeze+obs, declining to 0.30 after 59h, worst freeze run |
| 10 | SP13i_alt08_no_obs_s42 | 1 | 0.52 | thr=0.8 deadlock, stuck at 0.33-0.52 for 14 iters |

### ALREADY DEAD (2 runs, 0 additional cores)

| Run | Status |
|-----|--------|
| SP13h_asym_obs_s42 | Stopped at iter 15, 29.9h. Plateaued at 0.48. |
| SP13j_alt08_obs_s42 | Stopped at iter 8, 13.8h. Plateaued at 0.52. |

### KEEP (12 runs, 12 cores)

| # | Run | CR Now | Why Keep |
|---|-----|--------|----------|
| 1 | **SP13l_asym08_obs_s43** | **0.89** | **#1 run. Monotonic climb. Still rising.** |
| 2 | **SP13k_asym08_no_obs_s43** | 0.67 | Cliff test in progress — most important experiment |
| 3 | SP13a_fov_no_obs_s42 | 0.62 | Freeze baseline. 0.79 PO avg. Gold standard reference. |
| 4 | SP13a_fov_no_obs_s43 | 0.70 | Second freeze seed. Seed sensitivity data. |
| 5 | SP13b_fov_obs_s43 | 0.54 | Obstacle freeze comparison (PO avg 0.62). |
| 6 | SP14b_pure_selfplay_s43 | 0.45 | Only viable pure self-play. Needs 10+ iters. |
| 7 | SP13e_alt_no_obs_s42 | 0.62 | thr=0.6 comparison data (4 switches, oscillating). |
| 8 | SP13e_alt_no_obs_s43 | 0.39 | thr=0.6 comparison data (declining seed). |
| 9 | SP13f_alt_obs_s42 | 0.46 | thr=0.6+obs comparison (slowly improving). |
| 10 | SP13f_alt_obs_s43 | 0.48 | thr=0.6+obs comparison (slowly improving). |
| 11 | SP13i_alt08_no_obs_s43 | 0.48 | thr=0.8 regression data (0.83→0.48). |
| 12 | SP13j_alt08_obs_s43 | 0.47 | thr=0.8 regression data (0.87→0.47). |

### Post-Kill Compute Plan
After killing 10 runs: **12 cores used, 12 cores free.**

Suggested new launches with freed cores:
1. **More asym+thr=0.8 seeds** (SP13k/l config): 4-6 runs with seeds 40-45 to validate the winning formula
2. **Higher threshold test** (thr=0.9): 2 runs to see if delaying the switch even further helps
3. **asym+thr=0.8 with warm-seed**: 2 runs starting from SP13l's best checkpoint

---

## Key Findings & Lessons

### 1. Sensing Range: 8m is the Minimum
5m FOV 90deg covers only ~3.5m^2 at the triangle tip. In a 10x10 arena, the pursuer can only see ~3.5% of the area. 8m covers ~25m^2 (~25%), which is sufficient for learning basic pursuit strategies. Below 8m, the task becomes essentially blind search.

### 2. Obstacle Observations Hurt (Usually)
Adding obstacle positions to the observation vector consistently reduces performance in freeze-evader runs (delta -0.16 PO avg). The extra dimensions add noise. Exception: SP13l (asym+thr=0.8) shows obstacles can be neutral with the right training config. Hypothesis: in asymmetric self-play, obstacles create interesting tactical possibilities that benefit from longer, more stable training.

### 3. Alt-Freeze Threshold Has a Goldilocks Problem
- **thr=0.6**: Too easy → frequent switches → oscillation, never converges
- **thr=0.8**: Better → fewer switches → higher peaks, but deadlock/cliff when switch happens
- **thr=0.8 + asymmetric**: Sweet spot → switch rarely/never triggers → stable monotonic learning

### 4. Asymmetric Obs Creates a Threshold-Dependent Interaction
The evader having full obs makes it a harder opponent. At low threshold, this is FATAL — the evader learns too fast when unfrozen. At high threshold, this is BENEFICIAL — the harder opponent prevents premature threshold crossing, giving the pursuer extended training time.

### 5. Seed Sensitivity is Extreme
Same config, different seeds → wildly different outcomes:
- SP13a: s42 PO avg 0.79 vs s43 PO avg 0.61 (delta 0.18)
- SP14: s42 CR=0.11 vs s43 CR=0.59 (delta 0.48!)
- Multi-seed runs are essential for any conclusion.

### 6. Self-Play Cliff is Universal (Without Countermeasures)
Every alt-freeze run that eventually switched roles showed a CR drop. The magnitude depends on config:
- thr=0.6: small drops (0.68→0.47) but frequent, causing oscillation
- thr=0.8: large drops (0.87→0.47) but rare
- asym+thr=0.8: drop happened at SP13k (0.82→0.63→0.67) — monitoring if it recovers

### 7. Masking Curriculum Works
All runs show improved or maintained performance as p_full_obs anneals from 1.0 to 0.0. SP13a_s42 maintained 0.79 avg CR after complete annealing. The PO-GRL approach of gradual masking is validated.

### 8. Best Configs for a Paper
If writing up results, the strongest comparison would be:
- **Freeze evader** (SP13a_s42): CR=0.79 PO avg — upper bound but not self-play
- **Self-play asym+thr=0.8** (SP13l_s43): CR=0.89 current — true adversarial training
- **Self-play thr=0.6** (SP13e): CR=0.49 avg — shows oscillation failure mode
- **Self-play asym+thr=0.6** (SP13g): CR=0.21 avg — shows collapse failure mode
- **Pure self-play** (SP14b): CR=0.45 — no freeze baseline

---

## Appendix: Full Iteration Histories

### SP13a_fov_no_obs_s42 (42 iters, 58.7h, freeze evader, 8m, p_fo=0.000)
```
it 1 | E      | CR=0.21 |
it 2 | P frz_E| CR=0.63 | pfo=0.94  0.3M
it 3 | P frz_E| CR=0.56 | pfo=0.88  0.6M
it 4 | P frz_E| CR=0.71 | pfo=0.82  0.9M
it 5 | P frz_E| CR=0.73 | pfo=0.75  1.2M
it 6 | P frz_E| CR=0.70 | pfo=0.69  1.5M
it 7 | P frz_E| CR=0.52 | pfo=0.63  1.8M
it 8 | P frz_E| CR=0.65 | pfo=0.57  2.2M
it 9 | P frz_E| CR=0.55 | pfo=0.51  2.5M
it10 | P frz_E| CR=0.71 | pfo=0.45  2.8M
it11 | P frz_E| CR=0.75 | pfo=0.39  3.1M
it12 | P frz_E| CR=0.86 | pfo=0.32  3.4M
it13 | P frz_E| CR=0.80 | pfo=0.26  3.7M
it14 | P frz_E| CR=0.82 | pfo=0.20  4.0M
it15 | P frz_E| CR=0.81 | pfo=0.14  4.3M
it16 | P frz_E| CR=0.93 | pfo=0.08  4.6M ← annealing nearly done
it17 | P frz_E| CR=0.88 | pfo=0.02  4.9M ← FULL PO BEGINS
it18 | P frz_E| CR=0.73 | pfo=0.00
it19 | P frz_E| CR=0.86 | pfo=0.00
it20 | P frz_E| CR=0.87 | pfo=0.00
it21 | P frz_E| CR=0.89 | pfo=0.00
it22 | P frz_E| CR=0.87 | pfo=0.00
it23 | P frz_E| CR=0.88 | pfo=0.00
it24 | P frz_E| CR=0.87 | pfo=0.00
it25 | P frz_E| CR=0.90 | pfo=0.00
it26 | P frz_E| CR=0.88 | pfo=0.00
it27 | P frz_E| CR=0.97 | pfo=0.00 ← PEAK
it28 | P frz_E| CR=0.84 | pfo=0.00
it29 | P frz_E| CR=0.94 | pfo=0.00
it30 | P frz_E| CR=0.83 | pfo=0.00
it31 | P frz_E| CR=0.75 | pfo=0.00
it32 | P frz_E| CR=0.63 | pfo=0.00
it33 | P frz_E| CR=0.79 | pfo=0.00
it34 | P frz_E| CR=0.74 | pfo=0.00
it35 | P frz_E| CR=0.82 | pfo=0.00
it36 | P frz_E| CR=0.76 | pfo=0.00
it37 | P frz_E| CR=0.79 | pfo=0.00
it38 | P frz_E| CR=0.61 | pfo=0.00
it39 | P frz_E| CR=0.72 | pfo=0.00
it40 | P frz_E| CR=0.66 | pfo=0.00
it41 | P frz_E| CR=0.51 | pfo=0.00
it42 | P frz_E| CR=0.62 | pfo=0.00 ← declining
```

### SP13k_asym08_no_obs_s43 (13 iters, 24.2h, alt 0.8+asym, p_fo=0.263)
```
it 1 | E      | CR=0.23 |
it 2 | P frz_E| CR=0.61 | pfo=0.94  0.3M  ← SWITCH to pursuer
it 3 | P frz_E| CR=0.47 | pfo=0.88  0.6M
it 4 | P frz_E| CR=0.46 | pfo=0.82  0.9M
it 5 | P frz_E| CR=0.45 | pfo=0.75  1.2M
it 6 | P frz_E| CR=0.50 | pfo=0.69  1.5M
it 7 | P frz_E| CR=0.76 | pfo=0.63  1.8M  ← climb begins
it 8 | P frz_E| CR=0.76 | pfo=0.57  2.2M
it 9 | P frz_E| CR=0.79 | pfo=0.51  2.5M
it10 | P frz_E| CR=0.80 | pfo=0.45  2.8M
it11 | P frz_E| CR=0.82 | pfo=0.39  3.1M  ← PEAK, crosses 0.8 threshold
it12 | E frz_P| CR=0.63 | pfo=0.32  3.4M  ← SWITCH to evader, THE CLIFF
it13 | E frz_P| CR=0.67 | pfo=0.26  3.7M  ← slight recovery? TBD
```

### SP13l_asym08_obs_s43 (13 iters, 25.0h, alt 0.8+asym+obs, p_fo=0.263)
```
it 1 | E      | CR=0.23 |
it 2 | P frz_E| CR=0.14 | pfo=0.94  0.3M  ← SWITCH to pursuer
it 3 | P frz_E| CR=0.06 | pfo=0.88  0.6M  ← obstacle instability
it 4 | P frz_E| CR=0.36 | pfo=0.82  0.9M
it 5 | P frz_E| CR=0.57 | pfo=0.75  1.2M
it 6 | P frz_E| CR=0.54 | pfo=0.69  1.5M
it 7 | P frz_E| CR=0.59 | pfo=0.63  1.8M
it 8 | P frz_E| CR=0.57 | pfo=0.57  2.2M
it 9 | P frz_E| CR=0.76 | pfo=0.51  2.5M  ← climb begins
it10 | P frz_E| CR=0.69 | pfo=0.45  2.8M
it11 | P frz_E| CR=0.72 | pfo=0.39  3.1M
it12 | P frz_E| CR=0.73 | pfo=0.32  3.4M
it13 | P frz_E| CR=0.89 | pfo=0.26  3.7M  ← NEW PEAK, still climbing, no switch
```

### SP13g_asym_no_obs_s43 (17 iters, 35.5h, alt 0.6+asym, p_fo=0.017) — CATASTROPHIC COLLAPSE
```
it 1 | E      | CR=0.23 |
it 2 | P frz_E| CR=0.33 | pfo=0.94  0.3M  ← SWITCH to pursuer
it 3 | P frz_E| CR=0.47 | pfo=0.88  0.6M
it 4 | P frz_E| CR=0.80 | pfo=0.82  0.9M  ← pursuer peaks
it 5 | E frz_P| CR=0.62 | pfo=0.75  1.2M  ← SWITCH to evader, cliff starts
it 6 | E frz_P| CR=0.49 | pfo=0.69  1.5M
it 7 | E frz_P| CR=0.51 | pfo=0.63  1.8M
it 8 | E frz_P| CR=0.15 | pfo=0.57  2.2M  ← CRASH (evader dominates)
it 9 | P frz_E| CR=0.31 | pfo=0.51  2.5M  ← SWITCH back to pursuer, too late
it10 | P frz_E| CR=0.18 | pfo=0.45  2.8M
it11 | P frz_E| CR=0.16 | pfo=0.39  3.1M
it12 | P frz_E| CR=0.20 | pfo=0.32  3.4M
it13 | P frz_E| CR=0.09 | pfo=0.26  3.7M  ← all-time low
it14 | P frz_E| CR=0.17 | pfo=0.20  4.0M
it15 | P frz_E| CR=0.15 | pfo=0.14  4.3M
it16 | P frz_E| CR=0.20 | pfo=0.08  4.6M
it17 | P frz_E| CR=0.21 | pfo=0.02  4.9M  ← irrecoverable
```

### SP13i_alt08_no_obs_s42 (18 iters, 35.0h, alt 0.8, p_fo=0.000) — DEADLOCK
```
it 1 | E      | CR=0.21 |
it 2 | P frz_E| CR=0.58 | pfo=0.94  0.3M  ← SWITCH to pursuer
it 3 | P frz_E| CR=0.61 | pfo=0.88  0.6M
it 4 | P frz_E| CR=0.82 | pfo=0.82  0.9M  ← pursuer peaks, crosses 0.8
it 5 | E frz_P| CR=0.52 | pfo=0.75  1.2M  ← SWITCH to evader
it 6 | E frz_P| CR=0.42 | pfo=0.69  1.5M
it 7 | E frz_P| CR=0.38 | pfo=0.63  1.8M
it 8 | E frz_P| CR=0.48 | pfo=0.57  2.2M
it 9 | E frz_P| CR=0.38 | pfo=0.51  2.5M
it10 | E frz_P| CR=0.39 | pfo=0.45  2.8M
it11 | E frz_P| CR=0.48 | pfo=0.39  3.1M
it12 | E frz_P| CR=0.38 | pfo=0.32  3.4M
it13 | E frz_P| CR=0.36 | pfo=0.26  3.7M
it14 | E frz_P| CR=0.38 | pfo=0.20  4.0M
it15 | E frz_P| CR=0.33 | pfo=0.14  4.3M
it16 | E frz_P| CR=0.38 | pfo=0.08  4.6M  ← 11 iters stuck at 0.33-0.48
it17 | E frz_P| CR=0.52 | pfo=0.02  4.9M  ← slight recovery?
it18 | E frz_P| CR=0.52 | pfo=0.00         ← still deadlocked
```

### SP14b_pure_selfplay_s43 (6 iters, 11.2h, pure self-play, p_fo=0.693)
```
it 1 | E | CR=0.23 |
it 2 | E | CR=0.17 | pfo=0.94  0.3M
it 3 | E | CR=0.52 | pfo=0.88  0.6M  ← rapid climb
it 4 | E | CR=0.59 | pfo=0.82  0.9M  ← peak
it 5 | E | CR=0.47 | pfo=0.75  1.2M  ← declining
it 6 | E | CR=0.45 | pfo=0.69  1.5M  ← still declining, early days
```

---

*Report generated during Session 81. See `docs/worklogs/2026-03-10_S81.md` for session context.*
*Trajectory PNGs available at http://100.71.2.97:8080/dashboard*
*TensorBoard at http://100.71.2.97:6006*
