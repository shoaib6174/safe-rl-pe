# SP11e Training Report

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | SAC |
| **Self-play mode** | Alternate freeze (`--alternate_freeze`) |
| **Freeze switch threshold** | 0.9 (effectively never switches) |
| **Arena** | 10×10m |
| **Speed** | Equal (pursuer 1.0, evader 1.0) |
| **Obstacles** | 0–3 random per episode, 3 in observation |
| **Partial obs** | Combined masking (sensing radius 3.0m + LOS) |
| **Max steps/episode** | 2000 (100 seconds) |
| **Micro-phase steps** | 2048 |
| **Eval interval** | Every 150 micro-phases (100 episodes) |
| **Opponent pool** | 50 (reservoir sampling) |
| **Collapse rollback** | threshold=0.10, streak_limit=5 |
| **PFSP-lite** | Enabled |
| **SAC hyperparams** | lr=3e-4, ent_coef=0.03, train_freq=1, gradient_steps=1, buffer=1M |
| **Seed** | 42, n_envs=4 |
| **Max total steps** | 30M |
| **Shaping rewards** | None (no w_vis_pursuer, no w_search) |
| **Cold-start eval** | SR_P=0.22 (random vs random) |
| **Output** | `results/SP11e_sac_maxsteps2000/` |

### Key Difference from SP11d
Two changes from SP11d: `freeze_switch_threshold=0.9` (vs 0.6) and `max_steps=2000` (vs 1000). The high threshold means the evader never unfreezes (pursuer never reaches 90%), giving the pursuer uninterrupted training against a fixed random evader.

## Training Timeline

### Stage 1: Rapid Learning (M0–M900, 0–1.8M steps)
- Fast initial ramp from random (22%) to 76%
- Pursuer quickly learned basic search-and-capture against random evader
- New best set at every eval: 38% → 58% → 67% → **76%**

| Eval | SR_P | Steps | Note |
|------|------|-------|------|
| M150 | 0.380 | 307K | First eval |
| M300 | 0.580 | 614K | +20pp |
| M750 | 0.670 | 1.5M | |
| M900 | **0.760** | 1.8M | Early peak |

### Stage 2: First Plateau (M1050–M2100, 2.2M–4.3M steps)
- SR_P oscillated between 56–65%
- Dipped from 76% peak, stabilized around 62%
- The pursuer had learned the "easy" captures but was struggling with harder spawn configurations
- 8 consecutive evals in the 56–65% band

| Eval Range | SR_P Range | Avg |
|------------|-----------|-----|
| M1050–M2100 | 0.560–0.650 | 0.62 |

### Stage 3: Second Climb (M2250–M2700, 4.6M–5.5M steps)
- Broke through the first plateau
- Steady improvement: 62% → 68% → 72% → 73%
- Pursuer developing more systematic search patterns

| Eval | SR_P |
|------|------|
| M2250 | 0.620 |
| M2400 | 0.680 |
| M2550 | 0.720 |
| M2700 | 0.730 |

### Stage 4: Second Plateau (M2850–M4950, 5.8M–10.1M steps)
- The longest plateau: 14 evals spanning 4.3M steps
- SR_P oscillated between 46–67%, averaging ~59%
- Dipped as low as 46% (M4200, M4500) — significant variance
- No clear upward trend for ~4M steps

| Eval Range | SR_P Range | Avg | Duration |
|------------|-----------|-----|----------|
| M2850–M4950 | 0.460–0.670 | 0.59 | 4.3M steps |

### Stage 5: Third Climb — Breakthrough (M5250–M7200, 10.8M–14.7M steps)
- Broke through the second plateau decisively
- Steady upward trend: 68% → 70% → 72% → 76% → 79% → **85%**
- New all-time bests set at M6750 (79%) and M7050 (85%)
- The pursuer appears to have developed qualitatively better search strategies

| Eval | SR_P | Note |
|------|------|------|
| M5250 | 0.680 | Climb begins |
| M5550 | 0.700 | First 70%+ |
| M6000 | 0.720 | |
| M6300 | 0.760 | |
| M6750 | **0.790** | New best |
| M7050 | **0.850** | **All-time high** |
| M7200 | 0.750 | Slight dip |

## SR_P Trajectory (Full History)

```
SR_P
0.85 ┤                                                           ·
0.80 ┤                                                         ·
0.75 ┤   ·                                               · ·
0.70 ┤                                            ··  ··    ·  ·
0.65 ┤  ·  ··  ·     · ·       · ·           ··         ·
0.60 ┤    ·  ··  ··· ·  ·· ··    · ··  · ·
0.55 ┤       ·    ·    ·           ·  ·
0.50 ┤                                 · ·
0.45 ┤                                ·
0.40 ┤
0.35 ┤·
     └──────────────────────────────────────────────────────────
      M0    M1500   M3000   M4500   M6000   M7050
      (0)   (3M)    (6M)    (9M)    (12M)   (14.4M)
```

## Best Model Checkpoints

| Milestone | SR_P | Steps | Note |
|-----------|------|-------|------|
| M150 | 0.380 | 307K | First |
| M300 | 0.580 | 614K | |
| M750 | 0.670 | 1.5M | |
| M900 | 0.760 | 1.8M | Early peak |
| M6750 | 0.790 | 13.8M | Broke plateau |
| **M7050** | **0.850** | **14.4M** | **Current best** |

## Behavior Analysis

### What the pursuer learned

Based on GIF analysis across multiple checkpoints:

1. **Active search**: The pursuer moves purposefully through the arena rather than standing still. It sweeps toward likely evader positions and changes direction when the evader isn't found.

2. **Detection → pursuit transition**: When the evader enters the 3m sensing cone with clear LOS, the pursuer immediately switches from search to direct pursuit.

3. **Obstacle navigation**: The pursuer routes around obstacles rather than getting stuck. In 0-obstacle episodes, captures are faster (56–250 steps). With 2-3 obstacles, captures take longer (400–600+ steps) but still succeed.

4. **Capture in open space**: At equal speed, the pursuer catches the random evader by cutting angles and predicting straight-line movement. The random evader has no evasion strategy, making interception relatively straightforward once detected.

### Why ~15-25% of episodes still escape

The remaining escapes come from unfavorable spawn configurations:
- **Large initial distance**: Evader spawns far away (up to 15m in a 10m arena diagonal ~14m). With 3m sensing, the pursuer must search most of the arena.
- **Obstacle occlusion**: Even when close, 2-3 obstacles can block LOS, causing the pursuer to lose track and the random evader to wander away.
- **Time pressure**: 2000 steps (100s) at 1.0 m/s means the pursuer covers ~100m total. In a 10×10 arena, a systematic sweep requires ~50-60m minimum. Unlucky search paths waste time.

### GIF evidence

| GIF | Eval SR | GIF Capture Rate | Observations |
|-----|---------|-------------------|-------------|
| `20260304_SP11e_71pct` | 71% | 8/9 (89%) | Fast captures (208-592 steps), 1 escape |
| `20260304_SP11e_M6750_79pct` | 79% | 8/9 (89%) | Similar pattern, 1 escape |
| `20260304_SP11e_M7050_85pct` | 85% | 8/9 (89%) | Fastest yet (56-615 steps), 1 escape |

GIF capture rate consistently higher than eval SR (89% vs 71-85%) due to small sample (9 episodes vs 100).

## Training Dynamics

### Learning rate profile
- **Fast phase**: 0–1.8M steps (38% → 76%) — 38pp in 1.8M steps = **21pp/M steps**
- **Plateau 1**: 2.2M–4.3M — 2.1M steps for 0pp net gain
- **Climb 2**: 4.6M–5.5M — 0.9M steps for +11pp = **12pp/M steps**
- **Plateau 2**: 5.8M–10.1M — 4.3M steps for 0pp net gain
- **Climb 3**: 10.8M–14.4M — 3.6M steps for +19pp = **5pp/M steps**

**Key insight**: Learning is not monotonic. The pursuer goes through extended plateaus (2-4M steps) followed by breakthrough climbs. Each climb produces diminishing gains but reaches higher peaks. This suggests the pursuer is discovering qualitatively new strategies at each breakthrough rather than incrementally optimizing.

### Variance analysis
- **High variance throughout**: ±15pp swings common (e.g., M4200=46% → M4350=64%)
- **Eval noise**: 100-episode eval produces ±5-8% statistical noise for true rates around 60-70%
- **Variance decreasing over time**: Stage 5 shows tighter oscillation (71-85%) vs Stage 2 (46-67%)

## Comparison with SP11d

| Metric | SP11d | SP11e |
|--------|-------|-------|
| Threshold | 0.6 | 0.9 |
| Max steps | 1000 | 2000 |
| Freeze switch | M2850 (fatal) | Never |
| Peak SR_P | 0.620 | **0.850** |
| Steps to peak | 5.8M | 14.4M |
| Final SR_P | 0.070 (collapsed) | 0.750 (running) |
| Evader pool | E30 (contaminated) | E0 (empty) |

The two config differences fully explain the divergence:
1. **threshold=0.9** prevented the premature freeze switch that killed SP11d
2. **max_steps=2000** gave the pursuer double the search time, critical for the find-then-catch task under partial observability

## Current Status & Outlook

- **Running**: 14.7M / 30M steps, 287 steps/s, ~880 minutes elapsed
- **Still improving**: The M7050 peak (85%) was set recently, and the plateau-breakthrough pattern suggests further peaks are possible
- **Diminishing returns**: Each climb takes longer and yields less. Likely approaching asymptotic performance against a random evader.
- **Ceiling estimate**: 85-90% is likely the ceiling. The remaining 10-15% escapes are from inherently hard configurations (large distance + obstacle occlusion) that may be unsolvable within 2000 steps at equal speed.

### What happens next?
SP11e demonstrates the pursuer can learn effective search-and-capture against a **random** evader. The critical question is what happens when the evader unfreezes:
- If threshold stays at 0.9, the pursuer continues training indefinitely against random evader
- The real test is co-evolution — can this strong pursuer maintain performance against a training evader?
- SP11d showed that even 62% pursuer collapsed instantly when the evader got 307K steps of training
- SP11e at 85% may be more robust, but the same dynamics could apply

## Key Lessons

| Lesson | Evidence |
|--------|----------|
| **Uninterrupted training works** | 14.4M steps of pursuer-only training reached 85%, whereas SP11d collapsed after a single switch at 5.8M |
| **Plateaus break eventually** | Two major plateaus (2M and 4M steps) both led to breakthroughs. Patience pays off. |
| **max_steps=2000 is critical** | Double the episode length compared to SP11d. Extra search time converts more "near misses" into captures. |
| **Random evader is not the final test** | 85% vs random ≠ 85% vs trained evader. The real challenge begins at co-evolution. |
| **Diminishing returns are real** | 21pp/M → 12pp/M → 5pp/M. Each breakthrough requires increasingly more training. |

## GIFs Generated

All local at `results_new/SP11e/`:

| File | Checkpoint | GIF Capture |
|------|-----------|-------------|
| `20260304_SP11e_71pct.{gif,png}` | M6450 | 8/9 |
| `20260304_SP11e_M6750_79pct.{gif,png}` | M6750 | 8/9 |
| `20260304_SP11e_M7050_85pct.{gif,png}` | M7050 | 8/9 |
| + 12 earlier GIFs from pre-M6000 | Various | Various |
