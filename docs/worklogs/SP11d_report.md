# SP11d Training Report

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | SAC |
| **Self-play mode** | Alternate freeze (`--alternate_freeze`) |
| **Freeze switch threshold** | 0.6 (switch when training agent's SR > 60%) |
| **Arena** | 10×10m |
| **Speed** | Equal (pursuer 1.0, evader 1.0) |
| **Obstacles** | 0–3 random per episode, 3 in observation |
| **Partial obs** | Combined masking (sensing radius 3.0m + LOS) |
| **Max steps/episode** | 1000 (50 seconds) |
| **Micro-phase steps** | 2048 |
| **Eval interval** | Every 150 micro-phases (100 episodes) |
| **Opponent pool** | 50 (reservoir sampling) |
| **Collapse rollback** | threshold=0.10, streak_limit=5 |
| **PFSP-lite** | Enabled |
| **SAC hyperparams** | lr=3e-4, ent_coef=0.03, train_freq=1, gradient_steps=1, buffer=1M |
| **Seed** | 42, n_envs=4 |
| **Max total steps** | 30M |
| **Shaping rewards** | None (no w_vis_pursuer, no w_search) |
| **Output** | `results/SP11d_sac_maxsteps1000/` |

## Training Timeline

### Phase 1: Pursuer Ramp-Up (M0–M2850, 0–5.8M steps)
- Evader frozen, pursuer training exclusively
- **SR_P oscillated 37–62%**, never sustaining above 60%
- High variance: swings of ±15pp between consecutive evals
- Pool filled to P50/E0 (pursuer snapshots only)
- Peak: **SR_P=0.620 at M2850** (5.8M steps)

| Eval | SR_P | Note |
|------|------|------|
| M150 | 0.400 | First eval |
| M300 | 0.520 | |
| M600 | 0.590 | New best |
| M900 | 0.370 | Dip |
| M1050 | 0.590 | Recovery |
| M2850 | **0.620** | **Peak — triggered freeze switch** |

### Phase 2: Freeze Switch (M2850–M3150)
- At M2850, pursuer SR=0.620 > 0.6 threshold → **pursuer frozen, evader trains**
- Evader trained for just 150 micro-phases (307K steps)
- At M3000: SR_P dropped to 0.320 (evader SR=0.680 > 0.6) → **evader frozen again, pursuer resumes**
- Evader pool jumped from E0 → E30 during this brief window
- **This single switch was catastrophic** — the evader snapshots from M3000 permanently corrupted the learning signal

### Phase 3: Irreversible Collapse (M3150–M7500, 6.5M–15.4M steps)
- Pursuer training resumed with evader frozen, but now evaluating against the M3000 evader (which had 307K steps of training)
- **Immediate drop**: 42% → 24% → 28% → 18% in 4 evals
- **Steady decline**: settled at 7–19% by M4350 (8.9M steps)
- **Never recovered**: 9M+ steps of additional training (M4350→M7500) produced no improvement
- Final 10 evals averaged **~11% SR_P**

| Eval | SR_P | Trend |
|------|------|-------|
| M3150 | 0.420 | Post-switch |
| M3300 | 0.240 | Collapse begins |
| M3600 | 0.180 | |
| M4350 | 0.110 | Bottom |
| M5550 | 0.070 | Worst |
| M5850 | 0.050 | **All-time low** |
| M7500 | 0.070 | Still collapsed at kill |

## SR_P Trajectory (Full History)

```
SR_P
0.70 ┤
0.60 ┤    ·    ·  ·            ·
0.55 ┤ ·    · ·  ·· ·  ·  ·
0.50 ┤  ·  ·       ··  · ·
0.45 ┤       ·        ·  ·
0.40 ┤·              ·       ·
0.35 ┤      ·
0.30 ┤                          ·
0.25 ┤                           · ·
0.20 ┤                              · ·
0.15 ┤                         ▼SWITCH  ·  · ··     ·   ·
0.10 ┤                                    · ··  ··· ·· ·····
0.05 ┤                                          ·  ·       ·
     └─────────────────────────────────────────────────────────
      M0          M2850        M5000        M7500
                  (5.8M)       (10M)        (15.4M)
```

## Root Cause Analysis

### Why the collapse was irreversible

1. **Evader learned enough in 307K steps to break the pursuer's strategy.** The M3000 evader — despite minimal training — learned basic evasion in an open arena at equal speed. The pursuer's M2850 policy (62% SR) was tuned against a *random* evader and couldn't adapt.

2. **Sparse reward death spiral under partial observability.** At equal speed with 3m sensing radius and LOS masking, the pursuer must first *find* the evader, then *catch* it. Against a trained evader, the find→catch pipeline fails more often. As capture rate drops, the pursuer receives fewer positive reward signals, causing the policy to degrade further.

3. **"Giving up" is a local optimum.** With capture reward at ~0 expected return, the pursuer has no incentive to move. Standing still and moving randomly yield similar (near-zero) reward. Without a continuous search incentive, the policy converges to passivity.

4. **max_steps=1000 worsened the problem.** Shorter episodes (50 seconds) gave the pursuer less time to find the evader through random exploration. Compare with SP11e (max_steps=2000, 100 seconds) which held 72-85% against the same frozen random evader.

5. **Pool contamination locked in the failure.** The E30 evader pool entries from M3000 were the only evader snapshots available. Even with PFSP-lite biasing toward weaker opponents, all opponents were from the same brief 307K-step training window — no diversity.

## Key Lessons

| Lesson | Evidence |
|--------|----------|
| **threshold=0.6 is too low** | Pursuer barely crossed 0.6 once (M2850) in 5.8M steps. The switch happened at peak variance, not genuine mastery. |
| **max_steps=1000 insufficient for partial obs search** | SP11e (max_steps=2000) achieved 85% vs SP11d's 62% peak — extra time for search is critical. |
| **A single premature freeze switch can be fatal** | 150 micro-phases of evader training destroyed 5.8M steps of pursuer progress permanently. |
| **No search reward = no recovery** | Once capture becomes sparse, the pursuer has zero gradient signal. SP11g/SP11h (w_search=0.003) address this. |
| **Collapse rollback didn't help** | Despite collapse_threshold=0.10, the pursuer couldn't recover because the underlying problem was reward sparsity, not a bad checkpoint. |

## Comparison with SP11e

| Metric | SP11d | SP11e |
|--------|-------|-------|
| Threshold | 0.6 | 0.9 |
| Max steps | 1000 | 2000 |
| Peak SR_P | 0.620 (M2850) | 0.850 (M7050) |
| Freeze switch | Yes (M2850) | Never (still climbing) |
| Final SR_P | 0.070 | 0.850 |
| Total steps | 15.5M (killed) | 14.4M (running) |

The two differences — threshold and max_steps — fully explain the divergence. SP11e's threshold=0.9 effectively keeps the evader frozen permanently, letting the pursuer train uninterrupted. max_steps=2000 gives double the search time per episode.

## GIFs Generated

GIFs were captured at various points during training (local at `results_new/SP11d/`):

| File | Eval SR_P | Phase |
|------|-----------|-------|
| `20260303_231324_52pct.gif` | 52% | Pre-collapse |
| `20260304_025943_62pct.gif` | 62% | Peak (M2850) |
| Various 47-57% GIFs | — | Oscillation period |

## Verdict

**SP11d is a conclusive failure** due to premature freeze switching. The run demonstrates that at equal speed under partial observability:
- The pursuer needs extensive uninterrupted training (>10M steps) to build search competence
- A 0.6 threshold triggers the switch too early, before the pursuer has robust strategies
- Without continuous search reward shaping, the pursuer cannot recover from capture rate collapse
- max_steps=1000 is insufficient for the search-then-pursue task

**Successor runs**: SP11g (w_search=0.003, threshold=0.6) and SP11h (w_search=0.003, threshold=0.9) test whether staleness-based search reward prevents the "giving up" failure mode.
