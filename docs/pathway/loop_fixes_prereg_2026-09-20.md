# S16 — Phase 2: five remedies for the loop, head to head (2026-09-20, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D125** (the correction), **D126** (escape = rotate-then-move; not our discretisation),
**D127** (the loop survives exploration), **D128** (mechanism: frozen input — raw-world observation delta
0.000, opponent visible on 2.4 % of stalled steps — plus a discount horizon ~6× too short: escaping multiplies
capture 7.1× but 629 steps later, worth 0.009 at γ = 0.99, and γ ≈ 0.9974 would be needed to reach 0.05),
**D113** (a stall penalty at 1e-3 recovered nothing, 0/3).

## 1. The question

Phase 1 identified two candidate causes and excluded two others. This round asks which remedies actually
work, and in particular **whether any learned remedy matches the trivial scripted one**. That comparison, not
the ranking, is the round's point: if a twenty-line recovery beats every learned fix, that is the sentence a
practitioner needs.

## 2. Arms

All training arms share one configuration so the comparison is like-for-like: **`SearchMDP` from scratch**,
CBF shield at defaults, `shield_obs=True`, `r_acq=0`, `hold=10`, PPO with SB3 defaults, **5 M env steps**,
seeds **60, 61, 62**, probes every 500 k against `GreedyEvader` on stream **1 730 000+**, `OMP_NUM_THREADS=2`.

| arm | change from `BASE` | tests |
|---|---|---|
| `BASE` | none — the S5 RC configuration (`stall_penalty = 1e-3`), γ = 0.99 | the control |
| `GAM` | **γ = 0.999** (effective horizon 1000 steps, vs 100) | D128's discount mechanism, directly |
| `OBS` | two extra observation features: the **admissible ω range at v = 0**, `(ω_min, ω_max)` normalised by `ω_max`, computed from the DCBF rows at `v = 0` and clipped to the action box | D128's frozen-input half: tell the policy which way is free |
| `MEM` | `RecurrentPPO` (LSTM), same features as `BASE` | can memory alone break the loop? |
| `PEN` | `stall_penalty = 1e-2` (10× `BASE`) | does simply penalising immobility harder work? Replicates and extends D113's negative |
| `REC` | **no training** — `BASE`'s three policies with a rotation recovery at evaluation: after 20 immobile steps, request `(0, ω)` in whichever sign the filter admits | the scripted baseline every learned arm is measured against |

**Budget caveat, stated now:** 5 M steps from scratch is half what RC19 received, so absolute capture rates in
this round are NOT comparable to D112–D124. Only within-round comparisons are valid, and the round is judged
only on those.

## 3. Gate

Stream **1 720 000+**, N = **500**, all arms and all seeds on the same episodes. Opponents: the learned
structured attacker `ATK4-45`, `ObstacleHuggingEvader`, `GreedyEvader`. Primary metric the **immobile-step
rate**; reported alongside: deadlock-episode rate, capture rate, intervention rate, violations.

## 4. Hypotheses (fixed now)

**H-FIX-1 (does an arm solve it?).** An arm **SOLVES** the loop iff, against `ATK4-45`, its immobile-step rate
is **< 0.10** on **≥ 2 of 3** seeds AND its `GreedyEvader` capture is not significantly below `BASE`'s
(McNemar, p < 0.05, paired, `BASE` better ⇒ the arm is labelled `SOLVES (weaker)`).

**H-FIX-2 (the headline).** A learned arm **MATCHES** the scripted recovery iff its mean immobile-step rate
against `ATK4-45` is within **0.05** of `REC`'s. **Pre-registered expectation: no learned arm matches `REC`.**

**H-FIX-3 (the discount test).** `GAM`'s immobile-step rate is lower than `BASE`'s by **≥ 0.05** on **≥ 2 of 3**
seeds. This is the direct test of D128's mechanism and is reported whether or not `GAM` solves the loop.
A failure here is informative: it would mean the horizon is not the binding constraint after all.

**H-SAFE:** 0 pursuer contact steps in training, probes and gates, every arm including `REC`.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **`REC` solves, no learned arm matches** | The expected and most useful outcome: the remedy is an engineering one, not a learning one, and the paper says so plainly. |
| **`OBS` matches `REC`** | The frozen-input half is the binding constraint and the fix is an observation-design rule: expose the filter's admissible set to the policy. |
| **`GAM` matches `REC`** | The discount half is binding; the fix is a horizon rule and generalises to any long-delay filtered task. |
| **`MEM` matches `REC`** | Memory alone suffices; the claim becomes one about memoryless architectures specifically. |
| **`PEN` matches `REC`** | D113 was a budget artefact and the simplest fix works after all — report and re-scope D113. |
| **No arm solves, including `REC`** | The recovery's effect in D125 was specific to the policies it was applied to; the round is reported as a failure and Phase 2 is redesigned. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, retuned or re-seeded after any number of this round is seen. The arms,
seeds, budget, streams, thresholds (0.10, 0.05, 20 immobile steps for `REC`) are fixed above.

## 6. Artefacts

`results/derisk/s16_arm_<ARM>-<seed>.json`, `s16_probe_*`, `ckpts_<ARM>-<seed>/`, `s16_start.json`,
`s16_round.{json,png}`; runner `scripts/derisk_s16_fixes.py`, run from a frozen worktree.
