# T1.0 — the horizon round: pre-registration (2026-09-08, S124)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), D100/D102 (the learner rounds and
their outcome), D103 (this round, and the user's explicit override recorded below).

## 0. The override, recorded plainly

D101 §5 committed, before the T0.5 data existed, that no further **learner** pre-registration
would follow. That commitment was kept: T0.5 ran, returned 0 IMPROVED, and Route C was
declared final (D102).

**The user overrode that commitment on 2026-09-08**, directing a round that varies the
**task** rather than the learner. This document records the override as an override, not as a
reinterpretation of what "learner pre-registration" meant. The distinction is real — every
prior round held the environment fixed and swapped mechanisms; this one holds the mechanism
fixed and changes one environment parameter — but the honest handling is to say the
commitment was set aside deliberately, by the person whose project it is, and not to lawyer
it away. If this round returns 0 IMPROVED, Route C stands as D102 left it, strengthened.

## 1. Why this round exists

Every learner round varied the mechanism against a fixed task. The task was never examined.
Measured on the gated clone's own 1M-step audit (`results/derisk/t0_signal_audit_pursuer.json`,
980 episodes):

| quantity | value |
|---|---|
| episode length | 1,200 steps (60 s at dt = 0.05) |
| mean episode actually run | 1,021 steps |
| **median capture time** | **220 steps** |
| capture time, 75th percentile | 276 steps |
| captures before step 400 | **175 of 184 = 95 %** |
| reward density at 1,200 steps | 184 positives per 1M steps |
| reward density if truncated at 400 | **485 per 1M** |

**The game is decided by step ~276; the episode runs to 1,200.** The remaining ~900 steps are
both agents milling about after the outcome is settled. They lengthen the credit-assignment
horizon threefold and dilute the per-step signal to a third, and they buy 9 extra captures out
of 184.

**Hypothesis H:** the learning failure recorded in D100/D102 is caused by the credit-assignment
horizon, not by the learner mechanism. If H holds, a learner that failed at 1,200 steps
improves on its clone at 400 with everything else held identical.

## 2. Common protocol (binding)

- Role: pursuer. Opponent: frozen `GreedyEvader`. Learner recipe fixed and identical across
  every arm: `finetune_br` — LayerNorm critics, `ent_coef = 1e-5`, `critic_warmup = 50_000`
  (D78/D80/D81). **No arm varies the mechanism.** This is deliberate: the round's whole claim
  is about the task, so the learner must be a constant.
- Budget **1,000,000 env steps** per arm; probes every **100,000**, N = 200; final probe
  N = **500**.
- **Two eval streams, one per horizon** (fresh; ≤ 860999 is consumed):
  **870000+** for every 400-step arm, **880000+** for the 1,200-step arm.
- **Two start bars, each measured before any training**, from the same gated clone
  `results/visual_gates/step_05/sac_bc_pursuer.zip`, N = 500 on that horizon's own stream:
  `START_400` and `START_1200`. Written to `results/derisk/t1_start.json`. An arm that cannot
  read its bar refuses to run. Expected from the truncation table: `START_400` ≈ 0.179,
  `START_1200` ≈ 0.19 — **expected, not assumed; the measured values govern.**
- **Pass rule (no collapse):** win rate ≥ **0.137** (`CLONE_CI_LO`) at every probe from
  **500,000** steps onward. A missing probe in the window → `UNDETERMINED`.
- **Improvement rule:** the final N = 500 probe's Wilson lower bound (z = 1.96) exceeds that
  arm's own start bar. **Survivor = IMPROVED.**
- Verdicts `COLLAPSED` / `PRESERVED` / `IMPROVED` / `UNDETERMINED` as in D99 §C2, computed by
  the existing `pe2.derisk.round_verdict`.
- Training seed **7** for every arm (A1's off-policy seed, as every prior SAC arm used).
- No budget extension, no criterion change, no seed re-roll, no horizon change after any
  number of this round is seen.

## 3. The arms

The design is a paired horizon comparison with the learner held constant.

| arm | `max_steps` | γ | eval stream | what it isolates |
|---|---|---|---|---|
| **HA** | **400** | 0.999 | 870000+ | **The primary test.** Episode length ALONE — every other parameter, including the discount, is exactly what failed at 1,200. |
| **HC** | **1200** | 0.999 | 880000+ | **The matched control for the horizon claim.** HA's exact configuration at the original length. Expected to fail, like every prior learner. Without it, an HA success invites "you never ran that configuration at 1,200". |
| **HB** | **400** | **0.9975** | 870000+ | Adds the matched discount. 1/(1−γ) = 400 sets the effective horizon to the episode length, the same rule the original used (0.999 ↔ 1000 ≈ 1200). Separates "shorter episode" from "shorter episode plus matched discount". |
| **HNEG** | 400 | 0.999 | 870000+ | **Negative control.** HA with the environment reward NEGATED (the pursuer is paid for *not* capturing). Must collapse by `control_collapsed`. Horizon-independent by construction, so it validates the harness at the new episode length without competing with the hypothesis. |

**Why the control is a sign flip and not the naive fine-tune.** The usual control here is the
D78 naive configuration, which collapsed to 0.010 at 1,200 steps. It is unusable in this
round: if H is true, the naive configuration might *succeed* at 400, and a control that the
hypothesis predicts may pass is no control at all. Negating the reward must fail whatever the
horizon does.

## 4. Reported, never gated

Per D89: per-arm `positives_per_1M` with its per-segment decomposition; the realised capture
rate and mean episode length per arm; both start bars with their Wilson intervals; and the
per-probe win rates. Also, because it is the round's own premise: for each arm the **fraction
of its training episodes that ended in capture**, so a reader can confirm the reward density
the truncation table predicts.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **HA IMPROVED and HC does not** | The horizon was the binding constraint — a substantive positive finding, and the strongest result this project has produced. The learner rounds' failure is re-read as an artefact of a task whose episode was three times longer than its own game. Track 2 (the belief-augmented evader) and Track 3 (the pair, shielded vs unshielded) are re-planned at the short horizon, each with its own pre-registration. |
| **HA and HC both IMPROVED** | The horizon is not what changed; the configuration itself (`finetune_br` with warm-up at terminal-only reward) was simply never run at 1,200 and works. Report as a gap in the earlier rounds' coverage, and re-plan Track 2/3 at the ORIGINAL horizon. |
| **HB IMPROVED, HA does not** | Shortening alone is insufficient; the matched discount is doing the work. Report both, and carry γ as a matched parameter into Track 2/3. |
| **0 IMPROVED** | The failure is not the horizon either. **Route C stands as D102 left it, strengthened**: the learning failure now survives reward shaping, algorithm family, anchoring at both strengths, bootstrapping, weighting calibration, AND a threefold reduction in the credit-assignment horizon with a 2.6× denser reward. |
| **HNEG does not collapse** | Every arm is withheld; the harness is unvalidated at the new horizon and the round decides nothing. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round
is seen. A crashed arm is `UNDETERMINED` with its probe history preserved.

## 6. The risk this round does not address, stated in advance

The clone sees its opponent on **5.2 %** of steps, against **~22 %** for the scripted teacher it
was cloned from. That gap is an imitation failure, not a property of the episode length, and
shortening the episode will not touch it. If H is false, this measurement is the most likely
place the real problem lives, and it is where I would look next. Recording it here so that a
0-IMPROVED outcome is not mistaken for "nothing left to try".

## 7. Artefacts

`results/derisk/t1_start.json`, `t1_arm_{HA,HB,HC,HNEG}.json` (flushed per probe),
`t1_round.{json,png}`, checkpoints under `results/derisk/ckpts_{ARM}/`. Visual gate: the
figure, both start bars drawn, signed in the worklog. Estimated wall time ≈ 6 h with the four
arms concurrent.
