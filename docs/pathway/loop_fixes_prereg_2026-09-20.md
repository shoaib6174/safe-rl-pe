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

## Amendment S16-A1 (2026-09-20, before any arm of this round exists): two claims corrected

**1. The recovery's safety argument was overstated — by me, in D125 and in §2 above.** I wrote that the
recovery "cannot create contact" because `v = 0` never translates the body. The code does not force `v = 0`:
it asks the filter for `(0, ±ω_max)` and executes **the filter's output**, which the reviewer showed can carry
`v > 0` at roughly one contact-free state in 1200. **The correct argument is the ordinary one: the recovery's
action is filtered exactly like any other action, so its safety rests on the filter, not on the zero speed.**
Empirically it emitted `v = 0` on all 5150 firings observed across 400 000 contact-free states, and safety is
never worse than `BASE` because the same filter judges both. H-SAFE is unchanged and still counts every
contact step; `REC` has no privileged status in it.

**2. The escape feature is the LINEARISED admissible set, not the filter's exact one-step test.** §5 says the
`OBS` outcome would establish a rule to "expose the filter's admissible set to the policy". What the feature
exposes is the ω range admissible under the DCBF rows evaluated at `v = 0`, intersected with the action box —
a linearisation. It matched the filter digit-for-digit at the state probed in review, but it is not
guaranteed to. **Any claim from the `OBS` arm must say "linearised admissible set", and a failure of `OBS`
does not rule out that the exact set would have worked.**

Pending, non-blocking: two stale docstrings (`evaluate_structured` still says a shield carries no
cross-episode state, which is false for `RecoveryShield`; and the recovery's own docstring repeats the
overstated argument corrected above) are to be fixed after the S16 runner lands, to avoid editing `pe2/`
while it is being built against.

## Amendment S16-A2 (2026-09-20, before any arm exists): the recovery may translate, and that is kept

A deeper probe in review falsified A1's empirical half: a **deterministic** firing exists that executes
`v = 0.2308 m/s` (23 % of `v_max`) at a contact-free pose. The base rate is ~4e-6, which is why 5150 firings
showed none. So the recovery is not a pure rotation: it *requests* `(0, ±ω_max)` and executes whatever the
filter returns, which is occasionally a translation.

**Ruling: the behaviour is KEPT, not fixed to force `v = 0`.** Three reasons. It is what §2 pre-registers
("request `(0, ω)` in whichever sign the filter admits" — the request is the rotation, the execution is
filtered as always). It is bit-for-bit the code path that produced the D125 diagnostic, so changing it now
would mean the `REC` arm no longer matches the result that motivated this whole round. And when the filter
does return forward motion for a turn request, that motion is admissible and escapes the immobility better
than a turn would, which is the arm's purpose.

**What must change instead:** (a) the docstring claiming `v = 0` is structural is false and is corrected;
(b) a test asserting `moving_recoveries == 0` asserts a non-invariant and is demoted to **recording** the
count; (c) the round reports, as a free diagnostic, the **fraction of recovery firings that translated rather
than rotated** — a number worth having, since it says how often the immobility was escapable by moving all
along. H-SAFE is unaffected: every recovery action is CBF-filtered exactly like every other action, and
contact steps are counted for `REC` as for every arm.

## Amendment S16-A3 (2026-09-20, before any arm exists): H-FIX-2's match test is one-sided

§4's H-FIX-2 says a learned arm MATCHES `REC` iff its mean immobile-step rate is "within 0.05" of `REC`'s.
Taken as a two-sided band that is wrong: an arm **0.10 below** `REC` — i.e. substantially BETTER than the
scripted recovery — would fall outside the band and be recorded as a non-match, which is absurd.

**Corrected: a learned arm MATCHES iff its mean immobile-step rate against `ATK4-45` is ≤ `REC`'s + 0.05.**
One-sided. An arm at or below `REC` matches trivially, and an arm strictly below `REC` by more than 0.05 is
additionally flagged **`BEATS REC`**, which would be the most surprising and most publishable outcome of the
round and must not be hidden inside a "non-match".

`BASE` is not a remedy and is excluded from the match list; whether `BASE` itself happens to sit within the
band of `REC` is reported separately, since that would indicate the recovery did nothing in this round and
would put every other comparison in question.

## Amendment S16-A4 (2026-09-20, recorded ~20 min after launch, BEFORE any arm has produced a gate number)

**`MEM` is confounded and the confound is quantified here so it cannot be discovered afterwards.** §2 says
every arm shares one configuration and differs in exactly one thing. That is true of `GAM`, `OBS` and `PEN`.
It is **not** true of `MEM`: `sb3_contrib.RecurrentPPO`'s defaults differ from `PPO`'s in two hyperparameters
that have nothing to do with recurrence —

| parameter | `PPO` (every other arm) | `RecurrentPPO` (`MEM`) |
|---|---|---|
| `n_steps` | 2048 | **128** |
| `batch_size` | 64 | **128** |

so `MEM` collects rollouts 16× shorter and updates on 2× larger minibatches. With `hold = 10`, a 128-step
rollout spans roughly 13 decisions, which is short relative to the ~629-step delay D128 measured between an
escape and its consequence. **`MEM` therefore tests "RecurrentPPO at its defaults", not "memory, all else
equal".**

**Ruling: the arm runs as pre-registered and is REPORTED with this caveat attached wherever its number
appears.** I am not re-specifying it mid-round to match `PPO`'s rollout length, because the arms are already
training and changing one now would be the precise kind of after-the-fact adjustment the stop rule exists to
prevent. The consequence is fixed here: **a `MEM` failure does NOT license the conclusion that memory fails**
— only that this configuration did — and if `MEM` is the arm that matters to the eventual paper, it must be
re-run in its own round with `n_steps` and `batch_size` matched to `BASE`. A `MEM` success is unaffected by
the caveat and would be the stronger result for having been obtained under a shorter horizon.
