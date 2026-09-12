# R1 — requalification: does RL improve on a COMPETENT prior? (2026-09-12, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), **D106** (the teacher was the binding
constraint), **D107** (the improvement is general across opponents), D100/D102/D103 (the
negative conclusions this round requalifies), D104/D105 (the diagnosis chain).

## 1. Why this round exists

Every learner round this project has run — Tier 0, T0.4, T0.5, T1.0, eight learners across
four pre-registrations — fine-tuned the **0.212 clone**. D105/D106 established that clone was
an imitation of a teacher standing still on 92.6 % of its steps, and that changing one branch
of the teacher produces a **0.476 clone** that ranks first of eight policies on every
opponent that is catchable at all (D107).

**Those four rounds are therefore evidence about fine-tuning a near-stationary prior. They
are not evidence about whether reinforcement learning works in this game**, and D102's
conclusion — a learning failure "independent of reward shaping, algorithm family, anchoring,
bootstrapping and the weighting's calibration" — cannot stand as written.

**Hypothesis R:** the collapse was a property of the prior, not of the learner. From a
competent prior, the plainest fine-tune improves rather than collapses.

This round is deliberately the **smallest experiment that can decide it**. It re-runs one
configuration, not eight. If the collapse reproduces from a competent prior, the negative
result survives and is strengthened; if it does not, four rounds of conclusions were artefacts
and the paper changes completely.

## 2. Protocol

- Role: pursuer. Opponent: frozen `GreedyEvader` at `EnvConfig()` defaults — **the same
  opponent every prior round used**, so the comparison to them is direct. D107 already
  establishes the prior's quality is not opponent-specific; adding opponents here would
  change two things at once.
- Learner: `finetune_br` — LayerNorm critics, `ent_coef = 1e-5`, `critic_warmup = 50_000`,
  γ = 0.999 (D78/D80/D81). **Unchanged from the configuration that collapsed.** No anchor, no
  dense reward, no horizon change. The prior is the only thing that differs from T1.0's HC arm.
- Budget **1,000,000** env steps; probes every **100,000** on N = 200 paired seeds; final
  probe N = **500**.
- Eval stream **990000+** (fresh; ≤ 980999 consumed).
- **The bar is measured, not inherited.** The new clone
  (`results/derisk/sac_bc_searching_pursuer.zip`) is scored once at N = 500 on 990000+ before
  any training, written to `results/derisk/r1_start.json` as `START_NEW`. An arm that cannot
  read it refuses to run.
- Training seed **7** (A1's off-policy seed, as every prior SAC arm used).
- Pass rule, improvement rule and verdicts are the existing `pe2.derisk.round_verdict`:
  win rate ≥ **0.137** at every probe from **500,000**; IMPROVED iff the final N = 500 Wilson
  lower bound exceeds `START_NEW`'s point estimate.

## 3. The arms

| arm | prior | purpose |
|---|---|---|
| **RQ-NEW** | the 0.476 searching clone | **the test.** Does the plainest fine-tune improve on a competent prior? |
| **RQ-OLD** | the 0.212 original clone | **the matched control.** The identical configuration on the identical stream from the old prior. Expected to collapse, reproducing T1.0's HC. Without it, an RQ-NEW result cannot be attributed to the prior rather than to the fresh seed stream. |

Two arms, one variable: the prior. Nothing else differs — same learner, same budget, same
seeds, same opponent, same everything.

No separate negative control: **RQ-OLD is one.** It is expected to collapse, and its collapse
validates the harness on this stream. If RQ-OLD does *not* collapse, that is itself a finding
(the earlier collapses were stream-specific) and both arms are reported with that caveat
rather than withheld — the two arms are independent, so neither invalidates the other.

## 4. What each outcome fixes

| result | direction |
|---|---|
| **RQ-NEW IMPROVED, RQ-OLD collapses** | Hypothesis R confirmed. The collapse was a property of the prior. **D100, D102 and D103 are formally requalified**: their conclusions are re-scoped to "fine-tuning a near-stationary prior", Route C is withdrawn as the project's direction, and the next pre-registration is the learner round that four rounds were trying to run — now from a prior worth improving. |
| **Both collapse** | The negative result **survives and is strengthened**: the collapse holds from a prior that ranks first of eight on every catchable opponent, so it is a property of the learner, not of the demonstration data. D100/D102/D103 are re-scoped but their conclusion stands, and Route C is restated on this stronger footing. |
| **RQ-NEW PRESERVED (holds, does not improve)** | The competent prior is preserved where the weak one collapsed — a real and reportable difference — but RL still adds nothing. Route C stands, with anchoring's role (D102) re-read: preservation may have been about the prior all along. |
| **RQ-OLD improves** | Unexpected and would contradict four rounds on the same configuration. Report it, change nothing else, and stop: something about this stream or this build differs and must be found before any further reading. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this
round is seen. A crashed arm is `UNDETERMINED` with its probe history preserved.

## 5. What this round does NOT do

It does not re-run the other seven learners, add the shield, touch the evader family, or
attempt the belief-augmented observation. Those follow from the outcome, not alongside it.
The hidden-state defect (D79/D105) remains unfixed and is still the most likely next
constraint if RQ-NEW improves but plateaus.

## 6. Artefacts

`results/derisk/r1_start.json`, `r1_arm_{RQNEW,RQOLD}.json` (flushed per probe),
`r1_round.{json,png}`, checkpoints under `results/derisk/ckpts_{ARM}/`. Figure: both curves,
`START_NEW` and the old clone's rate drawn as reference lines, the 0.137 floor and the 500k
pass window. Signed in the worklog. Estimated wall time ≈ 6 h with the two arms concurrent.
