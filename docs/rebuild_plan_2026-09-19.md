# Rebuild plan after D125 — the absorbing-loop paper

**Status: PLAN. Nothing below is a claim; Phase 0 exists to decide whether there is a paper at all.**

## 1. The candidate thesis

> *(sharpened by D126)* **Minimal-intervention safety filters generate states that can only be escaped by a
> multi-step manoeuvre — rotate, then move — which the filter's own objective never proposes (the nearest
> admissible action to a blocked forward command is a stop) and which a memoryless policy cannot execute
> (once stopped, its observation is frozen). The original wording follows.*
>
> A safety filter can put a memoryless learner into an **absorbing observation loop**. The filter blocks the
> commanded action, the agent stops, its observation stops changing, and a deterministic policy re-emits the
> same blocked action indefinitely. The safety guarantee holds throughout, no violation is ever recorded, and
> the standard safety metric therefore looks perfect while the task silently fails. The escape is usually one
> admissible action away, and the agent cannot see which one.

Why this is worth a paper, given the literature check (`docs/literature-review/adversarial_shield_deadlock_2026-09-19.md`):
cooperative CBF deadlock is a *force equilibrium between robots*; ours is a *single* agent against static
geometry, and the trap is in observation space, not configuration space. Q3 and Q4 of that check returned no
prior art for either the loop or a filter-aware remedy. The nearest neighbour (Raven, USENIX Sec '25) is an
injection attack on reciprocal collision avoidance, not a learning pathology.

**What we already have that supports it** (re-interpreted, not wasted):
- D125: at a stalled state, one turn direction is blocked and the other is returned untouched; the policy asks
  for the blocked one for 1085 consecutive steps with command σ = 0.002.
- D125: a 20-step rotation recovery drops the immobile-step rate 0.392 → 0.024 with **zero** violations and
  capture unchanged or better in every cell tested.
- S14 (D124): the loop occurs for **all six** pursuers measured, at rates 0.09–0.73 — it is not one bad policy.
- S9 (D118): its frequency scales with the filter's margin, with a more permissive class-K gain, and requires
  obstacles — now read as "how often the policy is put in a position it cannot think its way out of".
- S5 (D113): an immobility **reward penalty alone did not fix it** (0/3) — a useful negative for the fix section.
- D112/D124 capture numbers become the *cost* of the loop, once recomputed with recovery.

## 2. Phase 0 — validity checks that can kill this (≈ 3 days, mostly evaluation)

> **V1 and V2 are DONE (D126, 2026-09-19). V1 PASSED — an exact QP behaves identically to our candidate
> enumeration (0/150 forward motion either way), so the loop is not our discretisation. V2 refined the
> thesis: translation is genuinely inadmissible at a stalled state (0/150) but rotation is admissible in
> 150/150, so escape requires a ROTATE-THEN-MOVE manoeuvre that a minimal-intervention objective never
> proposes and a memoryless policy cannot execute. The thesis in §1 should be read in that sharper form.
> **V3 is DONE too (D127): the loop survives exploration — deterministic vs stochastic immobility is
> 0.276 → 0.275 for FT-B-41 and 0.293 → 0.333 for RC19 at their converged σ (0.36–0.43). An untrained net at
> σ = 1.0 halves it but still deadlocks 30 % of episodes. Under stochastic sampling the trained policy is
> barely better than the untrained one, despite an immobility penalty in its reward. PHASE 0 IS COMPLETE AND
> ALL THREE CHECKS PASSED — proceed to Phase 1.**

Run these BEFORE writing a line. Each is pre-registered separately.

**V1 — is the blocked direction real geometry or our candidate enumeration?** `CBFShield` picks the nearest
admissible point from a finite candidate set (`_qp_candidates`), not from a true QP solve. If a real solver
would return an admissible rotation where our enumeration returns `(0, 0)`, then the loop is partly our
discretisation and the finding changes a third time. Implement an exact QP solve (scipy) for the same
constraints, compare the returned action on every stalled state of a held-out set, and report the
disagreement rate. **This is the single highest-risk check; do it first.**

**V2 — how pervasive is the loop?** Over a stream, at every stalled step, ask the filter whether *any*
admissible action would have restored motion, and whether the policy's nominal was within reach of one.
Report the fraction of stalled steps that are "escapable but not escaped", per pursuer and per filter family.
The thesis needs this to be high; if most stalls are genuinely inescapable, the story reverts to geometry.

**V3 — does it happen during TRAINING, not just at deterministic evaluation?** At evaluation the policy is
deterministic, which guarantees the loop. During training it is stochastic and should sometimes escape. Log
stall entries, escapes and returns over a training run. If the stochastic policy escapes easily, the claim
narrows to "deterministic deployment of a stochastically trained policy", which is still real but smaller.

## 3. Phase 1 — mechanism: why does learning not fix it? (≈ 1 week)

The honest question a reviewer will ask. Hypotheses to test, in order of likelihood:
1. **Credit assignment.** The reward is terminal (capture = 1). Escaping a stall at step 600 rarely produces a
   capture, so the advantage of escaping is near zero. Measure the empirical return difference between
   escaping and not escaping, from training rollouts.
2. **Commitment.** Decisions are held 10 env steps, so an escape requires the *held* action to be right.
3. **Feature saturation.** The S5 stall-duration feature clips at 2 × DEADLOCK_STEPS; beyond that the
   observation is constant by construction. Measure how much of the loop happens past saturation.
4. **Entropy collapse.** Check the policy's entropy at stalled states versus elsewhere.

## 4. Phase 2 — the fixes, pre-registered head to head (≈ 1.5 weeks compute)

One round, four arms, same budget and seeds, judged on three axes at once: immobile-step rate, capture against
ordinary evaders, and violations (which must stay zero).

| arm | what |
|---|---|
| `REC` | scripted rotation recovery, no retraining (the D125 diagnostic, promoted to an arm) |
| `OBS` | admissible-turn-range features added to the observation, retrained |
| `MEM` | recurrent policy, retrained, no new features |
| `PEN` | immobility penalty only, retrained — the pre-registered replication of S5's negative |

The interesting outcome is not which wins but whether the *learned* fixes match the scripted one. If a trivial
recovery beats every learned remedy, that is the paper's most useful sentence for a practitioner.

## 5. Phase 3 — generality (≈ 1 week, optional depending on Phase 0)

- A second filter that is genuinely different in construction (true QP, or a projection filter), not sharing
  our candidate enumeration.
- A second environment if and only if Phase 0 and 1 are clean and time remains. A thin second domain is worse
  than none.

## 6. Venue and timing

AAMAS 2027 (2026-10-08) is **withdrawn** — the claim is not supported and will not be in nineteen days.
Realistic targets, in order: **IROS 2027** (deadline ~March 2027) or **RA-L** (rolling, submit when ready).
A workshop version could go earlier if a short deadline appears. The absorbing-loop framing is a better fit
for a learning or robot-learning audience than for a pure multiagent one, which also relaxes the AAMAS reframe
pressure.

## 7. What gets retired

The shielded-games arms-race narrative, the exploitability table as a headline, the two-pursuer idea, and the
"attack surface scales with conservativeness" claim in its current form. They become supporting material for
"how often the loop is triggered", or they are dropped. `docs/paper_exploitability_table.md`,
`docs/paper_mechanism_stall.md` and `docs/paper_ral_outline.md` must be marked superseded by D125 before any
of their numbers are reused.
