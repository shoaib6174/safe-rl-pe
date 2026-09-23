# Plan: the boundary-condition paper (RA-L), S18–S22

Written 2026-09-23. Supersedes the "diagnosis + remedy" framing that D132 found to be prior art.
Governed by `docs/retrospective_2026-09-23.md` §4 and the standing pre-registration discipline (D83).

## 0. Why the framing changes

D132 established that the phenomenon — a minimum-intervention safety filter producing safe-but-stalled
behaviour — was published as BIG-CBF on 2026-09-13, nine days before we checked. A paper whose headline is
"we found stalling" is dead. **But the thing we measured that BIG-CBF did not is a contradiction with a
second published result, and that contradiction is the paper.**

- **Pizarro Bejarano et al.** (arXiv:2410.11671) report that *penalising safety-filter interventions during
  training improves the learned policy.* This is an established, cited recommendation.
- **D129 (ours)** found the opposite, decisively and across three seeds: penalising the stall **backfired** —
  the agent learned to stop asking to move, which satisfies the penalty perfectly while destroying the task.

Both can be true. The plan is to show *exactly when each holds*, name the condition, prove it under stated
assumptions, and demonstrate it in two environments. **That is a discovery, not a diagnosis.**

## 1. The claim

> **Penalising safety-filter intervention helps when the task reward provides gradient inside the
> intervened region, and hurts when it does not.**
>
> Under a dense, per-step task reward (reference tracking) the penalty is a small perturbation on an
> already-informative gradient, so it steers the policy toward maneuvers the filter does not have to
> correct. Under a sparse terminal reward the penalty is the *only* gradient signal on the intervened set,
> so the policy optimises the penalty rather than the task, and its minimiser is the do-nothing action.
> **The remedy and the pathology are the same mechanism evaluated under different reward density.**

This explains Pizarro Bejarano et al.'s result, explains ours, and predicts which one a practitioner will
get. It is falsifiable, cheap to test, and nobody has stated it.

## 2. The four components

### S18 — the boundary experiment (the paper's core)

A 2×2, pre-registered before any number exists.

| | sparse terminal reward | dense shaped reward |
|---|---|---|
| **no intervention penalty** | BASE-sparse | BASE-dense |
| **intervention penalty** | PEN-sparse | PEN-dense |

- **Pre-registered prediction (the whole point):** a **crossover interaction**. PEN-sparse < BASE-sparse
  (reproducing D129); PEN-dense > BASE-dense (reproducing Pizarro Bejarano et al.). A main effect with no
  interaction refutes the claim and the paper does not exist in this form.
- 3 seeds per cell × 4 cells = 12 training runs; gates at N = 500 paired episodes on a fresh eval stream.
- Primary metric: task success (capture rate). **Secondary, and pre-registered as the mechanism check:**
  the deadlock rate and the fraction of policy-requested actions that are zero-motion. The mechanism claim
  is that PEN-sparse wins the *penalty* while losing the *task* — that must be visible, not inferred.
- **Construct gate (retrospective §4), run BEFORE any number is interpreted:** render three unselected
  episodes per cell; confirm the primary metric cannot be satisfied by doing nothing; confirm the dense
  reward actually differentiates actions on the intervened set (measure its gradient norm there, non-zero).

### S19 — the second environment

The claim is about reward density, not about pursuit–evasion. One environment makes it an anecdote.

- **Choice: 2-D goal-reaching through clutter with a CBF obstacle filter.** Reuses `pe2`'s dynamics,
  shield and harness wholesale; the only new pieces are a goal sampler and the two reward variants.
  Sparse = terminal goal bonus. Dense = negative distance-to-goal per step. This is the *minimum* change
  that varies reward density while holding the filter fixed, which is exactly the claim's independent
  variable.
- Rejected alternatives: a manipulator arm (new dynamics, new shield, weeks of work, tests nothing extra);
  a third pursuit–evasion variant (does not answer the generality objection at all).
- Same 2×2, same gates, same construct gate. **Pre-registered as confirmatory:** the interaction must
  reproduce in sign. If it does not, that is reported as a limit on the claim's scope, not buried.

### S20 — the formal statement

Half a page, stated as a proposition with its assumptions visible. Sketch:

Let `I ⊆ S` be the set of states where the filter intervenes, and suppose the task reward `r` is constant
on `I` (the sparse case: no terminal event occurs there). Then the policy gradient restricted to `I` is
determined entirely by the penalty term, whose per-step minimiser is any action the filter passes unchanged.
In the stalled configuration the zero-motion action is such an action, and it is absorbing under the
dynamics. Hence immobility is an attracting fixed point of the penalised objective on `I`. When `r` is
non-constant on `I`, the task gradient competes and the fixed point need not attract.

The proposition earns its place only if it is honest about what it assumes. It is a *sufficient* condition,
not a characterisation, and the paper says so.

### S21 — the reframe and write-up

Headline is the boundary condition. Structure:
1. Two published results disagree (cite both, fairly).
2. A condition that reconciles them, stated formally.
3. The crossover, measured, in two environments, pre-registered.
4. BIG-CBF's phenomenon cited as established context — **not** claimed as ours.
5. Practical rule: check whether your task reward has gradient where your filter fires, before you
   penalise interventions.

Our apparatus becomes an asset here rather than an overhead: pre-registered predictions with a crossover
that could have failed, seed-paired gates, negative controls, ~10⁹ shielded steps at zero violations, and
a documented error trail. Reviewers reward a prediction that was registered before it was tested.

## 3. What this is worth

Prior estimate for the diagnosis paper at RA-L: 15–25%. With all four components landing: **~2 in 5.**
The gain comes from converting a contribution reviewers read as "you characterised a known failure" into
one they read as "you resolved a disagreement in the literature and told me which regime I am in."

**The kill condition is honest and comes early.** If S18 shows no interaction, the claim is false and we
learn that in roughly a week for twelve training runs, before any writing. That is the cheapest possible
place to find out.

## 4. Order and gating

| round | content | gate to proceed |
|---|---|---|
| **S18** | pre-register, then the 2×2 in pursuit–evasion | **crossover interaction present.** No interaction ⇒ stop, claim refuted, reconsider |
| **S19** | second environment, same 2×2 | interaction reproduces in sign |
| **S20** | the proposition | assumptions match what S18/S19 actually held |
| **S21** | full draft | internal review before submission |

S20 may be drafted in parallel with S19 — it depends on S18's mechanism reading, not on S19's numbers.

## 5. Standing discipline (unchanged)

Pre-register in `docs/pathway/` **before any number exists**; implement via subagent-driven development with
review; launch from a frozen git worktree with `PYTHONPATH` pointed at it; record a numbered D-entry plus
worklog, tracker and memory. **Added by the retrospective:** the construct gate in S18 is mandatory and
blocking, and no round's numbers are interpreted before it passes.

## 6. First action

Write `docs/pathway/boundary_condition_prereg_2026-09-2X.md`: the 2×2, the crossover prediction with its
direction fixed in advance, the exact reward definitions for both density variants, the construct gate,
the stop rule, and the eval stream. Nothing is run until that document is committed.
