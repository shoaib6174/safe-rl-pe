# Plan: "Safety filters move failure to the metric nobody checks" — the construct-validity paper

Written 2026-10-03. Replaces `ral_plan_boundary_condition_2026-09-23.md` as the project's paper plan.
That plan is withdrawn: S18's behavioural gate (CG4) showed the boundary-condition claim rests on an
architecture that makes its question ill-posed (the intervened region is mostly the scripted law's, not the
learner's). The reasons are in the S18 worklog and below as Case 5.

## 1. The thesis, in one sentence

**A safety filter guarantees one metric perfectly and silently transforms the other; standard safe-RL
evaluation verifies the guaranteed metric and merely interprets the transformed one, and that is where
conclusions go wrong.**

Zero violations is always checked and always true. Task performance is always reported and almost never
validated as a construct. Across a year of pre-registered rounds, every major conclusion we had to retract
failed in exactly that gap — not in the statistics, which were sound every time, but in what the task
number *meant*. Pre-registration did not catch one of them. A behavioural check caught the last one.

This is not a confession paper. Each case is stated as a general claim about safety-filter evaluation,
demonstrated with a reproducible instance, and paired with the check that would have caught it. If a case
cannot be stated that way, it does not go in (section 5, gate G1).

## 2. What the paper contributes

1. **A taxonomy of five construct failures specific to evaluation under safety filters**, each with a
   reproducible demonstration from a pre-registered round — so the failure is *documented as it happened*,
   not reconstructed after the fact. That provenance is the asset; almost no lab has it.
2. **Evidence that pre-registration alone is insufficient.** All five occurred under pre-registration.
   It disciplines inference *from* a number; it says nothing about whether the number measures its name.
3. **A construct gate** (metric-level checks + a mandatory behavioural check), with the honest result that
   the metric-level checks caught two label errors and only the behavioural check caught the architectural
   confound — and it could only do so after training.
4. **A fixed-construct re-run** showing that correcting the construct changes the conclusion (section 4,
   Phase 2). This is what separates a methods paper from a lab notebook.

## 3. The five cases

| # | round | what the metric was named | what it measured | what caught it |
|---|---|---|---|---|
| 1 | D82–D108 | learner capability under fine-tuning | one-step TD critic divergence (Q to +21 on a [−0.3, 1] problem) | autopsy against the bounded return range |
| 2 | D112–D125 | adversarial exploitation of the filter ("attack surface") | a memoryless policy in an absorbing observation loop | a human watching a rendered episode |
| 3 | D126/D130 | filter infeasibility ("geometric pocket") | the remedy's own action set (rotation-only; a straight command moved 100 %) | a direct action probe |
| 4 | D129 | the remedy failing ("immobile steps") | the remedy's own mechanism (rotating to escape counted as stuck) | a pre-registered secondary metric |
| 5 | S18 CG4 | penalty on the *learner's* filter interventions | penalty on the composite agent, mostly the scripted law's | the behavioural construct gate |

Cases 2–5 all have the shape: a filter-induced *liveness* failure (stall, loop, standoff) is scored by a
*task* metric whose meaning assumed the agent was still trying. Case 1 is the bootstrap variant: the metric
reported the learner, the number came from the critic. The paper names the shape, then the instances.

## 4. Work plan — deliberately paced

Pace is a design decision. The retrospective's cause #5 was a feedback loop measured in days; S18 was
planned, built and launched in 48 hours after writing it, and that pace produced the confound. **No phase
starts before the previous phase's gate is met.**

### Phase 0 — close S18 honestly (≈1 week)

- Run `--summarize`. Report the pre-registered verdicts as recorded, with CG4's findings attached. No
  reinterpretation. If H-CROSS passes, it is reported as *unreliable because of Case 5*, not as a result.
- **The one new number:** replay all 18 arms' gate episodes (seeded, deterministic, checkpoints exist)
  with a branch-tagging recorder and split per-episode intervention magnitude into law-driven vs
  learner-driven. This turns Case 5 from an observation into a measurement. No training. Pre-register
  the split's reading before running it.
- Write the Case 5 autopsy as its own D-entry.
- **Gate P0:** the law/learner split is measured and the S18 record is closed with its caveats in the
  decisions log.

### Phase 1 — five reproducible cases (≈2 weeks)

- One script and one figure per case, runnable from a clean checkout against archived artefacts.
  Most exist as `derisk_*` scripts; the work is consolidation, not creation.
- For each case, a one-paragraph *general* statement that does not mention this project, followed by the
  instance. If the general statement cannot be written, the case is cut.
- **Gate G1:** an outside reader (a reviewer agent run hostile, then ideally a human) reads the five
  general statements *without* the instances and judges whether each is a claim about safety-filter
  evaluation or a description of a mistake. Any case judged the latter is cut or rewritten.

### Phase 2 — the fixed-construct re-run (≈1 week compute, 6 arms)

- Re-run S18's SPARSE column only (BASE/PEN × 3 seeds), with the penalty restricted to **learner-driven
  steps**, everything else frozen. Pre-registered question: *does correcting the construct change the
  penalty's measured effect?* Either answer is reportable; a changed conclusion is the stronger paper.
- Launch discipline from this round's lessons: `OMP_NUM_THREADS` set; literal arm lists; verify by
  checkpoint presence, not PID; no competing jobs on the training host.
- **Gate P2:** the pre-registered reading is applied as written.

### Phase 3 — write (≈3 weeks)

- Outline → full draft → hostile internal review (`/ars-reviewer`) → revise → second review.
- Required sections: the thesis; the shape; five cases; the gate and what each level caught; the
  fixed-construct result; **limitations stated without softening** — one lab, one simulator, one task
  family, taxonomy generality argued not demonstrated.
- The surviving positive result (D128 mechanism, D131 forward-offering recovery) appears as *a worked
  example of what a validated construct yields*, not as a headline.
- **Gate P3:** the second hostile review raises no construct-level objection the paper does not already
  address in its own text.

### Phase 4 — submit

## 5. Kill conditions and honest odds

- **G1 fails for ≥2 cases** → the material is a lab notebook, not a taxonomy. Stop; do not write.
- **Phase 2 shows the corrected construct changes nothing** → still reportable, but contribution 4 becomes
  "the confound was benign here," and odds drop.
- **Venue:** TMLR primary (correctness + interest, no novelty bar; the right home for a methods result
  with an n = 1 lab). A safe-RL / RL-evaluation workshop first to test reception is reasonable and costs
  little. RA-L is the wrong venue for this paper and is not pursued.
- **Odds, stated now so they can be checked later:** a well-executed version at TMLR ≈ 40–50 %. The risks
  reviewers will name: "one lab's mistakes," "obvious in hindsight," "n = 1 environment." Contributions 1,
  3 and 4 are the answers; if any of them is weak at Phase 3, the odds are lower and the paper should say so.

## 6. What this paper is not

It is not the discovery that was wanted. It is a methodological result with real data behind it, and it is
true. The crowded field, the simple environment and the absence of a unique asset make a discovery here
something to be *followed from a surprise*, not assembled from existing results — and this plan does not
pretend otherwise. If a surprise appears during Phase 0 or 2 (the law/learner split is the likeliest place),
it is pre-registered and chased on its own terms, not folded into this paper to inflate it.

## 7. Immediate next action

Phase 0, step 1: `--summarize` on S18, on the user's word. Then pre-register the law/learner replay.
