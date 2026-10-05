# Plan: making the pursuer competent — S19–S22

Written 2026-10-05. Companion to `construct_validity_paper_plan_2026-10-03.md`; does not replace it.
The two converge on one measurement (section 6).

## 0. Why this, and why in this order

S18's behavioural gate showed where the pursuer is dumb, with episodes rather than averages:

| failure watched | component | evidence |
|---|---|---|
| drives into a wall it can see and sits pinned at clearance 0.000 for ~1100 steps | **intercept law** (scripted, shield-blind) | `SPARSE-PEN_1880001`, `SPARSE-BASE_1880000` |
| roams the arena after losing sight, never converges | **learner** (memoryless) | `DVARY-BASE` 0/3 |
| static standoff with the evader adjacent | learner + horizon | `DVARY-PEN_1880001` |
| not consulted while the law is pinned, though it holds `escape_obs` | **control handoff** | all law-pinned episodes |

Four upgrades follow, one per row. **They are tested one at a time**, each pre-registered, each watched
before it is read, each gated before the next starts. The order is by impact per cost, and the first one
needs no reinforcement learning at all.

## 1. Shared instrument, built first: the branch-tagged gate recorder

Every S19+ claim about "the law" or "the learner" needs per-step attribution the S18 gate records do not
keep. One recorder, built once, used by every round below *and* by the paper's Phase 0 replay:

- per step: `branch ∈ {law, search}`, nominal command, executed command, intervention magnitude,
  `stalled`, `immobile`, clearance.
- per episode: intervention magnitude split by branch; stalled/immobile run lengths by branch; the
  longest law-pinned run.

**Construct check on the instrument itself (before any round uses it):** replay 20 S18 gate episodes and
confirm the per-episode totals reproduce the recorded `intervention/ep` bit-for-bit. If the new recorder
cannot reproduce the old numbers, nothing built on it is read.

## 2. S19 — the planner intercept (upgrade 1; no RL)

### Design
`InterceptLaw` steers toward a bearing: the evader's when visible, the last-seen point otherwise. The
planner variant keeps **exactly** that structure and changes only the point it steers toward:

- Grid A\* (0.1 m cells) over the known static map — arena inset by `agent_radius`, circular obstacles
  inflated by `agent_radius` + the shield's margin — from own pose to the target (evader position when
  visible; last-seen otherwise). Replanned every step; the map is tiny.
- Steer toward the path point ≈0.5 m ahead instead of toward the target directly.
- **Property that makes the comparison clean:** when the straight line is unobstructed, the lookahead
  point lies on it and the planner law is *identical* to the line law. Only obstructed episodes differ.
  A unit test asserts bit-identity on an empty arena.

### Cells — 2 × 2, no training
| | line law | planner law |
|---|---|---|
| unshielded | existing baseline | new |
| shielded (`CBFShield`, S18 config) | existing baseline | new |

Search policy fixed across all cells: the scripted `SearchingPursuer`'s search branch (teacher), so the
only variable is the law. Opponents `GREEDY`, `HUG`, and one frozen structured attacker (`ATK4-45`) — the
attacker is included *here* because the law–shield deadlock is exactly what attackers learned to induce
(D114/D115), and a planner law that is immune to it is the claim.

N = 500 paired, gate stream **2 000 000+**, construct stream **1 990 000+**.

### Hypotheses, directions fixed now
- **H-PLAN-1:** shielded capture, planner > line, on ≥ 2/3 opponents, `paired_indicator_test` p < 0.05.
- **H-PLAN-2 (the paper's number):** the shield's cost, (unshielded − shielded) capture, is smaller
  under the planner than under the line, on ≥ 2/3 opponents. *The filter's cost depends on the nominal's
  competence.*
- **H-PLAN-3:** the longest **law-branch** pinned run per episode (recorder, section 1) falls by ≥ 50 %
  in median, planner vs line, shielded.
- **H-SAFE:** zero contacts in every shielded cell.
- **Null that must hold:** unshielded, planner vs line capture differ by < 0.05 on open-arena scenarios
  (`pe2/scenarios.py`) — the planner must not be a *different* law in open space.

### Construct gate (blocking)
- CG-A: 200 planned paths on random maps never intersect an inflated obstacle (unit test + figure).
- CG-B: the recorder reproduces S18 totals (section 1).
- CG-C: 3 unselected episodes per cell (12 GIFs), watched before the table — specifically for: does the
  planner ever oscillate between two routes around an obstacle (a known A\*-replanning pathology)?

### Then, conditional on H-PLAN-1 and H-PLAN-2
Retrain the S18 `SPARSE-BASE` search learner under the planner law, 3 seeds, shielded, same budget — to
check the learner's gain is not law-specific. 3 arms, ≈ 6 h with threads set properly.

### Gate to S20
H-PLAN-1 or H-PLAN-2 fails → the law was not the binding constraint; **stop and re-diagnose from the GIFs**
before building anything else. Both pass → S20.

## 3. S20 — belief for the learner (upgrade 2)

### Design
Particle filter over the evader's position, ~500 particles, occlusion-aware: particles inside the
pursuer's current visibility region (FOV ∩ LOS, `pe2/sensing.py`) are killed when the evader is not seen;
survivors propagate with the evader's known speed bound; resample on sighting. Fed to the learner as five
features: belief mean (2), spread (1), bearing to the strongest mode (1), fraction of mass visible (1).
`pe2/belief.py`'s dead-reckoning is the baseline it must beat.

### Cells — 3 seeds each, planner law from S19, shielded
`NO-BELIEF` (S19 retrained learner) · `DEAD-RECKON` (existing stub) · `PARTICLE`. 9 arms.

### Hypotheses
- **H-BEL-1:** capture, `PARTICLE` > `NO-BELIEF`, ≥ 2/3 seeds, p < 0.05 on `GREEDY`.
- **H-BEL-2:** time-to-first-sight after a loss (recorder) is shorter under `PARTICLE`, ≥ 2/3 seeds.
- **H-BEL-3 (the belief must be a belief):** median error of the belief mean to the true evader position
  during occlusion is below dead-reckoning's. **Construct gate, measured before any capture number is
  read** — if the particle filter is not more accurate than dead reckoning, the column is mislabelled.

### Construct gate
CG-D: H-BEL-3 as above. CG-E: 3 unselected episodes per cell, watched for search that *looks* directed.

## 4. S21 — learner takeover on law failure (upgrade 3; conditional)

Only if S19's recorder still shows law-pinned runs ≥ 40 steps in ≥ 5 % of shielded episodes. If the
planner removed them, this round does not exist. Design: after K = 20 law-pinned steps, control passes to
the learner (which holds `escape_obs`) until the evader is reacquired or 50 steps elapse. 3 seeds,
takeover vs none. Hypothesis: law-pinned run length falls; capture does not fall.

## 5. S22 — horizon (upgrade 4; conditional)

Only if S20's GIFs still show standoffs. Design: PPO with λ-returns at γ = 0.995 vs the S20 winner at
0.99; nothing else changes. D128's mechanism predicts a gain; this is its test.

## 6. Where this meets the paper

H-PLAN-2 is the general form of the paper's Case 5: *every "cost of the shield" we ever reported was
measured on an incompetent nominal controller.* S19 measures the same shield on a competent one. Whatever
the sign, it goes into the paper as the demonstration that the construct "shield cost" is undefined
without naming the nominal. If the planner-based agent turns out strongly capable, that is a surprise to
be pre-registered and chased on its own terms — not folded in.

The paper's Phase 0 (closing S18, the law/learner replay) runs **first**, using section 1's recorder. It
is cheap and it is owed.

## 7. Launch discipline (from this round's failures, binding)

`OMP_NUM_THREADS` ≤ cores / processes · literal arm lists, never shell variables · a launch is confirmed
by checkpoint files, never by PIDs · no competing jobs on the host during a round · one stop-rule waiver
per round at most, archived with a README, and only before any arm's *gate* has run.

## 8. Pace

S19 ≈ 2 weeks (planner 3–4 days with tests; recorder 2 days; gates; run; watch; read). S20 ≈ 3 weeks.
S21/S22 ≈ 1 week each if they exist. No round starts before the previous round's gate is met and its
D-entry is written. Total ≈ 6–8 weeks to a competent, measured pursuer.

## 9. First action

Build the branch-tagged recorder (section 1) and pass its reproduction check. It unblocks S19's CG-B and
the paper's Phase 0 at once. Then pre-register S19 in full before a line of planner code is written.
