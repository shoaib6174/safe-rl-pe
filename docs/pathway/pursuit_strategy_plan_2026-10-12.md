# Plan: the fifth upgrade — pursuit strategy against a faster, obstacle-hugging evader (S23)

Written 2026-10-12. Follows `agent_intelligence_plan_2026-10-05.md` (upgrades 1–4) and D142, whose gate
withheld S20. This is a plan, not the pre-registration; the pre-registration is written only after §4's
construct checks have been built and before any gate number exists.

## 0. What D142 established, stated as mechanism

Removing the law's zero-clearance pins (−94…−99 %) moved capture by nothing. CG-C showed why: the
unshielded line law captures by brushing past the obstacle; with contact forbidden — by the shield or by a
planner that respects the same inflated map — the evader keeps the obstacle between them and the chase
becomes an orbit. **The pursuer's intercept rule is "steer at where the evader is."** Against a target that
is 1.2× faster (`evader_speed_ratio = 1.2`, D5) and free to orbit, following cannot close. Capture radius is
0.30 m (2 × `agent_radius`); `GreedyEvader` flees directly away from the pursuer's current position.

The binding constraint is therefore **pursuit strategy**, not intercept competence. This upgrade changes
only the point the law aims at when the evader is visible. Everything else — the planner's obstruction
handling, the shield, hold, observation flags — is S19's, frozen.

## 1. The strategy family — one variable, three members

| law | aim point when the evader is visible | what it tests |
|---|---|---|
| `PLAN` (S19, the reference) | the evader's current position | — |
| `LEAD-τ` | current position + τ · estimated velocity, τ ∈ {0.5, 1.0} s | "following never closes": does aiming ahead on the orbit close the gap? |
| `MPC-H` | the first action of the best sampled H-step pursuer trajectory against a constant-velocity evader prediction, cost = predicted minimum distance (0 at capture) + small path length | "reasoning about where it will be": does a short-horizon plan over our own exact dynamics find the cut-off, the reversal or the wall-cornering by itself? |

Velocity is estimated from two consecutive world positions reconstructed from the observation (the law
already keeps `last_seen`; one more remembered position is the same kind of state). `MPC-H` is
sampling-based (K = 64 action sequences at `hold`-length segments, H = 2.0 s = 40 steps, vectorised numpy
over our own `pursuer_step`), plans through the planner's inflated map for feasibility, and is executed
through the shield like every other law. No learning anywhere.

**Invariant that keeps it one variable:** with the evader stationary (estimated velocity 0) every member
must produce a bit-identical action to `PLAN` — `LEAD` trivially, `MPC` by construction when the best
sampled trajectory is the direct one (tested on an obstacle-free arena with a frozen evader).

## 2. What the opponent model is, and what it is not

`MPC-H` predicts the evader by constant velocity — inertia, no assumption about its policy. Fitting the
model to `GreedyEvader` would be overfitting to one opponent and would be caught by the three-opponent
gate, but it would also be the wrong science: the claim is about the pursuer reasoning, not about knowing
the opponent. **The model is itself a construct and gets its own check (CG-S2).**

## 3. The diagnostic that tells us WHICH mechanism any gain comes from — run first, 1 hour

A faster evader orbiting an isolated obstacle far from any wall can, in principle, orbit forever; capture
without contact then comes only from the **arena walls** cutting the orbit. Before the round:

- **D-cell:** a single obstacle at the arena centre, `GreedyEvader`, 100 construct-stream episodes per law.
  If no strategy captures here, any gain in the full round is wall-cornering and the paper says so. If
  `LEAD` or `MPC` captures here, the orbit itself is breakable and the mechanism is different.
- This is a diagnostic, not a hypothesis; its reading is written before it runs and it moves no bar.

## 4. Construct checks — built and passed before the pre-registration is written

- **CG-S1 (one variable):** the stationary-evader bit-identity above, 50 obstacle-free seeds per member.
- **CG-S2 (the model is a model):** on the S19 gate stream, per opponent, the distribution of
  constant-velocity prediction error at H = 2 s against the evader's actual position. Reported as a
  median and p90 in metres beside the 0.30 m capture radius. If the median exceeds the radius by an order
  of magnitude, `MPC-H` is planning against fiction and its result is read as such.
- **CG-S3 (informative episodes, chosen by a rule, not by outcome):** S19's CG-C spent two of its three
  seeds on episodes where the line was never obstructed. The three watched seeds are the **first three
  construct seeds on which `PLAN-SHIELD`'s law branch fires with the line obstructed for ≥ 1 step** — a
  rule on the baseline's geometry, fixed before any treatment runs, blind to every outcome.
- **CG-S4 (the recorder):** the branch-tagged recorder from S19, unchanged; `aim_point` added to the
  per-step record so a reviewer can see what the law was aiming at.

## 5. The round — evaluation only, no training

Cells: `{PLAN, LEAD-0.5, LEAD-1.0, MPC-2s} × SHIELD × {GREEDY, HUG, ATK}` = 12, N = 500 paired, fresh gate
stream **2 100 000+**, construct **2 090 000+**. `PLAN-SHIELD` is re-run on the new stream as the
reference; S19's numbers are not reused across streams.

Hypotheses (directions fixed at pre-registration):
- **H-STRAT-1:** shielded capture, member > `PLAN`, per opponent `paired_indicator_test` p < 0.05, on
  ≥ 2/3 opponents — evaluated per member; passing on any member passes the round.
- **H-STRAT-2 (the mechanism, so that S19's lesson is not repeated):** define an **orbit episode** on the
  `PLAN-SHIELD` baseline: evader visible for ≥ 100 consecutive law steps with no capture. The paired
  conversion rate (baseline lost → member captured) must be **higher on orbit episodes than on non-orbit
  episodes**. A gain that is not concentrated where the mechanism says it should be is a gain from
  something else and is reported as such.
- **H-SAFE** and the **NULL** (CG-S1 carried into the round record), as S19.

**Refutation, written now:** all three members fail H-STRAT-1 → pursuit strategy is not the constraint
either, and with the diagnostic of §3 the remaining explanation is structural: *with contact forbidden and
a 1.2× faster evader, the shield's task cost is irreducible by the nominal controller.* That is a bound,
reportable in the paper as the end of the "competent nominal" line, and the agent-intelligence plan stops
there rather than reaching for a sixth upgrade.

## 6. What this is worth to the paper

Either outcome lands. A pass is the first round in which a task metric moves under the filter, and
H-STRAT-2 says why. A fail closes the line with a measured bound and joins D129/D138/D142 as evidence
that filter-adjacent metrics and task metrics come apart for structural reasons, not for lack of trying.

## 7. Pace and discipline

Build order: velocity estimate + `LEAD` (1 day) → CG-S1 → §3 diagnostic (reads before `MPC` is built) →
`MPC-H` vectorised (≈ 1 week, tested against the exact dynamics) → CG-S2, CG-S3 → **pre-registration** →
review → frozen worktree → run (≈ 3–4 h; `MPC` is the slow row) → CG-S3 watched → `--summarize`.
Launch discipline per `agent_intelligence_plan` §7 and the S19 wrapper (CG gate in the runner, records
validated, no rate printed before the GIFs, captions without rates). ≈ 2–3 weeks.

## 8. First action

Implement the velocity estimate and `LEAD-τ` behind the same `law=` hook, with the CG-S1 identity test,
and run the §3 diagnostic for `PLAN` and `LEAD` only. The diagnostic's reading decides whether `MPC-H`
is built at all: if `LEAD` already breaks the orbit on an isolated obstacle, the sampling planner may be
unnecessary and the round shrinks.


---

## Amendment A1 (2026-10-12, after the §3 diagnostic ran once; before any round number)

**§3's design did not isolate the orbit.** "A single obstacle far from any wall" isolated the obstacle, not
the evader: `GreedyEvader` flees radially and the 10 m arena's walls corner it, so 61 % of the 100 episodes
were captured by wall-cornering in every law. The diagnostic's decision (`MPC-H` is built; LEAD +0.01/+0.02
< 0.10) stands because the reading was written first and the bar is conservative in that direction. The
mechanism attribution it was meant to provide moves entirely to **H-STRAT-2** (orbit episodes defined on
the baseline). No corrected diagnostic is run: the round answers the question at N = 500 with three
opponents. Recorded as D143.
