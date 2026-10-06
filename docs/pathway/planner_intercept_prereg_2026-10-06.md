# S19 — the planner intercept: is the shield's cost a property of the nominal controller?

**Status: PRE-REGISTERED. Written and committed BEFORE any planner code exists.**
Governing: `agent_intelligence_plan_2026-10-05.md` §2, D138 (S18 closed), S126 CG4 (the law drives into
walls), D83 (pre-registration discipline), `retrospective_2026-09-23.md` §4 (construct gate).

## 1. Claim

`InterceptLaw` steers at a visible evader along a straight bearing. When an obstacle or wall lies on that
bearing the shield pins it at zero clearance (CG4: ~1100 steps in one episode). **Every "cost of the
shield" this project has reported was measured on that controller.**

**Hypothesis P:** the shield's task cost is largely the nominal controller's incompetence. A law that
plans around the known static map, and is *identical* to the line law whenever the line is clear, will
(i) capture more under the shield and (ii) show a smaller shield cost, with (iii) law-pinned runs largely
gone — and no RL is needed to show it.

## 2. The planner law — binding design

`pe2/planner.py::PlannerInterceptLaw(cfg)`, same interface and same two branches as `InterceptLaw`
(visible → target = evader position from obs; else last-seen; else `None`). It changes ONLY the point
steered toward:

1. **Clear-line check first.** If the straight segment own→target clears every obstacle inflated by
   `cfg.agent_radius + margin` (margin = the S18 shield's, **0.0**) and the arena inset, steer at the
   target through the **identical code path** as `InterceptLaw` (`K_OMEGA`, `_clip_omega`, `_normalize`).
2. **Otherwise** grid A\* at **0.1 m** cells (8-connected) over the inflated map, then line-of-sight
   shortcutting of the path; steer at the path point **0.5 m** ahead (the goal if the path is shorter).
   Replanned every step.
3. Map access: `SearchMDP` gains `law: InterceptLaw | None = None` (default constructs `InterceptLaw` —
   bit-identical default) and `_begin` calls `self.law.reset(env=self.inner.env)`; `InterceptLaw.reset`
   accepts and ignores `env`. The planner reads `env.obstacles` at reset. Scripted laws are full-state by
   project convention (L1.5); the map is static and known.

**Invariant, tested:** on an arena with no obstacles, `PlannerInterceptLaw` and `InterceptLaw` produce
bit-identical actions on every step of 50 seeded episodes. Without this the comparison measures two laws.

## 3. Cells — 2 × 2, no training

| | `LINE` (`InterceptLaw`) | `PLAN` (`PlannerInterceptLaw`) |
|---|---|---|
| `OPEN` (no shield) | | |
| `SHIELD` (`CBFShield(cfg)`, S18 config: hold 10, `shield_obs`, `escape_obs`) | | |

Search policy in every cell: `OrbitSearch` (the teacher's search). Opponents: `GREEDY`, `HUG`,
`ATK4-45` (frozen structured attacker, `results/derisk/ckpts_ATK4-45/ATK4-45_10000000.zip`, hold 10 —
included because inducing law–shield deadlock is what attackers learned to do, D114/D115).
N = **500** paired, gate stream **2 000 000+**, construct stream **1 990 000+**. Recorder: the branch-
tagged `BranchTaggedSearchMDP` (`pe2/recorder.py`), gated by CG-B below.

## 4. Hypotheses — directions fixed now

- **H-PLAN-1:** SHIELD capture, PLAN > LINE, on ≥ 2/3 opponents, `paired_indicator_test` p < 0.05.
- **H-PLAN-2 (the paper's number):** shield cost = OPEN capture − SHIELD capture is **smaller** under PLAN
  than LINE on ≥ 2/3 opponents (difference-in-differences; the SHIELD gain significant as in H-PLAN-1 and
  the OPEN difference within the null band below).
- **H-PLAN-3:** SHIELD, median over episodes of `longest_pinned_run_by_branch["law"]` falls by ≥ 50 %,
  PLAN vs LINE, on ≥ 2/3 opponents.
- **H-SAFE:** 0 pursuer contact steps in both SHIELD cells, every opponent.
- **NULL (must hold):** OPEN, |PLAN − LINE capture| < 0.05 on every opponent AND bit-identity on the
  open-arena scenarios of `pe2/scenarios.py`. If the null fails, PLAN is a *different* law in open space
  and H-PLAN-1/2 are not about obstruction; report and stop.

**Refutation:** H-PLAN-1 fails → the law was not the binding constraint under the shield; re-diagnose
from CG-C's episodes before building anything else (the plan's gate to S20).

## 5. Construct gate — BLOCKING, before any capture rate is read

- **CG-A:** 200 planned paths on 200 random maps (construct stream) never enter an inflated obstacle or
  leave the inset arena; a figure of 12 of them.
- **CG-B:** `BranchTaggedSearchMDP` reproduces an S18 arm's recorded gate `intervention.total` and
  `capture_rate` **exactly** on replay (the recorder brief's check). A recorder that cannot reproduce the
  old numbers reads nothing new.
- **CG-C (behavioural):** 3 unselected episodes per cell on construct seeds 1 990 000–1 990 002 against
  `GREEDY` (12 GIFs), watched before the table, specifically for route oscillation around obstacles (an
  A\*-replanning pathology) and for any pinned run the planner did not remove.

## 6. Conditional second stage (only if H-PLAN-1 and H-PLAN-2 pass)

Retrain S18's `SPARSE-BASE` learner configuration under `PLAN`, seeds 70/71/72, 5 M steps, probe stream
**2 050 000+**, gate as above. **H-PLAN-4:** capture ≥ the S18 `SPARSE-BASE` arm of the same seed on
≥ 2/3 seeds (non-inferiority; the learner's gain is not law-specific). 3 arms, launched per the plan's §7
discipline (`OMP_NUM_THREADS` ≤ 4, literal arm lists, confirmed by checkpoints).

## 7. Stop rule

Nothing is re-run, retuned or re-seeded after any number of this round is seen. Amendments only before
the number they touch exists, appended here with a timestamp. A crashed cell is `UNDETERMINED`.

## 8. Artefacts

`results/derisk/s19_cells.json` (per cell × opponent), `s19_round.{json,png}`, `s19_construct.json`,
`s19_gifs/`, `s19_paths.png`. Launched from a frozen worktree at the commit that passes CG-A and CG-B.

---

## Amendment A1 (2026-10-06, before any S19 number exists)

Raised by the recorder's reproduction run on S18 `SPARSE-PEN-70` vs `GREEDY` (D139), not by any S19 data.

**A1.1 — H-PLAN-3's statistic had no power.** The law's longest pinned run has **median 0** on that cell:
only 85/500 episodes carry any law pin at all; it is a heavy tail, not a typical episode. "Median falls by
≥ 50 %" would read 0 against 0 — the same uninformative shape as S17's H3 (D131). **H-PLAN-3 is restated:**
the **fraction of episodes with a law-branch pinned run ≥ 40 steps** (`aggregate.episode_frac_law_pin_ge_40`,
already recorded; 0.152 on the reference cell) falls by ≥ 50 % relative, PLAN vs LINE, SHIELD, on ≥ 2/3
opponents. The median and the 90th percentile of the longest law run are reported beside it, not gated.

**A1.2 — intervention scope is the all-step scope.** S18's `intervention.total` began at the first learner
decision (`step()` zeroes the accumulator), omitting the law steps before it — 9.3 % of the magnitude on
the reference cell, all law. S19 is evaluation-only and uses the recorder's `intervention_total` (every
step). Any comparison to an S18 number must say which scope it uses.

**A1.3 — record schema.** `longest_pinned_run` is an int with `branch_at_start` as a sibling key (`None`
when no immobile step), per the implementation; the runner reads that schema.

---

## Amendment A2 (2026-10-06, before any S19 gate number exists)

Raised by the planner implementation's report (commit `29b6d7e`). **Disclosure:** the implementer ran a
20-episode OPEN-column smoke vs `GREEDY` (PLAN 0.25, LINE 0.30, n = 20, not a gate) which I have seen; the
amendments below are motivated by mechanism (A2.1) and by a logical error in my own null (A2.2), and
would be made identically had I not seen it. Both change *what is measured*, not any bar's value.

**A2.1 — §2 step 2's fallback was wrong, and wrong exactly where it matters.** "If A\* finds no path,
steer at the target directly" fires on **~14 % of planner calls** on the default arena. In 169 of 359
smoke cases the target *point* is clear and only its 0.1 m cell *centre* is blocked — pure
discretisation; the rest are the evader at zero clearance against an obstacle. Either way the rule
degrades PLAN to LINE precisely when the evader hugs an obstacle — the `HUG` opponent's and the learned
attackers' entire behaviour (D114/D115) — so H-PLAN-1 on `HUG` would have measured a discretisation
artefact, not planning. **Restated:** the A\* goal is the **nearest free cell to the target point within
one inflated obstacle radius** (goal-snapping); direct steering only if no free cell exists in that
radius. `n_unreachable` continues to count the residual and is reported per cell.

**A2.2 — my NULL conflated "open space" with "the OPEN column".** The OPEN column is *unshielded*, not
obstacle-free: obstacles are present, and a law that plans around them legitimately differs from one
that drives into them even without a shield. A capture band of |PLAN − LINE| < 0.05 in that column
would fail for the right reason. **Restated:** the NULL is the obstacle-free bit-identity alone (passed,
50/50, `test_planner.py`). The OPEN-column difference is **reported**, not a null, and **H-PLAN-2 drops
its "OPEN difference within the null band" conjunct** — the difference-in-differences
(OPEN − SHIELD per law) already isolates the shield's marginal cost per law regardless of the OPEN level.

**A2.3 — arena-bound tolerance, recorded.** The clear-line check tests the inset arena bound to
`8·eps_f32·arena_size = 9.54e-6 m`, the precision of the float32 observation it decodes from; an exact
test rejected a clear line on seed 1 990 009 and fired the planner on an empty map. Obstacle clearance
keeps the exact `≤ 0 ⇒ blocked` rule. No bar changed; the prereg was silent.

**A2.1 clarification (2026-10-06, commit `e356ae6`).** "One inflated obstacle radius" is the radius of the
obstacle **nearest the target point** — the one whose disc blocked the cell. It bounds only the search
window; the goal is still the nearest free cell, so no reading of this changes a result. After the change:
`n_unreachable` 359/2554 (14.1 %) → 0/2704 (0.0 %) on the same 20-episode smoke; invariant 50/50; CG-A 0/200.

---

## Amendment A3 (2026-10-06, before any S19 gate number exists) — runner rulings

From the runner implementation (commit `fa3b0d9`, 63 tests). Rulings on its stated readings:

**A3.1 — the NULL runs on an obstacle-free arena, not `pe2/scenarios.py`.** §4's phrase "open-arena
scenarios of `pe2/scenarios.py`" cannot implement the invariant: `close_open` / `mid_open` / `far_open`
are start geometries on the canonical two-obstacle layout ("open" = clear line of sight at the start). A2.2
already restated the NULL as the obstacle-free bit-identity; the runner implements that
(`EnvConfig(n_physical_obstacles=0)`, 50 construct seeds, `OrbitSearch`, `==` on the full log including
`branch`, built through the round's own `build_cell_env`). §4's wording is superseded.

**A3.2 — CG-B is enforced at launch, not in the runner.** The runner does not read the replay artefact.
The launch script refuses to start any cell unless all 36 `s18_branch_split_*.json` carry an exact
reproduction PASS, and prints the tally into the launch log. A manual gate is weaker than a mechanical one
and this is recorded as such.

**A3.3 — H-PLAN-3 VACUOUS scope.** Per opponent: LINE fraction 0 → that opponent is VACUOUS (never PASS,
never FAIL). Overall: PASS on ≥ 2 real passes; VACUOUS (not FAIL) when real passes < 2 but
passes + vacuous ≥ 2 — the shortfall is an absent phenomenon, not evidence against the planner. Accepted as
the reading closest to A1's intent.

**A3.4 — OPEN-column pin/intervention fields are structurally zero** (no shield → the recorder writes its
"nothing acted" record). H-PLAN-3 and H-SAFE are SHIELD-only; the OPEN contact count is reported, not gated.
"OPEN has no law pins" is not a finding.

**A3.5 — contact counting.** `pe2.metrics.summarize_shield` refuses shield-less logs, so contact steps
come from `SearchMDP.episode_violations` in both columns (the env's own contact predicates, every step).

**Disclosure.** The implementer ran a 4-cell smoke at N = 20 vs `GREEDY` (not a gate; N not
pre-registered): LINE-SHIELD capture 0.25 / law-pin fraction 0.25; PLAN-SHIELD 0.25 / 0.00; LINE-OPEN
0.30; PLAN-OPEN 0.25; mean all-step intervention 552 → 210 under the shield. Seen; must anchor nothing.

---

## Amendment A4 (2026-10-06, before any S19 gate number exists) — review rulings

Independent review of the runner (commit `fa3b0d9`): APPROVE-WITH-FIXES, two confirmed defects. Rulings:

**A4.1 — H-PLAN-2's significance conjunct is PER OPPONENT (binding).** §4/A2.2 read "cost(PLAN) < cost(LINE)
on ≥ 2/3 opponents AND the SHIELD gain significant as in H-PLAN-1". The runner applied the round-wide
H-PLAN-1 verdict, so the 2/3 with the smaller cost could be a different pair from the 2/3 with a
significant gain. **Binding reading:** an opponent counts toward H-PLAN-2 only if, on that opponent, both
cost(PLAN) < cost(LINE) AND the SHIELD gain is significant at p < 0.05. Pass on ≥ 2/3 such opponents. The
round-wide conjunction is reported beside it, not gated. Stricter than the implementation, chosen because
"the paper's number" must not rest on a gain that was not significant where the cost was smaller.

**A4.2 — the NULL runs on BOTH the OPEN and SHIELD panels.** On OPEN every shield-derived field is
structurally zero (A3.4), so the identity there cannot witness a divergence in `violations`, immobility or
intervention. The 50-seed obstacle-free identity is run on `LINE-SHIELD` vs `PLAN-SHIELD` as well; **both**
must hold for the NULL to pass.

**A4.3 — CG-B is enforced INSIDE the runner, before the first episode.** `run_cells` asserts that all 36
`s18_branch_split_*.json` exist with exact-reproduction PASS and records the tally in provenance. A3.2's
"manual gate is weaker" concession is withdrawn; the launch wrapper remains as a second check and is
committed to `scripts/` so it is provenance, not a file on a desk.

**A4.4 — record validity.** A cell record is measurable only if `n_episodes == 500` and
`seed0 == 2 000 000`; any other record (a smoke, a test artefact) is refused loudly, never skipped as "done"
and never summarised. **A4.5 — a crashed cell writes `UNDETERMINED`** and is never re-run (§7).
**A4.6 — no capture rate is printed during `--cells`**; the only reader of the numbers is `--summarize`,
which runs after `--gifs` (CG-C). **A4.7 — `device` is in provenance.**
