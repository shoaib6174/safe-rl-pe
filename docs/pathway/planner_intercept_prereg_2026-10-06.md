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
