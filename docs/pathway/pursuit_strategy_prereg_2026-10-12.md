# S23 — pursuit strategy against a faster, obstacle-hugging evader

**Status: PRE-REGISTERED. Committed BEFORE any round number exists.** Steps 1–4 (laws, diagnostic, construct
checks) are complete (Steps 1 and 3 reviewed CLEAN; Steps 2 and 4 are once-run measurement scripts with
tests and pre-written readings, self-checked — recorded as a ruling in the orchestration ledger); their measured results are quoted below as inputs, not outcomes.
Governing: `pursuit_strategy_plan_2026-10-12.md` (+A1, A2 and the A2 review notes), D142 (S19), D143
(steps 1–2), `s23_orchestration_2026-10-12.md`, D83.

## 1. Claim

**Hypothesis S.** With contact forbidden, a pursuer that aims at where the evader *is* cannot close on a
1.2× faster evader that keeps an obstacle between them (D142). A law that aims at where the evader *will
be* — by a fixed lead (`LEAD-τ`) or by short-horizon reasoning over the simulator's own exact dynamics
(`MPC-H`) — captures more under the shield, and the gain is concentrated in the episodes where the
baseline was orbiting.

## 2. The laws (frozen; one variable — the aim point when the evader is visible)

| law | aim point | commit |
|---|---|---|
| `PLAN` (reference, S19) | the evader's current position; A\* around the known map when the line is obstructed | `a252bc8` |
| `LEAD-0.5`, `LEAD-1.0` | current position + τ·v̂, τ ∈ {0.5, 1.0} s; v̂ from consecutive-step sightings via `env.current_step`; bit-identical to `PLAN` when v̂ reads zero (50/50 seeds, both τ) | `8769115` |
| `MPC-2s` | first segment of the min-cost of K = 64 sampled 40-step (2.0 s) action sequences (4 segments × 10 steps) rolled through `pursuer_step_batch` (bit-equal to `pursuer_step`); evader predicted by constant velocity v̂ clipped to the arena; cost = terminal predicted distance, capture bonus −(H − first capture step), + 0.01 × path length as tie-breaker; sequences touching an inflated obstacle or the inset arena bound are infeasible; the direct row (`PLAN`'s own action held 40 steps) is always included; replanned every call; RNG seeded from (digest of the numeric post-reset generator state, env step); bit-identical to `PLAN` when v̂ reads zero (50/50) | `d2ffeae` + Step 4 |

Stated limitations, from Steps 1–3 (not changed): v̂ is zero after every learner hold span, so the treated
laws' exposure varies per episode (0–161 planned calls on S19 seeds) — the per-episode planned-call count
is recorded; `MPC` returns `PLAN`'s action whenever 0/64 sampled rows are feasible (wall-adjacent states;
`feasible_count` recorded); predicted capture within 2 s is rare against a 1.2× evader, so `MPC` selects by
terminal distance throughout.

## 3. Cells — evaluation only, no training

`{PLAN, LEAD-0.5, LEAD-1.0, MPC-2s} × SHIELD × {GREEDY, HUG, ATK}` = **12 cells**, N = **500** paired on
gate stream **2 100 000+**; shield `CBFShield(cfg)` defaults (margin 0.0, γ 0.2 — asserted), hold 10,
`shield_obs`, `escape_obs`, `reward_wrapper=None`, search `OrbitSearch`, opponents exactly as S19
(ATK = `StructuredEvaderPolicy(ATK4-45 10 M, hold 10)`). `PLAN-SHIELD` is re-run on this stream; S19's
numbers are not reused. Recorder: `BranchTaggedSearchMDP` with `aim_point`, `v_est`, `feasible_count`,
`chose_direct` per step (Step 4). All-step intervention scope (S19 A1.2).

## 4. Hypotheses — directions fixed now

- **H-STRAT-1 (per law):** shielded capture, law > `PLAN`, per opponent `paired_indicator_test` p < 0.05,
  on ≥ 2/3 opponents. Any law passing passes the round; each law's verdict is reported.
- **H-STRAT-2 (mechanism, per law that passes H-STRAT-1, and reported for all):** on the `PLAN-SHIELD`
  baseline, an **orbit episode** is one with ≥ 100 consecutive law-branch steps with the evader visible and
  no capture. Paired by seed, the conversion rate (baseline lost → law captured) on orbit episodes must
  exceed the conversion rate on non-orbit episodes, per opponent, on ≥ 2/3 opponents. If `PLAN-SHIELD` has
  fewer than 20 orbit episodes against an opponent, that opponent is `VACUOUS` for H-STRAT-2 (never PASS).
  A gain not concentrated in orbit episodes is reported as a gain from something else.
- **H-SAFE:** 0 contact steps in all 12 cells.
- **NULL (CG-S1, carried into the round record):** 50 obstacle-free construct seeds (**2 090 000+**),
  `StationaryEvader`, each law vs `PLAN`: `play_episode` logs and per-step trace `==`. Any failure →
  `direction()` says no other verdict is about the aim point.

**Refutation (plan §5):** all three laws fail H-STRAT-1 → pursuit strategy is not the constraint either.
With contact forbidden and a 1.2× faster evader, the shield's task cost is **irreducible by the nominal
controller** at this horizon and these strategies — reported as a bound; the agent-intelligence line stops.
**H-STRAT-1 passes but H-STRAT-2 fails** → the gain is real but not the orbit mechanism; reported as such;
the mechanism question stays open.

## 5. Construct gate — before any capture rate is read

- **CG-S1:** the NULL above (Steps 1 and 3 each passed it on 1 990 000+; the round record re-produces it on
  2 090 000+).
- **CG-S2 (the model is a model):** constant-velocity 2-s prediction error of the evader on the S19 gate
  stream, `PLAN-SHIELD`, per opponent — **measured in Step 4 (`dc04e63`, `results/derisk/s23_cg_s2.json`):**

  | opponent | median | p90 | max | n law steps | frac v̂ ≠ 0 |
  |---|---|---|---|---|---|
  | GREEDY | 0.822 m | 2.967 | 3.668 | 53 018 | 0.494 |
  | HUG | 2.483 m | 2.765 | 3.668 | 87 210 | 0.658 |
  | ATK | 2.433 m | 2.739 | 3.667 | 176 774 | 0.788 |

  **Stated plainly, not only as a verdict:** H·dt·v_evader = 40 × 0.05 × 1.2 = 2.40 m, and the HUG and ATK
  medians sit exactly there — against those two opponents the constant-velocity prediction is typically
  wrong by as much as it predicts (8× the capture radius), i.e. inertia carries no information at 2 s.
  Against GREEDY it carries some (0.82 m). **Pre-registered expectation, not a bar:** if `MPC-2s` beats
  `PLAN` anywhere it is expected on GREEDY; a gain on HUG/ATK would be *despite* the model and would need
  explaining, not celebrating. `n_zero_feasible` is an upper bound (it also counts delegated calls); the
  exact 0/64 count is `feasible_count == 0 and aim_point is not None` from the per-step record.. Reading fixed before it
  ran: *median ≥ 3 m (an order above the 0.30 m capture radius) ⇒ `MPC` is planning against fiction and its
  result is read as such.* Verdict by the pre-written rule: **not crossed on any opponent** (no median ≥ 3 m); `MPC-2s` is run and read with the expectation above attached.
- **CG-S3 (watched episodes by rule):** the first three construct seeds (2 090 000+) on which
  `PLAN-SHIELD`'s law branch took its A\* path for ≥ 1 step vs `GREEDY` — **2 090 000, 2 090 005, 2 090 006** (scanned 7; hits carried 23 / 10 / 58 obstructed law steps; `results/derisk/s23_cg_s3_seeds.json`). 4 laws × 3 seeds = 12 GIFs, captions without any measured rate (S19 A5.1), watched
  before the table, specifically for: `MPC` wandering, `LEAD` overshooting the aim into a wall, and whether
  any law breaks an orbit or only corners.
- **CG-B:** the S19 wrapper's reproduction gate is not re-run (the recorder is unchanged except for
  additive fields); the runner asserts the additive fields do not alter `intervention_total` on 20 replayed
  S18 episodes before the first cell.

## 6. Stop rule

Nothing is re-run, retuned or re-seeded after any number of this round is seen. Amendments only before
the number they touch exists, appended with a timestamp. A crashed cell is `UNDETERMINED`. Records are
measurable only at N = 500 on seed0 2 100 000; duplicates reject both (S19 A4/A5).

## 7. Artefacts

`results/derisk/s23_cells/{LAW}-SHIELD_{OPP}.json`, `s23_round.{json,png}`, `s23_null.json`, `s23_gifs/`,
`s23_cg_s2.json`, `s23_cg_s3_seeds.json`. Frozen worktree at the commit that passes review. Launched
through a wrapper with the S19 gates (CG assertion in the runner, records validated, no rate printed before
the GIFs). Expected wall time ≈ 3–4 h (`MPC` row ≈ 1.6 ms/call).
