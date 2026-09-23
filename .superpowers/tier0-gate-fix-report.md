# Tier-0 gate fix: the T0.2 visibility criterion (amendment A7 / D96)

## What changed

**`pe2/rewards.py`**

- `visibility_term(w_vis, obs) -> float` — the `vis` dense term extracted as one
  named function: `w_vis * float(obs[6] > 0.5)`. `DenseRewardWrapper.step` now pays
  the term *through* it instead of inlining the expression, so there is a single
  definition of the term.
- `visibility_monotone_report(cfg, *, w_vis, n_bearings=361, range_m=None, term_fn=visibility_term)`
  — the specification's criterion, implemented analytically. It places the agent at
  the origin heading +x (so a world bearing *is* the relative bearing), holds the
  opponent at a fixed range of `fov_range / 2` = 4.0 m with **no obstacles**, sweeps
  the bearing over [−π, π] on a 361-point grid (step 2π/360 = 1° = 0.0174532925 rad),
  and scores each sample by calling `build_observation` (the env's own observation
  builder, hence `pe2.sensing.is_visible`, hence the FOV cone ∧ clear-LOS predicate
  of D6) and then `visibility_term`. The geometry is never reimplemented.
  Verdict = all three of:
  1. `monotone_into_cone` — non-decreasing as the bearing **magnitude** falls,
     checked on each side of the heading separately;
  2. `varies` — the reward is not constant;
  3. `transition_matches` — the outermost positive bearing sits at the configured
     half-angle `fov_angle / 2` (= π/4 = 45° by default) within one grid step.
  It returns the evidence: `bearings`, `rewards`, the three booleans,
  `transition_bearing`, `expected_transition_bearing`, `transition_tolerance`,
  `grid_step`, `n_bearings`, `range_m`, `fov_angle`, `fov_range`, `obstacles`,
  `w_vis`, `passes`.
- Dispatch: `TERM_KINDS["vis"]` is now `"monotone"`. `evis` stays `"indicator"`,
  `search`/`pbrs` stay `"continuous"`. `BAND = (0.05, 0.95)` and
  `MAX_MODAL_FRACTION = 0.95` are **unchanged**. New `CRITERIA` maps each kind to
  the name of its check and `DETERMINISTIC_KINDS = {"monotone"}` marks the kinds
  that carry no sampling error.
- `nondegeneracy_report(terms, cfg=None, w_vis=None)` — a `"monotone"` term is
  decided by `visibility_monotone_report(cfg, w_vis=w_vis)` and **raises** if `cfg`
  or `w_vis` is missing rather than silently falling back to a frequency criterion.
  Every entry now carries `criterion` and `stats_role`; for `vis` the rollout
  statistics are `stats_role: "observation"` and the geometric evidence is nested
  under `monotone`.

**`scripts/derisk_t0_pilots.py`**

- `gate_decisiveness` excludes `DETERMINISTIC_KINDS` from the episode-level
  bootstrap outright — such a term gets no share. The bootstrap is otherwise
  untouched and still gates the sampled terms.
- `nondegeneracy_gate` calls `nondegeneracy_report(terms, cfg=cfg, w_vis=W_VIS)`.
  A term with no bootstrap share is reported `decisive: true`,
  `bootstrap_pass_share: null`, `decisive_reason` = "monotone_into_cone is an
  analytic property of the reward's geometry: deterministic, no sampling error,
  decisive by construction". Sampled terms gain the symmetric worded reason.
  `bootstrap.terms` records which terms the bootstrap was applied to.
- The console line prints each term's `criterion`, whether its statistics are the
  criterion or an observation, and, for the analytic term, a second line with the
  sweep: monotonicity, variation, the located transition against the configured one,
  the tolerance, the resolution and the range.
- The `UNDECIDED` stop message is rewritten: only sampled criteria can land there,
  and A7 is cited as the precedent for what to do.
- The `NONDEG_EPISODES` budget comment now says the 500-episode budget gates the
  **sampled** terms; the measured evidence about the band is kept in place as the
  record of why the criterion was replaced.

**Docs** — `docs/worklogs/2026-09-04_S121.md` gains amendment **A7** in the existing
Pre-registration amendments table plus six rows in the output-fields table;
`docs/decisions.md` gains **D96** in the existing row format, self-contained.

## Covering tests

`tests/pe2/test_rewards.py` — purely geometric, no learning-library import, so no
import skip (and none was added). Every number is hand-computed in a comment: the
half-angle π/4 = 0.7853981634 rad = 45°, the swept range 8.0/2 = 4.0 m, the grid
step 2π/360 = π/180 = 1°, and each swept bearing.

- `TestVisibilityTerm::test_the_term_is_the_indicator_the_wrapper_pays`
- `TestVisibilityMonotoneCheck`
  - `test_the_shipped_visibility_term_passes`
  - `test_the_swept_curve_is_the_step_the_configuration_specifies` — 0°, ±44° pay
    0.1; ±46°, 90°, 180° pay 0.0
  - `test_the_evidence_is_a_full_sweep_at_a_range_inside_sensing_range`
  - `test_the_check_is_deterministic` — two calls agree exactly
  - `test_a_narrower_cone_moves_the_transition` — `fov_angle = π/3` moves the
    expected and located transition to π/6 = 30°
  - `test_the_check_scores_the_reward_the_wrapper_actually_pays` — the same two
    bearings (30°, 75° at 4 m) scored by the analytic sweep and by a real
    `DenseRewardWrapper` step must agree exactly
- `TestVisTermDispatch`
  - `test_the_vis_term_is_judged_by_geometry_not_by_frequency` — a rollout with a
    positive fraction of **0.03**, below the retired band edge, now passes
  - `test_a_degenerate_vis_reward_fails_the_report`
  - `test_the_monotone_criterion_refuses_to_run_without_the_configuration`
  - `test_the_search_term_keeps_its_non_constancy_check`
- Retargeted, unchanged in substance: the two band tests now exercise `evis`, the
  term that still carries the band; the shipped-configuration rollout test passes
  `cfg`/`w_vis` and asserts the `stats_role` split.

`tests/pe2/test_derisk_pilots_script.py` — keeps its `stable_baselines3` /
`matplotlib` import skips.

- `TestAnalyticTermIsNotBootstrapped::test_the_analytic_term_gets_no_bootstrap_share`
  — with both terms present, only `evis` comes back with a share.
- `TestAnalyticTermIsNotBootstrapped::test_the_gate_marks_it_decisive_by_construction`
  — the whole gate on a 2-episode rollout that could decide nothing by sampling:
  `vis` still returns decisive, share `null`, reason containing "analytic",
  `stats_role: "observation"`, while `search` is bootstrapped and
  `bootstrap.terms == ["search"]`.
- `TestGateDecisiveness` retargeted from `vis` to `evis` with its numbers unchanged
  (the band it exercises is untouched), including the band-edge coin-flip test that
  the machinery exists for.

## Negative control: the check can fail

Not asserted — constructed and run. Three degenerate rewards are swept through the
identical machinery via `term_fn` and each is rejected:

| construction | what it models | rejected by |
|---|---|---|
| pays `w_vis` inside the cone **except** in a blind spot of \|bearing\| < π/8 = 22.5° — 0.1 at 30°, 0.0 at 10° | turning toward the opponent **loses** reward: the exact degeneracy the criterion exists for. It varies, and its outermost positive bearing is still 45°, so only monotonicity catches it | `monotone_into_cone` false |
| pays `w_vis` at every bearing | non-decreasing by construction, so monotonicity alone would pass it | `varies` false |
| pays `w_vis` only inside \|bearing\| ≤ π/8 | monotone and varying, but the reward and the sensing configuration disagree by 22.5° against a 1° tolerance | `transition_matches` false |

And end-to-end, to show the criterion is coupled to the reward that trains rather
than to a second copy of the geometry: replacing the single shared
`visibility_term` with the blind-spot version made `DenseRewardWrapper` pay **0.0**
dead ahead (0.1 shipped) **and** `nondegeneracy_report` return `all_pass: false`,
together, in one run.

## Gate verdict on the shipped configuration

`results/derisk/t0_nondegeneracy.json` was **regenerated** (the only file touched in
`results/derisk/`; the 1M-step signal audit running alongside was left alone and
completed on its own). The gate was re-run at the same seed with the same gated
clone, so the rollout reproduces the previous numbers exactly — 500 episodes /
503,220 clone steps, seed 790500 — and only the verdict fields changed.

```
[nondeg] vis     kind=monotone   criterion=monotone_into_cone     passes=True decisive=True bootstrap_pass_share=analytic (n/a) | observation: frac_positive=0.0548 frac_modal=0.945 std=0.0228
[nondeg] vis     sweep: monotone_into_cone=True varies=True transition=0.7854 rad vs configured 0.7854 (+/-0.0175) over 361 bearings at 4.00 m
[nondeg] search  kind=continuous criterion=non_constancy          passes=True decisive=True bootstrap_pass_share=1.000 | criterion: frac_positive=0.9247 frac_modal=0.431 std=9.97e-06
all_pass= True all_decisive= True
```

The located transition is 0.7853981634 rad — exactly `fov_angle / 2`. `all_pass` and
`all_decisive` are both true, so the gate no longer stops the run and P-A's
remaining Tier-0 blocker is cleared. `search` is unchanged.

## Tests and lint

- `tests/pe2`: **331 passed** (3 m 38 s).
- Whole suite: **1081 passed, 68 failed, 3 skipped** (14 m 29 s). All 68 failures
  are in `tests/test_barriernet_actor.py`, `tests/test_barriernet_training.py`,
  `tests/test_comparison_framework.py` and `tests/test_differentiable_qp.py` —
  legacy files unrelated to `pe2`. Verified pre-existing by stashing every change
  and re-running `tests/test_differentiable_qp.py`: the same 27 failures.
- `ruff format` + `ruff check pe2 tests/pe2 scripts/derisk_t0_signal_audit.py
  scripts/derisk_t0_pilots.py`: clean. The whole `scripts/` directory was never
  linted.

## Files changed

- `pe2/rewards.py`
- `scripts/derisk_t0_pilots.py`
- `tests/pe2/test_rewards.py`
- `tests/pe2/test_derisk_pilots_script.py`
- `docs/worklogs/2026-09-04_S121.md`
- `docs/decisions.md`
- `results/derisk/t0_nondegeneracy.json` (regenerated; gitignored)

Commits on `main`, not pushed: `883fbd5`, `eb9e0c9`, `a8a48fb`.

## Not as prescribed / judgement calls

- **A `term_fn` seam in `visibility_monotone_report`.** The negative controls need
  to run through the identical machinery, so the production function takes the term
  it scores as a keyword-only parameter defaulting to the shipped
  `visibility_term`. This is the only added configurability;
  `nondegeneracy_report` got **no** injection point — the one test that needs the
  report itself to return false monkeypatches the module instead.
- **A two-sided sweep.** The task describes one sweep from behind to dead ahead.
  The implementation sweeps [−π, π] and checks monotonicity on each side of the
  heading separately, so a term that misbehaves on only one side is still caught.
  The `is_visible` predicate is symmetric in the bearing, so this costs nothing and
  closes a hole.
- **A third pass condition.** Beyond monotone and varying, the check also requires
  the step to sit at the configured half-angle. The task names this as evidence to
  report ("where the transition was found against where the configuration says it
  should be"); making it a pass condition is a small strengthening, and it is what
  catches a reward that disagrees with the sensing configuration.
- **`criterion` / `stats_role` added to every term**, not only to `vis`. Marking one
  term's statistics as "no longer the criterion" only reads clearly against terms
  that say theirs still is.
- **`tests/pe2/test_rewards.py::test_nondegeneracy_on_a_real_rollout` was left
  untouched.** It asserts `0.05 < frac_positive < 0.95` on a three-episode rollout,
  which is the retired band — but it exercises `signal_nondegeneracy`, a
  descriptive function that is unchanged and still produces the recorded
  observation; it does not call `nondegeneracy_report` and encodes no criterion.
  Changing it would have been an unrequested edit.
- Nothing else was weakened: `BAND`, `MAX_MODAL_FRACTION`, the search and evader
  criteria, `pilot_passes`, the 0.137 bound and the D94 collapse criterion are all
  untouched.
