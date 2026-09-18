# Second Filter (S11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second hard safety filter (`PredictiveShield`, exact lookahead over the environment's own dynamics) with the `CBFShield` interface and info keys, its validity preflight, and an S9-style evaluation sweep over its horizon with the pre-registered verdicts.

**Architecture:** `PredictiveShield(CBFShield)` overrides `filter` only: it reuses `CBFShield.constraints`/`_info` for the reported geometry and `_qp_candidates` for the candidate set, and decides acceptance by simulating `pe2.env` pursuer dynamics (factor the pursuer's one-step motion out of `PursuitEvasionEnv` into a pure function `pe2.env.pursuer_step(pose, v, omega, cfg, obstacles, x_lim, y_lim) -> (pose, wall_contact, obstacle_contact)` that the env itself calls, so the filter and the env share one implementation by construction). `scripts/derisk_s11_filter.py` reuses S9's cell/pool/record machinery via importlib (`scripts/derisk_s9_sweep.py`), overriding the cell table, shield factory (`PredictiveShield(cfg, horizon=H, margin=0)` or `CBFShield` for the base cells), prefix `s11`, stream, and verdicts.

**Tech Stack:** as S9.

**Spec:** `docs/pathway/second_filter_prereg_2026-09-18.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The env refactor (Task 1) must be bit-identical: `PursuitEvasionEnv.step` produces the same poses/contact flags before and after on 200 seeds × 50 random steps (test).
- `PredictiveShield.filter` contract: same signature and return as `CBFShield.filter`; info dict has exactly `CBFShield._info`'s keys plus nothing removed; `method ∈ {"exact", "backup"}`; `feasible` = an accepted candidate existed; `intervention` as `_info` computes it. Acceptance = simulate `H` held steps + 1 full-stop step with the shared `pursuer_step`; reject on any contact flag or any hard clearance < margin. Candidate set = `_qp_candidates(...)` ∪ {nominal} ∪ {u = 0}; nearest-to-nominal in the same metric `CBFShield` uses (check `_qp_candidates`/`filter` for the weights `w_v`, `w_omega`).
- Pre-registered cells/streams/tests exactly as spec §2–3: pursuers RC19/RCM28/STR14 (+ FT-B-41 reported), opponents ATT25/ATK2-31/HUG7/GREEDY, `HORIZONS = (1, 4, 16, 32)` (amendment A1), CBF base cells for FT-B-41, `SEED0 = 1_330_000`, `N = 300`, `MAX_WORKERS = 6`.
- V-P preflight per spec §3 (H=0 identity on 10 000 random states; 200 episodes RC19 vs ATT25 at H=4 with zero contact; 1 000 brute-force accept-decision checks against a fresh env instance).
- Never write under `results/derisk/` except the real `--preflight` (`s11_preflight.json`); tests/smokes → scratch. No real sweep launch (controller launches from a worktree).

---

### Task 1: `pe2.env.pursuer_step` (shared one-step dynamics) — behaviour-preserving

**Files:** Modify `pe2/env.py`; Test `tests/pe2/test_env.py` (append).

- [ ] Step 1 test: record 200 seeds × 50 random joint actions of `PursuitEvasionEnv` poses + contact flags at HEAD (compute expected by running the CURRENT code before editing and pin as a fixture generated in-test from a copy of the current functions — simplest: implement `pursuer_step` and assert the env's `_advance_agent` + `_project_out_of_obstacles` sequence equals `pursuer_step` on random states, plus an end-to-end equality against a saved trace produced by `git stash`-free means: run the env for 50 steps, then run `pursuer_step` on the same action sequence from the same initial pose and obstacles, and compare). Also assert the evader path is untouched.
- [ ] Step 2 implement: `pursuer_step(pose, v, omega, cfg, obstacles, x_lim, y_lim)` pure (no env state), used by `PursuitEvasionEnv.step` for the pursuer (evader continues with the existing methods, or both use the function with a role-agnostic signature — either is fine as long as bit-identity holds). Step 3 tests pass; ruff; commit `refactor(env): pure pursuer_step shared with the predictive shield`.

### Task 2: `PredictiveShield`

**Files:** Modify `pe2/shield.py`; Test `tests/pe2/test_shield.py` (append; find the existing shield tests).

- [ ] Step 1 tests: (a) H=0 returns the nominal (clipped) for random states; (b) exactness: on 1 000 random (state, candidate) the class's private `_accept(candidate, state, obstacles)` equals a brute-force re-simulation with `pursuer_step`; (c) guarantee: 50 random episodes of a random-action pursuer under `PredictiveShield(H=4)` inside `SearchMDP`/`SingleAgentWrapper` produce zero pursuer contact flags; (d) info keys == `CBFShield._info` keys on the same state; (e) `method` ∈ {exact, backup}, and u=0 is always in the candidate set (a state wedged in a pocket returns `backup` with `v_exec == 0`).
- [ ] Step 2 implement per Global Constraints (docstring: what it guarantees and why it is exact here). Step 3 pass; ruff; commit `feat(shield): PredictiveShield — exact lookahead hard filter (S11)`.

### Task 3: `scripts/derisk_s11_filter.py`

**Files:** Create runner; Test `tests/pe2/test_derisk_s11_sweep.py`.

- [ ] Step 1 tests: cell enumeration (64 + 4), verdict functions on fakes for FILT-1/2/3 (CONFIRMED / NOT), preflight record shape, skip logic, hsafe naming the cell. Step 2 implement (importlib S9; override `CELLS`, shield factory per cell, `OUT` prefix `s11`, `SEED0`; `--preflight` writes `s11_preflight.json` and refuses `--cells` without it). Step 3 pass; Step 4 scratch smoke (3 cells at N=4, workers=1, summarize) + the REAL `--preflight` (`OMP_NUM_THREADS=2`); Step 5 full suite, ruff, commit `feat(derisk): S11 second-filter sweep runner`.

## Self-review
- §1 filter definition → Task 2; exactness/guarantee → Task 1 shared dynamics + Task 2 tests + V-P; §2 cells → Task 3; §3 tests → Task 3 verdicts; §4 → direction; §5 artefacts → prefix `s11`.
