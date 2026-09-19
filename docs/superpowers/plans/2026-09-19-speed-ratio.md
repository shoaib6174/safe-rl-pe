# Speed-Ratio Sweep (S13) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An evaluation-only sweep of the stall attack across the evader speed ratio, for four frozen shielded pursuers against two learned attackers and two scripted probes, with the pre-registered survival and equivalence verdicts and a figure.

**Architecture:** `scripts/derisk_s13_speed.py` reuses `scripts/derisk_s9_sweep.py` via importlib exactly as `scripts/derisk_s11_filter.py` does (read S11 first — it is the closest precedent and solved the spawn-pool/global-override problem). Each cell builds `cfg_cell = dataclasses.replace(EnvConfig(), evader_speed_ratio=r)` and evaluates with the CBF shield at defaults. No new library code.

**Tech Stack:** as S9/S11.

**Spec:** `docs/pathway/speed_ratio_prereg_2026-09-19.md` (binding; read §2's transfer caveat, it shapes the verdicts).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Pre-registered, never tunable: `PURSUERS` = RC19 (`ckpts_RC19/RC19_10000000.zip`, shield_obs True), RCM28 (`ckpts_RCM28/RCM28_10000000.zip`, True), FT-B-41 (`ckpts_FT-B-41/FT-B-41_5000000.zip`, True), STR14 (`ckpts_STR14/STR14_10000000.zip`, **False**); `OPPONENTS` = ATT25 (`ckpts_CTL-D225/CTL-D225_10000000.zip`), ATK4-45 (`ckpts_ATK4-45/ATK4-45_10000000.zip`), HUG7 (`ObstacleHuggingEvader`), GREEDY (`GreedyEvader`); `RATIOS = (0.8, 1.0, 1.2, 1.4)`; `SEED0 = 1_360_000`; `N = 300`; `P = 0.05`; `EQ_MARGIN = 0.10`; `MAX_WORKERS = 6`; shield = `CBFShield(cfg_cell)` at defaults (δ = 0, γ = 0.2); `n_obstacles = 2`.
- Structured attackers are `pe2.evade.StructuredEvaderPolicy(model, cfg_cell, hold=10, d_pos=D_POS)` — pass the CELL config so the hug law and the action scaling use that ratio's `evader_v_max`.
- Verdicts exactly as spec §4. H-SPD-2's interval is the Wald interval on paired discordant counts, computed in the runner (no scipy).
- Never write under `results/derisk/` except `s13_cells/` and `s13_round.*` when the controller runs it; tests and smokes redirect `OUT` into scratch. No real launch from the implementer.
- 64 cells at N = 300 with the CBF filter is ≈ S9's cost (~3 min/cell, ~35 min wall at 6 workers).

---

### Task 1: `scripts/derisk_s13_speed.py`

**Files:** Create `scripts/derisk_s13_speed.py`; Test `tests/pe2/test_derisk_s13_sweep.py`.

**Interfaces:** `CELLS` (64, name `f"{pursuer}__{opponent}__r{ratio:.1f}"`), `run_cell(name, device)` (record carries `pursuer`, `opponent`, `ratio`, `deadlock_ind`, `capture_ind`, rates, Wilson, violations, clearance floors, `provenance`), `spd1_verdicts(records)`, `spd2_verdicts(records)`, `spd3_report(records)`, `hsafe(records)`, `direction(v)`, `summarize`, `write_figure`, CLI `--cells [names] | --summarize [--workers N] [--device]`.

- [ ] **Step 1: Write the failing tests.** Cover: 64 cells and their naming; `run_cell` builds the cell config with the right ratio (monkeypatch `evaluate_structured` and assert the `cfg` it receives has `evader_speed_ratio == r` and `evader_v_max == r * v_max`, and that a structured opponent factory got the same cfg); `spd1_verdicts` on fakes (attacker above both scripted probes at all four ratios → CONFIRMED; failing at one ratio → NOT; the "best scripted probe" is chosen per (pursuer, ratio) by the higher deadlock rate; overall needs ≥ 3 of 4 pursuers); `spd2_verdicts` (a fake with `b − c` small and a tight interval → CONFIRMED; a fake with a wide interval straddling ±0.10 → NOT; check the Wald formula against a hand-computed value); `spd3_report` saturation flag at capture > 0.95; `hsafe` names the cell; `direction` has a row per spec §5 including the "fails at 1.2 → harness" row; skip-existing logic.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement**, following `scripts/derisk_s11_filter.py`'s structure for the importlib load, the spawn pool and the atomic per-cell records.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Scratch smoke** (`OUT` → `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s13_smoke/`, `N=4`, workers 1, 4 cells spanning both attacker kinds and ratios 0.8 and 1.4, then `--summarize`): no exception, 0 violations, figure written, nothing under `results/derisk/`. Paste the output.
- [ ] **Step 6:** `tests/pe2/test_derisk_s13_sweep.py` + `tests/pe2/test_derisk_s9_sweep.py` + `tests/pe2/test_derisk_s11_sweep.py` green, ruff, commit `feat(derisk): S13 speed-ratio sweep runner`.

## Self-review
- §3 cells → `CELLS` + cell configs; §4 tests → the three verdict functions with the stated formulas; §5 → `direction`; §6 → prefix `s13`; §2's caveat → the scripted probe is always the same-ratio comparator (enforced in `spd1_verdicts`, asserted in a test).
