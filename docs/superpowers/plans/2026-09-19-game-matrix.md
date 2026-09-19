# Empirical Game Matrix (S14) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate all 6 defenders against all 7 attackers on one seed-paired stream, produce the two payoff matrices, the three pre-registered verdicts, the zero-sum meta-game solved by LP, a figure, and one GIF per cell on a single unselected seed.

**Architecture:** `scripts/derisk_s14_matrix.py` reuses `scripts/derisk_s9_sweep.py` via importlib exactly as `scripts/derisk_s11_filter.py` and `scripts/derisk_s13_speed.py` do (read S13 first — it is the most recent and cleanest precedent). No new library code. The GIF pass reuses the recorder classes from the ad-hoc script at `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s12_gif.py` (read it: `RecE` subclasses `EvadeMDP` for a learned/structured evader, `RecS` subclasses `SearchMDP` for a scripted evader); move those into the runner, do not import from scratch.

**Tech Stack:** as S13, plus `scipy.optimize.linprog` (scipy 1.15.1 is installed) for the meta-game.

**Spec:** `docs/pathway/game_matrix_prereg_2026-09-19.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Pre-registered, never tunable. **Defenders** (id → ckpt, shield_obs): `TEACHER` → scripted `pe2.scripted.SearchingPursuer` (no ckpt; hold = 1, see how `scripts/derisk_s4_shield.py` runs the TEACHER cells), `STR14` → `ckpts_STR14/STR14_10000000.zip` **False**, `RC19` → `ckpts_RC19/RC19_10000000.zip` True, `RCM28` → `ckpts_RCM28/RCM28_10000000.zip` True, `FT-B-41` → `ckpts_FT-B-41/FT-B-41_5000000.zip` True, `SEL-A-50` → `ckpts_SEL-A-50/SEL-A-50_5000000.zip` True (verify the filename). **Attackers**: `CTL-D125` → `ckpts_CTL-D125/CTL-D125_10000000.zip`, `CTL-D225` → `ckpts_CTL-D225/CTL-D225_10000000.zip`, `ATK2-31` → `ckpts_ATK2-31/ATK2-31_10000000.zip`, `ATK4-45` → `ckpts_ATK4-45/ATK4-45_10000000.zip` (all four as `pe2.evade.StructuredEvaderPolicy`, hold 10, `D_POS`), `SPLITHUG` → `pe2.evade.SplitHugEvader` wrapped as the mixture does, `HUG7` → `ObstacleHuggingEvader`, `GREEDY` → `GreedyEvader`. `TRAINED_AGAINST = {"CTL-D125": "STR14", "CTL-D225": "RC19", "ATK2-31": "RCM28", "ATK4-45": "FT-B-41"}`; `TRANSFER_ONLY_DEFENDERS = ("TEACHER", "SEL-A-50")`. `SEED0 = 1_370_000`, `N = 300`, `P = 0.05`, `MAX_WORKERS = 6`, shield `CBFShield(cfg)` at defaults, `EnvConfig()` defaults (ratio 1.2, 2 obstacles).
- Verdicts exactly as spec §3. The meta-game LP: row player (defender) minimises, column maximises, on the 6×7 deadlock-rate matrix; solve the standard zero-sum LP for the value and both maximin mixed strategies; repeat on the capture matrix with the roles that make sense (defender maximises capture). Report support sets at a tolerance of 1e-6.
- GIFs per spec §4: every cell on seed `SEED0` exactly, no search, title carries the cell's measured rates.
- Never write under `results/derisk/` from tests or smokes (redirect `OUT` to `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s14_smoke/`). Do not launch the real round.
- H-SAFE counts every cell.

---

### Task 1: matrix runner (cells, verdicts, meta-game, figure)

**Files:** Create `scripts/derisk_s14_matrix.py`; Test `tests/pe2/test_derisk_s14_matrix.py`.

**Interfaces:** `CELLS` (42, name `f"{defender}__{attacker}"`), `run_cell(name, device)`, `deadlock_matrix(records)` / `capture_matrix(records)` → (rows, cols, 2-D lists), `mat1_verdicts`, `mat2_verdicts`, `mat3_report`, `meta_game(matrix, rows, cols, minimise_rows=True)` → `{value, row_strategy, col_strategy, row_support, col_support}`, `hsafe`, `direction`, `summarize`, `write_figure` (a heatmap of each payoff matrix with the meta-game support marked), CLI `--cells [names] | --summarize [--workers N] [--device]`.

- [ ] **Step 1: failing tests.** 42 cells and naming; the TEACHER row uses the scripted pursuer with hold = 1 and no `shield_obs`; `run_cell` passes the right `shield_obs` per defender and the right opponent object per attacker (monkeypatch `evaluate_structured` and assert); `mat1_verdicts` on fakes (own-defender argmax → SPECIALISED; another defender higher → GENERAL with the paired p; ≥3/4 → CONFIRMED); `mat2_verdicts` on fakes (each `<` tested against the attacker attaining the larger side's max; one inversion → NOT CONFIRMED, naming the pair); `mat3_report` (a dominating defender is reported as such; otherwise the Pareto set); `meta_game` on a hand-solvable 2×2 zero-sum matrix with a known mixed equilibrium (verify value and both strategies to 1e-6) and on a matrix with a saddle point (pure strategies); `hsafe` names the cell; `direction` rows per spec §5; skip-existing.
- [ ] **Step 2: run, verify they fail. Step 3: implement. Step 4: tests pass.**
- [ ] **Step 5: scratch smoke** (`N=4`, workers 1, 4 cells covering a learned attacker, a scripted attacker, the TEACHER row and the SEL-A-50 row, then `--summarize`): no exception, 0 violations, both matrices and the meta-game computed, figure written, nothing under `results/derisk/`.
- [ ] **Step 6:** its tests plus `test_derisk_s9_sweep.py`, `test_derisk_s11_sweep.py`, `test_derisk_s13_sweep.py` green; ruff; commit `feat(derisk): S14 empirical game matrix runner`.

### Task 2: the GIF pass

**Files:** Modify `scripts/derisk_s14_matrix.py` (add `--gifs`); Test: extend `tests/pe2/test_derisk_s14_matrix.py`.

**Interfaces:** `--gifs` renders one GIF per cell into `OUT / "s14_gifs" / f"{defender}__{attacker}.gif"`, each on seed `SEED0` exactly. Reads `s14_cells/*.json` for the cell's deadlock and capture rates and puts them in the title. Skips a GIF that already exists. Refuses to run if the cell record is missing (so the title cannot be fabricated).

- [ ] **Step 1: failing tests** — the GIF path/name per cell; the title string contains both rates; a missing cell record raises `SystemExit`; an existing file is skipped; the seed used is exactly `SEED0` (assert by monkeypatching the recorder).
- [ ] **Step 2–4:** implement using the two recorder classes moved in from the scratch script (a learned/structured attacker takes the `EvadeMDP` route, a scripted one takes the `SearchMDP` route; the TEACHER defender needs the scripted-pursuer path in both routes — check `derisk_s4_shield.py` and `pe2.structured.StructuredPursuerPolicy` for how a scripted pursuer is driven, and if the `EvadeMDP` route cannot take a scripted defender, say so in the report rather than faking it).
- [ ] **Step 5: smoke** — render 2 GIFs at `every=8` into scratch, assert they open with PIL and have > 50 frames.
- [ ] **Step 6:** tests green, ruff, commit `feat(derisk): S14 per-cell GIFs`.

## Self-review
- §2 matrix → `CELLS`, the defender/attacker tables, the transfer-only labelling; §3 → the three verdict functions + `meta_game`; §4 → Task 2; §5 → `direction`; §6 → prefix `s14`.
