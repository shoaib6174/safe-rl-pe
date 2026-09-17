# S10 Confirmatory Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run S8's β = 0.01 configuration on five fresh seeds with S8's per-arm test, a ≥ 3/5 round verdict, three reference rows, and the stage-2 race, reusing S8's runner through importlib.

**Architecture:** Task 1 makes S8's runner prefix-parametric (`ARTEFACT_PREFIX = "s8"` replacing every literal `s8_`/`s8race` path piece; behaviour-preserving, all S8 tests unchanged). Task 2 adds `scripts/derisk_s10_confirm.py`, which loads S8 via importlib, sets S10's constants (prefix `s10`, paths, `SEEDS = (40..44)`, `ARMS`, streams, race seeds/streams/prefix `s10race`), adds FT-A2-35 as a third reference, defines `h5_verdicts`, `direction`, `summarize`, and reuses everything else.

**Tech Stack:** as S8.

**Spec:** `docs/pathway/shield_remedy_3b_prereg_2026-09-17.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Pre-registered: `SEEDS = (40, 41, 42, 43, 44)`, arms `FT-B-{seed}` with `{"seed", "family": "FT-B", "beta": 0.01}`; `PASS_MIN = 3`; `GATE_SEED0 = 1_250_000`; `PROBE_SEED0 = 1_260_000`; `SEEDS2 = (45, 46, 47)`; `PROBE2_SEED0 = 1_270_000`; `GATE2_SEED0 = 1_280_000`; `RACE_BAR = 0.58`; `S7_RACE_RATE = 0.147`; references `RC19`, `RCM28`, `FT-A2-35` (= `results/derisk/ckpts_FT-A2-35/FT-A2-35_5000000.zip` — verify the exact file name in that directory; it is the final-mark checkpoint); everything else (`WEIGHTS`, `MIXTURE`, `TOTAL`, `EVERY`, `GATE_N`, `EVAL_N`, `P_GATE`, RC config, init/anchor RC19) inherited from S8 unchanged.
- Task 1 must not change any S8 behaviour: `tests/pe2/test_derisk_s8_round.py` passes unchanged, and the literal outputs (`s8_start.json` name etc.) are identical when `ARTEFACT_PREFIX == "s8"`.
- Never write under `results/derisk/` (S8/S9/S10 artefacts) from tests or smokes; smokes redirect every path into scratch (S8's `OUT`, `START_PATH`, `PREFLIGHT_PATH`, `RACE_START_PATH`, and — via S8's `apply_overrides`/`race_setup` — S7's and S6's). No real launch (the controller launches from a worktree).
- H-SAFE counts arms (train/probe/gate), all three references, and the race.

---

### Task 1: `ARTEFACT_PREFIX` in `scripts/derisk_s8_remedy3.py`

**Files:** Modify `scripts/derisk_s8_remedy3.py`; Test `tests/pe2/test_derisk_s8_round.py` (add one test).

- [ ] Step 1: add a test asserting `R.ARTEFACT_PREFIX == "s8"` and that, with `OUT` monkeypatched and `ARTEFACT_PREFIX` set to `"zz"`, `require_arm_not_run("FT-A1-34")` looks for `zz_arm_FT-A1-34.json` (create that file → SystemExit) and `load_results()` reads `zz_arm_*.json`. Run → fail.
- [ ] Step 2: introduce `ARTEFACT_PREFIX = "s8"` and helper(s) `_arm_file(arm)`, `_probe_file(arm)`, `_race_arm_glob()`; replace EVERY literal `"s8_arm_"`, `"s8_probe_"`, `"s8race"` (grep `s8` in the file and account for each hit; `START_PATH`/`PREFLIGHT_PATH`/`RACE_START_PATH` stay module constants but are derived from the prefix at import so S10 can still override them directly). Keep `s8_round.json`/`s8_round.png` under the prefix too (`_round_file(ext)`).
- [ ] Step 3: run `tests/pe2/test_derisk_s8_round.py` (all pass, 27+1), ruff, commit `refactor(derisk): S8 runner takes ARTEFACT_PREFIX (for S10)`.

---

### Task 2: `scripts/derisk_s10_confirm.py`

**Files:** Create `scripts/derisk_s10_confirm.py`; Test `tests/pe2/test_derisk_s10_round.py`.

**Interfaces:** loads S8 (`_s8()` via importlib), then sets on the S8 module: `ARTEFACT_PREFIX = "s10"`, `START_PATH = OUT / "s10_start.json"`, `RACE_START_PATH = OUT / "s10_race_start.json"`, `SEEDS`, `ARMS = {f"FT-B-{s}": {"seed": s, "family": "FT-B", "beta": 0.01}}`, `GATE_SEED0`, `PROBE_SEED0`, `SEEDS2`, `PROBE2_SEED0`, `GATE2_SEED0`, `REFERENCES = ("RC19", "RCM28", "FT-A2-35")` plus the checkpoint map S8 uses to load references (find how `measure_start` maps a reference name to a checkpoint and `shield_obs`; extend it with FT-A2-35, `shield_obs=True`). Confirm `make_learner` treats family `"FT-B"` as an anchored fine-tune (anything not `CONTROL`), and that S8's `apply_overrides()` pushes the NEW `GATE_SEED0`/`GATE_N`/`OUT` into S7 — call it again after setting S10's values, or set S7's globals directly. Race: S8's `race_setup` must receive S10's prefix `s10race`, streams and `SEEDS2` — check what it reads and override before calling; ATK arm names `ATK4-{45,46,47}`. Verdicts: `h5_verdicts(results, starts)` → per-arm via `S8._arm_verdict` (unchanged), `H5 = PASS iff ≥ PASS_MIN arms PASS`, `INCOMPLETE` if any arm lacks `final`, artefact rule per spec §3, counts of fail flavours, FT-A2-35's row copied from `starts` into the summary as `reference_ft_a2_35`; `hsafe` = S8's with the third reference; `direction` = spec §5 rows; `summarize`/`write_figure` (three reference bars + five arms, two panels as S8); `--preflight` not needed (S8's V-A0 already holds for this code; the guard `require_preflight` must instead check S8's `s8_preflight.json` exists and passed — or be bypassed with an explicit note in the runner docstring; choose the check).

- [ ] Step 1: tests — constants; guards (arm already run under prefix `s10`; start lacks FT-A2-35 → SystemExit); `h5_verdicts` on fakes: 3/5 pass → PASS + artefact tiebreaks; 2/5 → FAIL with counts; missing arm → INCOMPLETE; FT-A2-35 never artefact; `hsafe` counts the third reference; `direction` rows. Run → fail.
- [ ] Step 2: implement. Step 3: tests pass. Step 4: scratch smoke (paths redirected; `TOTAL=4096`, `EVERY=2048`, `GATE_N=6`, `EVAL_N=4`): measure-start (3 references), one arm, summarize, race-start + one race arm + race-summarize; no exception, 0 violations, nothing under `results/derisk/`. Step 5: lint, `tests/pe2/test_derisk_s8_round.py` + new tests, commit `feat(derisk): S10 confirmatory runner (S8 β=0.01 on fresh seeds)`.

## Self-review
- Spec §2 arms/streams/references → Task 2 constants; §3 verdicts/artefact/reports → `h5_verdicts`, `summarize`; §4 race → overrides; §5 → `direction`; §6 → prefix `s10`/`s10race`.
