# Selection Round (S12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the S10 recipe on two independent batches of six seeds, gate all twelve on a selection stream, select one per batch by the pre-registered rule, confirm both on a fresh stream against RC19 (with FT-B-41 as a reference), and report the pooled per-seed frontier rate.

**Architecture:** `scripts/derisk_s12_select.py` loads the S8 runner via importlib (as S10 does) with prefix `s12`, `SEEDS = (48..59)`, `ARMS = {f"SEL-{A|B}-{s}": {"seed", "family": "SEL", "beta": 0.01, "batch"}}`, `GATE_SEED0 = 1_290_000` (selection), `PROBE_SEED0 = 1_320_000`, `REFERENCES = ("RC19",)` for the selection stream; adds a confirmation stage: `--confirm-start` (RC19 + FT-B-41 on `CONFIRM_SEED0 = 1_300_000`), `--select` (writes `s12_selection.json`), `--confirm` (gates the two selected checkpoints on the confirmation stream via S7's `gate_pursuer` with `GATE_SEED0` temporarily switched — implement as an explicit function that sets `S7.GATE_SEED0 = CONFIRM_SEED0` before calling and restores after), `--summarize` (H6, artefact, shrinkage, pooled rate over S8 + S10 + S12 seeds read from their round files).

**Tech Stack:** as S10.

**Spec:** `docs/pathway/selection_round_prereg_2026-09-18.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Pre-registered: batches A = (48, 49, 50, 51, 52, 53), B = (54, 55, 56, 57, 58, 59); β = 0.01; `SEL_SEED0 = 1_290_000`; `CONFIRM_SEED0 = 1_300_000`; `PROBE_SEED0 = 1_320_000`; `GATE_N = 500`; selection rule and H6 exactly as spec §2–3; references: selection stream RC19 only; confirmation stream RC19 and `FT-B-41 = results/derisk/ckpts_FT-B-41/FT-B-41_5000000.zip` (shield_obs True); everything else inherited from S8/S10 unchanged. No race stage.
- Per-arm verdict = S8's `_arm_verdict` (its `PASS_MIN = 2` held-out clause; capture guard via `gate_pass`).
- Pooled frontier rate: count `PASS` per-arm verdicts among S8's three `FT-A2-*` arms (from `s8_round.json`), S10's five `FT-B-*` arms (from `s10_round.json`) and S12's twelve arms on their selection stream; Wilson interval; the count for S8/S10 is READ from their round files, never recomputed.
- Never write under `results/derisk/` from tests/smokes; all smokes into scratch with every S8/S7/S6 path redirected. No real launch (the controller launches from a worktree).
- H-SAFE counts all twelve arms (train/probe/selection gate), RC19 on both streams, FT-B-41, and the two confirmation gates.

---

### Task 1: `scripts/derisk_s12_select.py`

**Files:** Create `scripts/derisk_s12_select.py`; Test `tests/pe2/test_derisk_s12_round.py`.

**Interfaces (module-level):** `BATCHES = {"A": (48,...,53), "B": (54,...,59)}`, `ARMS`, `SEL_SEED0`, `CONFIRM_SEED0`, `PROBE_SEED0`, `FT_B41_CKPT`; functions `select(results, starts) -> dict` (per batch: per-arm verdicts via `S8._arm_verdict`, `selected`, `selected_passed_selection: bool`, rule trace), `confirm_start(device)`, `confirm(device)` (gates each selected ckpt on `CONFIRM_SEED0`, writes `s12_confirm_<arm>.json` with the same record shape as an arm's `final`), `h6_verdicts(selection, confirms, confirm_starts) -> dict` (per selected: two-sided verdict vs RC19 on the confirmation stream via `S8._arm_verdict` with `starts = confirm_starts`; H6 per §3; artefact rule; shrinkage = selection-stream vs confirmation-stream deadlock/capture), `pooled_rate(s8_round, s10_round, selection) -> dict` (k, n, rate, Wilson), `hsafe`, `direction`, `summarize`, `write_figure` (two panels as S10; bars: RC19 (confirm), FT-B-41 (confirm), the two selected (confirm), plus faded bars for all 12 on the selection stream), CLI `--measure-start | --arms | --select | --confirm-start | --confirm | --summarize [--device]`.

- [ ] Step 1: tests (fakes): constants and arm set (12, batches disjoint); `select` picks lowest mean deadlock among passing, tie-breaks, and the no-pass fallback; `h6_verdicts` PASS / PASS (1/2) / FAIL / INCOMPLETE and artefact rule (better of the two if both pass); `pooled_rate` reads counts from fake round dicts (S8: 1 of 3, S10: 3 of 5) plus selection verdicts and returns Wilson; `hsafe` counts confirmation gates and both references; guards (arm already run under prefix `s12`; `--confirm` before `--select` → SystemExit; `--select` before all 12 arms → SystemExit naming the missing arms); direction rows. Run → fail.
- [ ] Step 2: implement (load S8 via importlib exactly as `scripts/derisk_s10_confirm.py` does; reuse `S8.run_arm`, `S8.measure_start` (with `REFERENCES=("RC19",)` and `REFERENCE_CKPT`), `S7.gate_pursuer`; for the confirmation gate switch `S7.GATE_SEED0` to `CONFIRM_SEED0` inside a try/finally). Step 3: tests pass. Step 4: scratch smoke at tiny budgets: measure-start, two arms (one per batch), select, confirm-start, confirm, summarize; no exception, 0 violations, nothing under `results/derisk/`. Step 5: full `tests/pe2` once under nohup, lint, commit `feat(derisk): S12 selection round runner`.

## Self-review
- §2 recipe/batches/streams/references → constants + reuse; selection rule + fallback → `select`; §3 H6/artefact/reports/pooled rate → `h6_verdicts`, `pooled_rate`, `summarize`; §4 → `direction`; §5 → prefix `s12` files.
