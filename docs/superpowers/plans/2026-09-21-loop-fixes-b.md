# Loop Fixes B (S17) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A forward-offering recovery shield, then a runner that trains one new arm, re-gates four existing checkpoint families on a fresh stream, and judges seven families on three task-based metrics with per-step nominal/executed logging.

**Architecture:** Task 1 adds `RecoveryShieldV2` to `pe2/shield.py` beside the existing `RecoveryShield` (which is NOT modified — S16's records must stay reproducible). Task 2 is `scripts/derisk_s17_fixes_b.py`, modelled closely on `scripts/derisk_s16_fixes.py` (arm table, gate, verdicts, atomic records, per-episode shield construction, RecurrentPredictor pattern not needed here since no arm is recurrent).

**Tech Stack:** as S16.

**Spec:** `docs/pathway/loop_fixes_b_prereg_2026-09-21.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **`RecoveryShield` (V1) must not change.** S16's numbers must stay reproducible; the full `tests/pe2` suite (1407) must stay green.
- Pre-registered: arms `BASE`, `OBS`, `GAM` (existing `results/derisk/ckpts_{BASE,OBS,GAM}-{60,61,62}/*_5000000.zip`), `OBSGAM` (**new**, `escape_obs=True` AND `gamma=0.999`, else the S16 BASE recipe: `SearchMDP` from scratch, `CBFShield`, `shield_obs=True`, `stall_penalty=1e-3`, `r_acq=0`, `hold=10`, PPO, 5 M steps, seeds 60/61/62, probes every 500 k vs Greedy on 1 750 000+), and `RECF-BASE`, `RECF-OBS`, `RECF-OBSGAM` (no training, V2 recovery on the corresponding policies). `GATE_SEED0 = 1_740_000`, `GATE_N = 500`, opponents `ATK4-45`/`HUG7`/`GREEDY`, `RECOVERY_AFTER = 20`, `MOVE_EPS = 0.05 * v_max`, bars `DEADLOCK_BAR = 0.05`, `SPIN_BAR = 0.05`, `P_GATE = 0.05`, `PASS_MIN = 2`.
- `OBS` and `OBSGAM` policies need `escape_obs=True` at evaluation (16-dim observation); `BASE` and `GAM` need it off (14-dim). Getting this wrong silently feeds a policy the wrong observation — pin it with a test that asserts each arm's observation width against its checkpoint's expected input size.
- **Spin fraction** = fraction of ALL gate steps with `v_exec <= 0.05 * v_max` AND `abs(omega_exec) >= 0.5 * omega_max`. Computed over every step of every gate episode, not per stall.
- **Per-step logging (D130):** each gate episode records the policy's nominal `(v_nom, w_nom)` and the executed `(v_exec, w_exec)` as separate fields, plus the recovery category fired (`forward`/`policy`/`rotation`/`none`). Aggregate counts per arm go in the record; do not store full per-step arrays for all 500 episodes (store aggregates plus the first 5 episodes' traces).
- A fresh `RecoveryShieldV2` per episode.
- Never write under `results/derisk/` from tests or smokes; redirect to `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s17_smoke/`. Do not launch the real round.

---

### Task 1: `RecoveryShieldV2`

**Files:** Modify `pe2/shield.py`; Test `tests/pe2/test_shield.py`.

**Interface:** `RecoveryShieldV2(cfg, recovery_after=20, move_eps=None, **cbf_kwargs)`. On a step whose immobile run has reached `recovery_after`, try candidates in this FIXED order — `(v_max, 0)`, `(0.5*v_max, 0)`, the policy's nominal, `(0, +omega_max)`, `(0, -omega_max)` — filtering each with `CBFShield.filter`, and execute the first whose output has `v_exec > move_eps` (default `0.05 * v_max`). If none moves, execute whichever rotation gives the larger `|omega_exec|` (V1's behaviour). Info gains `recovery_fired: bool` and `recovery_kind: "forward"|"policy"|"rotation"|"none"`. Counters `n_recoveries`, `n_recoveries_forward`, `n_recoveries_rotation`.

- [ ] **Step 1: failing tests.** The counter fires at 20 not 19; candidate ORDER is respected (construct a state where both `(v_max,0)` and `(0.5*v_max,0)` move and assert the full-speed one is chosen); a state where only rotation moves falls back to rotation with `recovery_kind="rotation"`; when not firing it is bit-identical to `CBFShield` over 500 random states (actions AND info); zero violations over 50 random episodes; `RecoveryShield` (V1) is untouched — its own tests still pass unchanged.
- [ ] **Step 2–4:** run/fail, implement, targeted tests pass, then the FULL suite green.
- [ ] **Step 5:** ruff, commit `feat(pe2): RecoveryShieldV2 — a forward-offering recovery (S17)`.

### Task 2: `scripts/derisk_s17_fixes_b.py`

**Files:** Create the runner; Test `tests/pe2/test_derisk_s17_round.py`.

**Interfaces:** `ARMS` (7 families × 3 seeds = 21 gate entries; only `OBSGAM-*` trains), `run_arm` (train, for OBSGAM only), `gate_arm`, `h1_verdicts`, `h2_verdicts`, `h3_verdicts`, `solves`, `hsafe`, `direction`, `summarize`, `write_figure`, CLI `--train <names> | --gate <names> | --summarize`.

- [ ] **Step 1: failing tests** — the arm table and each family's policy source, `escape_obs` flag and observation width; spin-fraction arithmetic on a hand-built trace (a step at `v=0.04*v_max, |w|=0.6*omega_max` counts; `v=0.06*v_max` does not; `|w|=0.4*omega_max` does not); `h1/h2/h3_verdicts` and `solves` on fakes, including an arm that passes H1 and H2 but fails H3 (the spinning remedy — this is the case the round exists to catch); `BASE` excluded from solving; `hsafe`; `direction` rows per §5; guards; skip-existing.
- [ ] **Step 2–4:** run/fail, implement, tests pass.
- [ ] **Step 5: scratch smoke** — `GATE_N=6`, train `OBSGAM-60` at `TOTAL=4096`, gate `BASE-60`, `RECF-BASE-60`, `OBSGAM-60`; report observation widths and the recovery-kind counts.
- [ ] **Step 6:** full suite green, ruff, commit `feat(derisk): S17 redesigned fixes runner`.

## Self-review
- §2 V2 semantics and per-step logging → Task 1 + the runner's record; §3 arms and reuse → the arm table; §4 the three bars and `solves` → the verdict functions; §5 → `direction`; §6 → prefix `s17`. V1 untouched is enforced by running the full suite in both tasks.
