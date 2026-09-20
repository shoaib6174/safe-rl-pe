# Loop Fixes (S16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Library support for two new remedies (an escape-direction observation feature and a rotation-recovery shield), then a runner that trains five arms × three seeds, evaluates all of them plus the no-training `REC` arm on one gate, and returns the three pre-registered verdicts.

**Architecture:** Task 1 adds `RecoveryShield` and the admissible-ω-range feature to `pe2/`, both behind flags that default OFF so every existing round is bit-identical. Task 2 is `scripts/derisk_s16_fixes.py`, modelled on `scripts/derisk_s5_remedy.py` (probe + train + gate per arm) and `scripts/derisk_s8_remedy3.py` (arm table, verdicts, atomic records).

**Tech Stack:** as the other derisk runners, plus `sb3_contrib.RecurrentPPO` (2.7.1, verified installed) for `MEM`.

**Spec:** `docs/pathway/loop_fixes_prereg_2026-09-20.md` (binding).

## Global Constraints

- `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Nothing may change the behaviour of any existing round.** Both new features default OFF; the full `tests/pe2` suite (1319 tests) must stay green, including the V1-P/V1-E bit-exactness tests.
- Pre-registered: arms `BASE` (γ 0.99, `stall_penalty` 1e-3), `GAM` (γ 0.999), `OBS` (+2 features), `MEM` (RecurrentPPO), `PEN` (`stall_penalty` 1e-2), each × seeds `(60, 61, 62)`; `REC` = `BASE`'s policies + recovery at eval, no training. `TOTAL = 5_000_000`, `EVERY = 500_000`, `PROBE_SEED0 = 1_730_000`, `GATE_SEED0 = 1_720_000`, `GATE_N = 500`, `EVAL_N = 200`, `RECOVERY_AFTER = 20`, `SOLVE_BAR = 0.10`, `MATCH_BAR = 0.05`, `GAM_BAR = 0.05`, `PASS_MIN = 2`; gate opponents `ATK4-45` (`ckpts_ATK4-45/ATK4-45_10000000.zip` as `StructuredEvaderPolicy`), `HUG7`, `GREEDY`; everything else the S5 RC configuration from scratch (`SearchMDP`, `CBFShield(cfg)`, `shield_obs=True`, `r_acq=0`, `hold=10`).
- H-SAFE counts contact steps in training, probes and gates for every arm including `REC`.
- Never write under `results/derisk/` from tests or smokes; redirect into `/tmp/claude-1000/-home-niro-1-Codes-safe-rl-pe/ff03fc68-2116-4512-a227-312f3099d143/scratchpad/s16_smoke/`. Do not launch the real round.

---

### Task 1: `RecoveryShield` and the escape-direction feature

**Files:** Modify `pe2/shield.py`, `pe2/structured.py`; Test `tests/pe2/test_shield.py`, `tests/pe2/test_structured.py`.

**Interfaces:**
- `pe2.shield.RecoveryShield(cfg, recovery_after=20, **cbf_kwargs)` subclassing `CBFShield`: counts consecutive steps whose info reports `immobile`; once the count reaches `recovery_after`, re-filter with nominal `(0, +ω_max)` and `(0, −ω_max)` and return whichever yields the larger `|ω_exec|` if that exceeds 0.1, else the normal result. Resets the counter on any non-immobile step. Reports the same info keys plus `recovery_fired: bool`. **`v = 0` never translates the body, so the recovery cannot create contact — assert this with a test of 50 random episodes at zero violations.**
- Escape-direction feature, flag `escape_obs: bool = False` on `SearchMDP.__init__` and `StructuredPursuerPolicy.__init__`: when on, append two features `(ω_min, ω_max) / cfg.omega_max` clipped to [−1, 1], where the bounds come from the DCBF rows evaluated at `v = 0` (`c1·ω ≥ c2` ⇒ a bound on ω) intersected with the action box; if the rows leave no feasible ω, emit `(0, 0)`. Observation width becomes `obs_dim(cfg) + 3 + 2` when both `shield_obs` and `escape_obs` are on.

- [ ] **Step 1: write the failing tests.** `RecoveryShield`: the counter fires at exactly 20 and not 19; it picks the admissible sign; it resets on motion; `recovery_fired` is reported; zero violations over 50 random episodes; with `recovery_after` never reached it is bit-identical to `CBFShield` (compare returned actions over 500 random states). Escape feature: off by default and the observation width is unchanged; on, the width grows by 2; at a state where the earlier probe showed `(0,+ω)` blocked and `(0,−ω)` free, the feature's sign reflects that; the feature is identical between the `SearchMDP` and `StructuredPursuerPolicy` code paths (this codebase computes `shield_obs` in BOTH places and they must not drift — find both and test they agree on 200 random states).
- [ ] **Step 2: run, verify failure. Step 3: implement. Step 4: targeted tests pass, then the FULL `tests/pe2` suite green (1319 tests) to prove nothing existing moved.**
- [ ] **Step 5:** ruff, commit `feat(pe2): RecoveryShield and the escape-direction observation feature (S16)`.

### Task 2: `scripts/derisk_s16_fixes.py`

**Files:** Create the runner; Test `tests/pe2/test_derisk_s16_round.py`.

**Interfaces:** `ARMS` (15 entries `f"{arm}-{seed}"` plus the `REC` evaluation variant), `run_arm`, `gate_arm`, `fix1_verdicts`, `fix2_verdicts`, `fix3_verdicts`, `hsafe`, `direction`, `summarize`, `write_figure`, CLI `--measure-start | --arms <names> | --rec | --summarize`.

- [ ] **Step 1: failing tests** — the arm table and per-arm config (γ, `stall_penalty`, `escape_obs`, algorithm) exactly per spec §2; `MEM` uses `RecurrentPPO` and everything else `PPO`; `fix1_verdicts` (SOLVES needs < 0.10 on ≥2/3 seeds AND the capture guard; `SOLVES (weaker)` when `BASE` is significantly better at capture); `fix2_verdicts` (MATCHES within 0.05 of `REC`'s mean); `fix3_verdicts` (`GAM` below `BASE` by ≥ 0.05 on ≥2/3 seeds); `hsafe` includes `REC`; `direction` rows per spec §5; guards (arm already run; `--rec` before `BASE` exists → `SystemExit`).
- [ ] **Step 2–4:** implement; tests pass.
- [ ] **Step 5: scratch smoke** at `TOTAL=4096`, `EVERY=2048`, `GATE_N=6`, `EVAL_N=4`: one `BASE` seed, one `OBS` seed, one `MEM` seed (prove RecurrentPPO runs end to end), then `--rec` and `--summarize`. Report the observation widths seen per arm.
- [ ] **Step 6:** its tests plus the full suite green, ruff, commit `feat(derisk): S16 loop-fixes runner`.

## Self-review
- §2 arms → the arm table and Task 1's two features; §3 gate → the gate function; §4 → the three verdict functions with their exact bars; §5 → `direction`; §6 → prefix `s16`. The "defaults OFF, nothing existing moves" constraint is enforced by running the full suite in both tasks.
