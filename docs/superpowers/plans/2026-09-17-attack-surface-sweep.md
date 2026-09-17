# Attack-Surface Sweep (S9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An evaluation-only sweep runner that measures the stall attack against three frozen shielded pursuers and four opponents across shield margin δ, physical obstacle count n and class-K gain γ on one paired stream, with the pre-registered monotonicity verdicts and a three-panel figure.

**Architecture:** `EnvConfig` gains `n_physical_obstacles` (None → `n_obstacles`), consumed only by `PursuitEvasionEnv._generate_obstacles`'s count. `scripts/derisk_s9_sweep.py` enumerates cells, runs each with `evaluate_structured(model, cfg_cell, opponent_factory, N, SEED0, hold, shield_factory=lambda: CBFShield(cfg_cell, gamma=γ, margin=δ), shield_obs=...)`, writes one atomic record per cell into `s9_cells/<cell>.json`, runs cells through a `multiprocessing` pool of ≤ 6 workers (`OMP_NUM_THREADS=2` inherited), and `--summarize` merges, tests and plots.

**Tech Stack:** as S7/S8; `scipy.stats.spearmanr` if scipy is installed in the venv (check), else a 20-line rank correlation.

**Spec:** `docs/pathway/attack_surface_sweep_prereg_2026-09-17.md` incl. amendment A1 (binding; read first).

## Global Constraints

- Always `./venv/bin/python`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; ruff clean; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Pre-registered: `PURSUERS = {"RC19": (ckpts_RC19/RC19_10000000.zip, shield_obs=True), "RCM28": (ckpts_RCM28/RCM28_10000000.zip, True), "STR14": (ckpts_STR14/STR14_10000000.zip, False)}`; `OPPONENTS = ("ATT25", "ATK2-31", "HUG7", "GREEDY")` with `ATT25 = ckpts_CTL-D225/CTL-D225_10000000.zip`, `ATK2-31 = ckpts_ATK2-31/ATK2-31_10000000.zip` (both as `StructuredEvaderPolicy`, hold 10, `D_POS`), `HUG7 = ObstacleHuggingEvader`, `GREEDY = GreedyEvader`; `DELTAS = (0.0, 0.05, 0.10, 0.20, 0.30)`, `NS = (0, 1, 2, 3, 4)` with `NS_CONFIRM = (2, 3, 4)`, `GAMMAS = (0.1, 0.2, 0.4)`; base = (δ 0, n 2, γ 0.2) shared by the three sweeps (one cell, not three); `SEED0 = 1_240_000`, `N = 300`, `P = 0.05`, `MAX_WORKERS = 6`.
- Verdicts exactly as spec §3 + A1: per pursuer, per hypothesis, against BOTH structured attackers; overall CONFIRMED iff ≥ 2 of 3 pursuers.
- H-SAFE: any pursuer contact step in any cell ⇒ `FAILS`, named cell.
- Seven S8 training processes are running on this machine: never exceed 6 workers, never write outside `results/derisk/s9_*` (tests/smokes → scratch).
- Never re-run a completed cell (`--cells` skips cells whose record exists; `--force` does not exist).

---

### Task 1: `n_physical_obstacles`

**Files:** Modify `pe2/config.py` (`EnvConfig`), `pe2/env.py` (`reset`: `self._generate_obstacles(self.cfg.n_physical_obstacles if self.cfg.n_physical_obstacles is not None else self.cfg.n_obstacles)`); Test `tests/pe2/test_env.py` (append; find the file that tests `PursuitEvasionEnv` reset — `grep -l "PursuitEvasionEnv" tests/pe2/*.py`).

**Interfaces:** `EnvConfig(n_physical_obstacles: int | None = None)`; `dataclasses.replace(cfg, n_physical_obstacles=n)` yields an env with `len(env.obstacles) == n` after reset while `obs_dim(cfg)` is unchanged; `n = 0` gives an empty list and a valid reset; the default is bit-identical to today (test: same seed → same obstacles and poses with and without the field set to 2).

- [ ] Step 1 tests: (a) default unchanged (obstacles + poses identical for seed 7 between `EnvConfig()` and `EnvConfig(n_physical_obstacles=2)`); (b) `n_physical_obstacles=4` → 4 obstacles, `obs_dim` unchanged, observation vector length unchanged; (c) `=0` → no obstacles, reset and 5 steps run, observation slots 7..10 are zero; (d) with the same seed, the first two obstacles for n=4 equal the two for n=2 (sequential generator — assert, and if it fails report; do not force it).
- [ ] Step 2 run → fail (unexpected field). Step 3 implement. Step 4 whole `tests/pe2/test_env.py` + `tests/pe2/test_obs.py` pass. Step 5 lint + commit `feat(env): n_physical_obstacles (S9 sweep)`.

---

### Task 2: `scripts/derisk_s9_sweep.py`

**Files:** Create `scripts/derisk_s9_sweep.py`; Test `tests/pe2/test_derisk_s9_sweep.py`.

**Interfaces:**
- `CELLS: dict[str, dict]` — name `f"{pursuer}__{opponent}__d{δ:.2f}_n{n}_g{γ:.1f}"` → `{"pursuer", "opponent", "delta", "n", "gamma", "sweep": set of "delta"/"n"/"gamma" tags}`; the base cell carries all three tags; 3 × 4 × 11 = 132 cells.
- `run_cell(name, device="cpu") -> dict` — builds `cfg_cell = dataclasses.replace(EnvConfig(), n_physical_obstacles=n)`, loads the pursuer model (cached per process), opponent factory (structured ones via `pe2.evade.StructuredEvaderPolicy(model, cfg_cell, hold=10, d_pos=D_POS)`; scripted via class), runs `evaluate_structured(model, cfg_cell, factory, N, SEED0, hold=HOLD, shield_factory=lambda: CBFShield(cfg_cell, gamma=γ, margin=δ), shield_obs=<pursuer's>)`, returns the record `{name, pursuer, opponent, delta, n, gamma, deadlock_ind, capture_ind, deadlock_rate, capture_rate, escape_rate (= 1 − capture − timeouts? no: escape = outcome == "timeout"|"escape" — use the env's outcome labels; report `outcomes` counts), stall_step_rate, intervention_rate, violations, min_clearance_floor, wilson_deadlock, n, seed0, provenance}`; written atomically to `OUT / "s9_cells" / f"{name}.json"`.
- `--cells [names]` (default all missing) runs through `multiprocessing.get_context("spawn").Pool(MAX_WORKERS)`; skips existing records; prints one line per completed cell.
- `verdicts(records) -> dict` — per pursuer × sweep: `rho`, `top_vs_bottom` (`paired_indicator_test(bottom_ind, top_ind, P)`), per-attacker CONFIRMED flags, per-pursuer CONFIRMED, overall; `hsafe`; the n-sweep uses `NS_CONFIRM` for the verdict and reports {0,1} under `exploratory`.
- `--summarize` → `s9_round.json` + `s9_round.png` (3 panels; x = level; y = deadlock rate with Wilson bars; solid lines = structured attackers, dashed = scripted; one colour per pursuer).
- Nested-obstacle assertion (A1): before the pool starts, for seed `SEED0`, build envs for n = 2 and n = 4 and assert the first two obstacles coincide; write the result into `s9_round.json` as `obstacles_nested`.

- [ ] Step 1 tests (fake records with `_ind` helper): cell enumeration count and base-cell sharing; `verdicts` CONFIRMED / NOT CONFIRMED for a monotone vs a flat fake; the n-verdict ignores {0,1}; hsafe FAILS names the cell; `--cells` skip logic with tmp OUT (monkeypatch `run_cell` to a stub).
- [ ] Step 2 fail → Step 3 implement → Step 4 pass.
- [ ] Step 5 scratch smoke: OUT → scratch, `N=4`, run 3 cells (one per sweep incl. an n=0 and a δ=0.3 cell) in-process (`MAX_WORKERS=1`) and `--summarize`; no exception, 0 violations. Do NOT run the real sweep — the controller launches it.
- [ ] Step 6 full `tests/pe2` suite, lint, commit `feat(derisk): S9 attack-surface sweep runner`.

## Self-review
- Spec §2 cells → `CELLS`; §3 tests → `verdicts`; A1 → `n_physical_obstacles`, `NS_CONFIRM`, exploratory block, nested assertion; §5 artefacts → paths; parallelism cap → `MAX_WORKERS`.
