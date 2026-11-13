# pelab Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new repo `/home/niro-1/Codes/pelab` in which the user can run scripted players against each other, read results with intervals, make GIFs, play by keyboard, and learn the environment from a guide, a workbook and a metrics document; and the old repo frozen.

**Architecture:** A small core library `pelab/` ported from the old repo's `pe2` core, with three additions (observation switch, safety stage inside `env.step`, Player contract with a truth snapshot). Four command-line tools work on any two registered players. Approaches live in numbered folders and register players by file.

**Tech Stack:** Python 3.12, numpy, gymnasium, matplotlib, imageio, pytest. No torch or stable-baselines3 in this milestone.

**Spec:** `/home/niro-1/Codes/safe-rl-pe/docs/superpowers/specs/2026-11-13-pelab-learning-repo-design.md` (sections 5.1–5.6, 5.7 METRICS only, 5.8, 6 steps 0–5 and 7, 8). Milestones 2 (approach 02) and 3 (FINDINGS, PAPERS) get their own plans.

## Global Constraints

- Old repo `/home/niro-1/Codes/safe-rl-pe` (called OLD below): nothing moved, nothing deleted, no history rewritten. Only additions: one commit of untracked work, a README banner, a tag. Tag is not pushed.
- New repo (NEW): own venv at `./venv/`; all commands use `./venv/bin/python`. Local git only, no remote.
- NEW never imports from OLD. Golden files are generated once from OLD and stored in `NEW/tests/golden/`.
- `EnvConfig()` with no arguments in NEW must give step-for-step the same trajectories as `pe2.config.EnvConfig()` in OLD.
- `runs/` is git-ignored. No loose scripts in the repo root.
- Every public function and class carries a docstring written for a learner: what it does, units, and why where the reason is not obvious. Decision ids from OLD (D5, D31, …) are kept only inside `docs/HISTORY.md`, not in code comments.
- Names fixed by the spec: package `pelab`; presets `open`, `obstacles`, `partial`, `partial_safe`; tools `duel`, `gif`, `play`, `compare`; documents `GUIDE.md`, `WORKBOOK.md`, `METRICS.md`, `HISTORY.md`.

## Review Focus

1. A player name that is not registered for the asked role → the tool prints the registered names for that role and exits non-zero; no traceback. (Task 6)
2. A preset or config whose obstacles cannot be placed (too many or too large for the arena) → `reset` raises an error naming the config field, instead of silently placing fewer. (Task 2)
3. `--episodes 0` or a negative count → the tool refuses with a message; statistics functions given `n = 0` return an interval of (0, 1), not a division error. (Tasks 4, 6)
4. A player returning NaN or a wrong-length action → `match.play_episode` raises naming the player; the NaN never reaches the physics. (Task 4)
5. `tools/gif.py` writing into a folder that does not exist → the folder is created. (Task 6)

---

### Task 1: Scaffold NEW and port the pure pieces

**Files (all in NEW):**
- Create: `pyproject.toml`, `.gitignore`, `pelab/__init__.py`, `pelab/config.py`, `pelab/dynamics.py`, `pelab/sensing.py`, `pelab/obs.py`, `pelab/seeding.py`
- Test: `tests/test_config.py`, `tests/test_dynamics.py`, `tests/test_sensing.py`, `tests/test_obs.py`, `tests/test_seeding.py`

**Interfaces — Produces:**
- `pelab.config.SensingConfig`, `pelab.config.EnvConfig` (frozen dataclasses; all `pe2` fields and defaults unchanged) plus two new fields: `observation: str = "partial"` (allowed: `"partial"`, `"full"`) and `safe_roles: tuple[str, ...] = ()` (subset of `("pursuer", "evader")`). `__post_init__` rejects other values with `ValueError`.
- `pelab.config.PRESETS: dict[str, EnvConfig]` and `preset(name: str) -> EnvConfig`:
  - `open`: `n_physical_obstacles=0, observation="full"`
  - `obstacles`: `observation="full"`
  - `partial`: defaults (equal to `EnvConfig()`)
  - `partial_safe`: `safe_roles=("pursuer",)`
  - all keep `n_obstacles=2`.
- `pelab.dynamics`: `wrap_angle`, `euler_step`, `robot_step` (the old `pursuer_step`, renamed because both robots use it), `robot_step_batch` — bodies copied from `pe2/env.py` and `pe2/sensing.py` without changing operation order.
- `pelab.sensing`: `in_fov`, `segment_intersects_disk`, `los_clear`, `is_visible`.
- `pelab.obs`: `obs_dim`, `observation_space`, `own_pose_from_obs`, `opponent_world_from_obs`, `build_observation(own_pose, opponent_pose, obstacles, cfg)` — dispatches on `cfg.observation`; `"full"` gives what `pe2.obs.full_state_observation` gave.
- `pelab.seeding`: `make_rng`, `derive_seed` (from `pe2/rng.py`).

- [ ] **Step 1:** Create the repo: `git init`, venv, `pyproject.toml` (package `pelab`, dependencies `numpy`, `gymnasium`, `matplotlib`, `imageio`; dev extra `pytest`), `.gitignore` (`venv/`, `runs/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`), `pip install -e ".[dev]"`.
- [ ] **Step 2:** Port tests from `OLD/tests/pe2/test_{config,sensing,obs,rng}.py` and the kinematics tests from `test_env.py`, with imports renamed. Add:

```python
def test_default_config_equals_partial_preset():
    assert preset("partial") == EnvConfig()

def test_presets_share_observation_length():
    assert {obs_dim(c) for c in PRESETS.values()} == {11}

@pytest.mark.parametrize("kw", [{"observation": "lidar"}, {"safe_roles": ("referee",)}])
def test_config_rejects_unknown_switch_values(kw):
    with pytest.raises(ValueError):
        EnvConfig(**kw)

def test_full_observation_always_shows_opponent():
    # opponent placed behind the observer, outside the field of view
    cfg = EnvConfig(observation="full")
    o = build_observation(np.array([0., 0., 0.]), np.array([-3., 0., 0.]), [], cfg)
    assert o[6] == 1.0 and o[3] == pytest.approx(-0.3)

def test_partial_observation_hides_opponent_behind():
    o = build_observation(np.array([0., 0., 0.]), np.array([-3., 0., 0.]), [], EnvConfig())
    assert o[6] == 0.0 and o[3] == 0.0
```

- [ ] **Step 3:** Run `./venv/bin/python -m pytest tests -q` → fails on missing modules.
- [ ] **Step 4:** Write the five modules.
- [ ] **Step 5:** Run the tests → all pass.
- [ ] **Step 6:** Commit `core: config with presets, dynamics, sensing, observation, seeding`.

### Task 2: The world, with the safety stage, proven equal to `pe2`

**Files:**
- Create: `pelab/safety.py`, `pelab/env.py`, `tests/golden/make_golden.md`, `tests/golden/plain.json`, `tests/golden/filtered.json`
- Test: `tests/test_env.py`, `tests/test_safety.py`, `tests/test_golden.py`

**Interfaces — Consumes:** Task 1.

**Interfaces — Produces:**
- `pelab.safety.CBFShield(cfg, gamma=0.2, vcp_d=0.1, w_v=150.0, w_omega=1.0, margin=0.0)` with `filter(action, state, obstacles, v_max=None) -> (np.ndarray, dict)` — the base class from `pe2/shield.py` lines 73–517 with its helpers; no subclasses.
- `pelab.safety.ShieldMetrics` with `record(shield_info, violation)` and `summary() -> dict`, and `summarize_shield(logs)` — from `pe2/metrics.py`, keeping only what these need.
- `pelab.env.PursuitEvasionEnv(config: EnvConfig | None = None)`; `reset(seed=...)`, `step(action)` with joint action `[v_p, w_p, v_e, w_e]`; constants `OUTCOME_NONE`, `OUTCOME_CAPTURE`, `OUTCOME_TIMEOUT`; attributes `cfg`, `obstacles`, `pursuer_pose`, `evader_pose`, `current_step`, `outcome`.
- `step` stages, in this order: clip → prep-phase rule → filter each role in `cfg.safe_roles` (speed cap `v_max` for the pursuer, `evader_v_max` for the evader) → physics.
- `info` keys: the `pe2` ones, plus `requested_action` and `applied_action` (each `{"pursuer": (v, w), "evader": (v, w)}`), plus `shield` (`{role: filter_info}` for filtered roles, `{}` otherwise).

- [ ] **Step 1:** Generate the golden files from OLD. Write a throwaway script in the session scratchpad, run with `OLD/venv/bin/python` from OLD's root. For seeds 1000–1004 on `pe2.config.EnvConfig()`: play `GreedyPursuer` vs `GreedyEvader` with `run_duel`-equivalent stepping and record, per step, both poses, both observations, and the action sent; `plain.json` holds these and the final outcome. `filtered.json`: the same with `CBFShield(cfg)` filtering the pursuer as in `run_shielded_duel`, recording the applied pursuer action too. Floats stored with `float.hex()` so equality is bitwise. `make_golden.md` records the OLD commit hash and the exact procedure in five lines.
- [ ] **Step 2:** Write `tests/test_golden.py`:

```python
def test_plain_world_matches_pe2_bitwise():
    # replay the recorded actions through pelab's env on EnvConfig();
    # every pose and every observation equals the recording exactly (==, not approx)

def test_filtered_world_matches_pe2_bitwise():
    # EnvConfig(safe_roles=("pursuer",)); feed the recorded REQUESTED actions;
    # info["applied_action"]["pursuer"] and every pose equal the recording exactly
```

- [ ] **Step 3:** Port `OLD/tests/pe2/test_env.py` (remaining tests) and, from `test_shield.py`, only tests that use `CBFShield`, `NoOpShield`-free paths, `Constraint`, `compute_vcp` and `ShieldMetrics` (drop every test naming `RecoveryShield`, `RecoveryShieldV2`, `PredictiveShield`, `escape_*`). Keep `test_hard_rows_equal_env_predicates_bitwise`. Add:

```python
def test_safety_off_reports_no_shield_info():
    # info["shield"] == {} and applied == clipped requested

def test_safe_preset_has_zero_pursuer_contacts():
    # preset("partial_safe"), a pursuer that always drives full speed straight,
    # seeds 0..19, full episodes: no pursuer wall/obstacle contact flag ever set

def test_same_pursuer_without_safety_does_touch():
    # preset("partial"), same pursuer and seeds: at least one contact (negative control)

def test_unplaceable_obstacles_raise():
    cfg = EnvConfig(n_physical_obstacles=40, obstacle_radius_min=1.0, obstacle_radius_max=1.0)
    with pytest.raises(RuntimeError, match="n_physical_obstacles"):
        PursuitEvasionEnv(cfg).reset(seed=0)
```

  The last test is a deliberate difference from `pe2`, which silently placed fewer; it cannot affect the default config (two small obstacles always fit). State this in the env docstring.
- [ ] **Step 4:** Run → fail. Write `safety.py`, `env.py`. Run → pass.
- [ ] **Step 5:** Commit `core: environment with safety stage; bitwise-equal to pe2 on defaults`.

### Task 3: Players

**Files:**
- Create: `pelab/players.py`
- Test: `tests/test_players.py`

**Interfaces — Produces:**
- `Truth` — frozen dataclass: `pursuer_pose: np.ndarray`, `evader_pose: np.ndarray`, `obstacles: tuple[dict, ...]`, `step: int`, `cfg: EnvConfig`; `Truth.from_env(env)` copies the poses.
- `Player` — base class: `__init__(self, role: str, cfg: EnvConfig)`, `sees_truth: bool = False`, `reset(self, seed: int) -> None`, `act(self, obs: np.ndarray, truth: Truth | None) -> tuple[float, float]`.
- Scripted players ported from `pe2/scripted.py`, reading `truth` instead of `env`, all `sees_truth = True` unless noted: `GreedyPursuer`, `GreedyEvader`, `StationaryEvader`, `OpenSpaceEvader`, `ObstacleHuggingEvader`; and the observation-only ones `FovLimitedPursuer`, `SearchingPursuer`, `FovLimitedEvader` (`sees_truth = False`). `BoxRoundTripEvader` is dropped (it exists only to stress the shield). Add `RandomPlayer` (uniform over the role's action box, seeded in `reset`).
- Registry: `register(role: str, name: str, factory: Callable[[str, EnvConfig], Player])`, `make_player(role, name, cfg) -> Player` (raises `KeyError` whose message lists the names registered for that role), `registered(role) -> list[str]`, `load_approaches(root: Path) -> None` (imports every `approaches/*/players.py` by file path, skipping `_template`).
- Core registrations: pursuer `greedy`, `random`, `fov_limited`, `searching`; evader `greedy`, `random`, `stationary`, `open_space`, `obstacle_hugging`, `fov_limited`.

- [ ] **Step 1:** Tests: port the behaviour tests of the carried players from `OLD/tests/pe2/test_scripted.py`; add

```python
def test_unknown_player_lists_registered_names():
    with pytest.raises(KeyError, match="greedy"):
        make_player("pursuer", "gredy", EnvConfig())

def test_truth_is_a_copy():
    # mutating env.pursuer_pose after Truth.from_env leaves truth.pursuer_pose unchanged

def test_random_player_is_repeatable_per_seed_and_inside_bounds(): ...
```

- [ ] **Step 2:** Run → fail. Implement. Run → pass. Commit `core: Player contract, truth snapshot, scripted players, registry`.

### Task 4: Matches and statistics

**Files:**
- Create: `pelab/match.py`, `pelab/stats.py`
- Test: `tests/test_match.py`, `tests/test_stats.py`

**Interfaces — Produces:**
- `match.play_episode(env, pursuer: Player, evader: Player, seed: int, on_step: Callable[[PursuitEvasionEnv, dict], None] | None = None) -> dict` — resets players with `derive_seed(seed, role)`, passes `Truth` only to players with `sees_truth`, validates each action (two finite floats, else `ValueError` naming the player's class and role). Returned log keys: `seed`, `pursuer`, `evader` (class names), `outcome`, `steps`, `final_distance`, `los_breaks`, `pursuer_violations`, `evader_violations`, `total_violations`, `shield` (`{role: ShieldMetrics.summary()}` for filtered roles).
- `match.run_battery(cfg, pursuer_factory, evader_factory, n_episodes: int, seed0: int = 1000) -> list[dict]` — seeds `seed0 + i`; `n_episodes < 1` raises `ValueError`.
- `stats`: `wilson_ci(k, n, z=1.96) -> (lo, hi)` (returns `(0.0, 1.0)` when `n == 0`), `binomial_pvalue`, `mcnemar_pvalue`, `capture_count(logs) -> (k, n)`, `summarize(logs) -> dict` (capture rate with interval, mean steps to capture, timeouts, violations per role) — from `pe2/eval.py`; `select_best_checkpoint` is not carried.

- [ ] **Step 1:** Tests: port `OLD/tests/pe2/test_eval.py` for the carried functions; add

```python
def test_golden_plain_outcomes_via_play_episode():
    # greedy vs greedy on EnvConfig(), seeds 1000..1004: outcome and steps equal tests/golden/plain.json

def test_learned_style_player_never_receives_truth():
    # a Player with sees_truth=False records its truth argument: always None

def test_nan_action_is_rejected_with_player_name(): ...

def test_same_seed_same_log():
    assert play_episode(env, p, e, 7) == play_episode(env, p, e, 7)

def test_wilson_with_zero_trials():
    assert wilson_ci(0, 0) == (0.0, 1.0)

def test_battery_rejects_zero_episodes(): ...
```

- [ ] **Step 2:** Run → fail. Implement. Run → pass. Commit `core: matches and statistics`.

### Task 5: Pictures and run records

**Files:**
- Create: `pelab/render.py`, `pelab/runs.py`
- Test: `tests/test_render.py`, `tests/test_runs.py`

**Interfaces — Produces:**
- `render.Renderer(cfg)` with `draw(env, ...) -> np.ndarray` (RGB frame), `render.Trails`, `render.FovOverlay`, `render.save_gif(frames, path, fps=20.0)` (creates parent folders), and `render.record_episode(env, pursuer, evader, seed, every: int = 2) -> list[np.ndarray]` built on `play_episode(on_step=...)`. From `pe2/render.py`; `BeliefOverlay` is dropped, `ShieldOverlay` kept only if it needs nothing beyond `info["shield"]`.
- `runs.save_run(path: Path, cfg: EnvConfig, role: str, extra: dict) -> None` writes `config.json` (config fields, role, observation mode, observation length, extra).
- `runs.load_run_config(path) -> dict`.
- `runs.check_compatible(run: dict, cfg: EnvConfig, role: str) -> None` raises `IncompatibleRun` naming the first difference among role, observation mode, observation length.
- `runs.describe_difference(run: dict, cfg: EnvConfig) -> list[str]` — the other differing fields, for printing.

- [ ] **Step 1:** Tests:

```python
def test_frame_shape_and_dtype(): ...            # (H, W, 3) uint8
def test_gif_written_into_new_folder(tmp_path): ...   # tmp_path / "a" / "b" / "x.gif" exists, > 0 bytes
def test_run_roundtrip(tmp_path): ...
def test_role_mismatch_raises(tmp_path): ...     # message contains "role"
def test_observation_mode_mismatch_raises(tmp_path): ...  # saved partial, asked full → message contains "observation"
def test_obstacle_count_difference_is_allowed_and_described(tmp_path): ...
```

- [ ] **Step 2:** Run → fail. Implement. Run → pass. Commit `core: rendering, GIFs, run records`.

### Task 6: The four commands

**Files:**
- Create: `tools/duel.py`, `tools/gif.py`, `tools/compare.py`, `tools/play.py`, `pelab/keyboard.py`
- Test: `tests/test_tools.py`, `tests/test_keyboard.py`

**Interfaces — Produces (command lines):**
- `python tools/duel.py <pursuer> <evader> [--preset partial] [--episodes 200] [--seed0 1000]` → prints one table: capture rate with 95 % interval, mean steps to capture, timeouts, contacts per role, a `*` beside players that see the truth.
- `python tools/gif.py <pursuer> <evader> [--preset] [--seed 1000] [--out runs/gifs/<p>_vs_<e>_<preset>_<seed>.gif]`.
- `python tools/compare.py [--preset] [--episodes 100]` → every registered pursuer × every registered evader; prints the table and writes `runs/leaderboard_<preset>.md`; an incompatible pairing shows `n/a`.
- `python tools/play.py --me pursuer|evader --vs <name> [--preset] [--seed]` → a matplotlib window; arrow keys: up = full speed while held, left/right = turn while held; `r` restarts with the next seed, `q` quits. Without a display it exits with a message saying a display is needed.
- `pelab.keyboard.KeyboardPlayer(role, cfg)` — a `Player`; `press(key)`, `release(key)`; `act` maps the held keys to `(speed, turn rate)`.
- All tools call `load_approaches(repo_root / "approaches")` first; unknown names, unknown presets and `--episodes < 1` exit with code 2 and a one-line message.

- [ ] **Step 1:** Tests (commands run with `subprocess`, `--episodes 3`):

```python
def test_duel_prints_capture_rate(): ...          # exit 0, "capture" in stdout
def test_duel_unknown_player_exits_2_and_lists_names(): ...
def test_duel_zero_episodes_exits_2(): ...
def test_gif_creates_file_in_missing_folder(tmp_path): ...
def test_compare_writes_leaderboard(tmp_path): ...   # run with cwd-independent --out-dir tmp_path
def test_keyboard_up_is_full_speed(): ...          # press("up") → (v_max, 0.0); evader role → evader_v_max
def test_keyboard_left_and_up(): ...               # → (v_max, +omega_max)
def test_keyboard_release_stops(): ...
def test_keyboard_player_completes_an_episode_from_scripted_keys(): ...
def test_play_without_display_exits_with_message(): ...   # env without DISPLAY, non-interactive backend
```

  `compare.py` therefore takes `--out-dir` (default `runs/`).
- [ ] **Step 2:** Run → fail. Implement. Run → pass. Commit `tools: duel, gif, compare, play`.

### Task 7: Approach 01, the template, repo front matter

**Files:**
- Create: `approaches/README.md`, `approaches/_template/{README.md,train.py,players.py}`, `approaches/01_scripted/{README.md,run.py,players.py}`, `README.md`, `CLAUDE.md`, `docs/HISTORY.md`
- Test: `tests/test_approaches.py`

**Interfaces — Produces:**
- `approaches/01_scripted/players.py` registers one new player written here as the worked example: evader `zigzag` (flees, alternating its turn every fixed number of steps), `sees_truth = True`.
- `approaches/01_scripted/run.py` runs every registered pursuer × evader on the four presets (`--episodes`, default 200) and writes `runs/01_scripted/results.md`.
- `approaches/01_scripted/README.md`: question, method, command, the result tables produced by that command, what was learned.
- `CLAUDE.md` (NEW): the role statement from spec 5.8; venv rule; "results are never committed".
- `docs/HISTORY.md`: one page: where the code came from (OLD path, tag `archive-2026-11-13`), and the lessons carried — seeded evaluation, scripted players see the truth, safety outside the physics, quote worst-case numbers — each with its OLD decision id.

- [ ] **Step 1:** Tests:

```python
def test_load_approaches_registers_zigzag(): ...
def test_template_is_not_loaded(): ...
def test_approach_01_run_small(tmp_path): ...   # --episodes 2 --out-dir tmp_path → results.md exists with four preset headings
```

- [ ] **Step 2:** Run → fail. Implement. Run → pass.
- [ ] **Step 3:** Run `./venv/bin/python approaches/01_scripted/run.py` at 200 episodes; paste the produced tables into its README. Check greedy-vs-greedy against the reviewer's numbers (no obstacles ≈ 100/100 captures; default ≈ 37/100): a large disagreement is a bug to diagnose, not a number to paste.
- [ ] **Step 4:** Commit `approaches: template and 01_scripted with measured results`.

### Task 8: GUIDE, WORKBOOK (parts 1–7), METRICS

**Files:**
- Create: `docs/GUIDE.md`, `docs/WORKBOOK.md`, `docs/METRICS.md`, `docs/figures/*.png` (made by `docs/make_figures.py`), `exercises/check_01.py` … `check_07.py`, `exercises/answers/ex_01.py` … `ex_07.py`, `exercises/README.md`
- Test: `tests/test_exercises.py`, `tests/test_docs.py`

**Content fixed by the spec:** GUIDE parts 1–7 as listed in spec 5.6, each naming file and lines, one command that shows the idea, and a "you understand this when you can…" line; it states both facts from spec section 8 row 7. WORKBOOK exercises, one per guide part:

| # | Exercise | Check |
|---|---|---|
| 1 | Run a duel and a GIF; write down three things seen | answers three questions about the printed table (which side won more, the interval width, who saw the truth) |
| 2 | Step a robot by hand for 3 steps from a given pose and action | the user's three poses equal `euler_step`'s to 1e-9 |
| 3 | Predict capture/contact/timeout for four given situations | compares with the env |
| 4 | Fill in the 11 observation numbers for a given scene, partial and full | equals `build_observation` to 1e-6 |
| 5 | Predict which of three requested actions the filter changes near a wall; then measure the capture cost of `partial_safe` vs `partial` for greedy | compares predictions with `CBFShield.filter`; checks the user's reported rates are the ones the command prints |
| 6 | Write a new scripted evader in `exercises/my_evader.py` that survives greedy on `obstacles` more often than `greedy` evader does | runs both over 100 paired seeds; passes if the user's captures are fewer |
| 7 | Given two result tables, say whether the difference is real; compute a Wilson interval by hand | compares with `wilson_ci` and `mcnemar_pvalue` |

Each `check_NN.py` reads the user's answer from `exercises/my_answers/ex_NN.py` (or `my_evader.py`), prints PASS or what is wrong, exits 0/1, and accepts `--answers exercises/answers` to run against the reference.

METRICS.md: the entries of spec 5.7. Outcome, uncertainty and safety metrics are marked **live** with a worked example whose numbers come from the named function. Stall, game and training-curve entries are marked **arrives with approach …** and give definition and the known trap only.

- [ ] **Step 1:** Tests:

```python
@pytest.mark.parametrize("n", range(1, 8))
def test_check_passes_on_reference_answer(n): ...      # exit 0
@pytest.mark.parametrize("n", range(1, 8))
def test_check_fails_on_wrong_answer(n, tmp_path): ... # a deliberately wrong answer file → exit 1
def test_every_command_in_docs_runs(): ...
    # extract fenced lines starting "python tools/" or "python approaches/" from GUIDE, WORKBOOK, METRICS, README;
    # run each with episode counts overridden small; all exit 0
def test_every_file_line_reference_in_guide_exists(): ...
    # each `pelab/x.py:LINE` named in GUIDE.md points inside the file
def test_metrics_worked_examples_match_functions(): ...
    # the numbers printed in METRICS.md's live examples equal the function outputs to the shown precision
```

- [ ] **Step 2:** Run → fail. Write `make_figures.py` and the figures (arena with both robots and a field-of-view cone; unicycle step; observation slots; filter near a wall), the three documents, checks and answers. Run → pass.
- [ ] **Step 3:** Commit `docs: guide, workbook with checks, metrics`.

### Task 9: Whole-repo verification and review

- [ ] **Step 1:** `./venv/bin/python -m pytest tests -q` → all pass; record the count.
- [ ] **Step 2:** From a fresh clone into the scratchpad with a fresh venv: `pip install -e ".[dev]"`, run the tests, run the README's five commands. This proves nothing depends on files outside git.
- [ ] **Step 3:** Open two produced GIFs (one `open`, one `partial_safe`) as frames and look: robots, obstacles, trails, field of view drawn; the filtered pursuer does not touch.
- [ ] **Step 4:** One fresh reviewer (most capable model) reads the whole NEW repo against the spec; fix what is confirmed.
- [ ] **Step 5:** Commit; tag NEW `m1`.

### Task 10: Freeze OLD

Done last, so the archive contains this session's final worklog.

- [ ] **Step 1:** In OLD: finish `docs/worklogs/2026-11-13_S136.md`; add the S136 line to the Session Index in `docs/workflow_tracker.md`; add the decision to `docs/decisions.md` (next free id after D165; check for a parallel session's use of D166 first) and to the worklog table.
- [ ] **Step 2:** Add `docs/paper_b/draft/main.bbl` and `main.blg` patterns to `.gitignore`; add the ARCHIVED banner at the top of `README.md` (date, tag name, path of NEW).
- [ ] **Step 3:** `git add` the untracked work by name (`docs/paper_a/`, `docs/paper_b/study_guide_2026-11-02.md`, worklogs S133–S136, `docs/superpowers/`), plus the modified files; commit `archive: freeze — paper work moves to pelab`, naming in the body the files that came from other sessions.
- [ ] **Step 4:** `git tag archive-2026-11-13`; `git status` clean. Do not push.
- [ ] **Step 5:** Update auto-memory: one project memory stating the new working mode and where NEW is.
