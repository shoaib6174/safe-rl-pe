"""S11 — is the stall attack a property of hard shields, or of the CBF-QP?
(spec: docs/pathway/second_filter_prereg_2026-09-18.md + amendment S11-A1).

No training. The same frozen shielded pursuers S9 measured are re-evaluated
against the same two structured attackers and two scripted baselines under a
SECOND hard filter built a different way — `pe2.shield.PredictiveShield`, a
sampling lookahead through the environment's own `pursuer_step` — while its own
conservativeness knob, the lookahead horizon H, moves:

  H (lookahead steps)  {1, 4, 16, 32}   delta 0, n 2   (A1: 0.05-1.6 s)

4 pursuers (RC19, RCM28, STR14 + FT-B-41, reported) x 4 opponents x 4 horizons
= 64 predictive cells, plus the 4 CBF base cells (delta 0, n 2, gamma 0.2) that
complete S9's base row for FT-B-41 — 68 cells, every one on the same 300-seed
stream from SEED0.

This module is S9's machinery under S11's constants: it loads
`scripts/derisk_s9_sweep.py` privately (importlib, NOT sys.modules) and reuses
its record IO, its H-SAFE rule and its statistics, re-pointing only the two
globals those helpers read (`CELLS`, `cells_dir`). S9's own script and its
artefacts on disk are untouched.

What is NOT reused: `run_cell` (the shield differs per cell) and the pool loop
(its gate is S11's V-P preflight, not S9's nested-obstacle check, and its worker
must live in THIS module so `spawn` re-imports S11's world, not S9's).

Usage:
  ./venv/bin/python scripts/derisk_s11_filter.py --preflight            (V-P)
  ./venv/bin/python scripts/derisk_s11_filter.py --cells                (missing)
  ./venv/bin/python scripts/derisk_s11_filter.py --cells RC19__ATT25__pred_H4_d0.00_n2
  ./venv/bin/python scripts/derisk_s11_filter.py --summarize

Stop rule (spec section 4): a cell that already has a record is NEVER re-run;
no level is added or removed after any number is seen. `--cells` refuses to run
at all until `s11_preflight.json` exists and passes.

`OUT` is the only output root and `--workers 1` is the only mode in which
reassigning it (a smoke) reaches the code that writes: spawned workers re-import
this module and get the module-level default.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import multiprocessing
import os
import pathlib
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.eval import win_count
from pe2.metrics import summarize_shield
from pe2.structured import HOLD


def _s9():
    p = pathlib.Path(__file__).resolve().parent / "derisk_s9_sweep.py"
    spec = importlib.util.spec_from_file_location("derisk_s9_sweep", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S9 = _s9()

CKPT_ROOT = pathlib.Path("results/derisk")  # read-only inputs
OUT = pathlib.Path("results/derisk")
ARTEFACT_PREFIX = "s11"

# --- pre-registered (spec sections 2-3, amendment A1). Do not tune. ---
SEED0 = 1_330_000
N_EPISODES = 300
PREFLIGHT_SEED0 = 1_329_000  # below the cell stream: V-P plays no cell episode (its (i) reset seeds overlap the stream numerically but nothing is read out)
P_GATE = S9.P_GATE  # 0.05
MAX_WORKERS = 6
OMP_THREADS = S9.OMP_THREADS  # "2"
HORIZONS = (1, 4, 16, 32)  # A1 (was 1, 2, 4, 8)
H_FILT3 = 4  # the level H-FILT-3 reads, unchanged by A1
DELTA = 0.0
N_OBSTACLES = 2
GAMMA = 0.2
PREDICTIVE = "predictive"
CBF = "cbf"
# V-P sizes (spec section 3). Not knobs: `main` passes none of them.
PREFLIGHT_STATES = 10_000
PREFLIGHT_EPISODES = 200
PREFLIGHT_PAIRS = 1_000
PREFLIGHT_HORIZON = 4  # the horizon V-P (ii) runs its episodes at

PURSUERS = {
    **S9.PURSUERS,  # RC19, RCM28, STR14 — same checkpoints, same shield_obs
    "FT-B-41": {
        "ckpt": CKPT_ROOT / "ckpts_FT-B-41" / "FT-B-41_5000000.zip",
        "shield_obs": True,  # S10 artefact, RC configuration (D120)
    },
}
SCORING = ("RC19", "RCM28", "STR14")  # the 3 pursuers a verdict is counted over
REPORT_ONLY = "FT-B-41"  # reported everywhere, votes nowhere
CBF_BASE_PURSUERS = ("FT-B-41",)  # whose base row S9 is missing
OPPONENTS = S9.OPPONENTS
STRUCTURED = S9.STRUCTURED  # ATT25, ATK2-31
SCRIPTED = S9.SCRIPTED  # HUG7, GREEDY


def cell_name(
    pursuer: str, opponent: str, filt: str, horizon: int | None = None
) -> str:
    if filt == PREDICTIVE:
        return f"{pursuer}__{opponent}__pred_H{horizon}_d{DELTA:.2f}_n{N_OBSTACLES}"
    return f"{pursuer}__{opponent}__cbf_d{DELTA:.2f}_n{N_OBSTACLES}_g{GAMMA:.1f}"


def _build_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}

    def add(p: str, o: str, filt: str, horizon: int | None) -> None:
        name = cell_name(p, o, filt, horizon)
        cells[name] = {
            "name": name,
            "pursuer": p,
            "opponent": o,
            "filter": filt,
            "horizon": horizon,
            "delta": DELTA,
            "n": N_OBSTACLES,
            # The predictive filter inherits CBFShield's gamma: its barrier rows
            # LOCATE the candidate set (and feed shield_obs), the rollout alone
            # decides admissibility. Recorded so both filters state it.
            "gamma": GAMMA,
        }

    for p in PURSUERS:
        for o in OPPONENTS:
            for h in HORIZONS:
                add(p, o, PREDICTIVE, h)
    for p in CBF_BASE_PURSUERS:
        for o in OPPONENTS:
            add(p, o, CBF, None)
    return cells


CELLS = _build_cells()


def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


def preflight_path() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_preflight.json"


# S9's record IO and H-SAFE rule read exactly these two globals; re-pointing
# them makes every borrowed helper operate on S11's cells and S11's OUT (which
# `cells_dir` reads at call time, so a test that moves OUT still works).
S9.CELLS = CELLS
S9.cells_dir = cells_dir

_ci = S9._ci
_write_json = S9._write_json
provenance = S9.provenance
spearman = S9.spearman
write_cell = S9.write_cell
load_records = S9.load_records
hsafe = S9.hsafe
_paired_top_vs_bottom = S9._paired_top_vs_bottom


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def shield_factory(spec: dict, cfg_cell: EnvConfig):
    """The cell's filter, as a zero-argument factory (`evaluate_structured`
    builds ONE shield per call). Predictive cells carry a horizon; CBF base
    cells carry the class-K gain."""
    from pe2.shield import CBFShield, PredictiveShield

    if spec["filter"] == PREDICTIVE:

        def build_predictive():
            return PredictiveShield(
                cfg_cell, horizon=spec["horizon"], margin=spec["delta"]
            )

        return build_predictive

    def build_cbf():
        return CBFShield(cfg_cell, gamma=spec["gamma"], margin=spec["delta"])

    return build_cbf


def run_cell(name: str, device: str = "cpu") -> dict:
    """One cell: the frozen pursuer under this cell's filter over the
    pre-registered stream. Same readouts as S9 plus the predictive filter's
    own backup-tier rate (A1: identically 0 at delta = 0, reported as such)."""
    from pe2.structured import evaluate_structured

    spec = CELLS[name]
    cfg_cell = dataclasses.replace(EnvConfig(), n_physical_obstacles=spec["n"])
    pursuer = PURSUERS[spec["pursuer"]]
    model = S9._model(pursuer["ckpt"], device)
    logs = evaluate_structured(
        model,
        cfg_cell,
        S9.opponent_factory(spec["opponent"], cfg_cell, device),
        N_EPISODES,
        SEED0,
        hold=HOLD,
        shield_factory=shield_factory(spec, cfg_cell),
        shield_obs=pursuer["shield_obs"],
    )
    sh = summarize_shield(logs)
    k, n_ep = win_count(logs, "pursuer")
    deadlock_ind = [bool(lg["shield"]["deadlock"]) for lg in logs]
    outcomes = Counter(lg["outcome"] for lg in logs)
    return {
        "name": name,
        "pursuer": spec["pursuer"],
        "opponent": spec["opponent"],
        "filter": spec["filter"],
        "horizon": spec["horizon"],
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "shield_obs": pursuer["shield_obs"],
        "n_episodes": n_ep,
        "seed0": SEED0,
        "deadlock_ind": deadlock_ind,
        "capture_ind": [lg["outcome"] == "capture" for lg in logs],
        "deadlock_rate": sh["deadlock_episode_rate"],
        "capture_rate": k / n_ep,
        # The env terminates on capture and truncates on timeout and nothing
        # else (pe2/env.py OUTCOME_*), so the evader "escapes" exactly when the
        # episode times out.
        "escape_rate": outcomes["timeout"] / n_ep,
        "outcomes": dict(outcomes),
        "stall_step_rate": sh["stall_step_rate"],
        "intervention_rate": sh["intervention_rate"],
        "backup_rate": sh["backup_rate"],
        "violations": int(sh["violation_steps"]),
        "min_clearance_floor": sh["min_clearance_floor"],
        "min_clearance_hard_floor": sh["min_clearance_hard_floor"],
        "wilson_deadlock": _ci(sum(deadlock_ind), n_ep),
        "provenance": provenance(),
    }


def _cell_line(rec: dict) -> str:
    return (
        f"[{rec['name']}] deadlock {rec['deadlock_rate']:.3f} "
        f"capture {rec['capture_rate']:.3f} escape {rec['escape_rate']:.3f} "
        f"stall {rec['stall_step_rate']:.3f} int {rec['intervention_rate']:.3f} "
        f"backup {rec['backup_rate']:.3f} viol {rec['violations']}"
    )


def _run_and_write(name: str, device: str) -> dict:
    rec = run_cell(name, device)
    write_cell(rec)
    return rec


def _worker(args: tuple) -> str:
    name, device = args
    return _cell_line(_run_and_write(name, device))


def run_cells(
    names: list[str] | None = None, device: str = "cpu", workers: int = MAX_WORKERS
) -> list[str]:
    """Run every named cell that has no record yet (default: all 68).

    The gate is V-P: no cell of this round may be measured before the preflight
    has passed, because the predictive filter's guarantee is exact and H-SAFE
    therefore tests the implementation (spec section 4 — a failure is not
    fixable inside the round)."""
    wanted = list(CELLS) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}")
    require_preflight()
    cells_dir().mkdir(parents=True, exist_ok=True)
    todo = [n for n in wanted if not (cells_dir() / f"{n}.json").exists()]
    print(
        f"{len(todo)}/{len(wanted)} cells to run "
        f"({len(wanted) - len(todo)} already have a record; stop rule: never re-run)",
        flush=True,
    )
    if workers <= 1:
        for name in todo:
            print(_cell_line(_run_and_write(name, device)), flush=True)
        return todo
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(workers, len(todo)) or 1) as pool:
        for line in pool.imap_unordered(_worker, [(n, device) for n in todo]):
            print(line, flush=True)
    return todo


# ----------------------------------------------------------------------------
# V-P preflight (spec section 3)
# ----------------------------------------------------------------------------
def _random_states(n: int, seed: int, cfg: EnvConfig | None = None) -> list[tuple]:
    """n (pose, obstacles) pairs straight from the env's own reset — so every
    pose is a state the env can actually be in (contact-free by construction,
    `_sample_free_pose`) rather than one this script invented."""
    from pe2.env import PursuitEvasionEnv

    cfg = cfg or EnvConfig()
    env = PursuitEvasionEnv(cfg)
    states = []
    for i in range(n):
        env.reset(seed=seed + i)
        states.append((env.pursuer_pose.copy(), [dict(o) for o in env.obstacles]))
    return states


def _executed_nominal(action, cfg: EnvConfig, v_max: float) -> tuple[float, float]:
    """The command the env would execute for this nominal with NO safety
    modification: the action-box clip (`CBFShield._clip_nominal`), then the
    float32 round-trip the rollout is driven through.

    Transcribed HERE from the two env-side functions —
    `pe2.structured._normalize` (into a float32 Box action) and
    `pe2.exploitability.SingleAgentWrapper._scale_action` (clip and
    de-normalise) — and deliberately NOT by calling `PredictiveShield._snap`,
    so V-P (i) compares the filter against an independent reference rather than
    against itself. The perturbation is <= 1.5e-8 m/s, reported below."""
    v = float(np.clip(float(action[0]), 0.0, v_max))
    w = float(np.clip(float(action[1]), -cfg.omega_max, cfg.omega_max))
    a = np.array([2.0 * v / v_max - 1.0, w / cfg.omega_max], dtype=np.float32)
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    return float((a[0] + 1.0) / 2.0 * v_max), float(a[1] * cfg.omega_max)


# The grid round-trip may move a command by at most one float32 ulp of the
# normalised action; anything larger is a safety modification, not a snap.
IDENTITY_TOL = 1e-6


def preflight_identity(
    n_states: int = PREFLIGHT_STATES, seed: int = PREFLIGHT_SEED0
) -> dict:
    """V-P (i): at H = 0 the predictive filter degenerates to a clip — on every
    state it must return the nominal with no safety modification, i.e. exactly
    the executed-grid nominal above, and within IDENTITY_TOL of the raw
    `CBFShield._clip_nominal` box clip."""
    from pe2.shield import PredictiveShield

    cfg = EnvConfig()
    shield = PredictiveShield(cfg, horizon=0, margin=0.0)
    rng = np.random.default_rng(seed)
    mismatches = []
    worst_deviation = 0.0
    for pose, obstacles in _random_states(n_states, seed, cfg):
        action = (
            float(rng.uniform(-0.5, cfg.v_max + 0.5)),
            float(rng.uniform(-cfg.omega_max - 1.0, cfg.omega_max + 1.0)),
        )
        u, info = shield.filter(action, pose, obstacles)
        expect = _executed_nominal(action, cfg, cfg.v_max)
        clipped = shield._clip_nominal(action, cfg.v_max)
        deviation = max(abs(float(u[i]) - clipped[i]) for i in (0, 1))
        worst_deviation = max(worst_deviation, deviation)
        if (
            (float(u[0]), float(u[1])) != expect
            or info["method"] != "exact"
            or deviation > IDENTITY_TOL
        ):
            mismatches.append(
                {
                    "pose": [float(x) for x in pose],
                    "action": list(action),
                    "u": [float(x) for x in u],
                    "expect": list(expect),
                    "clipped": list(clipped),
                    "deviation": deviation,
                    "method": info["method"],
                }
            )
    return {
        "check": "identity_at_horizon_zero",
        "pass": not mismatches,
        "horizon": 0,
        "n_states": n_states,
        "seed": seed,
        "mismatches": len(mismatches),
        "examples": mismatches[:5],
        "max_deviation_from_clip": worst_deviation,
        "tolerance": IDENTITY_TOL,
        "reference": (
            "CBFShield._clip_nominal of the same nominal, through the env's own "
            "float32 action-grid round-trip (pe2.structured._normalize + "
            "SingleAgentWrapper._scale_action), transcribed independently"
        ),
    }


def preflight_guarantee(
    n_episodes: int = PREFLIGHT_EPISODES, device: str = "cpu"
) -> dict:
    """V-P (ii): the guarantee itself — RC19 vs ATT25 under
    `PredictiveShield(H = 4)`, on a stream disjoint from the cells, must raise
    the env's contact flag zero times."""
    from pe2.shield import PredictiveShield
    from pe2.structured import evaluate_structured

    cfg = EnvConfig()
    model = S9._model(PURSUERS["RC19"]["ckpt"], device)

    def build():
        return PredictiveShield(cfg, horizon=PREFLIGHT_HORIZON, margin=0.0)

    logs = evaluate_structured(
        model,
        cfg,
        S9.opponent_factory("ATT25", cfg, device),
        n_episodes,
        PREFLIGHT_SEED0,
        hold=HOLD,
        shield_factory=build,
        shield_obs=PURSUERS["RC19"]["shield_obs"],
    )
    sh = summarize_shield(logs)
    return {
        "check": "zero_contact_under_the_predictive_filter",
        "pass": int(sh["violation_steps"]) == 0,
        "horizon": PREFLIGHT_HORIZON,
        "pursuer": "RC19",
        "opponent": "ATT25",
        "n_episodes": n_episodes,
        "seed0": PREFLIGHT_SEED0,
        "n_steps": sh["n_steps"],
        "violation_steps": int(sh["violation_steps"]),
        "episodes_with_violation": int(sh["episodes_with_violation"]),
        "min_clearance_hard_floor": sh["min_clearance_hard_floor"],
        "backup_rate": sh["backup_rate"],
        "deadlock_episode_rate": sh["deadlock_episode_rate"],
    }


def _bruteforce_accept(
    pose, u, horizon: int, cfg: EnvConfig, obstacles: list[dict], x_lim, y_lim
) -> bool:
    """The acceptance predicate re-derived from scratch: hold `u` for `horizon`
    steps then stop, through `pe2.env.pursuer_step` on the ENV's own bounds,
    rejecting on either contact flag or a body clearance below delta = 0.

    Deliberately shares no code with `PredictiveShield._rollout` /
    `_body_clearance` — that is the whole point of V-P (iii)."""
    from pe2.env import pursuer_step

    p = np.array(pose, dtype=float)
    for k in range(horizon + 1):
        v, w = (float(u[0]), float(u[1])) if k < horizon else (0.0, 0.0)
        p, wall, obstacle = pursuer_step(p, v, w, cfg, obstacles, x_lim, y_lim)
        if wall or obstacle:
            return False
        clearance = min(
            x_lim[1] - p[0], p[0] - x_lim[0], y_lim[1] - p[1], p[1] - y_lim[0]
        )
        for o in obstacles:
            clearance = min(
                clearance,
                float(np.hypot(p[0] - o["x"], p[1] - o["y"]))
                - (o["radius"] + cfg.agent_radius),
            )
        if clearance < DELTA:
            return False
    return True


def _aimed_at_surface(pose, obstacles: list[dict], cfg: EnvConfig, x_lim, y_lim, rng):
    """A contact-free pose just outside an obstacle, heading INTO it.

    V-P (iii) has to pin both answers of the accept predicate, and a uniform
    (state, candidate) draw almost never produces a rejection: the longest
    horizon moves a straight candidate 1.6 m inside a 9.7 m arena, and a
    candidate turning near omega_max sweeps 4.5 rad and stays inside a ~0.35 m
    circle. So half the pairs are placed here, a few centimetres off an
    obstacle and pointed at it. The pose is still an ordinary state of this env
    — in bounds and touching nothing — it is just not a uniform one. Returns
    None when the placement would overlap something (the caller then keeps the
    env's own reset pose).
    """
    if not obstacles:
        return None
    obs = obstacles[int(rng.integers(len(obstacles)))]
    phi = float(rng.uniform(-np.pi, np.pi))
    gap = float(rng.uniform(0.01, 0.25))
    r = obs["radius"] + cfg.agent_radius + gap
    x, y = obs["x"] + r * np.cos(phi), obs["y"] + r * np.sin(phi)
    if not (x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1]):
        return None
    for other in obstacles:
        if (
            np.hypot(x - other["x"], y - other["y"])
            < other["radius"] + cfg.agent_radius
        ):
            return None
    heading = phi + np.pi + float(rng.normal(0.0, 0.2))
    return np.array([x, y, heading])


def preflight_bruteforce(
    n_pairs: int = PREFLIGHT_PAIRS, seed: int = PREFLIGHT_SEED0
) -> dict:
    """V-P (iii): the filter's accept decision on random (state, candidate)
    pairs must equal an independent re-simulation built on a fresh
    `PursuitEvasionEnv`'s own bounds. Every pre-registered horizon is
    exercised (the pairs cycle through `HORIZONS`), and half the pairs are
    aimed at an obstacle so both answers are pinned (`n_accepted` reports the
    balance actually achieved)."""
    from pe2.env import PursuitEvasionEnv
    from pe2.shield import PredictiveShield

    cfg = EnvConfig()
    env = PursuitEvasionEnv(cfg)  # its _x_min/_x_max are the reference bounds
    x_lim = (env._x_min, env._x_max)
    y_lim = (env._y_min, env._y_max)
    shields = {h: PredictiveShield(cfg, horizon=h, margin=0.0) for h in HORIZONS}
    rng = np.random.default_rng(seed + 1)
    disagreements = []
    n_accepted = 0
    n_aimed = 0
    for i, (pose, obstacles) in enumerate(_random_states(n_pairs, seed, cfg)):
        horizon = HORIZONS[i % len(HORIZONS)]
        aimed = (
            _aimed_at_surface(pose, obstacles, cfg, x_lim, y_lim, rng)
            if i % 2
            else None
        )
        if aimed is not None:
            pose, n_aimed = aimed, n_aimed + 1
            u = (float(rng.uniform(0.5 * cfg.v_max, cfg.v_max)), 0.0)
        else:
            u = (
                float(rng.uniform(0.0, cfg.v_max)),
                float(rng.uniform(-cfg.omega_max, cfg.omega_max)),
            )
        got = bool(shields[horizon]._accept(u, pose, obstacles))
        want = _bruteforce_accept(pose, u, horizon, cfg, obstacles, x_lim, y_lim)
        n_accepted += int(want)
        if got != want:
            disagreements.append(
                {
                    "pose": [float(x) for x in pose],
                    "u": list(u),
                    "horizon": horizon,
                    "filter": got,
                    "brute_force": want,
                }
            )
    return {
        "check": "accept_equals_brute_force_resimulation",
        "pass": not disagreements,
        "n_pairs": n_pairs,
        "seed": seed,
        "horizons": sorted(HORIZONS),
        "n_accepted": n_accepted,
        "n_aimed_at_surface": n_aimed,
        "disagreements": len(disagreements),
        "examples": disagreements[:5],
        "bounds": {"x": list(x_lim), "y": list(y_lim)},
    }


def run_preflight(
    n_states: int = PREFLIGHT_STATES,
    n_episodes: int = PREFLIGHT_EPISODES,
    n_pairs: int = PREFLIGHT_PAIRS,
    device: str = "cpu",
) -> dict:
    """Run V-P and write `s11_preflight.json`. All three checks must pass
    before a single cell of this round may be measured."""
    checks = {
        "identity": preflight_identity(n_states=n_states),
        "guarantee": preflight_guarantee(n_episodes=n_episodes, device=device),
        "bruteforce": preflight_bruteforce(n_pairs=n_pairs),
    }
    blob = {
        "pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "seed0": PREFLIGHT_SEED0,
        "horizons": list(HORIZONS),
        "delta": DELTA,
        "provenance": provenance(),
    }
    _write_json(preflight_path(), blob)
    return blob


def require_preflight() -> dict:
    path = preflight_path()
    if not path.exists():
        raise SystemExit(
            f"no preflight at {path}: run --preflight before any cell (V-P, "
            "spec section 3). The predictive filter's guarantee is exact, so a "
            "failed cell cannot be fixed inside the round."
        )
    blob = json.loads(path.read_text())
    if not blob.get("pass"):
        raise SystemExit(
            f"the preflight at {path} did NOT pass: "
            f"{ {k: v.get('pass') for k, v in blob.get('checks', {}).items()} }. "
            "No cell of this round may run."
        )
    return blob


# ----------------------------------------------------------------------------
# tests (spec section 3)
# ----------------------------------------------------------------------------
def _hkey(h: int) -> str:
    return f"H{h}"


def _overall(per_pursuer: dict) -> tuple[str, int]:
    """The shared rule: INCOMPLETE if any SCORING pursuer is, else CONFIRMED
    iff at least 2 of the 3 confirm. `REPORT_ONLY` never counts."""
    scored = [per_pursuer[p]["verdict"] for p in SCORING]
    n_conf = sum(v == "CONFIRMED" for v in scored)
    if "INCOMPLETE" in scored:
        return "INCOMPLETE", n_conf
    return ("CONFIRMED" if n_conf >= 2 else "NOT CONFIRMED"), n_conf


def _filt1_horizon(records: dict, pursuer: str, horizon: int) -> dict:
    """H-FILT-1 at one (pursuer, H): both attackers, paired, against the
    STRONGER of the two scripted probes at that same cell."""
    names = {o: cell_name(pursuer, o, PREDICTIVE, horizon) for o in OPPONENTS}
    missing = [n for n in names.values() if n not in records]
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    best = max(SCRIPTED, key=lambda o: records[names[o]]["deadlock_rate"])
    probe = records[names[best]]
    attackers = {}
    for a in STRUCTURED:
        attackers[a] = {
            "cell": names[a],
            "test": "paired_indicator",
            **paired_indicator_test(
                records[names[a]]["deadlock_ind"], probe["deadlock_ind"], P_GATE
            ),
        }
    return {
        "verdict": (
            "CONFIRMED"
            if all(t["pass"] for t in attackers.values())
            else "NOT CONFIRMED"
        ),
        "best_scripted_opponent": best,
        "best_scripted_cell": names[best],
        "scripted_rates": {o: records[names[o]]["deadlock_rate"] for o in SCRIPTED},
        "attackers": attackers,
    }


def filt1(records: dict) -> dict:
    """H-FILT-1 (existence): under the predictive filter, at every horizon, the
    structured attackers deadlock the pursuer MORE than the strongest scripted
    probe. A pursuer confirms iff both attackers pass at all four horizons."""
    per_pursuer = {}
    for p in PURSUERS:
        horizons = {_hkey(h): _filt1_horizon(records, p, h) for h in HORIZONS}
        verdicts_ = [b["verdict"] for b in horizons.values()]
        if "INCOMPLETE" in verdicts_:
            verdict = "INCOMPLETE"
        elif all(v == "CONFIRMED" for v in verdicts_):
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"
        per_pursuer[p] = {
            "verdict": verdict,
            "scored": p in SCORING,
            "horizons": horizons,
        }
    overall, n_conf = _overall(per_pursuer)
    return {
        "hypothesis": "H-FILT-1",
        "levels": list(HORIZONS),
        "verdict": overall,
        "n_confirmed": n_conf,
        "scoring": list(SCORING),
        "pursuers": per_pursuer,
    }


def _filt2_opponent(records: dict, pursuer: str, opponent: str) -> dict:
    names = [cell_name(pursuer, opponent, PREDICTIVE, h) for h in HORIZONS]
    missing = [n for n in names if n not in records]
    if missing:
        return {"verdict": "INCOMPLETE", "levels": list(HORIZONS), "missing": missing}
    cells = [records[n] for n in names]
    rates = [c["deadlock_rate"] for c in cells]
    rho = spearman(rates)
    tvb = _paired_top_vs_bottom(cells[-1], cells[0])  # H = 32 vs H = 1
    block = {
        "levels": list(HORIZONS),
        "cells": names,
        "rates": rates,
        "wilson": [c["wilson_deadlock"]["wilson"] for c in cells],
        "rho": rho,
        "top_vs_bottom": tvb,
        "rate_top": rates[-1],
        "rate_bottom": rates[0],
        "capture_rates": [c["capture_rate"] for c in cells],
        "stall_step_rates": [c["stall_step_rate"] for c in cells],
        "backup_rates": [c["backup_rate"] for c in cells],
    }
    # S9's delta rule, verbatim, with H for delta: rho > 0 AND the top level
    # deadlocks more by the paired test (the levels share a seed stream).
    block["verdict"] = (
        ("CONFIRMED" if rho > 0 and tvb["pass"] else "NOT CONFIRMED")
        if opponent in STRUCTURED
        else None
    )
    return block


def filt2(records: dict) -> dict:
    """H-FILT-2 (scaling): the deadlock rate under the structured attackers is
    non-decreasing in the horizon — S9's delta rule with H as the level."""
    per_pursuer = {}
    for p in PURSUERS:
        opponents = {o: _filt2_opponent(records, p, o) for o in OPPONENTS}
        structured = [opponents[o]["verdict"] for o in STRUCTURED]
        if "INCOMPLETE" in structured:
            verdict = "INCOMPLETE"
        elif all(v == "CONFIRMED" for v in structured):
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"
        per_pursuer[p] = {
            "verdict": verdict,
            "scored": p in SCORING,
            "opponents": opponents,
        }
    overall, n_conf = _overall(per_pursuer)
    return {
        "hypothesis": "H-FILT-2",
        "levels": list(HORIZONS),
        "verdict": overall,
        "n_confirmed": n_conf,
        "scoring": list(SCORING),
        "pursuers": per_pursuer,
    }


# The pre-registered ordering at H = 4: RCM28 < FT-B-41 < RC19 (STR14 reported).
FILT3_ORDER = (("FT-B-41", "RCM28"), ("RC19", "FT-B-41"))
FILT3_REPORTED = ("STR14",)


def filt3(records: dict) -> dict:
    """H-FILT-3 (transfer of the ordering): at H = 4 the CBF-shield ordering of
    defenders by deadlock under attack is preserved, both inequalities paired,
    against both attackers."""
    needed = sorted({p for pair in FILT3_ORDER for p in pair} | set(FILT3_REPORTED))
    attackers = {}
    for a in STRUCTURED:
        names = {p: cell_name(p, a, PREDICTIVE, H_FILT3) for p in needed}
        missing = [n for n in names.values() if n not in records]
        if missing:
            attackers[a] = {"verdict": "INCOMPLETE", "missing": missing}
            continue
        tests = {
            f"{hi}>{lo}": {
                "test": "paired_indicator",
                **paired_indicator_test(
                    records[names[hi]]["deadlock_ind"],
                    records[names[lo]]["deadlock_ind"],
                    P_GATE,
                ),
            }
            for hi, lo in FILT3_ORDER
        }
        attackers[a] = {
            "verdict": (
                "CONFIRMED"
                if all(t["pass"] for t in tests.values())
                else "NOT CONFIRMED"
            ),
            "cells": names,
            "rates": {p: records[names[p]]["deadlock_rate"] for p in names},
            "tests": tests,
            "reported": {p: records[names[p]]["deadlock_rate"] for p in FILT3_REPORTED},
        }
    verdicts_ = [b["verdict"] for b in attackers.values()]
    if "INCOMPLETE" in verdicts_:
        overall = "INCOMPLETE"
    elif all(v == "CONFIRMED" for v in verdicts_):
        overall = "CONFIRMED"
    else:
        overall = "NOT CONFIRMED"
    return {
        "hypothesis": "H-FILT-3",
        "horizon": H_FILT3,
        "ordering": [f"{lo} < {hi}" for hi, lo in FILT3_ORDER],
        "verdict": overall,
        "attackers": attackers,
    }


def cbf_base(records: dict) -> dict:
    """The 4 CBF base cells (delta 0, n 2, gamma 0.2) that complete S9's base
    row for FT-B-41. Reported beside the predictive cells; no verdict."""
    rates: dict = {}
    missing = []
    for p in CBF_BASE_PURSUERS:
        for o in OPPONENTS:
            name = cell_name(p, o, CBF)
            if name in records:
                rates.setdefault(o, {})[p] = records[name]["deadlock_rate"]
            else:
                rates.setdefault(o, {})[p] = None
                missing.append(name)
    return {
        "filter": CBF,
        "setting": {"delta": DELTA, "n": N_OBSTACLES, "gamma": GAMMA},
        "pursuers": list(CBF_BASE_PURSUERS),
        "rates": rates,
        "missing": missing,
        "note": (
            "S9's base row for FT-B-41, measured on S11's stream; reported for "
            "comparison with the predictive cells, no verdict is derived"
        ),
    }


def verdicts(records: dict) -> dict:
    missing = [n for n in CELLS if n not in records]
    return {
        "filt1": filt1(records),
        "filt2": filt2(records),
        "filt3": filt3(records),
        "cbf_base": cbf_base(records),
        "hsafe": hsafe(records),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "horizons": list(HORIZONS),
        "seed0": SEED0,
        "n_episodes": N_EPISODES,
    }


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_COLORS = {
    "RC19": "tab:blue",
    "RCM28": "tab:orange",
    "STR14": "tab:green",
    "FT-B-41": "tab:red",
}


def _yerr(cells: list[dict], ys: list[float]) -> list[list[float]]:
    lo = [y - c["wilson_deadlock"]["wilson"][0] for y, c in zip(ys, cells, strict=True)]
    hi = [c["wilson_deadlock"]["wilson"][1] - y for y, c in zip(ys, cells, strict=True)]
    return [lo, hi]


def _panel_filt1(ax, records: dict) -> None:
    """Bars: each attacker and the stronger scripted probe, per pursuer x H."""
    labels, groups = [], []
    for p in PURSUERS:
        for h in HORIZONS:
            names = {o: cell_name(p, o, PREDICTIVE, h) for o in OPPONENTS}
            if any(n not in records for n in names.values()):
                continue
            best = max(SCRIPTED, key=lambda o: records[names[o]]["deadlock_rate"])
            labels.append(f"{p}\nH={h}")
            groups.append(
                (
                    p,
                    [records[names[a]]["deadlock_rate"] for a in STRUCTURED]
                    + [records[names[best]]["deadlock_rate"]],
                )
            )
    if not groups:
        ax.text(0.5, 0.5, "no complete pursuer x H group yet", ha="center")
        return
    xs = np.arange(len(groups))
    width = 0.27
    series = list(STRUCTURED) + ["best scripted"]
    for j, label in enumerate(series):
        ax.bar(
            xs + (j - 1) * width,
            [g[1][j] for g in groups],
            width,
            label=label,
            color=("0.6" if j == 2 else None),
            edgecolor="k",
            lw=0.3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(f"deadlock episode rate (N={N_EPISODES}, seeds {SEED0}+)")
    ax.set_title("H-FILT-1: attacker vs the strongest scripted probe")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=7)


def _panel_filt2(ax, records: dict) -> None:
    """Lines over the horizon levels, one per pursuer x opponent."""
    xs_idx = np.arange(len(HORIZONS))
    drew = False
    for p in PURSUERS:
        for o in OPPONENTS:
            names = [cell_name(p, o, PREDICTIVE, h) for h in HORIZONS]
            have = [(i, n) for i, n in enumerate(names) if n in records]
            if not have:
                continue
            drew = True
            cells = [records[n] for _, n in have]
            ys = [c["deadlock_rate"] for c in cells]
            ax.errorbar(
                [i for i, _ in have],
                ys,
                yerr=_yerr(cells, ys),
                marker="o" if o in STRUCTURED else "s",
                ms=4,
                lw=1.5,
                capsize=2,
                color=_COLORS[p],
                ls="-" if o in STRUCTURED else "--",
                alpha=1.0 if o in STRUCTURED else 0.5,
                label=f"{p} vs {o}",
            )
    ax.set_xticks(xs_idx)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlim(-0.3, len(HORIZONS) - 0.7)
    ax.set_xlabel("predictive-filter horizon H (steps, dt = 0.05 s)")
    ax.set_ylabel("deadlock episode rate")
    ax.set_title("H-FILT-2: does the surface scale with the lookahead?")
    ax.set_ylim(-0.02, 1.02)
    if drew:
        ax.legend(fontsize=5, ncol=2)


def _panel_filt3(ax, records: dict) -> None:
    """The defender ordering at H = 4, one group per attacker."""
    order = ["RCM28", "FT-B-41", "RC19", "STR14"]
    xs = np.arange(len(STRUCTURED))
    width = 0.2
    drew = False
    for j, p in enumerate(order):
        names = [cell_name(p, a, PREDICTIVE, H_FILT3) for a in STRUCTURED]
        have = [(i, n) for i, n in enumerate(names) if n in records]
        if not have:
            continue
        drew = True
        cells = [records[n] for _, n in have]
        ys = [c["deadlock_rate"] for c in cells]
        ax.bar(
            [xs[i] + (j - 1.5) * width for i, _ in have],
            ys,
            width,
            yerr=_yerr(cells, ys),
            capsize=2,
            label=p,
            color=_COLORS[p],
            edgecolor="k",
            lw=0.3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(list(STRUCTURED))
    ax.set_ylabel("deadlock episode rate")
    ax.set_title(f"H-FILT-3: ordering at H = {H_FILT3} (RCM28 < FT-B-41 < RC19)")
    ax.set_ylim(0.0, 1.02)
    if drew:
        ax.legend(fontsize=7)


def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    _panel_filt1(axes[0], records)
    _panel_filt2(axes[1], records)
    _panel_filt3(axes[2], records)
    fig.suptitle(
        "S11 — the stall surface under a SECOND hard filter (predictive "
        "lookahead, delta = 0, n = 2); solid = structured attacker, "
        "dashed = scripted baseline"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize() -> dict:
    records = load_records()
    preflight = (
        json.loads(preflight_path().read_text()) if preflight_path().exists() else None
    )
    blob = {
        "verdicts": verdicts(records),
        "preflight": (
            None
            if preflight is None
            else {
                "pass": preflight.get("pass"),
                "checks": {
                    k: v.get("pass") for k, v in preflight.get("checks", {}).items()
                },
            }
        ),
        "cells": {
            n: {k: v for k, v in r.items() if k not in ("deadlock_ind", "capture_ind")}
            for n, r in records.items()
        },
    }
    _write_json(OUT / f"{ARTEFACT_PREFIX}_round.json", blob)
    write_figure(records, OUT / f"{ARTEFACT_PREFIX}_round.png")
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)  # spec section 5
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        blob = run_preflight(device=args.device)
        print(json.dumps({k: v["pass"] for k, v in blob["checks"].items()}, indent=2))
        print(f"V-P {'PASSES' if blob['pass'] else 'FAILS'} -> {preflight_path()}")
    if args.cells is not None:
        run_cells(args.cells or None, args.device, args.workers)
    if args.summarize:
        blob = summarize()
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    h: {
                        "verdict": v[h]["verdict"],
                        **(
                            {"n_confirmed": v[h]["n_confirmed"]}
                            if "n_confirmed" in v[h]
                            else {}
                        ),
                    }
                    for h in ("filt1", "filt2", "filt3")
                },
                indent=2,
            )
        )
        print(json.dumps(v["hsafe"], indent=2))
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
