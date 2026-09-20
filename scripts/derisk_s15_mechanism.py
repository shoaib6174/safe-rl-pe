"""S15 — why does learning not escape the loop? Instrumentation only
(spec: docs/pathway/loop_mechanism_prereg_2026-09-20.md).

No policy is trained. Two frozen shielded pursuers (FT-B-41, RC19) are replayed
against the learned attacker ATK4-45 and against GreedyEvader while every env
step is traced, so the four pre-registered measurements can be read off the
trace rather than argued:

  M1  credit assignment   discounted return-to-go at each stall entry,
                          escaped vs not                      (predicted HOLD)
  M2  frozen action       per-step L2 change of the emitted (nominal) action
                          during stall runs, with the observation change and
                          the visibility fraction beside it    (predicted HOLD)
  M3  entropy collapse    action sigma at stalled vs non-stalled states
                                                              (predicted FAIL)
  M4  commitment          the same cells re-run at hold = 1   (predicted FAIL)

M3 and M4 are CONTROLS: the round is only informative because two of its four
predictions are expected to fail.

Why this script drives its own episode loop: the round needs per-step traces and
`pe2.structured.evaluate_structured` returns per-episode logs. The loop here is
`SingleAgentWrapper(cfg, "pursuer", opponent_factory)` driven by
`StructuredPursuerPolicy`, which is `SearchMDP.play_episode` episode-for-episode
(validity check V1-P in `pe2/structured.py`) — the same evaluation every prior
round used, only observed step by step.

Usage:
  ./venv/bin/python scripts/derisk_s15_mechanism.py --cells                (all missing)
  ./venv/bin/python scripts/derisk_s15_mechanism.py --cells RC19__GREEDY__h10
  ./venv/bin/python scripts/derisk_s15_mechanism.py --summarize

Stop rule (spec section 5): a cell that already has a record is NEVER re-run; no
threshold is changed after any number is seen. Cells are independent, so several
processes may run disjoint `--cells` lists concurrently.

`OUT` is the only output root; a smoke reassigns it before calling `run_cells`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

import numpy as np

from pe2.config import EnvConfig
from pe2.evade import D_POS
from pe2.exploitability import SingleAgentWrapper
from pe2.metrics import DEADLOCK_STEPS
from pe2.scripted import GreedyEvader

CKPT_ROOT = pathlib.Path("results/derisk")  # read-only inputs
OUT = pathlib.Path("results/derisk")

# --- pre-registered (spec sections 2-3). Do not tune. ---
SEED0 = 1_710_000
N_EPISODES = 200
GAMMA = 0.99  # the training discount
ENTRY_IMMOBILE_STEPS = 10  # a stall entry is the first step of a run this long
ESCAPE_WINDOW = 40  # an escape must BEGIN within this many steps of the entry
ESCAPE_MOVING_STEPS = 5  # and last at least this many consecutive steps
ESCAPE_V_FRAC = 0.1  # "moving" is v_exec > this fraction of v_max
P1_RTG_BAR = 0.05  # |Δ discounted return-to-go| below this: P1 holds
P1_CAPTURE_BAR = 0.15  # capture rate after an escape below this: P1 holds
P3_SIGMA_RATIO_BAR = 0.8  # sigma_stalled / sigma_free above this: P3 holds
P4_IMMOBILE_BAR = 0.05  # |Δ immobile-step rate| below this: P4 holds
HOLDS = (10, 1)  # M4: the same policies at the trained hold and at 1
MAIN_HOLD = 10  # M1-M3 read the hold-10 cells
EVADER_HOLD = 10  # the attacker's own hold, unchanged by M4 (as in S6/S9)
OMP_THREADS = "2"

POLICIES = {
    "FT-B-41": {
        "ckpt": CKPT_ROOT / "ckpts_FT-B-41" / "FT-B-41_5000000.zip",
        "shield_obs": True,
    },
    "RC19": {
        "ckpt": CKPT_ROOT / "ckpts_RC19" / "RC19_10000000.zip",
        "shield_obs": True,  # S5 remedy configuration (D113)
    },
}
OPPONENTS = {
    "ATK4-45": {"ckpt": CKPT_ROOT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"},
    "GREEDY": {"scripted": GreedyEvader},
}

_MODELS: dict = {}


def cell_name(policy: str, opponent: str, hold: int) -> str:
    return f"{policy}__{opponent}__h{hold}"


def _build_cells() -> dict[str, dict]:
    cells = {}
    for p in POLICIES:
        for o in OPPONENTS:
            for h in HOLDS:
                name = cell_name(p, o, h)
                cells[name] = {
                    "name": name,
                    "policy": p,
                    "opponent": o,
                    "hold": h,
                }
    return cells


CELLS = _build_cells()


def cells_dir() -> pathlib.Path:
    return OUT / "s15_cells"


def main_cells() -> list[str]:
    """The hold-10 cells, which M1-M3 are read from."""
    return [n for n, s in CELLS.items() if s["hold"] == MAIN_HOLD]


def _model(path, device: str):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def provenance() -> dict:
    import subprocess

    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "sb3_version": stable_baselines3.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


# ----------------------------------------------------------------------------
# the pre-registered definitions (spec section 3)
# ----------------------------------------------------------------------------
def _true_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal runs of True as (start, length)."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return runs


def stall_entries(immobile: Sequence[bool]) -> list[tuple[int, int]]:
    """Every stall entry as (entry step, run length): the first step of a
    MAXIMAL run of >= ENTRY_IMMOBILE_STEPS consecutive immobile steps. Runs are
    maximal, so two long runs separated by one moving step are two entries and a
    short run between two long ones is not an entry of its own."""
    return [(s, n) for s, n in _true_runs(immobile) if n >= ENTRY_IMMOBILE_STEPS]


def escape_after(
    entry: int, v_exec: Sequence[float], v_max: float
) -> tuple[bool, int | None]:
    """Did the agent escape this stall entry, and after how many steps?

    An escape is a MAXIMAL run of >= ESCAPE_MOVING_STEPS consecutive steps with
    v_exec > ESCAPE_V_FRAC * v_max that BEGINS within ESCAPE_WINDOW steps of the
    entry. The window is on the beginning only: a qualifying run that starts at
    entry + 40 counts however long it lasts, one that starts at entry + 41 does
    not. Runs are maximal, so a 4-step burst is not an escape even when a longer
    burst follows it, and a burst cut short by the episode end is not an escape.
    """
    moving = [float(v) > ESCAPE_V_FRAC * v_max for v in v_exec]
    for s, n in _true_runs(moving):
        if n < ESCAPE_MOVING_STEPS:
            continue
        offset = s - entry
        if 0 < offset <= ESCAPE_WINDOW:
            return True, offset
        if offset > ESCAPE_WINDOW:
            break
    return False, None


def return_to_go(outcome: str, steps_remaining: int, gamma: float = GAMMA) -> float:
    """The discounted return-to-go at a step `steps_remaining` before the end of
    an episode, under the training discount and the training terminal reward
    (+1 capture, -1 timeout, nothing in between). The env terminates on capture
    and truncates on timeout and nothing else (pe2/env.py OUTCOME_*), so any
    other label is an error rather than a silently signed zero."""
    if outcome == "capture":
        sign = 1.0
    elif outcome == "timeout":
        sign = -1.0
    else:
        raise ValueError(f"unknown outcome {outcome!r}: expected capture or timeout")
    return sign * gamma**steps_remaining


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def opponent_factory(name: str, cfg: EnvConfig, device: str):
    spec = OPPONENTS[name]
    if "scripted" in spec:
        return spec["scripted"]
    from pe2.evade import StructuredEvaderPolicy

    model = _model(spec["ckpt"], device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=EVADER_HOLD, d_pos=D_POS)


def play_traced_episode(wrapper, pol, cfg: EnvConfig, seed: int) -> dict:
    """One seeded episode, traced step by step.

    Identical dynamics to `SearchMDP.play_episode` (validity check V1-P): the
    intercept law acts when it applies, otherwise the PPO decision held `hold`
    env steps, and the shield filters the executed command with the true pose.
    The arrays returned are per ENV STEP and all the same length.

    `obs` is the POLICY'S input — the raw observation with the three shield
    features appended when the policy was trained with `shield_obs` — so M2's
    "does the observation freeze" and M3's sigma are both read off exactly what
    the network sees. `action` is the NOMINAL command in the normalised [-1, 1]
    action space, i.e. what the policy asks for before the filter acts on it.
    """
    wrapper.reset(seed=seed)
    pol.reset()
    obs_rows, act_rows = [], []
    immobile, stalled, v_exec, visible = [], [], [], []
    violations = 0
    while True:
        pobs = wrapper._last_joint_obs["pursuer"]
        # `_features()` is what `act` is about to append; read it first,
        # because `act` overwrites the shield info it is computed from.
        lobs = (
            np.concatenate([np.asarray(pobs, dtype=np.float32), pol._features()])
            if pol.shield_obs
            else np.asarray(pobs, dtype=np.float32)
        )
        v, w = pol.act(pobs, wrapper.env)
        sinfo = pol.last_sinfo
        v_nom, w_nom = pol.last_nom
        obs_rows.append(lobs)
        act_rows.append([2.0 * v_nom / cfg.v_max - 1.0, w_nom / cfg.omega_max])
        immobile.append(bool(sinfo["immobile"]))
        stalled.append(bool(sinfo["stalled"]))
        v_exec.append(float(sinfo["v_exec"]))
        visible.append(bool(pobs[6] > 0.5))
        _, _, term, trunc, info = wrapper.step(
            np.array([2.0 * v / cfg.v_max - 1.0, w / cfg.omega_max], dtype=np.float32)
        )
        contact = bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"])
        violations += int(contact)
        pol.record(contact)
        if term or trunc:
            break
    return {
        "seed": seed,
        "outcome": info["outcome"],
        "steps": int(wrapper.env.current_step),
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "action": np.asarray(act_rows, dtype=np.float64),
        "immobile": immobile,
        "stalled": stalled,
        "v_exec": v_exec,
        "visible": visible,
        "violations": violations,
    }


def _median(xs) -> float:
    return float(np.median(np.asarray(xs, dtype=float))) if len(xs) else float("nan")


def _entries_of(episode: int, tr: dict, cfg: EnvConfig) -> list[dict]:
    """M1 + M2 per stall entry of one traced episode."""
    out = []
    total = len(tr["immobile"])
    for start, length in stall_entries(tr["immobile"]):
        escaped, steps_to_escape = escape_after(start, tr["v_exec"], cfg.v_max)
        steps_remaining = total - start
        run = slice(start, start + length)
        d_act = np.linalg.norm(np.diff(tr["action"][run], axis=0), axis=1)
        d_obs = np.linalg.norm(np.diff(tr["obs"][run], axis=0), axis=1)
        out.append(
            {
                "episode": episode,
                "step": start,
                "escaped": escaped,
                "steps_to_escape": steps_to_escape,
                "outcome": tr["outcome"],
                "steps_remaining": steps_remaining,
                "return_to_go": return_to_go(tr["outcome"], steps_remaining),
                "action_deltas_median": _median(d_act),
                "obs_deltas_median": _median(d_obs),
                "visible_fraction": float(np.mean(tr["visible"][run])),
            }
        )
    return out


def action_sigma(model, obs: np.ndarray) -> np.ndarray:
    """Per-observation action sigma of a Gaussian SB3 policy, shape (N, act_dim).

    For SB3's default `MlpPolicy` the Gaussian head carries ONE learned
    `log_std` parameter with no state input, so every row comes back identical
    — that is a property of the architecture, not a measurement, and
    `m3_verdict` reports it as such instead of inventing a per-state number.
    """
    import torch

    rows = []
    with torch.no_grad():
        for i in range(0, len(obs), 4096):
            chunk = torch.as_tensor(np.asarray(obs[i : i + 4096], dtype=np.float32))
            dist = model.policy.get_distribution(chunk).distribution
            scale = np.asarray(dist.scale.cpu().numpy(), dtype=float)
            if scale.ndim == 1:  # a shared scale broadcast over the batch
                scale = np.broadcast_to(scale, (len(chunk), scale.shape[0]))
            rows.append(np.array(scale, dtype=float))
    return np.concatenate(rows) if rows else np.zeros((0, 0))


def run_cell(name: str, device: str = "cpu") -> dict:
    """One cell: the frozen policy at this cell's hold against this cell's
    opponent over the pre-registered stream, traced step by step."""
    from pe2.shield import CBFShield
    from pe2.structured import StructuredPursuerPolicy

    spec = CELLS[name]
    cfg = EnvConfig()
    pspec = POLICIES[spec["policy"]]
    model = _model(pspec["ckpt"], device)
    wrapper = SingleAgentWrapper(
        cfg, "pursuer", opponent_factory(spec["opponent"], cfg, device)
    )
    pol = StructuredPursuerPolicy(
        model, cfg, CBFShield(cfg), spec["hold"], pspec["shield_obs"]
    )
    entries: list[dict] = []
    per_episode: list[dict] = []
    obs_all: list[np.ndarray] = []
    stalled_all: list[np.ndarray] = []
    n_steps = n_immobile = n_capture = violations = 0
    deadlocks = 0
    for i in range(N_EPISODES):
        tr = play_traced_episode(wrapper, pol, cfg, SEED0 + i)
        ep_entries = _entries_of(i, tr, cfg)
        entries += ep_entries
        longest_immobile = max((n for _, n in _true_runs(tr["immobile"])), default=0)
        longest_stall = max((n for _, n in _true_runs(tr["stalled"])), default=0)
        per_episode.append(
            {
                "seed": tr["seed"],
                "outcome": tr["outcome"],
                "steps": tr["steps"],
                "n_immobile": int(sum(tr["immobile"])),
                "n_stalled": int(sum(tr["stalled"])),
                "longest_immobile_run": int(longest_immobile),
                "longest_stall_run": int(longest_stall),
                "n_entries": len(ep_entries),
                "violations": tr["violations"],
            }
        )
        n_steps += len(tr["immobile"])
        n_immobile += int(sum(tr["immobile"]))
        n_capture += int(tr["outcome"] == "capture")
        violations += tr["violations"]
        deadlocks += int(longest_stall >= DEADLOCK_STEPS)
        obs_all.append(tr["obs"])
        stalled_all.append(np.asarray(tr["stalled"], dtype=bool))
    obs = np.concatenate(obs_all) if obs_all else np.zeros((0, 0), dtype=np.float32)
    is_stalled = np.concatenate(stalled_all) if stalled_all else np.zeros(0, dtype=bool)
    sigma = action_sigma(model, obs)
    per_step = sigma.mean(axis=1) if sigma.size else np.zeros(0)
    spread = float(per_step.max() - per_step.min()) if per_step.size else 0.0
    return {
        "name": name,
        "policy": spec["policy"],
        "opponent": spec["opponent"],
        "hold": spec["hold"],
        "n": N_EPISODES,
        "seed0": SEED0,
        "entries": entries,
        "per_episode": per_episode,
        "immobile_step_rate": n_immobile / n_steps if n_steps else float("nan"),
        "deadlock_episode_rate": deadlocks / N_EPISODES,
        "capture_rate": n_capture / N_EPISODES,
        "violations": violations,
        "sigma_stalled": (
            float(per_step[is_stalled].mean()) if is_stalled.any() else None
        ),
        "sigma_free": (
            float(per_step[~is_stalled].mean()) if (~is_stalled).any() else None
        ),
        # False = the policy's sigma does not depend on the observation at all
        # (SB3's default Gaussian head), which makes M3 vacuous by construction.
        "sigma_state_dependent": bool(spread > 1e-9),
        "sigma_spread": spread,
        "n_steps": n_steps,
        "provenance": provenance(),
    }


def write_cell(record: dict) -> pathlib.Path:
    path = cells_dir() / f"{record['name']}.json"
    _write_json(path, record)
    return path


def _cell_line(rec: dict) -> str:
    return (
        f"[{rec['name']}] immobile {rec['immobile_step_rate']:.3f} "
        f"deadlockEp {rec['deadlock_episode_rate']:.3f} "
        f"capture {rec['capture_rate']:.3f} entries {len(rec['entries'])} "
        f"viol {rec['violations']}"
    )


def run_cells(names: list[str] | None = None, device: str = "cpu") -> list[str]:
    """Run every named cell that has no record yet (default: all 8)."""
    wanted = list(CELLS) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}")
    cells_dir().mkdir(parents=True, exist_ok=True)
    todo = [n for n in wanted if not (cells_dir() / f"{n}.json").exists()]
    print(
        f"{len(todo)}/{len(wanted)} cells to run "
        f"({len(wanted) - len(todo)} already have a record; stop rule: never re-run)",
        flush=True,
    )
    for name in todo:
        rec = run_cell(name, device)
        write_cell(rec)
        print(_cell_line(rec), flush=True)
    return todo


def load_records() -> dict:
    d = cells_dir()
    if not d.exists():
        return {}
    out = {}
    for p in sorted(d.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("name"):
            out[rec["name"]] = rec
    return out


# ----------------------------------------------------------------------------
# the four verdicts (spec section 2)
# ----------------------------------------------------------------------------
def _missing_main(records: dict) -> list[str]:
    return [n for n in main_cells() if n not in records]


def _mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=float))) if len(xs) else float("nan")


def _m1_block(entries: list[dict]) -> dict:
    esc = [e for e in entries if e["escaped"]]
    non = [e for e in entries if not e["escaped"]]
    rtg_e, rtg_n = (
        _mean([e["return_to_go"] for e in esc]),
        _mean([e["return_to_go"] for e in non]),
    )
    return {
        "n_entries": len(entries),
        "n_escaped": len(esc),
        "n_not_escaped": len(non),
        "escape_rate": len(esc) / len(entries) if entries else float("nan"),
        "mean_return_to_go_escaped": rtg_e,
        "mean_return_to_go_not_escaped": rtg_n,
        "delta_return_to_go": rtg_e - rtg_n,
        "capture_rate_escaped": _mean([e["outcome"] == "capture" for e in esc]),
        "capture_rate_not_escaped": _mean([e["outcome"] == "capture" for e in non]),
        "mean_steps_remaining_escaped": _mean([e["steps_remaining"] for e in esc]),
        "mean_steps_remaining_not_escaped": _mean([e["steps_remaining"] for e in non]),
        "mean_steps_to_escape": _mean([e["steps_to_escape"] for e in esc]),
    }


def m1_verdict(records: dict) -> dict:
    """P1: |Δ discounted return-to-go| < 0.05 AND the capture rate following an
    escape < 0.15. Read from the hold-10 cells POOLED, which is how the spec
    states the measurement ("over episodes of FT-B-41 and RC19 against ATK4-45
    and GreedyEvader"); the per-cell numbers are reported beside it."""
    missing = _missing_main(records)
    per_cell = {
        n: _m1_block(records[n]["entries"]) for n in main_cells() if n in records
    }
    pooled = _m1_block(
        [e for n in main_cells() if n in records for e in records[n]["entries"]]
    )
    if missing or not pooled["n_escaped"] or not pooled["n_not_escaped"]:
        verdict = "INCOMPLETE"
    else:
        verdict = (
            "HOLDS"
            if abs(pooled["delta_return_to_go"]) < P1_RTG_BAR
            and pooled["capture_rate_escaped"] < P1_CAPTURE_BAR
            else "FAILS"
        )
    return {
        "hypothesis": "P1",
        "prediction": "HOLD",
        "verdict": verdict,
        "missing": missing,
        "per_cell": per_cell,
        **pooled,
    }


def m2_verdict(records: dict) -> dict:
    """P2: the emitted action is near-constant during stall runs — median
    per-step action change < 0.01 on the [-1, 1] action space. Each stall run
    contributes its own median per-step change (the record's
    `action_deltas_median`) and the verdict reads the median over runs, so no
    single very long run can carry the number. The observation change and the
    visibility fraction are reported and NOT gated: the spec states M2 so that
    "the observation is constant" can be refuted."""
    missing = _missing_main(records)
    entries = [e for n in main_cells() if n in records for e in records[n]["entries"]]
    per_cell = {
        n: {
            "n_entries": len(records[n]["entries"]),
            "action_delta_median": _median(
                [e["action_deltas_median"] for e in records[n]["entries"]]
            ),
            "obs_delta_median": _median(
                [e["obs_deltas_median"] for e in records[n]["entries"]]
            ),
            "visible_fraction": _mean(
                [e["visible_fraction"] for e in records[n]["entries"]]
            ),
        }
        for n in main_cells()
        if n in records
    }
    action = _median([e["action_deltas_median"] for e in entries])
    if missing or not entries:
        verdict = "INCOMPLETE"
    else:
        verdict = "HOLDS" if action < 0.01 else "FAILS"
    return {
        "hypothesis": "P2",
        "prediction": "HOLD",
        "verdict": verdict,
        "missing": missing,
        "n_entries": len(entries),
        "action_delta_median": action,
        "obs_delta_median": _median([e["obs_deltas_median"] for e in entries]),
        "visible_fraction": _mean([e["visible_fraction"] for e in entries]),
        "per_cell": per_cell,
    }


def m3_verdict(records: dict) -> dict:
    """P3 (CONTROL, predicted to FAIL): sigma at stalled states is NOT
    materially lower — ratio to non-stalled sigma > 0.8.

    If no cell's sigma depends on the observation at all, the ratio is 1 by
    construction and the measurement is vacuous: the verdict is then
    "N/A (state-independent sigma)" and no number is reported as if it had been
    measured. That is the honest answer to M3 for an SB3 `MlpPolicy`, whose
    Gaussian head carries a single learned `log_std` parameter."""
    missing = _missing_main(records)
    present = [records[n] for n in main_cells() if n in records]
    dependent = [r for r in present if r.get("sigma_state_dependent")]
    per_cell = {
        r["name"]: {
            "sigma_stalled": r["sigma_stalled"],
            "sigma_free": r["sigma_free"],
            "state_dependent": bool(r.get("sigma_state_dependent")),
        }
        for r in present
    }
    ratio = None
    if dependent:
        stalled = _mean([r["sigma_stalled"] for r in dependent])
        free = _mean([r["sigma_free"] for r in dependent])
        ratio = stalled / free if free else None
    if missing:
        verdict = "INCOMPLETE"
    elif not dependent:
        verdict = "N/A (state-independent sigma)"
    elif ratio is None:
        verdict = "INCOMPLETE"
    else:
        verdict = "HOLDS" if ratio > P3_SIGMA_RATIO_BAR else "FAILS"
    return {
        "hypothesis": "P3",
        "prediction": "FAIL (control)",
        "verdict": verdict,
        "missing": missing,
        "state_dependent": bool(dependent),
        "state_independent_cells": sorted(
            r["name"] for r in present if not r.get("sigma_state_dependent")
        ),
        "ratio": ratio,
        "sigma_stalled": _mean([r["sigma_stalled"] for r in dependent])
        if dependent
        else None,
        "sigma_free": _mean([r["sigma_free"] for r in dependent])
        if dependent
        else None,
        "per_cell": per_cell,
    }


def m4_verdict(records: dict) -> dict:
    """P4 (CONTROL, predicted to FAIL): reducing the hold from 10 to 1 does not
    materially change immobility — |Δ immobile-step rate| < 0.05 for every
    (policy, opponent) pair."""
    pairs = []
    missing = []
    for p in POLICIES:
        for o in OPPONENTS:
            names = {h: cell_name(p, o, h) for h in HOLDS}
            absent = [n for n in names.values() if n not in records]
            missing += absent
            if absent:
                pairs.append({"policy": p, "opponent": o, "missing": absent})
                continue
            rates = {h: records[names[h]]["immobile_step_rate"] for h in HOLDS}
            delta = rates[1] - rates[MAIN_HOLD]
            pairs.append(
                {
                    "policy": p,
                    "opponent": o,
                    "immobile_hold10": rates[MAIN_HOLD],
                    "immobile_hold1": rates[1],
                    "delta": delta,
                    "pass": bool(abs(delta) < P4_IMMOBILE_BAR),
                }
            )
    if missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "HOLDS" if all(p["pass"] for p in pairs) else "FAILS"
    return {
        "hypothesis": "P4",
        "prediction": "FAIL (control)",
        "verdict": verdict,
        "missing": sorted(set(missing)),
        "pairs": pairs,
    }


def hsafe(records: dict) -> dict:
    """H-SAFE: 0 pursuer contact steps in every cell."""
    offenders = [n for n, r in records.items() if int(r.get("violations", 0))]
    steps = sum(int(r.get("violations", 0)) for r in records.values())
    missing = [n for n in CELLS if n not in records]
    return {
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "violation_steps": steps,
        "offenders": sorted(offenders),
        "cells_measured": len(records),
    }


# ----------------------------------------------------------------------------
# what each outcome fixes (spec section 4)
# ----------------------------------------------------------------------------
DIRECTIONS = (
    {
        "key": "predicted",
        "result": "P1 and P2 hold, P3 and P4 fail",
        "direction": (
            "The predicted picture: no gradient signal to escape, and a "
            "near-constant emitted action. The paper's mechanism section is "
            "these two measurements, and the fix must supply either the signal "
            "(dense escape reward) or the state (memory / escape features) — "
            "Phase 2's arms are then well motivated."
        ),
    },
    {
        "key": "p1_fails",
        "result": "P1 fails (a real advantage exists)",
        "direction": (
            "The signal was there and learning still missed it; the explanation "
            "is optimisation, not credit assignment, and Phase 1 must continue "
            "before Phase 2 is worth running."
        ),
    },
    {
        "key": "p2_fails",
        "result": "P2 fails (the action varies)",
        "direction": (
            "The policy is trying different things and still not escaping; the "
            "constraint is the filter, not the policy, and the paper's balance "
            "shifts back toward the filter's objective."
        ),
    },
    {
        "key": "p3_fails",
        "result": "P3 fails (entropy collapses at stalls)",
        "direction": (
            "A simpler and more familiar story — exploration dies exactly where "
            "needed — which must then be reported as the primary mechanism."
        ),
    },
    {
        "key": "p4_fails",
        "result": "P4 fails (hold matters)",
        "direction": (
            "Commitment is implicated; Phase 2 adds a hold-length arm and the "
            "claim narrows to temporally-committed policies."
        ),
    },
)


def direction(blocks: dict) -> list[dict]:
    """The spec section 4 rows this round's verdicts select. An outcome the
    table does not cover (every prediction holding, or a vacuous M3 alongside
    holding controls) selects no row; `verdicts` says so in `notes` rather than
    inventing a sixth row here."""
    v = {k: blocks[k]["verdict"] for k in ("m1", "m2", "m3", "m4")}
    by_key = {row["key"]: row for row in DIRECTIONS}
    if (v["m1"], v["m2"], v["m3"], v["m4"]) == ("HOLDS", "HOLDS", "FAILS", "FAILS"):
        return [by_key["predicted"]]
    return [
        by_key[f"p{i}_fails"]
        for i, k in enumerate(("m1", "m2", "m3", "m4"), start=1)
        if v[k] == "FAILS"
    ]


def verdicts(records: dict) -> dict:
    missing = [n for n in CELLS if n not in records]
    blocks = {
        "m1": m1_verdict(records),
        "m2": m2_verdict(records),
        "m3": m3_verdict(records),
        "m4": m4_verdict(records),
    }
    notes = []
    if blocks["m3"]["verdict"].startswith("N/A"):
        notes.append(
            "M3 is vacuous for these policies: their action sigma is "
            "state-independent (one learned log_std parameter, SB3's default "
            "Gaussian head), so sigma at stalled states EQUALS sigma "
            "elsewhere by construction and P3 is reported as "
            "'N/A (state-independent sigma)', not as a measured ratio. The "
            "spec section 4 row for the predicted picture requires P3 to "
            "FAIL and therefore cannot be selected."
        )
    if not missing and not direction(blocks):
        notes.append(
            "no spec section 4 row matches this combination of verdicts; the "
            "measurements stand on their own."
        )
    return {
        **blocks,
        "hsafe": hsafe(records),
        "direction": direction(blocks),
        "notes": notes,
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "seed0": SEED0,
        "n_episodes": N_EPISODES,
    }


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = verdicts(records)
    # only cells that HAVE a record are drawn: a partial round must not show a
    # missing cell as a measured zero.
    names = [n for n in main_cells() if n in records]
    labels = [n.replace("__h10", "").replace("__", "\n") for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ax = axes[0]
    esc = [v["m1"]["per_cell"][n]["mean_return_to_go_escaped"] for n in names]
    non = [v["m1"]["per_cell"][n]["mean_return_to_go_not_escaped"] for n in names]
    ax.bar(x - 0.2, esc, 0.4, label="escaped")
    ax.bar(x + 0.2, non, 0.4, label="not escaped")
    ax.set_title(
        f"M1 / P1 {v['m1']['verdict']}\n"
        f"Δ={v['m1']['delta_return_to_go']:.4f} (bar {P1_RTG_BAR}), "
        f"capture|escape={v['m1']['capture_rate_escaped']:.3f} "
        f"(bar {P1_CAPTURE_BAR})"
    )
    ax.set_ylabel(f"mean discounted return-to-go at the entry (γ={GAMMA})")
    ax.legend(fontsize=7)

    ax = axes[1]
    act = [v["m2"]["per_cell"][n]["action_delta_median"] for n in names]
    obs = [v["m2"]["per_cell"][n]["obs_delta_median"] for n in names]
    ax.bar(x - 0.2, act, 0.4, label="action")
    ax.bar(x + 0.2, obs, 0.4, label="observation")
    ax.axhline(0.01, color="k", ls="--", lw=1, label="P2 bar 0.01")
    # symlog, not log: a median change of exactly 0 is the interesting reading
    # here and must render as 0 rather than be pushed off a log axis.
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("median per-step L2 change during stall runs")
    ax.set_title(
        f"M2 / P2 {v['m2']['verdict']}\n"
        f"action {v['m2']['action_delta_median']:.2e}, "
        f"obs {v['m2']['obs_delta_median']:.2e}, "
        f"visible {v['m2']['visible_fraction']:.3f}"
    )
    ax.legend(fontsize=7)

    ax = axes[2]
    st = [v["m3"]["per_cell"][n]["sigma_stalled"] or 0.0 for n in names]
    fr = [v["m3"]["per_cell"][n]["sigma_free"] or 0.0 for n in names]
    ax.bar(x - 0.2, st, 0.4, label="stalled")
    ax.bar(x + 0.2, fr, 0.4, label="non-stalled")
    ax.set_ylabel("mean action sigma")
    ax.set_title(f"M3 / P3 {v['m3']['verdict']}")
    if not v["m3"]["state_dependent"]:
        ax.text(
            0.5,
            0.55,
            "sigma is state-independent\n(one learned log_std):\nM3 is vacuous",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    ax.legend(fontsize=7)

    ax = axes[3]
    pairs = [p for p in v["m4"]["pairs"] if "delta" in p]
    px = np.arange(len(pairs))
    ax.bar(px - 0.2, [p["immobile_hold10"] for p in pairs], 0.4, label="hold 10")
    ax.bar(px + 0.2, [p["immobile_hold1"] for p in pairs], 0.4, label="hold 1")
    ax.set_xticks(px)
    ax.set_xticklabels([f"{p['policy']}\n{p['opponent']}" for p in pairs], fontsize=7)
    ax.set_ylabel("immobile-step rate")
    ax.set_title(f"M4 / P4 {v['m4']['verdict']} (bar |Δ| < {P4_IMMOBILE_BAR})")
    ax.legend(fontsize=7)

    for a in axes[:3]:
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=7)
        a.set_xlim(-0.6, len(names) - 0.4)  # bar width stays sane on a partial round
    fig.suptitle(
        f"S15 — why the shielded policy does not escape the loop "
        f"(N={N_EPISODES}, seeds {SEED0}+; P3 and P4 are controls)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize() -> dict:
    records = load_records()
    blob = {
        "verdicts": verdicts(records),
        "cells": {
            n: {k: val for k, val in r.items() if k not in ("per_episode", "entries")}
            | {"n_entries": len(r.get("entries", []))}
            for n, r in records.items()
        },
    }
    _write_json(OUT / "s15_round.json", blob)
    write_figure(records, OUT / "s15_round.png")
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)  # spec section 5
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cells is not None:
        run_cells(args.cells or None, args.device)
    if args.summarize:
        v = summarize()["verdicts"]
        for key in ("m1", "m2", "m3", "m4"):
            b = v[key]
            print(f"{b['hypothesis']} ({b['prediction']}): {b['verdict']}")
        print(json.dumps(v["hsafe"], indent=2))
        for row in v["direction"]:
            print(f"\n[{row['result']}] {row['direction']}")
        for note in v["notes"]:
            print(f"\nNOTE: {note}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
