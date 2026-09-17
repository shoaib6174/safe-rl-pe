"""S9 — is the stall attack a property of the shield? Evaluation-only sweeps
(spec: docs/pathway/attack_surface_sweep_prereg_2026-09-17.md + amendments A1, A2).

No training. Three frozen shielded pursuers are re-evaluated against two frozen
structured attackers and two scripted baselines while ONE shield or geometry
knob moves at a time:

  delta  (shield margin, m)   {0, 0.05, 0.10, 0.20, 0.30}   gamma 0.2, n 2
  n      (obstacles PLACED)   {0, 1, 2, 3, 4}               delta 0,  gamma 0.2
  gamma  (class-K gain)       {0.1, 0.2, 0.4}               delta 0,  n 2

11 distinct settings (the base setting delta 0 / n 2 / gamma 0.2 belongs to all
three sweeps and is run once) x 3 pursuers x 4 opponents = 132 cells, every cell
on the same 300-seed stream from SEED0. A1: only the number of obstacles
PHYSICALLY generated moves; `cfg.n_obstacles` stays 2 so every policy's
observation layout is the one it was trained with.

Usage:
  ./venv/bin/python scripts/derisk_s9_sweep.py --cells                 (all missing)
  ./venv/bin/python scripts/derisk_s9_sweep.py --cells RC19__ATT25__d0.00_n2_g0.2
  ./venv/bin/python scripts/derisk_s9_sweep.py --summarize

Stop rule (spec section 5): a cell that already has a record is NEVER re-run;
no level is added or removed after any number is seen.

`OUT` is the only output root and `--workers 1` is the only mode in which
reassigning it (a smoke) reaches the code that writes: spawned workers re-import
this module and get the module-level default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import pathlib
from collections import Counter

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.evade import D_POS
from pe2.eval import wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD

CKPT_ROOT = pathlib.Path("results/derisk")  # read-only inputs
OUT = pathlib.Path("results/derisk")

# --- pre-registered (spec sections 2-3, amendment A1). Do not tune. ---
SEED0 = 1_240_000
N_EPISODES = 300
P_GATE = 0.05
MAX_WORKERS = 6
OMP_THREADS = "2"
EVADER_HOLD = 10  # StructuredEvaderPolicy's hold, as in S6/S7
DELTAS = (0.0, 0.05, 0.10, 0.20, 0.30)
NS = (0, 1, 2, 3, 4)
NS_CONFIRM = (2, 3, 4)  # A1: >= 2 obstacles, so both observed slots are filled
NS_EXPLORATORY = (0, 1)  # A1: reported, no verdict (zero-filled slots)
GAMMAS = (0.1, 0.2, 0.4)
BASE_DELTA, BASE_N, BASE_GAMMA = 0.0, 2, 0.2

PURSUERS = {
    "RC19": {
        "ckpt": CKPT_ROOT / "ckpts_RC19" / "RC19_10000000.zip",
        "shield_obs": True,  # S5 remedy configuration (D113)
    },
    "RCM28": {
        "ckpt": CKPT_ROOT / "ckpts_RCM28" / "RCM28_10000000.zip",
        "shield_obs": True,  # S7 RC_SETTINGS (D116)
    },
    "STR14": {
        "ckpt": CKPT_ROOT / "ckpts_STR14" / "STR14_10000000.zip",
        "shield_obs": False,  # trained unshielded (D111, S6 defender D1)
    },
}
OPPONENTS = {
    "ATT25": {"ckpt": CKPT_ROOT / "ckpts_CTL-D225" / "CTL-D225_10000000.zip"},
    "ATK2-31": {"ckpt": CKPT_ROOT / "ckpts_ATK2-31" / "ATK2-31_10000000.zip"},
    "HUG7": {"scripted": ObstacleHuggingEvader},
    "GREEDY": {"scripted": GreedyEvader},
}
STRUCTURED = ("ATT25", "ATK2-31")  # the two attackers a verdict is read from
SCRIPTED = ("HUG7", "GREEDY")  # the non-adversarial baseline, no verdict

_MODELS: dict = {}


def cell_name(pursuer: str, opponent: str, delta: float, n: int, gamma: float) -> str:
    return f"{pursuer}__{opponent}__d{delta:.2f}_n{n}_g{gamma:.1f}"


def _settings() -> dict[tuple[float, int, float], set]:
    """The 11 distinct (delta, n, gamma) settings, each tagged with the sweeps
    it belongs to. The base setting carries all three tags and is run once."""
    s: dict[tuple[float, int, float], set] = {}
    for d in DELTAS:
        s.setdefault((d, BASE_N, BASE_GAMMA), set()).add("delta")
    for n in NS:
        s.setdefault((BASE_DELTA, n, BASE_GAMMA), set()).add("n")
    for g in GAMMAS:
        s.setdefault((BASE_DELTA, BASE_N, g), set()).add("gamma")
    return s


def _build_cells() -> dict[str, dict]:
    cells = {}
    for p in PURSUERS:
        for o in OPPONENTS:
            for (d, n, g), tags in _settings().items():
                name = cell_name(p, o, d, n, g)
                cells[name] = {
                    "name": name,
                    "pursuer": p,
                    "opponent": o,
                    "delta": d,
                    "n": n,
                    "gamma": g,
                    "sweep": set(tags),
                }
    return cells


CELLS = _build_cells()


def cells_dir() -> pathlib.Path:
    return OUT / "s9_cells"


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


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson": [lo, hi]}


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def opponent_factory(name: str, cfg_cell: EnvConfig, device: str):
    """The cell's opponent. Structured attackers are built on `cfg_cell`, so the
    hug law and the model both see the arena the cell actually runs in."""
    spec = OPPONENTS[name]
    if "scripted" in spec:
        return spec["scripted"]
    from pe2.evade import StructuredEvaderPolicy

    model = _model(spec["ckpt"], device)
    return lambda: StructuredEvaderPolicy(
        model, cfg_cell, hold=EVADER_HOLD, d_pos=D_POS
    )


def run_cell(name: str, device: str = "cpu") -> dict:
    """One cell: the frozen pursuer, shielded at this cell's (delta, gamma), in
    an arena with this cell's obstacle count, over the pre-registered stream."""
    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    spec = CELLS[name]
    cfg_cell = dataclasses.replace(EnvConfig(), n_physical_obstacles=spec["n"])
    model = _model(PURSUERS[spec["pursuer"]]["ckpt"], device)
    logs = evaluate_structured(
        model,
        cfg_cell,
        opponent_factory(spec["opponent"], cfg_cell, device),
        N_EPISODES,
        SEED0,
        hold=HOLD,
        shield_factory=lambda: CBFShield(
            cfg_cell, gamma=spec["gamma"], margin=spec["delta"]
        ),
        shield_obs=PURSUERS[spec["pursuer"]]["shield_obs"],
    )
    sh = summarize_shield(logs)
    k, n_ep = win_count(logs, "pursuer")
    deadlock_ind = [bool(lg["shield"]["deadlock"]) for lg in logs]
    outcomes = Counter(lg["outcome"] for lg in logs)
    return {
        "name": name,
        "pursuer": spec["pursuer"],
        "opponent": spec["opponent"],
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "sweep": sorted(spec["sweep"]),
        "shield_obs": PURSUERS[spec["pursuer"]]["shield_obs"],
        "n_episodes": n_ep,
        "seed0": SEED0,
        "deadlock_ind": deadlock_ind,
        "capture_ind": [lg["outcome"] == "capture" for lg in logs],
        "deadlock_rate": sh["deadlock_episode_rate"],
        "capture_rate": k / n_ep,
        # The env terminates on capture and truncates on timeout and nothing
        # else (pe2/env.py OUTCOME_*), so the evader "escapes" exactly when the
        # episode times out. `outcomes` carries the raw labels so this stays a
        # reported count, not a definition invented here.
        "escape_rate": outcomes["timeout"] / n_ep,
        "outcomes": dict(outcomes),
        "stall_step_rate": sh["stall_step_rate"],
        "intervention_rate": sh["intervention_rate"],
        "violations": int(sh["violation_steps"]),
        "min_clearance_floor": sh["min_clearance_floor"],
        "wilson_deadlock": _ci(sum(deadlock_ind), n_ep),
        "provenance": provenance(),
    }


def write_cell(record: dict) -> pathlib.Path:
    path = cells_dir() / f"{record['name']}.json"
    _write_json(path, record)
    return path


def _cell_line(rec: dict) -> str:
    return (
        f"[{rec['name']}] deadlock {rec['deadlock_rate']:.3f} "
        f"capture {rec['capture_rate']:.3f} escape {rec['escape_rate']:.3f} "
        f"stall {rec['stall_step_rate']:.3f} int {rec['intervention_rate']:.3f} "
        f"viol {rec['violations']}"
    )


def _worker(args: tuple) -> str:
    name, device = args
    return _cell_line(_run_and_write(name, device))


def _run_and_write(name: str, device: str) -> dict:
    rec = run_cell(name, device)
    write_cell(rec)
    return rec


def check_obstacles_nested(seed: int = SEED0, n_a: int = 2, n_b: int = 4) -> dict:
    """A1: the obstacle generator is sequential (`for _ in range(n)`), so for a
    given seed the n_a-obstacle set is a prefix of the n_b-obstacle one. Asserted
    before any cell runs, because the n-sweep's reading depends on it.

    NOT asserted (and false): that the initial poses are paired across n. The
    obstacles are drawn from the episode RNG BEFORE the poses (pe2/env.py
    reset), so a different n shifts the pose draws. Only the delta and gamma
    sweeps, which hold n at 2, are seed-paired episode for episode.
    """
    from pe2.env import PursuitEvasionEnv

    def obstacles(n):
        env = PursuitEvasionEnv(
            dataclasses.replace(EnvConfig(), n_physical_obstacles=n)
        )
        env.reset(seed=seed)
        return [
            [float(o["x"]), float(o["y"]), float(o["radius"])] for o in env.obstacles
        ]

    a, b = obstacles(n_a), obstacles(n_b)
    return {
        "pass": bool(len(a) == n_a and len(b) == n_b and b[:n_a] == a),
        "seed": seed,
        f"obstacles_n{n_a}": a,
        f"obstacles_n{n_b}": b,
        "poses_paired_across_n": False,
    }


def run_cells(
    names: list[str] | None = None, device: str = "cpu", workers: int = MAX_WORKERS
) -> list[str]:
    """Run every named cell that has no record yet (default: all 132)."""
    wanted = list(CELLS) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}")
    nested = check_obstacles_nested()
    if not nested["pass"]:
        raise SystemExit(
            f"obstacle sets are NOT nested across n at seed {SEED0}: {nested}. "
            "The n-sweep's reading depends on it; the round may not proceed."
        )
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
# tests (spec section 3)
# ----------------------------------------------------------------------------
def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean of the ranks they span)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(y: list[float]) -> float:
    """Spearman rho between the LEVEL INDEX (0, 1, 2, ...) and y — the sign is
    all the spec reads. A constant y has no rank variance; it is reported as
    0.0 (no monotone trend), never NaN, so the verdict rules stay decidable."""
    n = len(y)
    if n < 2:
        return 0.0
    ry = _ranks(list(y))
    rx = _ranks([float(i) for i in range(n)])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return 0.0 if den == 0 else num / den


def _paired_top_vs_bottom(top: dict, bottom: dict) -> dict:
    """A2: the test IS "the top level deadlocks MORE than the bottom level",
    which in `pe2/derisk.py`'s convention is `paired_indicator_test(top, bottom)`
    — its FIRST argument is the side expected to be higher (PASS needs
    b = a-only > c = b-only). Used by the delta and gamma sweeps, which hold n at
    2 and are therefore seed-paired episode for episode."""
    return {
        "test": "paired_indicator",
        **paired_indicator_test(top["deadlock_ind"], bottom["deadlock_ind"], P_GATE),
    }


def _wilson_top_vs_bottom(top: dict, bottom: dict) -> dict:
    """A2: the n sweep is NOT seed-paired — the env draws the obstacles before
    the poses (`pe2/env.py` reset), so the RNG stream shifts with n and the
    episodes of two n-levels are different games. Its top-vs-bottom test is
    therefore Wilson separation at 95 %: the top level's lower bound above the
    bottom level's upper bound. Both intervals come from `pe2.eval.wilson_ci`
    (via `_ci`, stored in the record as `wilson_deadlock`)."""
    t, b = top["wilson_deadlock"], bottom["wilson_deadlock"]
    return {
        "test": "wilson_separation",
        "pass": bool(t["wilson"][0] > b["wilson"][1]),
        "rate_a": t["rate"],
        "rate_b": b["rate"],
        "wilson_a": t["wilson"],
        "wilson_b": b["wilson"],
        "n": t["n"],
    }


SWEEPS = {
    # levels: what the verdict reads; top/bottom: indices into `levels` for the
    # top-vs-bottom test (gamma runs conservative -> loose, so its "top" is
    # gamma = 0.1); test: paired where the levels share a seed stream (A2).
    "delta": {
        "levels": list(DELTAS),
        "settings": [(d, BASE_N, BASE_GAMMA) for d in DELTAS],
        "top": -1,
        "bottom": 0,
        "test": _paired_top_vs_bottom,
    },
    "n": {
        "levels": list(NS_CONFIRM),
        "settings": [(BASE_DELTA, n, BASE_GAMMA) for n in NS_CONFIRM],
        "top": -1,
        "bottom": 0,
        "test": _wilson_top_vs_bottom,
    },
    "gamma": {
        "levels": [0.4, 0.2, 0.1],
        "settings": [(BASE_DELTA, BASE_N, g) for g in (0.4, 0.2, 0.1)],
        "top": -1,
        "bottom": 0,
        "test": _paired_top_vs_bottom,
    },
}


def _cell_verdict(sweep: str, rates: list[float], rho: float, tvb: dict) -> str:
    if sweep == "delta":
        # rho > 0 AND the top level deadlocks more by the paired test.
        ok = rho > 0 and tvb["pass"]
    elif sweep == "n":
        # A1: lowest at n = 2 and non-decreasing over {2, 3, 4}. "Lowest" is read
        # strictly, so an all-equal (e.g. all-zero) sweep cannot confirm by tie.
        # A2: plus the unpaired top-vs-bottom test, Wilson separation of n = 4
        # from n = 2.
        ok = rates[0] < min(rates[1:]) and rho >= 0 and tvb["pass"]
    else:
        # gamma: the conservative end (0.1) must not deadlock LESS than the
        # loose end (0.4). Non-strict, exactly as the spec states it.
        ok = rates[-1] >= rates[0]
    return "CONFIRMED" if ok else "NOT CONFIRMED"


def _opponent_block(sweep: str, records: dict, pursuer: str, opponent: str) -> dict:
    cfgs = SWEEPS[sweep]
    names = [cell_name(pursuer, opponent, *s) for s in cfgs["settings"]]
    if any(n not in records for n in names):
        return {
            "verdict": "INCOMPLETE",
            "levels": cfgs["levels"],
            "missing": [n for n in names if n not in records],
        }
    cells = [records[n] for n in names]
    rates = [c["deadlock_rate"] for c in cells]
    rho = spearman(rates)
    tvb = cfgs["test"](cells[cfgs["top"]], cells[cfgs["bottom"]])
    block = {
        "levels": cfgs["levels"],
        "cells": names,
        "rates": rates,
        "wilson": [c["wilson_deadlock"]["wilson"] for c in cells],
        "rho": rho,
        "top_vs_bottom": tvb,
        "rate_top": rates[cfgs["top"]],
        "rate_bottom": rates[cfgs["bottom"]],
        "capture_rates": [c["capture_rate"] for c in cells],
        "stall_step_rates": [c["stall_step_rate"] for c in cells],
    }
    # Scripted baselines are reported in every sweep and never carry a verdict.
    block["verdict"] = (
        _cell_verdict(sweep, rates, rho, tvb) if opponent in STRUCTURED else None
    )
    return block


def _exploratory_n(records: dict) -> dict:
    """A1: n in {0, 1} run and reported, no verdict (zero-filled obstacle slots
    are an input the policies never saw in training)."""
    rates: dict = {}
    missing = []
    for p in PURSUERS:
        rates[p] = {}
        for o in OPPONENTS:
            names = [cell_name(p, o, BASE_DELTA, n, BASE_GAMMA) for n in NS_EXPLORATORY]
            missing += [n for n in names if n not in records]
            rates[p][o] = [
                records[n]["deadlock_rate"] if n in records else None for n in names
            ]
    return {
        "levels": list(NS_EXPLORATORY),
        "rates": rates,
        "missing": missing,
        "caveat": (
            "exploratory (A1): fewer obstacles than the observation's 2 reserved "
            "slots, which are then zero-filled; no verdict is derived"
        ),
    }


def _sweep_block(sweep: str, records: dict) -> dict:
    per_pursuer = {}
    for p in PURSUERS:
        opponents = {o: _opponent_block(sweep, records, p, o) for o in OPPONENTS}
        structured = [opponents[o]["verdict"] for o in STRUCTURED]
        if "INCOMPLETE" in structured:
            verdict = "INCOMPLETE"
        elif all(v == "CONFIRMED" for v in structured):
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"
        per_pursuer[p] = {"verdict": verdict, "opponents": opponents}
    n_conf = sum(v["verdict"] == "CONFIRMED" for v in per_pursuer.values())
    if any(v["verdict"] == "INCOMPLETE" for v in per_pursuer.values()):
        overall = "INCOMPLETE"
    else:
        overall = "CONFIRMED" if n_conf >= 2 else "NOT CONFIRMED"
    block = {
        "hypothesis": f"H-GEO-{sweep}",
        "levels": SWEEPS[sweep]["levels"],
        "verdict": overall,
        "n_confirmed": n_conf,
        "pursuers": per_pursuer,
    }
    if sweep == "n":
        block["exploratory"] = _exploratory_n(records)
        block["note"] = (
            "A2: levels differ in obstacle count, so episodes are NOT pose-paired "
            "across n (obstacles are drawn before the poses); top_vs_bottom is "
            "Wilson separation of n = 4 from n = 2 at 95 %, not a paired test"
        )
    return block


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
        "min_clearance_floor": min(
            (
                r["min_clearance_floor"]
                for r in records.values()
                if r.get("min_clearance_floor") is not None
            ),
            default=None,
        ),
    }


def verdicts(records: dict) -> dict:
    missing = [n for n in CELLS if n not in records]
    return {
        "sweeps": {s: _sweep_block(s, records) for s in SWEEPS},
        "hsafe": hsafe(records),
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

    colors = dict(zip(PURSUERS, ("tab:blue", "tab:orange", "tab:green"), strict=True))
    xs_by_sweep = {"delta": list(DELTAS), "n": list(NS), "gamma": list(GAMMAS)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, sweep in zip(axes, ("delta", "n", "gamma"), strict=True):
        levels = xs_by_sweep[sweep]
        for p in PURSUERS:
            for o in OPPONENTS:
                settings = [
                    {
                        "delta": (lv, BASE_N, BASE_GAMMA),
                        "n": (BASE_DELTA, lv, BASE_GAMMA),
                        "gamma": (BASE_DELTA, BASE_N, lv),
                    }[sweep]
                    for lv in levels
                ]
                names = [cell_name(p, o, *s) for s in settings]
                xs = [lv for lv, n in zip(levels, names, strict=True) if n in records]
                if not xs:
                    continue
                cells = [records[n] for n in names if n in records]
                ys = [c["deadlock_rate"] for c in cells]
                lo = [
                    y - c["wilson_deadlock"]["wilson"][0]
                    for y, c in zip(ys, cells, strict=True)
                ]
                hi = [
                    c["wilson_deadlock"]["wilson"][1] - y
                    for y, c in zip(ys, cells, strict=True)
                ]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    marker="o" if o in STRUCTURED else "s",
                    ms=4,
                    lw=1.5,
                    capsize=2,
                    color=colors[p],
                    ls="-" if o in STRUCTURED else "--",
                    alpha=1.0 if o in STRUCTURED else 0.55,
                    label=f"{p} vs {o}",
                )
        ax.set_xlabel(
            {
                "delta": "shield margin delta (m)",
                "n": "obstacles placed",
                "gamma": "class-K gain gamma",
            }[sweep]
        )
        ax.set_ylabel(f"deadlock episode rate (N={N_EPISODES}, seeds {SEED0}+)")
        ax.set_title(f"H-GEO-{sweep}")
        ax.set_ylim(-0.02, 1.02)
        # Axes span the whole sweep even when only part of it has run, so a
        # figure drawn over a partial round is read on the same scale.
        pad = 0.06 * (max(levels) - min(levels))
        ax.set_xlim(min(levels) - pad, max(levels) + pad)
        ax.set_xticks(levels)
    axes[1].axvspan(-0.3, 1.3, color="0.9", zorder=0)
    axes[1].text(0.5, 0.98, "exploratory (A1)", ha="center", va="top", fontsize=7)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "S9 — attack surface vs shield margin, obstacle count and class-K gain "
        "(solid = structured attacker, dashed = scripted baseline)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize() -> dict:
    records = load_records()
    blob = {
        "verdicts": verdicts(records),
        "obstacles_nested": check_obstacles_nested(),
        "cells": {
            n: {k: v for k, v in r.items() if k not in ("deadlock_ind", "capture_ind")}
            for n, r in records.items()
        },
    }
    _write_json(OUT / "s9_round.json", blob)
    write_figure(records, OUT / "s9_round.png")
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)  # spec section 5
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cells is not None:
        run_cells(args.cells or None, args.device, args.workers)
    if args.summarize:
        blob = summarize()
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    s: {
                        "verdict": b["verdict"],
                        "n_confirmed": b["n_confirmed"],
                        "pursuers": {
                            p: pb["verdict"] for p, pb in b["pursuers"].items()
                        },
                    }
                    for s, b in v["sweeps"].items()
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
