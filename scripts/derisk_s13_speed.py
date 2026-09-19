"""S13 — does the stall attack depend on the evader's speed advantage?
(spec: docs/pathway/speed_ratio_prereg_2026-09-19.md).

No training. The four frozen shielded pursuers S9/S10/S11 measured are
re-evaluated against the same two learned attackers and two scripted probes
while ONE knob moves — the evader's speed cap:

  evader_speed_ratio  {0.8, 1.0, 1.2, 1.4}   delta 0, gamma 0.2, n 2

0.8 is the regime that has never been run: the pursuer is 25 % FASTER than its
evader. 4 pursuers x 4 opponents x 4 ratios = 64 cells, every one on the same
300-seed stream from SEED0.

The transfer caveat (spec section 2) shapes the verdicts: every policy here was
trained at ratio 1.2, so every other ratio is out-of-distribution transfer and a
low deadlock rate at 0.8 could mean "the attack needs a speed advantage" OR
"this attacker failed to transfer". H-SPD-1 therefore always compares against
the SCRIPTED probes measured at the SAME ratio — those recompute their commands
from the geometry and `cfg.evader_v_max` every step and so are speed-agnostic by
construction. Nothing here licenses a claim about an attacker TRAINED at another
ratio.

This module is S9's machinery under S13's constants: it loads
`scripts/derisk_s9_sweep.py` privately (importlib, NOT sys.modules) and reuses
its record IO, its H-SAFE rule and its statistics, re-pointing only the three
globals those helpers read (`CELLS`, `cells_dir`, `OPPONENTS` — S13's attacker
is ATK4-45, S9's was ATK2-31). S9's own script and its artefacts on disk are
untouched.

What is NOT reused: `run_cell` (the config differs per cell) and the pool loop
(its gate is S13's ratio-pairing check, not S9's nested-obstacle check, and its
worker must live in THIS module so `spawn` re-imports S13's world, not S9's).

Usage:
  ./venv/bin/python scripts/derisk_s13_speed.py --cells            (all missing)
  ./venv/bin/python scripts/derisk_s13_speed.py --cells RC19__ATT25__r0.8
  ./venv/bin/python scripts/derisk_s13_speed.py --summarize

Stop rule (spec section 5): a cell that already has a record is NEVER re-run;
no ratio, pursuer or opponent is added or removed after any number is seen.

`OUT` is the only output root and `--workers 1` is the only mode in which
reassigning it (a smoke) reaches the code that writes: spawned workers re-import
this module and get the module-level default.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import multiprocessing
import os
import pathlib
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.eval import win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
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
ARTEFACT_PREFIX = "s13"

# --- pre-registered (spec sections 3-4). Do not tune. ---
SEED0 = 1_360_000
N_EPISODES = 300
P_GATE = S9.P_GATE  # 0.05
EQ_MARGIN = 0.10  # H-SPD-2's equivalence margin on the paired difference
Z95 = 1.96  # the 95 % Wald multiplier (no scipy)
MAX_WORKERS = 6
OMP_THREADS = S9.OMP_THREADS  # "2"
RATIOS = (0.8, 1.0, 1.2, 1.4)  # 0.8 = the pursuer is 25 % faster
SPD2_RATIOS = (0.8, 1.2)  # the pair H-SPD-2 reads (1.2 = the training ratio)
CONFIRM_MIN = 3  # a verdict needs >= 3 of the 4 pursuers
DELTA = 0.0
N_OBSTACLES = 2
GAMMA = 0.2
SPD3_OPPONENT = "GREEDY"  # H-SPD-3 reads capture against GreedyEvader
SATURATION = 0.95  # capture above this at 0.8 makes that column a ceiling

PURSUERS = {
    **S9.PURSUERS,  # RC19, RCM28, STR14 — same checkpoints, same shield_obs
    "FT-B-41": {
        "ckpt": CKPT_ROOT / "ckpts_FT-B-41" / "FT-B-41_5000000.zip",
        "shield_obs": True,  # S10 artefact, RC configuration (D120)
    },
}
SCORING = tuple(PURSUERS)  # all four vote (spec section 3: >= 3 of 4)
OPPONENTS = {
    "ATT25": {"ckpt": CKPT_ROOT / "ckpts_CTL-D225" / "CTL-D225_10000000.zip"},
    "ATK4-45": {"ckpt": CKPT_ROOT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"},
    "HUG7": {"scripted": ObstacleHuggingEvader},
    "GREEDY": {"scripted": GreedyEvader},
}
STRUCTURED = ("ATT25", "ATK4-45")  # the two learned attackers a verdict reads
SCRIPTED = ("HUG7", "GREEDY")  # the speed-agnostic probes (spec section 2)


def cell_name(pursuer: str, opponent: str, ratio: float) -> str:
    return f"{pursuer}__{opponent}__r{ratio:.1f}"


def _build_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for p in PURSUERS:
        for o in OPPONENTS:
            for r in RATIOS:
                name = cell_name(p, o, r)
                cells[name] = {
                    "name": name,
                    "pursuer": p,
                    "opponent": o,
                    "ratio": r,
                    "delta": DELTA,
                    "n": N_OBSTACLES,
                    "gamma": GAMMA,
                }
    return cells


CELLS = _build_cells()


def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


# S9's record IO and H-SAFE rule read `CELLS` and `cells_dir`, and its opponent
# factory reads `OPPONENTS`; re-pointing the three makes every borrowed helper
# operate on S13's cells, S13's OUT (read at call time, so a test that moves OUT
# still works) and S13's attacker roster.
S9.CELLS = CELLS
S9.cells_dir = cells_dir
S9.OPPONENTS = OPPONENTS

_ci = S9._ci
_model = S9._model
_write_json = S9._write_json
opponent_factory = S9.opponent_factory
provenance = S9.provenance
spearman = S9.spearman
write_cell = S9.write_cell
load_records = S9.load_records
hsafe = S9.hsafe


def _rkey(ratio: float) -> str:
    return f"r{ratio:.1f}"


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def cell_config(ratio: float) -> EnvConfig:
    """The cell's environment: the locked config with ONE field moved. The cap
    itself is `cfg.evader_v_max = ratio * v_max` (a property), so every consumer
    — the env's integrator, the scripted probes and the structured attacker's
    action scaling — picks the ratio up from the config alone. The pursuer's
    `v_max` never moves."""
    return dataclasses.replace(
        EnvConfig(), evader_speed_ratio=ratio, n_physical_obstacles=N_OBSTACLES
    )


def run_cell(name: str, device: str = "cpu") -> dict:
    """One cell: the frozen pursuer under the CBF shield at the base setting
    (delta 0, gamma 0.2, n 2), against this cell's opponent, in an arena whose
    evader is capped at this cell's ratio, over the pre-registered stream."""
    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    spec = CELLS[name]
    cfg_cell = cell_config(spec["ratio"])
    pursuer = PURSUERS[spec["pursuer"]]
    model = _model(pursuer["ckpt"], device)
    logs = evaluate_structured(
        model,
        cfg_cell,
        opponent_factory(spec["opponent"], cfg_cell, device),
        N_EPISODES,
        SEED0,
        hold=HOLD,
        shield_factory=lambda: CBFShield(cfg_cell, gamma=GAMMA, margin=DELTA),
        shield_obs=pursuer["shield_obs"],
    )
    sh = summarize_shield(logs)
    k, n_ep = win_count(logs, "pursuer")
    deadlock_ind = [bool(lg["shield"]["deadlock"]) for lg in logs]
    capture_ind = [lg["outcome"] == "capture" for lg in logs]
    outcomes = Counter(lg["outcome"] for lg in logs)
    return {
        "name": name,
        "pursuer": spec["pursuer"],
        "opponent": spec["opponent"],
        "ratio": spec["ratio"],
        "evader_v_max": cfg_cell.evader_v_max,
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "shield_obs": pursuer["shield_obs"],
        "n_episodes": n_ep,
        "seed0": SEED0,
        "deadlock_ind": deadlock_ind,
        "capture_ind": capture_ind,
        "deadlock_rate": sh["deadlock_episode_rate"],
        "capture_rate": k / n_ep,
        # The env terminates on capture and truncates on timeout and nothing
        # else (pe2/env.py OUTCOME_*), so the evader "escapes" exactly when the
        # episode times out.
        "escape_rate": outcomes["timeout"] / n_ep,
        "outcomes": dict(outcomes),
        "stall_step_rate": sh["stall_step_rate"],
        "intervention_rate": sh["intervention_rate"],
        "violations": int(sh["violation_steps"]),
        "min_clearance_floor": sh["min_clearance_floor"],
        "min_clearance_hard_floor": sh["min_clearance_hard_floor"],
        "wilson_deadlock": _ci(sum(deadlock_ind), n_ep),
        "wilson_capture": _ci(sum(capture_ind), n_ep),
        "provenance": provenance(),
    }


def _cell_line(rec: dict) -> str:
    return (
        f"[{rec['name']}] deadlock {rec['deadlock_rate']:.3f} "
        f"capture {rec['capture_rate']:.3f} escape {rec['escape_rate']:.3f} "
        f"stall {rec['stall_step_rate']:.3f} int {rec['intervention_rate']:.3f} "
        f"viol {rec['violations']}"
    )


def _run_and_write(name: str, device: str) -> dict:
    rec = run_cell(name, device)
    write_cell(rec)
    return rec


def _worker(args: tuple) -> str:
    name, device = args
    return _cell_line(_run_and_write(name, device))


def check_ratio_pairing(seed: int = SEED0, n: int = 3) -> dict:
    """The ratio must move no RNG draw: `pe2/env.py` reset generates the
    obstacles and then both poses, and `evader_speed_ratio` enters nowhere in
    that path, so episode i is the SAME game at every ratio.

    Asserted before any cell runs because H-SPD-2 is a PAIRED test ACROSS two
    ratios (S9's `check_obstacles_nested` is the same gate for the same reason:
    there the n-sweep broke the pairing and the test had to change)."""
    from pe2.env import PursuitEvasionEnv

    def draws(ratio: float) -> list[dict]:
        env = PursuitEvasionEnv(cell_config(ratio))
        out = []
        for i in range(n):
            env.reset(seed=seed + i)
            out.append(
                {
                    "obstacles": [
                        [float(o["x"]), float(o["y"]), float(o["radius"])]
                        for o in env.obstacles
                    ],
                    "pursuer": [float(x) for x in env.pursuer_pose],
                    "evader": [float(x) for x in env.evader_pose],
                }
            )
        return out

    ref = draws(RATIOS[0])
    identical = {_rkey(r): draws(r) == ref for r in RATIOS}
    return {
        "pass": all(identical.values()),
        "ratios": list(RATIOS),
        "reference_ratio": RATIOS[0],
        "seed": seed,
        "episodes_checked": n,
        "identical_layouts": identical,
    }


def run_cells(
    names: list[str] | None = None, device: str = "cpu", workers: int = MAX_WORKERS
) -> list[str]:
    """Run every named cell that has no record yet (default: all 64)."""
    wanted = list(CELLS) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}")
    paired = check_ratio_pairing()
    if not paired["pass"]:
        raise SystemExit(
            f"the ratios are NOT seed-paired at seed {SEED0}: {paired}. "
            "H-SPD-2 is a paired test across ratio 0.8 and ratio 1.2; the round "
            "may not proceed."
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


# ----------------------------------------------------------------------------
# tests (spec section 4)
# ----------------------------------------------------------------------------
def _overall(per_pursuer: dict) -> tuple[str, int]:
    """The shared rule: INCOMPLETE if any pursuer is, else CONFIRMED iff at
    least CONFIRM_MIN of the 4 confirm."""
    scored = [per_pursuer[p]["verdict"] for p in SCORING]
    n_conf = sum(v == "CONFIRMED" for v in scored)
    if "INCOMPLETE" in scored:
        return "INCOMPLETE", n_conf
    return ("CONFIRMED" if n_conf >= CONFIRM_MIN else "NOT CONFIRMED"), n_conf


def _spd1_ratio(records: dict, pursuer: str, ratio: float) -> dict:
    """H-SPD-1 at one (pursuer, ratio): both attackers, paired, against the
    STRONGER of the two scripted probes AT THAT SAME RATIO (spec section 2 —
    the probes are speed-agnostic, so they are the only comparator that
    separates 'the attack needs a speed advantage' from 'this attacker did not
    transfer')."""
    names = {o: cell_name(pursuer, o, ratio) for o in OPPONENTS}
    missing = [n for n in names.values() if n not in records]
    if missing:
        return {"verdict": "INCOMPLETE", "ratio": ratio, "missing": missing}
    best = max(SCRIPTED, key=lambda o: records[names[o]]["deadlock_rate"])
    probe = records[names[best]]
    attackers = {
        a: {
            "cell": names[a],
            "test": "paired_indicator",
            **paired_indicator_test(
                records[names[a]]["deadlock_ind"], probe["deadlock_ind"], P_GATE
            ),
        }
        for a in STRUCTURED
    }
    return {
        "verdict": (
            "CONFIRMED"
            if all(t["pass"] for t in attackers.values())
            else "NOT CONFIRMED"
        ),
        "ratio": ratio,
        "best_scripted_opponent": best,
        "best_scripted_cell": names[best],
        "scripted_rates": {o: records[names[o]]["deadlock_rate"] for o in SCRIPTED},
        "attackers": attackers,
    }


def spd1_verdicts(records: dict) -> dict:
    """H-SPD-1 (survival): at EVERY ratio, including the ones where the pursuer
    is faster, both structured attackers deadlock the pursuer more than the
    strongest scripted probe at that ratio. A pursuer confirms iff that holds at
    all four ratios; the round iff >= 3 of the 4 pursuers confirm."""
    per_pursuer = {}
    for p in PURSUERS:
        ratios = {_rkey(r): _spd1_ratio(records, p, r) for r in RATIOS}
        vs = [b["verdict"] for b in ratios.values()]
        if "INCOMPLETE" in vs:
            verdict = "INCOMPLETE"
        elif all(v == "CONFIRMED" for v in vs):
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"
        per_pursuer[p] = {"verdict": verdict, "ratios": ratios}
    overall, n_conf = _overall(per_pursuer)
    # Per-ratio tallies, so `direction` can tell "fails at 1.2" (a harness
    # failure: 1.2 is the training ratio and reproduces S9/S10/S11) from
    # "fails at 0.8 only" (a scope statement) without recomputing anything.
    by_ratio, holding = {}, {}
    for r in RATIOS:
        blocks = [per_pursuer[p]["ratios"][_rkey(r)] for p in SCORING]
        n = sum(b["verdict"] == "CONFIRMED" for b in blocks)
        by_ratio[_rkey(r)] = n
        holding[_rkey(r)] = (
            None
            if any(b["verdict"] == "INCOMPLETE" for b in blocks)
            else n >= CONFIRM_MIN
        )
    return {
        "hypothesis": "H-SPD-1",
        "ratios": list(RATIOS),
        "verdict": overall,
        "n_confirmed": n_conf,
        "scoring": list(SCORING),
        "n_confirmed_by_ratio": by_ratio,
        "ratios_holding": holding,
        "pursuers": per_pursuer,
        "comparator": (
            "the stronger scripted probe at the SAME ratio (spec section 2: the "
            "scripted probes recompute their commands from cfg.evader_v_max "
            "every step and are speed-agnostic by construction)"
        ),
    }


def wald_equivalence(
    ind_a: list[bool], ind_b: list[bool], margin: float = EQ_MARGIN
) -> dict:
    """H-SPD-2's test (spec section 4): on the paired per-episode indicators,
    with b = a-only and c = b-only discordant counts and n the episode count,

        d  = (b - c) / n
        hw = 1.96 * sqrt(b + c - (b - c)^2 / n) / n

    and the claim is EQUIVALENCE, so it CONFIRMS iff the whole 95 % interval
    [d - hw, d + hw] lies inside [-margin, +margin]. A wide interval therefore
    fails, exactly as it should: absence of a detected difference is not
    evidence of no difference."""
    if len(ind_a) != len(ind_b):
        raise ValueError("paired indicators must have equal length")
    n = len(ind_a)
    b = sum(1 for x, y in zip(ind_a, ind_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(ind_a, ind_b, strict=True) if y and not x)
    d = (b - c) / n
    hw = Z95 * math.sqrt(b + c - (b - c) ** 2 / n) / n
    return {
        "test": "paired_wald_equivalence",
        "pass": bool(d - hw >= -margin and d + hw <= margin),
        "d": d,
        "half_width": hw,
        "interval": [d - hw, d + hw],
        "margin": margin,
        "b": b,
        "c": c,
        "n": n,
        "rate_a": sum(map(bool, ind_a)) / n,
        "rate_b": sum(map(bool, ind_b)) / n,
    }


def _spd2_attacker(records: dict, pursuer: str, attacker: str) -> dict:
    lo, hi = (cell_name(pursuer, attacker, r) for r in SPD2_RATIOS)
    missing = [n for n in (lo, hi) if n not in records]
    if missing:
        return {"verdict": "INCOMPLETE", "cells": [lo, hi], "missing": missing}
    out = wald_equivalence(
        records[lo]["deadlock_ind"], records[hi]["deadlock_ind"], EQ_MARGIN
    )
    return {
        "verdict": "CONFIRMED" if out["pass"] else "NOT CONFIRMED",
        "cells": [lo, hi],
        "ratios": list(SPD2_RATIOS),
        **out,
    }


def spd2_verdicts(records: dict) -> dict:
    """H-SPD-2 (no speed dependence): for each pursuer and each structured
    attacker, the paired difference in deadlock rate between ratio 0.8 and ratio
    1.2 is within +/- 0.10. A pursuer confirms iff both attackers do; the round
    iff >= 3 of the 4 pursuers confirm."""
    per_pursuer = {}
    for p in PURSUERS:
        attackers = {a: _spd2_attacker(records, p, a) for a in STRUCTURED}
        vs = [b["verdict"] for b in attackers.values()]
        if "INCOMPLETE" in vs:
            verdict = "INCOMPLETE"
        elif all(v == "CONFIRMED" for v in vs):
            verdict = "CONFIRMED"
        else:
            verdict = "NOT CONFIRMED"
        per_pursuer[p] = {"verdict": verdict, "attackers": attackers}
    overall, n_conf = _overall(per_pursuer)
    return {
        "hypothesis": "H-SPD-2",
        "ratios": list(SPD2_RATIOS),
        "margin": EQ_MARGIN,
        "verdict": overall,
        "n_confirmed": n_conf,
        "scoring": list(SCORING),
        "pursuers": per_pursuer,
    }


def spd3_report(records: dict) -> dict:
    """H-SPD-3 (resolution): capture against GreedyEvader across the ratio.
    Reported, no verdict. If capture at ratio 0.8 exceeds SATURATION for EVERY
    pursuer the metric has saturated and that column is a ceiling."""
    rates: dict = {}
    missing = []
    for p in PURSUERS:
        rates[p] = {}
        for r in RATIOS:
            name = cell_name(p, SPD3_OPPONENT, r)
            if name in records:
                rates[p][_rkey(r)] = records[name]["capture_rate"]
            else:
                rates[p][_rkey(r)] = None
                missing.append(name)
    base = _rkey(RATIOS[0])
    at_base = {p: rates[p][base] for p in PURSUERS}
    return {
        "hypothesis": "H-SPD-3",
        "opponent": SPD3_OPPONENT,
        "ratios": list(RATIOS),
        "capture_rates": rates,
        f"capture_at_{RATIOS[0]:.1f}": at_base,
        # rho against the LEVEL INDEX, i.e. against a faster evader; capture is
        # expected to FALL, so a negative rho is the predicted direction.
        "rho_capture_vs_ratio": {
            p: (
                None
                if any(v is None for v in rates[p].values())
                else spearman([rates[p][_rkey(r)] for r in RATIOS])
            )
            for p in PURSUERS
        },
        "wilson": {
            p: {
                _rkey(r): (
                    records[cell_name(p, SPD3_OPPONENT, r)]["wilson_capture"]["wilson"]
                    if cell_name(p, SPD3_OPPONENT, r) in records
                    else None
                )
                for r in RATIOS
            }
            for p in PURSUERS
        },
        "saturated": (
            None if missing else bool(all(v > SATURATION for v in at_base.values()))
        ),
        "saturation_threshold": SATURATION,
        "missing": missing,
        "note": (
            "reported, no verdict; a saturated 0.8 column is a ceiling, not a "
            "measured capture rate"
        ),
    }


def verdicts(records: dict) -> dict:
    missing = [n for n in CELLS if n not in records]
    return {
        "spd1": spd1_verdicts(records),
        "spd2": spd2_verdicts(records),
        "spd3": spd3_report(records),
        "hsafe": hsafe(records),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "ratios": list(RATIOS),
        "seed0": SEED0,
        "n_episodes": N_EPISODES,
    }


def direction(v: dict) -> str:
    """Spec section 5, one row per outcome. Read in the spec's own order of
    precedence: H-SAFE first (a contact step stops the round), then
    completeness, then the harness row (ratio 1.2 is the training ratio and
    reproduces S9/S10/S11 — if the attack is not there, nothing else is read)."""
    spd1, spd2, hsafe_ = v["spd1"], v["spd2"], v["hsafe"]
    if hsafe_["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {hsafe_['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(hsafe_['offenders'])}. Stop — the shield's guarantee is "
            "the precondition of the whole round and nothing else is read."
        )
    if "INCOMPLETE" in (spd1["verdict"], spd2["verdict"], hsafe_["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record; every pursuer x opponent x ratio must run before a verdict "
            "is read (spec section 5)."
        )
    holding = spd1["ratios_holding"]
    failed = [r for r in RATIOS if not holding[_rkey(r)]]
    if not holding[_rkey(1.2)]:
        return (
            "H-SPD-1 fails at ratio 1.2 — the ratio every policy was TRAINED at "
            "and the one that reproduces S9/S10/S11. This is a harness failure, "
            "not a result: stop, diagnose the runner against the S9 base cells, "
            "and report nothing from this round."
        )
    if spd1["verdict"] == "CONFIRMED" and spd2["verdict"] == "CONFIRMED":
        return (
            "H-SPD-1 and H-SPD-2 CONFIRMED: the stall surface does not come from "
            "the speed asymmetry — a pursuer that can outrun its evader is pinned "
            "just as often, and the 0.8-vs-1.2 difference is inside +/- "
            f"{EQ_MARGIN:.2f}. The reviewer objection ('it only gets pinned "
            "because it cannot run the evader down') is closed and the geometric "
            "reading (D118/D121) is strengthened."
        )
    if spd1["verdict"] == "CONFIRMED":
        return (
            "H-SPD-1 CONFIRMED, H-SPD-2 NOT CONFIRMED "
            f"({spd2['n_confirmed']}/{len(SCORING)} pursuers): the attack survives "
            "at every ratio but its strength varies with the speed ratio. Report "
            "the curve and drop the 'independent of speed' claim; the survival "
            "statement is the one that carries."
        )
    if failed == [0.8]:
        return (
            "H-SPD-1 fails at ratio 0.8 only: the attack needs an evader that "
            "cannot be run down. The paper's scope narrows explicitly to that "
            "regime — which is the interesting one for pursuit anyway — and the "
            "boundary is stated. Spec section 2 still applies: no claim is made "
            "about an attacker TRAINED at 0.8."
        )
    return (
        "H-SPD-1 fails at ratio(s) "
        f"{', '.join(f'{r:.1f}' for r in failed)} "
        f"({spd1['n_confirmed']}/{len(SCORING)} pursuers confirm at all four). "
        "The attack does not survive across the swept range; report the surviving "
        "ratios as the measured scope and read H-SPD-2 only inside it."
    )


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_COLORS = {
    "RC19": "tab:blue",
    "RCM28": "tab:orange",
    "STR14": "tab:green",
    "FT-B-41": "tab:red",
}


def _yerr(cells: list[dict], ys: list[float], key: str) -> list[list[float]]:
    lo = [max(0.0, y - c[key]["wilson"][0]) for y, c in zip(ys, cells, strict=True)]
    hi = [max(0.0, c[key]["wilson"][1] - y) for y, c in zip(ys, cells, strict=True)]
    return [lo, hi]


def _panel_spd1(ax, records: dict) -> None:
    """Deadlock rate over the ratio, one line per pursuer x opponent; solid =
    learned attacker, dashed = the speed-agnostic scripted probe it must beat
    at that same ratio."""
    xs = np.arange(len(RATIOS))
    drew = False
    for p in PURSUERS:
        for o in OPPONENTS:
            have = [
                (i, cell_name(p, o, r))
                for i, r in enumerate(RATIOS)
                if cell_name(p, o, r) in records
            ]
            if not have:
                continue
            drew = True
            cells = [records[n] for _, n in have]
            ys = [c["deadlock_rate"] for c in cells]
            ax.errorbar(
                [i for i, _ in have],
                ys,
                yerr=_yerr(cells, ys, "wilson_deadlock"),
                marker="o" if o in STRUCTURED else "s",
                ms=4,
                lw=1.5,
                capsize=2,
                color=_COLORS[p],
                ls="-" if o in STRUCTURED else "--",
                alpha=1.0 if o in STRUCTURED else 0.45,
                label=f"{p} vs {o}",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r:.1f}" for r in RATIOS])
    ax.set_xlim(-0.3, len(RATIOS) - 0.7)
    ax.axvline(RATIOS.index(1.2), color="0.7", lw=1.0, ls=":", zorder=0)
    ax.set_xlabel("evader speed ratio (1.2 = the training ratio; 0.8 = pursuer faster)")
    ax.set_ylabel(f"deadlock episode rate (N={N_EPISODES}, seeds {SEED0}+)")
    ax.set_title("H-SPD-1: does the attack survive when the pursuer is faster?")
    ax.set_ylim(-0.02, 1.02)
    if drew:
        ax.legend(fontsize=5, ncol=2)


def _panel_spd2(ax, records: dict) -> None:
    """The paired 0.8 - 1.2 difference with its 95 % Wald interval, against the
    +/- 0.10 equivalence band."""
    labels, points = [], []
    blocks = spd2_verdicts(records)["pursuers"]
    for p in PURSUERS:
        for a in STRUCTURED:
            block = blocks[p]["attackers"][a]
            if block["verdict"] == "INCOMPLETE":
                continue
            labels.append(f"{p}\n{a}")
            points.append((p, block))
    if not points:
        ax.text(0.5, 0.5, "no complete 0.8/1.2 pair yet", ha="center")
        return
    xs = np.arange(len(points))
    ax.axhspan(-EQ_MARGIN, EQ_MARGIN, color="0.85", zorder=0)
    ax.axhline(0.0, color="0.4", lw=0.8, zorder=1)
    for x, (p, block) in zip(xs, points, strict=True):
        ax.errorbar(
            [x],
            [block["d"]],
            yerr=[[block["half_width"]], [block["half_width"]]],
            marker="o",
            ms=5,
            capsize=3,
            lw=1.5,
            color=_COLORS[p],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("paired deadlock difference, ratio 0.8 - ratio 1.2")
    ax.set_title(
        f"H-SPD-2: equivalence at +/- {EQ_MARGIN:.2f} (95 % Wald on discordant pairs)"
    )


def _panel_spd3(ax, records: dict) -> None:
    """Capture against GreedyEvader over the ratio, with the saturation line."""
    xs = np.arange(len(RATIOS))
    drew = False
    for p in PURSUERS:
        have = [
            (i, cell_name(p, SPD3_OPPONENT, r))
            for i, r in enumerate(RATIOS)
            if cell_name(p, SPD3_OPPONENT, r) in records
        ]
        if not have:
            continue
        drew = True
        cells = [records[n] for _, n in have]
        ys = [c["capture_rate"] for c in cells]
        ax.errorbar(
            [i for i, _ in have],
            ys,
            yerr=_yerr(cells, ys, "wilson_capture"),
            marker="o",
            ms=4,
            lw=1.5,
            capsize=2,
            color=_COLORS[p],
            label=p,
        )
    ax.axhline(SATURATION, color="0.4", lw=1.0, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r:.1f}" for r in RATIOS])
    ax.set_xlim(-0.3, len(RATIOS) - 0.7)
    ax.set_xlabel("evader speed ratio")
    ax.set_ylabel(f"capture rate vs {SPD3_OPPONENT}")
    ax.set_title(f"H-SPD-3: resolution (ceiling above {SATURATION:.2f})")
    ax.set_ylim(-0.02, 1.02)
    if drew:
        ax.legend(fontsize=7)


def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    _panel_spd1(axes[0], records)
    _panel_spd2(axes[1], records)
    _panel_spd3(axes[2], records)
    fig.suptitle(
        "S13 — the stall surface across the evader speed ratio (CBF shield, "
        "delta = 0, gamma = 0.2, n = 2); every policy was TRAINED at 1.2, so "
        "every other column is transfer and the comparator is always the "
        "scripted probe at the SAME ratio"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize() -> dict:
    records = load_records()
    v = verdicts(records)
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "cells": {
            n: {
                k: val
                for k, val in r.items()
                if k not in ("deadlock_ind", "capture_ind")
            }
            for n, r in records.items()
        },
    }
    _write_json(OUT / f"{ARTEFACT_PREFIX}_round.json", blob)
    write_figure(records, OUT / f"{ARTEFACT_PREFIX}_round.png")
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)  # spec section 6
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cells is not None:
        run_cells(args.cells or None, args.device, args.workers)
    if args.summarize:
        blob = summarize()
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    h: {"verdict": v[h]["verdict"], "n_confirmed": v[h]["n_confirmed"]}
                    for h in ("spd1", "spd2")
                },
                indent=2,
            )
        )
        print(json.dumps(v["hsafe"], indent=2))
        print(
            json.dumps(
                {
                    "saturated": v["spd3"]["saturated"],
                    f"capture_at_{RATIOS[0]:.1f}": v["spd3"][
                        f"capture_at_{RATIOS[0]:.1f}"
                    ],
                },
                indent=2,
            )
        )
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
