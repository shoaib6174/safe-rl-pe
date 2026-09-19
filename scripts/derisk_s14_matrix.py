"""S14 — the empirical game matrix: every defender against every attacker
(spec: docs/pathway/game_matrix_prereg_2026-09-19.md).

No training. The six shielded pursuers of the whole D112-D123 chain are
re-evaluated against the seven evaders of that chain ON ONE STREAM, so every
defender gets a worst-case exploitability measured against the same opponents
rather than a best-guess one assembled from different rounds:

  defenders (rows)  TEACHER, STR14, RC19, RCM28, FT-B-41, SEL-A-50
  attackers (cols)  CTL-D125, CTL-D225, ATK2-31, ATK4-45, SPLITHUG, HUG7, GREEDY

6 x 7 = 42 cells, delta 0 / gamma 0.2 / n 2, every one on the same 300-seed
stream from SEED0, so the whole matrix is seed-paired cell to cell.

The row that is NOT a checkpoint: `TEACHER` is the scripted pursuer. It is
driven exactly as S4 drove its TEACHER cells — `pe2.structured.OrbitSearch`
inside SearchMDP at hold 1, which is `pe2.scripted.SearchingPursuer`
episode-for-episode (spec section 2 validity check V1, pinned by
`tests/pe2/test_structured.py::test_v1_orbit_at_hold_1_reproduces_the_teacher
_episode_for_episode`). It was never trained, so it carries no `shield_obs`
flag at all; the shield filters its commands like anyone else's.

The matrix is not square in the best-response sense (spec section 2): four
defenders have an attacker trained against them, TEACHER and SEL-A-50 do not.
Their rows are transfer-only and `TRANSFER_ONLY` labels them in every record,
verdict and figure.

This module is S9's machinery under S14's constants: it loads
`scripts/derisk_s9_sweep.py` privately (importlib, NOT sys.modules) and reuses
its record IO, its H-SAFE rule and its Wilson helper, re-pointing only the two
globals those helpers read (`CELLS`, `cells_dir`). S9's own script and its
artefacts on disk are untouched.

What is NOT reused: `run_cell` (the defender may be scripted, and the hold is
per defender), `opponent_factory` (S14 adds SPLITHUG) and the pool loop (its
worker must live in THIS module so `spawn` re-imports S14's world, not S9's).

Usage:
  ./venv/bin/python scripts/derisk_s14_matrix.py --cells            (all missing)
  ./venv/bin/python scripts/derisk_s14_matrix.py --cells TEACHER__HUG7
  ./venv/bin/python scripts/derisk_s14_matrix.py --summarize

Stop rule (spec section 5): a cell that already has a record is NEVER re-run;
no defender, attacker or metric is added or removed after any number is seen.

`OUT` is the only output root and `--workers 1` is the only mode in which
reassigning it (a smoke) reaches the code that writes: spawned workers re-import
this module and get the module-level default.
"""

from __future__ import annotations

import os

OMP_THREADS = "2"  # spec section 6
# Set BEFORE numpy / torch are imported, for S13's reason: they read
# OMP_NUM_THREADS when they build their thread pools, so setting it inside
# main() is a no-op in the parent process.
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import dataclasses
import importlib.util
import json
import multiprocessing
import pathlib
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.evade import D_POS
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
ARTEFACT_PREFIX = "s14"

# --- pre-registered (spec sections 2-3). Do not tune. ---
SEED0 = 1_370_000
N_EPISODES = 300
P_GATE = S9.P_GATE  # 0.05
MAX_WORKERS = 6
EVADER_HOLD = S9.EVADER_HOLD  # 10, as in S6/S7/S9
DELTA = 0.0
N_OBSTACLES = 2
GAMMA = 0.2
CONFIRM_MIN = 3  # H-MAT-1 confirms iff >= 3 of the 4 learned attackers specialise
SUPPORT_TOL = 1e-6  # a meta-game strategy above this carries support
assert OMP_THREADS == S9.OMP_THREADS, "S14 and S9 must run at the same OMP setting"

# The six rows, in the spec's order. `hold` is per defender because TEACHER's
# search decision is the teacher's own (hold 1) and the learners' is HOLD.
# `shield_obs` is "as trained" — and TEACHER has none, deliberately: it was
# never trained, so there is no flag to carry and `run_cell` falls back to the
# unshielded-observation default.
DEFENDERS = {
    "TEACHER": {"scripted": "OrbitSearch", "hold": 1},
    "STR14": {
        "ckpt": CKPT_ROOT / "ckpts_STR14" / "STR14_10000000.zip",
        "shield_obs": False,  # trained unshielded (D111, S6 defender D1)
        "hold": HOLD,
    },
    "RC19": {
        "ckpt": CKPT_ROOT / "ckpts_RC19" / "RC19_10000000.zip",
        "shield_obs": True,  # S5 remedy configuration (D113)
        "hold": HOLD,
    },
    "RCM28": {
        "ckpt": CKPT_ROOT / "ckpts_RCM28" / "RCM28_10000000.zip",
        "shield_obs": True,  # S7 RC_SETTINGS (D116)
        "hold": HOLD,
    },
    "FT-B-41": {
        "ckpt": CKPT_ROOT / "ckpts_FT-B-41" / "FT-B-41_5000000.zip",
        "shield_obs": True,  # S10 artefact, RC configuration (D120)
        "hold": HOLD,
    },
    "SEL-A-50": {
        "ckpt": CKPT_ROOT / "ckpts_SEL-A-50" / "SEL-A-50_5000000.zip",
        "shield_obs": True,  # S12 selected policy, S10's recipe unchanged (D123)
        "hold": HOLD,
    },
}

# The seven columns, in the spec's order.
ATTACKERS = {
    "CTL-D125": {"ckpt": CKPT_ROOT / "ckpts_CTL-D125" / "CTL-D125_10000000.zip"},
    "CTL-D225": {"ckpt": CKPT_ROOT / "ckpts_CTL-D225" / "CTL-D225_10000000.zip"},
    "ATK2-31": {"ckpt": CKPT_ROOT / "ckpts_ATK2-31" / "ATK2-31_10000000.zip"},
    "ATK4-45": {"ckpt": CKPT_ROOT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"},
    "SPLITHUG": {"law": "SplitHugEvader"},
    "HUG7": {"scripted": ObstacleHuggingEvader},
    "GREEDY": {"scripted": GreedyEvader},
}
LEARNED = ("CTL-D125", "CTL-D225", "ATK2-31", "ATK4-45")
SCRIPTED = ("SPLITHUG", "HUG7", "GREEDY")
# Which defender each learned attacker was trained against (S6b defender D1 =
# STR14 and D2 = RC19; S7's race artefact was RCM28; S10's was FT-B-41).
TRAINED_AGAINST = {
    "CTL-D125": "STR14",
    "CTL-D225": "RC19",
    "ATK2-31": "RCM28",
    "ATK4-45": "FT-B-41",
}
# Spec section 2: the two rows with no attacker of their own.
TRANSFER_ONLY = tuple(d for d in DEFENDERS if d not in set(TRAINED_AGAINST.values()))

# H-MAT-2's pre-registered ordering on the worst case, written out pairwise.
# The spec's chain is
#     RCM28 < FT-B-41 ~ SEL-A-50 < RC19 <= STR14 <= TEACHER
# so the strict relations are every cross-group pair around each `<`; only
# those four are TESTED. `~` (the spec's "≈") and `<=` are reported, not tested.
ORDERING = (
    ("RCM28", "<", "FT-B-41"),
    ("RCM28", "<", "SEL-A-50"),
    ("FT-B-41", "~", "SEL-A-50"),
    ("FT-B-41", "<", "RC19"),
    ("SEL-A-50", "<", "RC19"),
    ("RC19", "<=", "STR14"),
    ("STR14", "<=", "TEACHER"),
)


def cell_name(defender: str, attacker: str) -> str:
    return f"{defender}__{attacker}"


def _build_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for d, dspec in DEFENDERS.items():
        for a in ATTACKERS:
            name = cell_name(d, a)
            cells[name] = {
                "name": name,
                "defender": d,
                "attacker": a,
                "delta": DELTA,
                "n": N_OBSTACLES,
                "gamma": GAMMA,
                "shield_obs": dspec.get("shield_obs", False),
                "hold": dspec["hold"],
                "transfer_only": d in TRANSFER_ONLY,
            }
    return cells


CELLS = _build_cells()


def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


# S9's record IO reads `cells_dir` and its H-SAFE rule reads `CELLS`;
# re-pointing the two makes every borrowed helper operate on S14's cells and
# S14's OUT (read at call time, so a test that moves OUT still works).
S9.CELLS = CELLS
S9.cells_dir = cells_dir

_ci = S9._ci
_model = S9._model
_write_json = S9._write_json
provenance = S9.provenance
write_cell = S9.write_cell
load_records = S9.load_records
hsafe = S9.hsafe


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def cell_config() -> EnvConfig:
    """Every cell runs the same arena: the locked config at the base obstacle
    count. Nothing sweeps in this round — the matrix is the sweep."""
    return dataclasses.replace(EnvConfig(), n_physical_obstacles=N_OBSTACLES)


def defender_policy(name: str, cfg_cell: EnvConfig, device: str):
    """The row's pursuer as something `evaluate_structured` can drive, i.e. an
    object with `.predict(obs, deterministic=True) -> (action, _)`.

    TEACHER is the scripted baseline: `OrbitSearch` on this cell's config, which
    at hold 1 reproduces `SearchingPursuer` episode for episode (V1). Every
    other row is its frozen PPO checkpoint."""
    spec = DEFENDERS[name]
    if "scripted" in spec:
        from pe2.structured import OrbitSearch

        return OrbitSearch(cfg_cell)
    return _model(spec["ckpt"], device)


def opponent_factory(name: str, cfg_cell: EnvConfig, device: str):
    """The column's evader, as an opponent factory for
    SingleAgentWrapper(role="pursuer").

    The four learned attackers and SPLITHUG occupy the SAME seat —
    `StructuredEvaderPolicy` at hold 10 with the hug law inside `D_POS` — so the
    only thing that differs between them is what decides outside that radius
    (S7's `splithug_factory`). HUG7 and GREEDY are the plain scripted classes."""
    spec = ATTACKERS[name]
    if "scripted" in spec:
        return spec["scripted"]
    from pe2.evade import SplitHugEvader, StructuredEvaderPolicy

    model = SplitHugEvader(cfg_cell) if "law" in spec else _model(spec["ckpt"], device)
    return lambda: StructuredEvaderPolicy(
        model, cfg_cell, hold=EVADER_HOLD, d_pos=D_POS
    )


def run_cell(name: str, device: str = "cpu") -> dict:
    """One cell: this row's pursuer under the CBF shield at the base setting
    (delta 0, gamma 0.2, n 2), against this column's evader, over the
    pre-registered stream."""
    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    spec = CELLS[name]
    cfg_cell = cell_config()
    logs = evaluate_structured(
        defender_policy(spec["defender"], cfg_cell, device),
        cfg_cell,
        opponent_factory(spec["attacker"], cfg_cell, device),
        N_EPISODES,
        SEED0,
        hold=spec["hold"],
        shield_factory=lambda: CBFShield(cfg_cell, gamma=GAMMA, margin=DELTA),
        shield_obs=spec["shield_obs"],
    )
    sh = summarize_shield(logs)
    k, n_ep = win_count(logs, "pursuer")
    deadlock_ind = [bool(lg["shield"]["deadlock"]) for lg in logs]
    capture_ind = [lg["outcome"] == "capture" for lg in logs]
    outcomes = Counter(lg["outcome"] for lg in logs)
    return {
        "name": name,
        "defender": spec["defender"],
        "attacker": spec["attacker"],
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "shield_obs": spec["shield_obs"],
        "hold": spec["hold"],
        "transfer_only": spec["transfer_only"],
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


def run_cells(
    names: list[str] | None = None, device: str = "cpu", workers: int = MAX_WORKERS
) -> list[str]:
    """Run every named cell that has no record yet (default: all 42)."""
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
# the two payoff matrices
# ----------------------------------------------------------------------------
def _matrix(records: dict, key: str) -> tuple[list[str], list[str], list[list]]:
    rows, cols = list(DEFENDERS), list(ATTACKERS)
    m = [
        [
            records[cell_name(d, a)][key] if cell_name(d, a) in records else None
            for a in cols
        ]
        for d in rows
    ]
    return rows, cols, m


def deadlock_matrix(records: dict) -> tuple[list[str], list[str], list[list]]:
    """The exploitability axis: the pursuer's deadlock-episode rate per cell."""
    return _matrix(records, "deadlock_rate")


def capture_matrix(records: dict) -> tuple[list[str], list[str], list[list]]:
    """The task axis: the pursuer's capture rate per cell."""
    return _matrix(records, "capture_rate")


def _worst(records: dict, defender: str, key: str, largest: bool) -> dict | None:
    """One defender's worst case on one axis: the max deadlock rate over the
    seven attackers (`largest=True`), or the min capture rate (`largest=False`),
    with the attacker that attains it."""
    have = [
        (a, records[cell_name(defender, a)][key])
        for a in ATTACKERS
        if cell_name(defender, a) in records
    ]
    if len(have) != len(ATTACKERS):
        return None
    pick = max(have, key=lambda t: t[1]) if largest else min(have, key=lambda t: t[1])
    return {"attacker": pick[0], key: pick[1]}


def worst_case_deadlock(records: dict) -> dict:
    return {d: _worst(records, d, "deadlock_rate", True) for d in DEFENDERS}


def worst_case_capture(records: dict) -> dict:
    return {d: _worst(records, d, "capture_rate", False) for d in DEFENDERS}


# ----------------------------------------------------------------------------
# hypotheses (spec section 3)
# ----------------------------------------------------------------------------
def mat1_verdicts(records: dict) -> dict:
    """H-MAT-1: each of the four learned attackers achieves its highest deadlock
    rate against the defender it was trained against.

    Per attacker the comparator is the highest-rate defender OTHER than its own
    — "the defender with the next-highest rate" when the attacker is specialised,
    and the argmax itself when it is not — and the test is
    `paired_indicator_test(own, other)`, which is meaningful because every cell
    of the round runs the same 300 seeds. SPECIALISED iff the own defender
    attains the column maximum; GENERAL otherwise, reported with the paired p.
    CONFIRMED iff >= CONFIRM_MIN of the 4 are SPECIALISED."""
    attackers: dict = {}
    for a in LEARNED:
        own = TRAINED_AGAINST[a]
        names = {d: cell_name(d, a) for d in DEFENDERS}
        missing = [n for n in names.values() if n not in records]
        if missing:
            attackers[a] = {
                "verdict": "INCOMPLETE",
                "own_defender": own,
                "missing": missing,
            }
            continue
        rates = {d: records[names[d]]["deadlock_rate"] for d in DEFENDERS}
        other = max((d for d in DEFENDERS if d != own), key=lambda d: rates[d])
        argmax = max(DEFENDERS, key=lambda d: rates[d])
        attackers[a] = {
            "verdict": "SPECIALISED" if rates[own] >= rates[other] else "GENERAL",
            "own_defender": own,
            "own_cell": names[own],
            "own_rate": rates[own],
            "argmax_defender": argmax,
            "comparator_defender": other,
            "comparator_cell": names[other],
            "rates": rates,
            "transfer_only_defenders": list(TRANSFER_ONLY),
            "test": {
                "test": "paired_indicator",
                **paired_indicator_test(
                    records[names[own]]["deadlock_ind"],
                    records[names[other]]["deadlock_ind"],
                    P_GATE,
                ),
            },
        }
    verdicts_ = [b["verdict"] for b in attackers.values()]
    n_spec = sum(v == "SPECIALISED" for v in verdicts_)
    if "INCOMPLETE" in verdicts_:
        overall = "INCOMPLETE"
    elif n_spec >= CONFIRM_MIN:
        overall = "CONFIRMED"
    else:
        overall = "NOT CONFIRMED"
    return {
        "hypothesis": "H-MAT-1",
        "verdict": overall,
        "n_specialised": n_spec,
        "confirm_min": CONFIRM_MIN,
        "scoring": list(LEARNED),
        "general": [a for a, b in attackers.items() if b["verdict"] == "GENERAL"],
        "attackers": attackers,
    }


def _relation(records: dict, worst: dict, lo: str, hi: str, rel: str) -> dict:
    """One pre-registered comparison `lo rel hi` on the worst-case deadlock rate.

    The comparator is the attacker that attains the LARGER side's maximum (spec
    section 3), and the paired test is run in that single column — the two
    defenders met the same evader on the same 300 seeds there, so the episodes
    pair. A `<` HOLDS iff the worst cases are in the claimed order AND that
    paired test passes at p < P_GATE; an ordering that inverts can never be
    rescued by a significant test in some other column, and a gap too small to
    separate is not a demonstrated `<`."""
    comparator = worst[hi]["attacker"]
    a_name, b_name = cell_name(hi, comparator), cell_name(lo, comparator)
    test = {
        "test": "paired_indicator",
        **paired_indicator_test(
            records[a_name]["deadlock_ind"], records[b_name]["deadlock_ind"], P_GATE
        ),
    }
    ordered = worst[lo]["deadlock_rate"] < worst[hi]["deadlock_rate"]
    tested = rel == "<"
    return {
        "lower": lo,
        "higher": hi,
        "relation": rel,
        "tested": tested,
        "comparator_attacker": comparator,
        "cells": [b_name, a_name],
        "worst_case": {
            lo: worst[lo]["deadlock_rate"],
            hi: worst[hi]["deadlock_rate"],
        },
        "worst_case_ordered": bool(ordered),
        "test": test,
        "holds": bool(ordered and test["pass"]) if tested else None,
    }


def mat2_verdicts(records: dict) -> dict:
    """H-MAT-2: on each defender's worst-case deadlock rate
    `w(d) = max over the 7 attackers`, the pre-registered ordering
    `RCM28 < FT-B-41 ~ SEL-A-50 < RC19 <= STR14 <= TEACHER` holds. Every strict
    `<` is tested; `~` and `<=` are reported. CONFIRMED iff every `<` holds."""
    worst = worst_case_deadlock(records)
    missing = [n for n in CELLS if n not in records]
    if any(w is None for w in worst.values()):
        return {
            "hypothesis": "H-MAT-2",
            "verdict": "INCOMPLETE",
            "ordering": [" ".join(t) for t in ORDERING],
            "missing": missing,
        }
    relations = {
        f"{lo} {rel} {hi}": _relation(records, worst, lo, hi, rel)
        for lo, rel, hi in ORDERING
    }
    tested = {k: b for k, b in relations.items() if b["tested"]}
    inversions = [k for k, b in tested.items() if not b["holds"]]
    return {
        "hypothesis": "H-MAT-2",
        "verdict": "CONFIRMED" if not inversions else "NOT CONFIRMED",
        "ordering": [" ".join(t) for t in ORDERING],
        "n_tested": len(tested),
        "n_holding": len(tested) - len(inversions),
        "inversions": inversions,
        "worst_case": worst,
        "transfer_only": list(TRANSFER_ONLY),
        "relations": relations,
    }


def mat3_report(records: dict) -> dict:
    """H-MAT-3 (no free lunch, reported without a verdict): no defender
    simultaneously attains the lowest worst-case deadlock rate `w(d)` and the
    highest worst-case capture rate `c(d) = min over the 7 attackers`. The
    Pareto frontier over `(w(d), -c(d))` is reported; if one defender dominates
    — it is both the argmin of w and the argmax of c — that is stated plainly
    instead and `free_lunch` is True."""
    wd, wc = worst_case_deadlock(records), worst_case_capture(records)
    complete = all(v is not None for v in wd.values()) and all(
        v is not None for v in wc.values()
    )
    if not complete:
        return {
            "hypothesis": "H-MAT-3",
            "complete": False,
            "pareto": None,
            "dominating": None,
            "free_lunch": None,
            "missing": [n for n in CELLS if n not in records],
        }
    w = {d: wd[d]["deadlock_rate"] for d in DEFENDERS}
    c = {d: wc[d]["capture_rate"] for d in DEFENDERS}
    best_w, best_c = min(w.values()), max(c.values())
    winners = [d for d in DEFENDERS if w[d] == best_w and c[d] == best_c]
    pareto = [
        d
        for d in DEFENDERS
        if not any(
            e != d and w[e] <= w[d] and c[e] >= c[d] and (w[e] < w[d] or c[e] > c[d])
            for e in DEFENDERS
        )
    ]
    return {
        "hypothesis": "H-MAT-3",
        "complete": True,
        "worst_case_deadlock": w,
        "worst_case_deadlock_attacker": {d: wd[d]["attacker"] for d in DEFENDERS},
        "worst_case_capture": c,
        "worst_case_capture_attacker": {d: wc[d]["attacker"] for d in DEFENDERS},
        "lowest_worst_case_deadlock": [d for d in DEFENDERS if w[d] == best_w],
        "highest_worst_case_capture": [d for d in DEFENDERS if c[d] == best_c],
        "dominating": winners[0] if len(winners) == 1 else None,
        "free_lunch": bool(winners),
        "pareto": pareto,
        "transfer_only": list(TRANSFER_ONLY),
        "note": "reported, no verdict (spec section 3)",
    }


# ----------------------------------------------------------------------------
# the meta-game
# ----------------------------------------------------------------------------
def _minimiser(m: np.ndarray) -> tuple[np.ndarray, float]:
    """min over x of max over j of (x^T M)_j, as the LP
        min v  s.t.  sum_i M[i][j] x_i <= v for every column j,
                     sum_i x_i = 1,  x >= 0.
    Returns the mixed strategy and the value."""
    from scipy.optimize import linprog

    n_rows, n_cols = m.shape
    res = linprog(
        c=np.concatenate([np.zeros(n_rows), [1.0]]),
        A_ub=np.hstack([m.T, -np.ones((n_cols, 1))]),
        b_ub=np.zeros(n_cols),
        A_eq=np.concatenate([np.ones((1, n_rows)), [[0.0]]], axis=1),
        b_eq=[1.0],
        bounds=[(0.0, None)] * n_rows + [(None, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"meta-game LP failed: {res.message}")
    return np.clip(res.x[:n_rows], 0.0, None), float(res.x[n_rows])


def _maximiser(m: np.ndarray) -> tuple[np.ndarray, float]:
    """max over y of min over i of (M y)_i, as the LP
    max u  s.t.  sum_j M[i][j] y_j >= u for every row i,
                 sum_j y_j = 1,  y >= 0."""
    from scipy.optimize import linprog

    n_rows, n_cols = m.shape
    res = linprog(
        c=np.concatenate([np.zeros(n_cols), [-1.0]]),
        A_ub=np.hstack([-m, np.ones((n_rows, 1))]),
        b_ub=np.zeros(n_rows),
        A_eq=np.concatenate([np.ones((1, n_cols)), [[0.0]]], axis=1),
        b_eq=[1.0],
        bounds=[(0.0, None)] * n_cols + [(None, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"meta-game LP failed: {res.message}")
    return np.clip(res.x[:n_cols], 0.0, None), float(res.x[n_cols])


def meta_game(
    matrix,
    rows: list[str],
    cols: list[str],
    minimise_rows: bool = True,
    tol: float = SUPPORT_TOL,
) -> dict:
    """Solve the zero-sum matrix game on `matrix` exactly by linear programming.

    `minimise_rows=True` is the DEADLOCK payoff: the row player (the defender)
    minimises and the column player (the attacker) maximises. `False` is the
    CAPTURE payoff, where the defender maximises. The game's value is the same
    quantity either way; the flag decides whose maximin strategy each side gets.

    Returns the value in the ORIGINAL payoff units, both maximin mixed
    strategies as name -> probability, and the support sets at `tol`."""
    m = np.asarray(matrix, dtype=object)
    if m.shape != (len(rows), len(cols)):
        raise ValueError(
            f"matrix shape {m.shape} does not match {len(rows)} rows x {len(cols)} cols"
        )
    if any(v is None for v in m.ravel()):
        raise ValueError("the meta-game needs every cell measured")
    m = m.astype(float)
    if not np.isfinite(m).all():
        raise ValueError("the meta-game needs every cell measured and finite")
    # The solvers below always treat rows as the minimiser; negating the payoff
    # swaps the roles and negating the value back undoes it.
    sign = 1.0 if minimise_rows else -1.0
    g = sign * m
    x, v_row = _minimiser(g)
    y, v_col = _maximiser(g)
    return {
        "value": sign * v_row,
        "row_player": "minimiser" if minimise_rows else "maximiser",
        "row_strategy": {r: float(p) for r, p in zip(rows, x, strict=True)},
        "col_strategy": {c: float(p) for c, p in zip(cols, y, strict=True)},
        "row_support": [r for r, p in zip(rows, x, strict=True) if p > tol],
        "col_support": [c for c, p in zip(cols, y, strict=True) if p > tol],
        "support_tol": tol,
        # the minimax theorem: the two LPs are duals and must agree
        "duality_gap": abs(v_row - v_col),
        "status": "optimal",
    }


def _meta_or_incomplete(records: dict, key: str, minimise_rows: bool) -> dict:
    rows, cols, m = _matrix(records, key)
    if any(v is None for row in m for v in row):
        return {
            "status": "INCOMPLETE",
            "value": None,
            "row_player": "minimiser" if minimise_rows else "maximiser",
            "payoff": key,
            "missing": [n for n in CELLS if n not in records],
        }
    return {"payoff": key, **meta_game(m, rows, cols, minimise_rows)}


def verdicts(records: dict) -> dict:
    missing = [n for n in CELLS if n not in records]
    return {
        "mat1": mat1_verdicts(records),
        "mat2": mat2_verdicts(records),
        "mat3": mat3_report(records),
        # the headline artefact (spec section 3): the defender minimises the
        # deadlock payoff, and maximises the capture payoff
        "meta_deadlock": _meta_or_incomplete(records, "deadlock_rate", True),
        "meta_capture": _meta_or_incomplete(records, "capture_rate", False),
        "hsafe": hsafe(records),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "defenders": list(DEFENDERS),
        "attackers": list(ATTACKERS),
        "transfer_only": list(TRANSFER_ONLY),
        "seed0": SEED0,
        "n_episodes": N_EPISODES,
    }


def direction(v: dict) -> str:
    """Spec section 5, one clause per row, in the spec's order of precedence:
    H-SAFE first (a contact step stops the round), then completeness, then
    H-MAT-1, H-MAT-2, the meta-game and H-MAT-3."""
    mat1, mat2, mat3, meta, hsafe_ = (
        v["mat1"],
        v["mat2"],
        v["mat3"],
        v["meta_deadlock"],
        v["hsafe"],
    )
    if hsafe_["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {hsafe_['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(hsafe_['offenders'])}. Stop — the shield's guarantee is "
            "the precondition of the whole matrix and nothing else is read."
        )
    if "INCOMPLETE" in (mat1["verdict"], mat2["verdict"], hsafe_["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record; every defender x attacker must run on the one stream before "
            "a verdict is read (spec section 5)."
        )
    parts = []
    if mat1["verdict"] == "CONFIRMED":
        parts.append(
            f"H-MAT-1 CONFIRMED ({mat1['n_specialised']}/{len(LEARNED)} attackers "
            "SPECIALISED): our best responses are genuinely defender-specific, so "
            "the chain is a real iterated best-response sequence and the "
            "exploitability numbers mean what we say."
        )
    else:
        parts.append(
            f"H-MAT-1 NOT CONFIRMED ({mat1['n_specialised']}/{len(LEARNED)} "
            f"SPECIALISED; GENERAL: {', '.join(mat1['general']) or 'none'}): the "
            "attack is a property of the shield, not of any defender — weaker as "
            "an arms-race story, STRONGER as a generality claim. Report it as the "
            "latter."
        )
    if mat2["verdict"] == "CONFIRMED":
        parts.append(
            "H-MAT-2 CONFIRMED: the progression the paper narrates survives a "
            "single-stream, worst-case test."
        )
    else:
        parts.append(
            f"H-MAT-2 NOT CONFIRMED ({mat2['n_holding']}/{mat2['n_tested']} "
            f"comparisons hold; inversions: {', '.join(mat2['inversions'])}): that "
            "defender's advantage was stream- or opponent-specific; the narrative "
            "is corrected to what the matrix shows."
        )
    if len(meta.get("row_support", [])) == 1:
        d = meta["row_support"][0]
        parts.append(
            f"the deadlock meta-game's defender support is concentrated on {d} "
            f"(equilibrium deadlock rate {meta['value']:.3f}, attacker support "
            f"{', '.join(meta['col_support'])}): {d} is the paper's "
            "recommendation, with a quantified equilibrium value."
        )
    else:
        parts.append(
            "the deadlock meta-game mixes "
            f"{len(meta.get('row_support', []))} defenders "
            f"({', '.join(meta.get('row_support', []))}) at equilibrium value "
            f"{meta['value']:.3f}: no single defender is the recommendation — "
            "report the mixture."
        )
    if mat3.get("dominating"):
        parts.append(
            f"H-MAT-3: {mat3['dominating']} dominates — it attains BOTH the lowest "
            "worst-case deadlock rate and the highest worst-case capture rate, so "
            "there is a free lunch here and it is stated plainly."
        )
    else:
        parts.append(
            "H-MAT-3: no defender attains both the lowest worst-case deadlock rate "
            f"and the highest worst-case capture rate; the Pareto frontier is "
            f"{', '.join(mat3.get('pareto') or [])}."
        )
    return " ".join(parts)


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
def _heatmap(ax, rows, cols, m, title, cmap, support_rows, support_cols) -> None:
    """One payoff matrix as a heatmap, with the meta-game support marked: a
    supported defender's row label and a supported attacker's column label are
    prefixed with a `*`, so the equilibrium is readable off the matrix itself."""
    grid = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in m], dtype=float
    )
    ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="0.4")
            else:
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grid[i, j] > 0.55 else "black",
                )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [f"* {c}" if c in support_cols else c for c in cols],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            f"* {r}{' (transfer)' if r in TRANSFER_ONLY else ''}"
            if r in support_rows
            else f"{r}{' (transfer)' if r in TRANSFER_ONLY else ''}"
            for r in rows
        ],
        fontsize=7,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("attacker (evader)", fontsize=8)
    ax.set_ylabel("defender (shielded pursuer)", fontsize=8)


def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = verdicts(records)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (fn, cmap, label, meta) in zip(
        axes,
        (
            (
                deadlock_matrix,
                "magma",
                "deadlock episode rate (defender MINIMISES)",
                v["meta_deadlock"],
            ),
            (
                capture_matrix,
                "viridis",
                "capture rate (defender MAXIMISES)",
                v["meta_capture"],
            ),
        ),
        strict=True,
    ):
        rows, cols, m = fn(records)
        _heatmap(
            ax,
            rows,
            cols,
            m,
            f"{label}\nmeta-game value "
            + (
                "incomplete"
                if meta.get("value") is None
                else f"{meta['value']:.3f} (* = equilibrium support)"
            ),
            cmap,
            set(meta.get("row_support", ())),
            set(meta.get("col_support", ())),
        )
    fig.suptitle(
        f"S14 — the empirical game matrix (CBF shield, delta = {DELTA}, "
        f"gamma = {GAMMA}, n = {N_OBSTACLES}; N = {N_EPISODES} seed-paired "
        f"episodes from {SEED0}). TEACHER and SEL-A-50 are transfer-only rows: "
        "no attacker was trained against them."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize() -> dict:
    records = load_records()
    v = verdicts(records)
    rows, cols, dl = deadlock_matrix(records)
    _r, _c, cap = capture_matrix(records)
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "matrices": {
            "deadlock": {"rows": rows, "cols": cols, "values": dl},
            "capture": {"rows": rows, "cols": cols, "values": cap},
        },
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
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cells is not None:
        run_cells(args.cells or None, args.device, args.workers)
    if args.summarize:
        blob = summarize()
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    "mat1": {
                        k: v["mat1"][k] for k in ("verdict", "n_specialised", "general")
                    },
                    "mat2": {
                        k: v["mat2"][k]
                        for k in ("verdict", "n_holding", "n_tested", "inversions")
                        if k in v["mat2"]
                    },
                },
                indent=2,
            )
        )
        print(json.dumps(v["hsafe"], indent=2))
        for key in ("meta_deadlock", "meta_capture"):
            print(
                json.dumps(
                    {
                        key: {
                            k: v[key].get(k)
                            for k in ("value", "row_support", "col_support", "status")
                        }
                    },
                    indent=2,
                )
            )
        print(
            json.dumps(
                {k: v["mat3"].get(k) for k in ("dominating", "free_lunch", "pareto")},
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
