"""S24.2 Step E.2 — the gate: 15 cells, the verdicts, the construct gates.

Pre-registration: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` sections 1,
2 and Amendments A1/A2. Where the brief and the prereg could be read to
disagree, the PREREGISTRATION wins and the reading is named at the line.

Evaluation only. Nothing here trains, selects or tunes anything, and nothing
re-runs or rewrites a comparator: `results/derisk/s24_1_cells/` is READ.

A rebinding derivative of `scripts/derisk_s24_1_gate.py` (S24.1's gate) and
`scripts/derisk_s24_os_gate.py` (that pattern's own derivative), loaded by file
path with that runner's own `_load` pattern. The ORCHESTRATION layer —
`run_cells`, `scan_records`, `load_records`, `read_cell`, `render_cell` — is
generic over a cell table and is CALLED, not copied, with `CELLS`,
`cells_dir`/`cell_file`, `record_status`, `provenance`, `run_cell`,
`write_cell`, `undetermined_record`, `ROWS`, `build_cell_env` and `ckpt_path`
rebound for the duration of a call (`_bound`/`_ORIG`, S24-OS's own technique).

**What is new rather than bound, and why.** This round's row is driven by
`EvaderSelectorMDP`, not `SelectorMDP`: the env, the per-episode log schema
(`n_zero_length_spans` is a plain int here — A3.4's span-accounting mismatch
was a property of a scripted PURSUER law re-entering a DONE tactic, which
cannot happen on this round's rows: A2.1's mask removes that path on every
cell) and `run_cell`'s own bookkeeping are therefore this file's own,
`build_cell_env` included. `cell_config`, `load_policy`, `require_ckpt`
(through the rebound `ckpt_path`) and `render_cell`'s caption are S24.1's own,
UNCHANGED: `render_cell` reads the frame label off `game.evader`'s and
`game.law`'s own current-tactic state (`s24_tactic_game.py::frame_label`),
which populates correctly for THIS round's evader exactly as it does for
S24.1's pursuer rows, because the evader adapter whose `select` this round
replaces is the same `TacticHolder`-shaped object either way — nothing about
the caption mechanism is scripted-evader-specific.

**The checkpoint path is the trainer's own.** `ckpt_path` is bound to
`derisk_s24_2_train.py`'s `ckpt_path` (loaded by file path as `T2`), so
S24.1's `require_ckpt`/`load_policy` — reused unchanged — resolve
`S24-2-{ARM}` checkpoints without this file ever repeating that stem as a
literal.

**The pursuer seats are resolved through S24.1's own gate function.**
`SEL-80/81/82`'s checkpoints are `G1.ckpt_path(name)` — never a literal path —
so a change to S24.1's checkpoint layout (not its content) cannot silently
desync a seat from the file CG-E1 already proved bit-identical against.

**Comparators are read OUTSIDE any binding, by the rule that measured them.**
The five scripted-evader rows of `results/derisk/s24_1_cells/`
(`{PURSUER}_TACTIC-V0`), validated by S24.1's OWN `record_status`/`cell_file`.
The join asserts `S1.CELLS is` this file's captured original before reading a
single file: inside a binding, `record_status` and `cell_file` would be THIS
round's and the comparator column would be checked against the wrong rule and
read from the wrong directory (S24-OS's own join states the same lesson).

Readings this module fixes, because the text can be read two ways:

1. **H-EV-1's and H-EV-2's "lowers capture ... DOWN direction" is
   `paired_indicator_test(comparator_ind, row_ind, p)`:** the test's own
   convention (`derisk_s24_os_gate.py`'s H-OS-1) is "pass iff the FIRST
   argument produces the event more often than the second"; the comparator
   (the scripted evader) must out-capture the row (the learned evader) for the
   row to have LOWERED capture, so the comparator's indicator is the first
   argument and the row's the second — never the reverse, which would test the
   opposite (UP) direction silently.
2. **H-EV-2 is read as "pass iff BOTH held-out pursuers (RULES-V0 AND
   LEAD-1.0) individually clear their own >= 2/3-seeds bar"** — the same
   both-must-pass shape `derisk_s24_os_gate.py::h_os_1` uses for its two
   opponents, because the brief's "each >= 2/3 evader seeds" states a
   per-pursuer bar and the round's claim is that the lowering generalises to
   EVERY held-out pursuer, not just one of them.
3. **H-EV-3's "for each evader seed" is read literally — ALL three seeds, not
   >= 2/3** (unlike every other hypothesis here): it is the ONE hypothesis the
   brief states without a "2/3" fraction, and it is a KILL criterion — a
   single seed whose evader cannot threaten ANY of the five pursuers at all
   means the speed ratio broke the alternation for that seed, which an
   aggregate pass-rate would hide.
4. **The 0.05 kill threshold and the exact-counts rule are the SAME
   discipline, extended to a bare threshold.** `derisk_s24_os_gate.py`'s
   review defect 2 is about a MARGIN (a float subtraction landing exactly on
   -0.03 in binary floating point); H-EV-3's comparison is `k >= 0.05 * N`,
   which has no subtraction and so no analogous failure mode at N = 500 (25.0
   is exactly representable and 25/500 rounds to the same double as the 0.05
   literal) — but it is computed on the integer capture count against
   `round(0.05 * N_EPISODES)` anyway, so the reader never has to re-derive that
   argument from the record to trust the verdict.
5. **CG-E3's "share of the most-used evader tactic" reads the per-cell
   `picks_share`'s maximum entry**, over the per-DECISION histogram (A2.2's
   convention, inherited): a tactic's SPAN length is not its decision count.

Usage:
  ./venv/bin/python scripts/derisk_s24_2_gate.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_2_gate.py --cells EVS-80_SEL-81
  ./venv/bin/python scripts/derisk_s24_2_gate.py --gifs           (CG-E2)
  ./venv/bin/python scripts/derisk_s24_2_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (S24.1's stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # as S24.1 / S24-OS; BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

from pe2.derisk import paired_indicator_test
from pe2.evader_selector_mdp import EVADER_ACTIONS, EvaderSelectorMDP
from pe2.structured import HOLD

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's gate (and through it S23's and S19's) and this round's OWN trainer
# (the authority on the checkpoint path and the pursuer mixture) — both loaded
# by FILE PATH.
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
T2 = _load("derisk_s24_2_train", _HERE / "derisk_s24_2_train.py")
GAME = S1.GAME

PREREG = "docs/pathway/tactics_prereg_s24_2_2026-10-17.md"
ARTEFACT_PREFIX = "s24_2"
OUT = pathlib.Path("results/derisk")

# --- pre-registered. Do not tune. ---
GATE_SEED0 = 2_400_000  # S24.1's GATE_SEED0, cross-checked at import below
N_EPISODES = 500
P_GATE = 0.05  # H-EV-1, H-EV-2
KILL_MIN_CAPTURE = 0.05  # H-EV-3
PASS_MIN = 2  # ">= 2/3", seeds
RATIO = 1.2
SEEDS: tuple[int, ...] = T2.SEEDS  # (80, 81, 82)

if S1.GATE_SEED0 != GATE_SEED0:
    raise SystemExit(
        f"S24.1's GATE_SEED0 is {S1.GATE_SEED0}, not {GATE_SEED0}; this round's "
        "gate stream must be the same one (cross-checked at import)."
    )
if S1.N_EPISODES != N_EPISODES or S1.P_GATE != P_GATE:
    # trivially true; stated so a later edit of either module trips it
    raise SystemExit(
        "S24.1's N or p-gate differs from this round's; the comparator join "
        "and the reused orchestration would disagree."
    )
if S1.cell_config().evader_speed_ratio != RATIO:
    raise SystemExit(
        f"S24.1's cell_config ratio moved to {S1.cell_config().evader_speed_ratio}, "
        f"not this round's {RATIO}; the comparator join would read the wrong ratio."
    )

ROW_NAMES: tuple[str, ...] = tuple(f"EVS-{s}" for s in SEEDS)
ROWS: dict[str, dict] = {
    row: {"kind": "learned", "arm": row, "seed": s}
    for row, s in zip(ROW_NAMES, SEEDS, strict=True)
}
# The five pursuer seats (brief, deliverable 2). SEL-80/81/82's checkpoints are
# resolved through S24.1's OWN gate function, never a literal path.
PURSUER_SPECS: dict[str, object] = {
    f"SEL-{s}": ("ckpt", str(S1.ckpt_path(f"SEL-{s}")), "v0") for s in SEEDS
} | {"RULES-V0": "RULES-V0", "LEAD-1.0": "LEAD-1.0"}
PURSUERS: tuple[str, ...] = tuple(PURSUER_SPECS)  # SEL-80, SEL-81, SEL-82, RULES-V0, LEAD-1.0
MIXTURE_PURSUERS: tuple[str, ...] = ("RULES-V0", "LEAD-1.0")  # H-EV-2: IN the training mixture (A3);
# the genuinely held-out seats are SEL-{s' != 81}, met inside H-EV-1 by EVS-80 and EVS-82.


def cell_name(row: str, pursuer: str) -> str:
    return f"{row}_{pursuer}"


def _build_cells() -> dict[str, dict]:
    """The 15 cells: every evader row x every pursuer seat. Each carries its
    row, pursuer and ratio STORED — nothing parses a name back apart."""
    return {
        cell_name(r, p): {
            "name": cell_name(r, p),
            "row": r,
            "pursuer": p,
            "pursuer_spec": PURSUER_SPECS[p],
            "ratio": RATIO,
            "hold": HOLD,
            **ROWS[r],
        }
        for r in ROW_NAMES
        for p in PURSUERS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance (this round's prefix; S24.1's shapes)
# ----------------------------------------------------------------------------
def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


def cell_file(name: str) -> pathlib.Path:
    return cells_dir() / f"{name}.json"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_gifs"


def gif_path(row: str, pursuer: str, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{row}_{pursuer}_{seed}.gif"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def spec_path() -> pathlib.Path:
    return _HERE.parent / PREREG


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    S1._write_json(path, record)
    return path


def provenance(device: str = "cpu") -> dict:
    return S1.S19.provenance(device) | {
        "spec": PREREG,
        "spec_sha256": S1._sha256(spec_path()),
        "cg_b": None,
        "shield": "RecoveryShieldV2 (D131), fresh per episode — the v0b assembly's",
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
        "trained_by": T2.PREREG,
        "comparator_source": S1.PREREG,
        "ckpt_stem": "the trainer's own `ckpt_path` (never a literal)",
    }


# ----------------------------------------------------------------------------
# record validity (S24.1's / S24-OS's shape: this round's own n/seed0/ratio/pursuer)
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == S1.UNDETERMINED:
        return S1.UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    n, seed0 = rec.get("n_episodes"), rec.get("seed0")
    ratio, pursuer = rec.get("ratio"), rec.get("pursuer")
    if n != N_EPISODES:
        return "INVALID", f"n_episodes {n!r} != the pre-registered {N_EPISODES}"
    if seed0 != GATE_SEED0:
        return "INVALID", f"seed0 {seed0!r} != the pre-registered {GATE_SEED0}"
    if ratio != CELLS[name]["ratio"]:
        return "INVALID", f"ratio {ratio!r} != the cell's declared {CELLS[name]['ratio']}"
    if pursuer != CELLS[name]["pursuer"]:
        return "INVALID", f"pursuer {pursuer!r} != the cell's declared {CELLS[name]['pursuer']}"
    return "MEASURABLE", ""


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    # `_ORIG`, never the bare `S1.undetermined_record`: while `_bound()` is
    # active that name IS this function, and calling it would recurse forever.
    return _ORIG["undetermined_record"](name, exc, device) | {
        "ratio": CELLS[name]["ratio"],
        "pursuer": CELLS[name]["pursuer"],
    }


# ----------------------------------------------------------------------------
# the cell's env: EvaderSelectorMDP, only the pursuer seat varies
# ----------------------------------------------------------------------------
def build_cell_env(
    name: str, cfg=None, device: str = "cpu", *, recorder: bool = False, label: str = ""
) -> EvaderSelectorMDP:
    spec = CELLS[name]
    cfg = S1.cell_config() if cfg is None else cfg
    return EvaderSelectorMDP(
        cfg.evader_speed_ratio,
        pursuer=spec["pursuer_spec"],
        memory=True,
        recorder=recorder,
        label=label,
        cfg=cfg,
        device=device,
    )


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """One cell: N paired episodes of the EVS row's learned evader under the
    pursuer's recovery shield, against this cell's pursuer seat."""
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    cfg = S1.cell_config()
    env = build_cell_env(name, cfg, device)
    predict = S1.load_policy(spec["row"], device).predict

    capture_ind: list[bool] = []
    ep_steps: list[int] = []
    violations: list[int] = []
    switches: list[int] = []
    zero_spans: list[int] = []
    decisions: list[int] = []
    bad_picks: list[int] = []
    tactics: list[dict] = []
    branch: list[dict] = []
    picks: Counter = Counter()
    usage: Counter = Counter()
    outcomes: Counter = Counter()
    seen_bad = 0
    for i in range(n):
        log = env.play_episode(predict, GATE_SEED0 + i)
        capture_ind.append(log["outcome"] == "capture")
        outcomes[log["outcome"]] += 1
        ep_steps.append(int(log["steps"]))
        violations.append(int(log["violations"]))
        switches.append(int(log["n_switches"]))
        zero_spans.append(int(log["n_zero_length_spans"]))
        decisions.append(int(log["n_decisions"]))
        total_bad = int(log["inapplicable_picks"])
        bad_picks.append(total_bad - seen_bad)
        seen_bad = total_bad
        tactics.append(log["tactics"])
        branch.append(log["branch"])
        picks.update(log["picks"])
        usage.update(log["usage"]["pursuer"])

    k = sum(capture_ind)
    total_picks = max(1, sum(picks.values()))
    return {
        "name": name,
        "row": spec["row"],
        "pursuer": spec["pursuer"],
        "pursuer_spec": [str(x) for x in spec["pursuer_spec"]] if isinstance(spec["pursuer_spec"], tuple) else str(spec["pursuer_spec"]),  # audit (review fix 6)
        "kind": "learned",
        "memory": True,
        "ratio": spec["ratio"],
        "hold": spec["hold"],
        "n_episodes": n,
        "seed0": GATE_SEED0,
        "device": device,
        "capture_ind": capture_ind,
        "capture_rate": k / n if n else float("nan"),
        "wilson_capture": S1._ci(k, n),
        "steps": ep_steps,
        "violations": violations,
        "violation_steps": int(sum(violations)),
        "n_switches": switches,
        "n_zero_length_spans": zero_spans,
        "n_decisions": decisions,
        "inapplicable_picks_per_episode": bad_picks,
        "inapplicable_picks": int(seen_bad),  # CG-T1
        "picks": {a: int(picks.get(a, 0)) for a in EVADER_ACTIONS},
        "usage_steps": dict(usage),
        "tactics": tactics,
        "branch": branch,
        "aggregate": {
            "capture_rate": k / n if n else float("nan"),
            "outcomes": dict(outcomes),
            "escape_rate": outcomes["timeout"] / n if n else float("nan"),
            "mean_steps": float(statistics.fmean(ep_steps)) if ep_steps else float("nan"),
            "mean_switches": float(statistics.fmean(switches)) if switches else float("nan"),
            "switches_p90": S1._q(switches, 0.90),
            "mean_decisions": float(statistics.fmean(decisions)) if decisions else float("nan"),
            "mean_zero_length_spans": (
                float(statistics.fmean(zero_spans)) if zero_spans else float("nan")
            ),
            "picks_share": {a: picks.get(a, 0) / total_picks for a in EVADER_ACTIONS},
        },
        "provenance": provenance(device),
    }


# ----------------------------------------------------------------------------
# the binding onto S24.1's gate (see the module docstring)
# ----------------------------------------------------------------------------
_BOUND_ALWAYS: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "record_status",
    "provenance", "run_cell", "write_cell", "undetermined_record",
    "ROWS", "build_cell_env", "ckpt_path",
)
_ORIG: dict = {
    k: getattr(S1, k)
    for k in (
        *_BOUND_ALWAYS, "run_cells", "scan_records", "load_records", "read_cell",
        "render_cell",
    )
}


@contextlib.contextmanager
def _bound():
    """Rebind the names above on S24.1's module for one call and restore every
    one afterwards. A delegating call is made THROUGH `_ORIG`, so it can never
    re-enter this module."""
    over = {
        "CELLS": CELLS, "CELL_ORDER": CELL_ORDER,
        "cells_dir": cells_dir, "cell_file": cell_file,
        "record_status": record_status, "provenance": provenance,
        "run_cell": run_cell, "write_cell": write_cell,
        "undetermined_record": undetermined_record,
        "ROWS": ROWS, "build_cell_env": build_cell_env,
        "ckpt_path": T2.ckpt_path,
    }
    saved = {k: getattr(S1, k) for k in over}
    for k, v in over.items():
        setattr(S1, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(S1, k, v)


def scan_records() -> tuple[dict, dict, dict]:
    with _bound():
        return _ORIG["scan_records"]()


def load_records(*, quiet: bool = False) -> dict:
    with _bound():
        return _ORIG["load_records"](quiet=quiet)


def read_cell(name: str) -> dict:
    with _bound():
        return _ORIG["read_cell"](name)


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell with no record yet (default: all 15). Literal names
    only — S24.1's own refusal order (unknown names, blocked cells, missing
    checkpoints, all before the first episode) runs unchanged."""
    with _bound():
        return _ORIG["run_cells"](names, device, n_episodes)


def require_ckpt(row: str) -> pathlib.Path:
    """This round's own, calling the TRAINER's `ckpt_path` directly (never a
    literal), so a precondition check outside `_bound()` (CG-E2's GIFs) still
    resolves the right path."""
    path = T2.ckpt_path(row)
    if not path.exists():
        raise SystemExit(
            f"missing the final checkpoint for row {row} at {path}. The budget "
            f"is fixed at {T2.TOTAL} env steps; there is no substitute."
        )
    return path


def render_cell(
    row: str, pursuer: str, seed: int, device: str = "cpu", every: int | None = None
):
    with _bound():
        return _ORIG["render_cell"](
            row, pursuer, seed, device, S1.GIF_EVERY if every is None else every
        )


# ----------------------------------------------------------------------------
# the comparator join — OUTSIDE any binding, by the rule that measured them
# ----------------------------------------------------------------------------
def _assert_outside_binding(what: str) -> None:
    if not (S1.CELLS is _ORIG["CELLS"] and S1.cells_dir is _ORIG["cells_dir"]):
        raise SystemExit(
            f"{what} may not run inside a binding: with S24.1's globals rebound "
            "to this round's, `record_status` and `cell_file` would be checked "
            "against the wrong rule and read from the wrong directory."
        )


def load_comparators() -> dict:
    """The five scripted-evader rows of `s24_1_cells` (`{PURSUER}_TACTIC-V0`),
    validated by S24.1's OWN `record_status`. Never re-run; REFUSES whole if
    any one is not measurable."""
    _assert_outside_binding("the comparator join")
    out: dict = {}
    problems: list[str] = []
    for pursuer in PURSUERS:
        name = f"{pursuer}_TACTIC-V0"
        path = S1.cell_file(name)
        if not path.exists():
            problems.append(f"{path}: absent")
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path}: unreadable: {exc}")
            continue
        status, why = S1.record_status(rec)
        if status != "MEASURABLE":
            problems.append(f"{path}: {status}: {why}")
            continue
        out[name] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the scripted-evader comparator column (s24_1_cells) is "
            "not fully measurable: " + "; ".join(problems)
        )
    return out


# ----------------------------------------------------------------------------
# the training probes (A3.1: absent evidence is an offender, never a zero)
# ----------------------------------------------------------------------------
def read_probes() -> dict:
    """The three arms' probe rows, read exactly as the trainer wrote them."""
    out: dict = {}
    for arm in T2.ARM_ORDER:
        path = T2._probe_file(arm)
        if not path.exists():
            out[arm] = {"path": str(path), "missing": True, "rows": []}
            continue
        try:
            rows = json.loads(path.read_text()).get("checkpoints", [])
        except (OSError, json.JSONDecodeError) as exc:
            out[arm] = {"path": str(path), "missing": True, "error": str(exc), "rows": []}
            continue
        out[arm] = {"path": str(path), "missing": False, "rows": rows}
    return out


def probe_block(probes: dict | None, arm: str) -> dict:
    block = (probes or {}).get(arm) or {"missing": True, "rows": [], "absent_key": True}
    rows = block.get("rows") or []
    return {
        "n_probe_rows": len(rows),
        "rows": rows,
        "evidence_absent": bool(block.get("missing")) or not rows,
        "path": block.get("path"),
        "error": block.get("error"),
    }


# ----------------------------------------------------------------------------
# hypotheses (prereg sections 1-2)
# ----------------------------------------------------------------------------
def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _direction_label(delta: float, a: str, b: str) -> str:
    if delta > 0:
        return f"{a} > {b}"
    if delta < 0:
        return f"{a} < {b}"
    return f"{a} == {b}"


def _down_test(row_rec: dict, comparator_rec: dict, row_name: str, comp_name: str) -> dict:
    """Reading 1: the comparator's indicator is the test's FIRST argument, so
    `pass` means the comparator out-captures the row — the row LOWERED
    capture relative to it (the DOWN direction)."""
    test = {
        "test": "paired_indicator",
        **paired_indicator_test(comparator_rec["capture_ind"], row_rec["capture_ind"], P_GATE),
    }
    delta = row_rec["capture_rate"] - comparator_rec["capture_rate"]
    return {
        "verdict": "PASS" if test["pass"] else "FAIL",
        "row": row_name,
        "comparator": comp_name,
        "capture_row": row_rec["capture_rate"],
        "capture_comparator": comparator_rec["capture_rate"],
        "delta": delta,
        "direction": _direction_label(delta, row_name, comp_name),
        "p": test["p"],
        "test": test,
    }


def h_ev_1(records: dict, comparators: dict) -> dict:
    """H-EV-1 (gated): `EVS-s` vs `SEL-s` (same seed) LOWERS capture relative
    to `SEL-s_TACTIC-V0` — same-seed pairing, >= 2/3 seeds."""
    per_seed: dict = {}
    for s in SEEDS:
        row_name = cell_name(f"EVS-{s}", f"SEL-{s}")
        comp_name = f"SEL-{s}_TACTIC-V0"
        a, b = records.get(row_name), comparators.get(comp_name)
        if a is None or b is None:
            missing = [n for n, v in ((row_name, a), (comp_name, b)) if v is None]
            per_seed[str(s)] = {"verdict": "INCOMPLETE", "missing": missing, "seed": s}
            continue
        per_seed[str(s)] = _down_test(a, b, row_name, comp_name) | {"seed": s}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "hypothesis": "H-EV-1",
        "gated": True,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "rule": "paired indicator test, DOWN direction, p < 0.05, same-seed "
        "pairing (s <-> s), on >= 2/3 seeds",
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "seeds": per_seed,
    }


def h_ev_2_pursuer(records: dict, comparators: dict, pursuer: str) -> dict:
    per_seed: dict = {}
    for s in SEEDS:
        row_name = cell_name(f"EVS-{s}", pursuer)
        comp_name = f"{pursuer}_TACTIC-V0"
        a, b = records.get(row_name), comparators.get(comp_name)
        if a is None or b is None:
            missing = [n for n, v in ((row_name, a), (comp_name, b)) if v is None]
            per_seed[str(s)] = {"verdict": "INCOMPLETE", "missing": missing, "seed": s}
            continue
        per_seed[str(s)] = _down_test(a, b, row_name, comp_name) | {"seed": s}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "pursuer": pursuer,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "seeds": per_seed,
    }


def h_ev_2(records: dict, comparators: dict) -> dict:
    """H-EV-2 (gated, reading 2): likewise vs the HELD-OUT pursuers RULES-V0
    and LEAD-1.0 — the round passes iff BOTH individually clear >= 2/3 seeds."""
    per = {p: h_ev_2_pursuer(records, comparators, p) for p in MIXTURE_PURSUERS}
    verds = [b["verdict"] for b in per.values()]
    return {
        "hypothesis": "H-EV-2",
        "gated": True,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verds
        else ("PASS" if all(v == "PASS" for v in verds) else "FAIL"),
        "rule": "paired indicator test, DOWN direction, p < 0.05, on >= 2/3 "
        "seeds, for EACH of RULES-V0 and LEAD-1.0",
        "pursuers": per,
    }


def h_ev_3(records: dict) -> dict:
    """H-EV-3 (kill, reading 3: ALL seeds, not 2/3): for each evader seed the
    MAX capture over the five pursuers must clear 0.05, on exact COUNTS
    (reading 4) — never a float rate."""
    threshold_k = round(KILL_MIN_CAPTURE * N_EPISODES)
    per_seed: dict = {}
    for s in SEEDS:
        cell_names = [cell_name(f"EVS-{s}", p) for p in PURSUERS]
        missing = [n for n in cell_names if n not in records]
        if missing:
            per_seed[str(s)] = {"verdict": "INCOMPLETE", "missing": missing, "seed": s}
            continue
        ks = {p: int(sum(records[cell_name(f"EVS-{s}", p)]["capture_ind"])) for p in PURSUERS}
        caps = {p: records[cell_name(f"EVS-{s}", p)]["capture_rate"] for p in PURSUERS}
        max_pursuer = max(ks, key=ks.get)
        per_seed[str(s)] = {
            "verdict": "PASS" if ks[max_pursuer] >= threshold_k else "FAIL",
            "seed": s,
            "max_pursuer": max_pursuer,
            "max_k": ks[max_pursuer],
            "max_capture": caps[max_pursuer],
            "threshold_k": threshold_k,
            "captures": caps,
            "counts": ks,
        }
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    passing = [s for s in SEEDS if per_seed[str(s)]["verdict"] == "PASS"]
    return {
        "hypothesis": "H-EV-3 (kill, reported first)",
        "gated": True,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if len(passing) == len(SEEDS) else "FAIL"),
        "rule": f"max capture over the five pursuers >= {KILL_MIN_CAPTURE:g} "
        f"(k >= {threshold_k}/{N_EPISODES}), for EVERY evader seed",
        "passing_seeds": passing,
        "seeds": per_seed,
    }


def cg_e3(records: dict) -> dict:
    """CG-E3 (reported, reading 5): the share of the most-used evader tactic
    per cell; flag a seed whose evader uses >= 0.95 of its decisions on ONE
    tactic against EVERY one of the five pursuers."""
    per_cell: dict = {}
    for n in CELL_ORDER:
        if n not in records:
            continue
        shares = records[n]["aggregate"]["picks_share"]
        top = max(shares, key=shares.get) if shares else None
        per_cell[n] = {
            "top_tactic": top,
            "top_share": None if top is None else shares[top],
        }
    flagged: list[int] = []
    for s in SEEDS:
        cells = [cell_name(f"EVS-{s}", p) for p in PURSUERS]
        if (
            all(c in per_cell for c in cells)
            and all((per_cell[c]["top_share"] or 0.0) >= 0.95 for c in cells)
            and len({per_cell[c]["top_tactic"] for c in cells}) == 1  # the SAME tactic (review fix 4)
        ):
            flagged.append(s)
    return {
        "gate": "CG-E3 (reported, not gated): the share of the most-used "
        "evader tactic per cell",
        "verdict": "REPORTED",
        "flag_threshold": 0.95,
        "per_cell": per_cell,
        "flagged_seeds": sorted(flagged),
    }


def h_safe(records: dict, probes: dict | None = None) -> dict:
    """H-SAFE: 0 contact steps in the 15 cells AND the three arms' probe rows
    (A3.1: absence = offender; the PURSUER's shield, as always)."""
    probes = read_probes() if probes is None else probes
    have = {n: records[n] for n in CELL_ORDER if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violation_steps", 0)))
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    train: dict = {}
    for arm in T2.ARM_ORDER:
        block = probe_block(probes, arm)
        rows = block["rows"]
        train[arm] = {
            "n_probe_rows": block["n_probe_rows"],
            "train_violations_max": max(
                (int(r["train_violations"]) for r in rows if "train_violations" in r),
                default=None,
            ),
            "probe_violations_total": sum(int(r["violations"]) for r in rows if "violations" in r),
            "probe_violations_absent": any("violations" not in r for r in rows),  # A3.1 (review fix 3)
            "evidence_absent": block["evidence_absent"],
            "path": block["path"],
        }
    train_offenders = sorted(
        a for a, b in train.items()
        if (b["train_violations_max"] or 0) or b["probe_violations_total"]
    )
    missing_training = sorted(
        a for a, b in train.items()
        if b["evidence_absent"] or b["train_violations_max"] is None or b["probe_violations_absent"]
    )
    missing = _missing(records)
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS"
        if (steps or train_offenders)
        else ("INCOMPLETE" if (missing or missing_training) else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": offenders,
        "training": train,
        "training_offenders": train_offenders,
        "missing_training": missing_training,
        "gated_cells": list(CELL_ORDER),
        "missing": missing,
    }


def cg_t1(records: dict, probes: dict | None = None) -> dict:
    """CG-T1, inherited (S24.1 prereg section 5): 0 `inapplicable_picks` over
    this round's 15 cells AND over every probe row's `train_inapplicable_picks`
    / `inapplicable_picks` (A3.1: an absent counter is an offender). Review fix 2."""
    cells = {n: int(records[n].get("inapplicable_picks", 0)) for n in CELL_ORDER if n in records}
    non_zero = sorted(n for n, v in cells.items() if v)
    probes = read_probes() if probes is None else probes
    training: dict = {}
    for arm, rows in probes.items():
        rows = rows if isinstance(rows, list) else (rows or {}).get("checkpoints", [])
        absent = not rows or any(("train_inapplicable_picks" not in r) or ("inapplicable_picks" not in r) for r in rows)
        mx = max((int(r.get("train_inapplicable_picks", 0)) + int(r.get("inapplicable_picks", 0)) for r in rows), default=0)
        training[arm] = {"evidence_absent": absent, "max_inapplicable": mx}
        if absent or mx:
            non_zero.append(f"probe:{arm}")
    return {
        "gate": "CG-T1 (the mask holds), inherited from S24.1 prereg section 5",
        "verdict": "REFUSE" if non_zero else "PASS",
        "cells": cells,
        "training": training,
        "offenders": non_zero,
    }


def direction(v: dict) -> str:
    """Section 2's refutation paragraph. Precedence: H-SAFE, then CG-T1,
    then completeness, then (reading 3) H-EV-3 REPORTED FIRST, then H-EV-1,
    H-EV-2 and CG-E3."""
    safe, gate1 = v["h_safe"], v["cg_t1"]
    if safe["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {safe['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(safe['offenders']) or 'the gate'}"
            + (
                f" and training contact in {', '.join(safe['training_offenders'])}"
                if safe["training_offenders"]
                else ""
            )
            + ". Stop — the filter's guarantee is the precondition of the whole "
            "round and nothing else is read."
        )
    if gate1["verdict"] == "REFUSE":
        return (
            f"CG-T1 REFUSES: inapplicable picks in {', '.join(gate1['offenders'])}. "
            "No capture rate here is read."
        )
    if v["missing"] or safe["verdict"] == "INCOMPLETE":
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record, or the arms' training probe evidence is absent (A3.1). "
            "Nothing is re-run after a number is seen (S24.1's stop rule)."
        )
    t3, t1, t2, e3 = v["h_ev_3"], v["h_ev_1"], v["h_ev_2"], v["cg_e3"]
    parts: list[str] = []
    for s in SEEDS:
        b = t3["seeds"][str(s)]
        parts.append(
            f"EVS-{s}: H-EV-3 {b['verdict']} (max capture "
            f"{b.get('max_capture', float('nan')):.3f} vs {b.get('max_pursuer', '?')})."
        )
    if t3["verdict"] != "PASS" and t3["verdict"] != "INCOMPLETE":
        failing = [s for s in SEEDS if t3["seeds"][str(s)]["verdict"] == "FAIL"]
        if failing:
            parts.append(
                f"H-EV-3 FAILS on {', '.join(f'EVS-{s}' for s in failing)}: ratio "
                "wrong for the alternation — stop."
            )
    for s in SEEDS:
        b = t1["seeds"][str(s)]
        parts.append(
            f"EVS-{s}: H-EV-1 {b['verdict']} vs SEL-{s}_TACTIC-V0 "
            f"({b.get('direction', '?')}, {b.get('delta', float('nan')):+.3f}, "
            f"p = {b.get('p', float('nan')):.3g})."
        )
    parts.append(f"H-EV-1 (round) {t1['verdict']} ({t1['n_pass']}/{len(SEEDS)} seeds).")
    for p, b in t2["pursuers"].items():
        parts.append(f"H-EV-2 vs {p}: {b['verdict']} ({b['n_pass']}/{len(SEEDS)} seeds).")
    parts.append(f"H-EV-2 (round) {t2['verdict']}.")
    if t1["verdict"] == "PASS" and t2["verdict"] == "PASS":
        parts.append(
            "H-EV-1 and H-EV-2 both PASS: the evader's learned tactic "
            "alternation lowers the shielded pursuer's capture against the "
            "selector seats (held-out SEL-80/SEL-82 for EVS-80/EVS-82, the "
            "training seat SEL-81 for EVS-81) and against the two scripted "
            "pursuers of its training mixture (RULES-V0 / LEAD-1.0; A3)."
        )
    elif t1["verdict"] != "PASS":
        parts.append(
            "H-EV-1 FAILS: the evader selector does not lower capture against "
            "the selector seats (two of three held-out, A3); S24.2's alternation "
            "claim is not licensed."
        )
    else:
        parts.append(
            "H-EV-1 passes but H-EV-2 fails: the lowering holds against the "
            "selector seats but not against the scripted pursuers of its own "
            "training mixture (RULES-V0 / LEAD-1.0; A3)."
        )
    if e3["flagged_seeds"]:
        parts.append(
            "CG-E3: a single tactic covers >= 95% of decisions against EVERY "
            "pursuer for " + ", ".join(f"EVS-{s}" for s in e3["flagged_seeds"])
            + " (reported, not gated)."
        )
    return " ".join(parts)


def verdicts(records: dict, comparators: dict | None = None, probes: dict | None = None) -> dict:
    """Every verdict of the round — a pure function of the records, the
    comparator column and the probe rows, so a test can state a round rather
    than run one."""
    missing = _missing(records)
    v = {
        "h_ev_3": h_ev_3(records),
        "h_ev_1": h_ev_1(records, comparators or {}),
        "h_ev_2": h_ev_2(records, comparators or {}),
        "cg_e3": cg_e3(records),
        "h_safe": h_safe(records, probes),
        "cg_t1": cg_t1(records, probes),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "rows": list(ROW_NAMES),
        "pursuers": list(PURSUERS),
        "seeds": list(SEEDS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": PREREG,
    }
    return v | {"direction": direction(v)}


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_BULK_KEYS = (
    "capture_ind", "steps", "violations", "n_switches", "n_zero_length_spans",
    "n_decisions", "inapplicable_picks_per_episode", "tactics", "branch",
)


def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(PURSUERS))
    width = 0.8 / len(ROW_NAMES)
    for i, row in enumerate(ROW_NAMES):
        vals = [
            np.nan if (r := records.get(cell_name(row, p))) is None else r["capture_rate"]
            for p in PURSUERS
        ]
        axes[0].bar(xs + i * width - 0.4 + width / 2, vals, width, label=row)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(list(PURSUERS), fontsize=8)
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-EV-1 / H-EV-2 / H-EV-3: capture per pursuer", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    width2 = 0.8 / (len(ROW_NAMES) * len(EVADER_ACTIONS))
    for i, row in enumerate(ROW_NAMES):
        for j, tactic in enumerate(EVADER_ACTIONS):
            vals = [
                np.nan
                if (r := records.get(cell_name(row, p))) is None
                else r["aggregate"]["picks_share"][tactic]
                for p in PURSUERS
            ]
            axes[1].bar(
                xs + (len(EVADER_ACTIONS) * i + j) * width2 - 0.4 + width2 / 2,
                vals, width2, label=f"{row} {tactic}",
            )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(list(PURSUERS), fontsize=8)
    axes[1].set_ylabel("share of DECISIONS (CG-E3)", fontsize=8)
    axes[1].set_title("the evader's pick profile per pursuer", fontsize=9)
    axes[1].legend(fontsize=5, ncol=2)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24.2 — learned evader tactic selector (RecoveryShieldV2, hold {HOLD}, "
        f"ratio {RATIO}; N = {N_EPISODES} seed-paired episodes from {GATE_SEED0})."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-E2). H-SAFE
    and the inherited CG-T1 REFUSE here, before any capture rate is read."""
    records = load_records()
    _scan = scan_records()  # (measurable, rejected, undetermined) as S24.1: audit trail into the blob (review fix 6)
    rejected = _scan[1] if isinstance(_scan, tuple) and len(_scan) > 1 else {}
    comparators = load_comparators()
    probes = read_probes()
    v = verdicts(records, comparators, probes)
    if v["cg_t1"]["verdict"] == "REFUSE":
        raise SystemExit(
            f"CG-T1 REFUSES the summary: {v['cg_t1']['offenders']}. No capture "
            "rate here is read."
        )
    if v["h_safe"]["missing_training"]:
        raise SystemExit(
            "H-SAFE REFUSES the summary: no training evidence for "
            f"{v['h_safe']['missing_training']} (A3.1: absent evidence is an "
            "offender, never a zero)."
        )
    if v["h_safe"]["verdict"] == "FAILS":
        raise SystemExit(
            "H-SAFE FAILS the summary: "
            f"{v['h_safe']['offenders'] + v['h_safe']['training_offenders']}."
        )
    blob = {
        "verdicts": v,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in records.items()
        },
        "rejected_records": rejected,
        "provenance": provenance(device),
    }
    S1._write_json(round_file("json"), blob)
    write_figure(records, round_file("png"))
    return blob


# ----------------------------------------------------------------------------
# CG-E2 (prereg section 1): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
GIF_ROWS: tuple[str, ...] = ROW_NAMES  # EVS-80/81/82
GIF_PURSUERS: tuple[str, ...] = ("SEL-81", "RULES-V0")
GIF_N = S1.GIF_N  # 3
GIF_EVERY = S1.GIF_EVERY  # 4


def gif_seeds() -> tuple[int, ...]:
    return S1.gif_seeds()  # (2 400 000, 2 400 001, 2 400 002) — a rule, not a pick


def gif_title(row: str, pursuer: str, seed: int) -> str:
    return f"{row} vs {pursuer} · episode {seed}"


def write_gifs(
    rows: list[str] | None = None, device: str = "cpu", every: int = GIF_EVERY, fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """CG-E2: EVS-80/81/82 x {SEL-81, RULES-V0} x the first three gate seeds —
    18 GIFs. Every precondition — each cell's measurable record and each row's
    checkpoint — is resolved before the first frame."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown CG-E2 row(s): {unknown}; this gate watches {GIF_ROWS}")
    seeds = gif_seeds()
    todo = [
        (row, p, seed)
        for row in wanted
        for p in GIF_PURSUERS
        for seed in seeds
        if not gif_path(row, p, seed).exists()
    ]
    for row, p in dict.fromkeys((r, p) for r, p, _ in todo):
        read_cell(cell_name(row, p))
        require_ckpt(row)
    n_all = len(wanted) * len(GIF_PURSUERS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"the first {GIF_N} gate seeds {seeds} per row vs {GIF_PURSUERS}. Watch "
        "for the evader actually alternating tactics, and for any pin (there "
        "should be none under the pursuer's recovery shield).",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for row, p, seed in todo:
        frames, log = render_cell(row, p, seed, device, every)
        path = gif_path(row, p, seed)
        save_gif(frames, str(path), fps=fps)
        print(
            f"wrote {path} ({len(frames)} frames; {log['outcome']} @ {log['steps']})",
            flush=True,
        )
    return todo


# ----------------------------------------------------------------------------
# CLI (S24.1's dispatch order, copied: cells, then the blocking GIF gate, then
# the table)
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--gifs", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cells is not None:
        run_cells(args.cells or None, args.device)
    if args.gifs is not None:
        write_gifs(args.gifs or None, args.device)
    if args.summarize:
        blob = summarize(args.device)
        v = blob["verdicts"]
        print(
            json.dumps(
                {h: v[h]["verdict"] for h in ("h_ev_3", "h_ev_1", "h_ev_2")}, indent=2
            )
        )
        print(json.dumps(v["cg_e3"], indent=2))
        print(json.dumps(v["h_safe"], indent=2))
        print(json.dumps(v["cg_t1"], indent=2))
        print(f"\n{v['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
