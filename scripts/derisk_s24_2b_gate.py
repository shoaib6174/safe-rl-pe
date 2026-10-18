"""S24.2 Step E.3 — the gate: 18 cells, the alternation's verdicts.

Pre-registration: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 3.
Where the brief and the prereg could be read to disagree, the PREREGISTRATION
wins and the reading is named at the line.

Evaluation only. Nothing here trains, selects or tunes anything, and nothing
re-runs or rewrites a comparator: `results/derisk/s24_1_cells/` and
`results/derisk/s24_2_cells/` are READ.

A rebinding derivative of `scripts/derisk_s24_1_gate.py` (S1, S24.1's gate) and
`scripts/derisk_s24_2_gate.py` (G2, E.2's own derivative of the same
technique), both loaded by FILE PATH. The ORCHESTRATION layer — `run_cells`,
`scan_records`, `load_records`, `read_cell`, `render_cell` — is generic over a
cell table and is CALLED, not copied, with `CELLS`, `cells_dir`/`cell_file`,
`record_status`, `provenance`, `write_cell`, `undetermined_record`, `ROWS`,
`build_cell_env` and `ckpt_path` rebound for the duration of a call
(`_bound`/`_ORIG`, E.2's own technique).

**`run_cell` itself is NOT rebound — S1's OWN is reused unchanged.** Every one
of this round's 18 cells is `kind == "learned"` and its env is `SelectorMDP`
(S24.1's own class, not `EvaderSelectorMDP`): `S1.run_cell`'s learned branch —
`build_cell_env` + `load_policy(spec["row"], ...).predict` +
`env.play_episode(predict, seed)` — is already exactly this round's shape, and
its record fields (`opponent`, `in_distribution`, `aggregate.picks_share` over
`pe2.selector_mdp.ACTIONS`, the pursuer's five tactics) are the ones this
round's verdicts read. Only `build_cell_env`'s OWN body is new: it extends
S24.1's opponent dispatch (`TACTIC-V0` / `GREEDY` / `HUG` through
`_set_opponent`) with the three learned evader seats (`EVS-80/81/82`, through
`game.evader.select = LearnedEvaderSelect(...)`, E.3's deliverable 1).

**The checkpoint path is this round's own trainer's.** `ckpt_path` is bound to
`derisk_s24_2b_train.py`'s `ckpt_path` (loaded by file path as `T`), so S1's
`require_ckpt`/`load_policy` — reused unchanged — resolve `S24-2B-{ARM}`
checkpoints without this file ever repeating that stem as a literal.

**The EVS seats and the pursuer seeds are resolved through E.2's and S24.1's
own gate functions.** `EVS-80/81/82`'s checkpoints are `T2.ckpt_path(name)`
(E.2's trainer); `SEL-80/81/82`'s stage-0 checkpoints never appear in THIS
round's env at all — they are read only as the comparator column below.

**Comparators are read OUTSIDE any binding, by the rule that measured them.**
`SEL-s_{TACTIC-V0,GREEDY,HUG}` from `results/derisk/s24_1_cells/`, validated by
S1's OWN `record_status`; `EVS-s'_SEL-s` from `results/derisk/s24_2_cells/`,
validated by G2's OWN `record_status` — G2 is a module object this file loads
independently of S1 (`_load` never populates `sys.modules`, so two `_load`
calls for the same path are two unrelated objects), so G2's globals are never
mutated by THIS round's `_bound()` and need no guard of their own; S1's are, so
the guard below applies to the s24_1_cells join only, exactly as G2's did.

Readings this module fixes, because the text can be read two ways:

1. **H-ALT-1's "raises capture" is `paired_indicator_test(row_ind,
   comparator_ind, p)`:** the UP direction — the row (`SEL3`) must
   out-capture the comparator (the stage-0 pursuer `SEL-s` against the SAME
   evader seed) for the row to have RAISED capture, so the row's indicator is
   the test's FIRST argument (G2's own DOWN-direction convention, reversed:
   there the comparator had to win for the row to have LOWERED capture; here
   the row has to win for it to have RAISED it).
2. **H-ALT-1 is two-level** (brief, deliverable 3): per pursuer seed `s`, over
   the THREE evader seeds `s'` (not just the two held-out ones — `EVS-81`'s
   own cell is read too, it is simply not the held-out claim), `>= 2/3` must
   PASS; then the round passes iff `>= 2/3` of the three pursuer seeds have
   THEIR per-seed verdict PASS.
3. **H-ALT-2's "non-inferiority" is `derisk_s24_os_gate.py`'s `h_os_1` shape,
   unpaired-direction**: per opponent (`TACTIC-V0`/`GREEDY`/`HUG`), per seed,
   PASS iff the LOSING-direction test (the comparator `SEL-s` beating the row
   `SEL3-s`) is NOT significant, OR the exact-count gap `k_row - k_comparator`
   clears `-0.03 * N_EPISODES` — compared on exact COUNTS (S24-OS's review
   defect 2's discipline), never a float subtraction. `>= 2/3` seeds per
   opponent, and the round passes iff ALL THREE opponents clear their bar.
4. **CG-E3's "one tactic" reads the PURSUER's picks** (this round's learner),
   over the per-cell `picks_share` of `pe2.selector_mdp.ACTIONS` (the five
   pursuer tactics) against EVERY one of the six evader opponents — the
   mirror of G2's CG-E3, which read the evader's picks against the five
   pursuer seats.
5. **CG-T1 and H-SAFE are this round's OWN (A2.4's inheritance is of the
   RULE, not of S24.1's row set)**: S24.1's `cg_t1`/`h_safe`/`cg_a2_1` close
   over `LEARNED_ROWS`/`CELL_ORDER` that are S24.1's three SEL/three NOMEM
   rows and 24 cells, which are not this round's; the functions below are
   this round's own, built the same way G2 built its own (reading THIS
   round's probe files, `T.ARM_ORDER`).

Usage:
  ./venv/bin/python scripts/derisk_s24_2b_gate.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_2b_gate.py --cells SEL3-80_EVS-81
  ./venv/bin/python scripts/derisk_s24_2b_gate.py --gifs           (the watched set)
  ./venv/bin/python scripts/derisk_s24_2b_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a
record is NEVER re-run (S24.1's stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # as S24.1 / E.2; BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib

from pe2.derisk import paired_indicator_test
from pe2.evader_selector_mdp import EvaderSelectorMDP
from pe2.learned_select import LearnedEvaderSelect

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's gate (and through it S23's and S19's), E.2's gate (for the
# `EVS-s'_SEL-s` comparator column, read through ITS OWN `record_status`), and
# this round's OWN trainer (the authority on the checkpoint path and the
# evader mixture) — all loaded by FILE PATH.
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
G2 = _load("derisk_s24_2_gate", _HERE / "derisk_s24_2_gate.py")
T2 = G2.T2  # E.2's trainer, the authority on EVS-80/81/82's checkpoint path
T = _load("derisk_s24_2b_train", _HERE / "derisk_s24_2b_train.py")
GAME = S1.GAME
HOLD = S1.HOLD

PREREG = "docs/pathway/tactics_prereg_s24_2_2026-10-17.md"
ARTEFACT_PREFIX = "s24_2b"
OUT = pathlib.Path("results/derisk")

# --- pre-registered. Do not tune. ---
GATE_SEED0 = 2_400_000  # cross-checked against S24.1 at import below
N_EPISODES = 500
P_GATE = 0.05  # H-ALT-1
NI_MARGIN = -0.03  # H-ALT-2's "count gap >= -0.03 * N"
PASS_MIN = 2  # ">= 2/3", seeds (and evader seeds, inside H-ALT-1)
RATIO = 1.2
SEEDS: tuple[int, ...] = T.SEEDS  # (80, 81, 82)

if S1.GATE_SEED0 != GATE_SEED0:
    raise SystemExit(
        f"S24.1's GATE_SEED0 is {S1.GATE_SEED0}, not {GATE_SEED0}; this round's "
        "gate stream must be the same one (cross-checked at import)."
    )
if S1.N_EPISODES != N_EPISODES:
    raise SystemExit(
        "S24.1's N differs from this round's; the comparator join and the "
        "reused orchestration would disagree."
    )
if S1.cell_config().evader_speed_ratio != RATIO:
    raise SystemExit(
        f"S24.1's cell_config ratio moved to {S1.cell_config().evader_speed_ratio}, "
        f"not this round's {RATIO}; the comparator join would read the wrong ratio."
    )

ROW_NAMES: tuple[str, ...] = tuple(f"SEL3-{s}" for s in SEEDS)
ROWS: dict[str, dict] = {
    row: {"kind": "learned", "arm": row, "seed": s, "memory": True}
    for row, s in zip(ROW_NAMES, SEEDS, strict=True)
}
# The six evader opponents (brief, deliverable 3). `EVS-80` (EVS-best) and
# TACTIC-V0/GREEDY/HUG are IN the training mixture (D156, deliverable 2);
# `EVS-81`/`EVS-82` are held out.
OPPONENTS: dict[str, dict] = {
    "TACTIC-V0": {"in_distribution": True},
    "EVS-80": {"in_distribution": True},
    "EVS-81": {"in_distribution": False},
    "EVS-82": {"in_distribution": False},
    "GREEDY": {"in_distribution": True},
    "HUG": {"in_distribution": True},
}
EVS_OPPONENTS: tuple[str, ...] = ("EVS-80", "EVS-81", "EVS-82")
NI_OPPONENTS: tuple[str, ...] = ("TACTIC-V0", "GREEDY", "HUG")  # H-ALT-2


def cell_name(row: str, opponent: str) -> str:
    return f"{row}_{opponent}"


def _build_cells() -> dict[str, dict]:
    """The 18 cells: every pursuer row x every evader opponent. Each carries
    its row, opponent and ratio STORED — S1's own field names (`opponent`,
    `in_distribution`), so S1's `run_cell` reads this table unchanged."""
    return {
        cell_name(r, o): {
            "name": cell_name(r, o),
            "row": r,
            "opponent": o,
            "in_distribution": OPPONENTS[o]["in_distribution"],
            "ratio": RATIO,
            "hold": HOLD,
            **ROWS[r],
        }
        for r in ROW_NAMES
        for o in OPPONENTS
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


def gif_path(row: str, opponent: str, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{row}_{opponent}_{seed}.gif"


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
        "shield": "RecoveryShieldV2 (D131), fresh per episode — the v0b assembly's",
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
        "trained_by": T.PREREG,
        "comparator_sources": [S1.PREREG, T2.PREREG],
        "ckpt_stem": "the trainer's own `ckpt_path` (never a literal)",
    }


# ----------------------------------------------------------------------------
# record validity (S24.1's / E.2's shape: this round's own n/seed0/ratio/opponent)
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    """S1's own `run_cell` (reused unchanged) never records a `ratio` field
    — S24.1 fixed it at 1.2 and never swept it, so its record schema has no
    such key — and this round inherits that exactly; `CELLS[name]["ratio"]`
    is metadata for provenance/tests only, never a field this checks against
    a record. `opponent`, like S1's own `name`/`n_episodes`/`seed0`, IS one
    of `run_cell`'s own output fields."""
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == S1.UNDETERMINED:
        return S1.UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    n, seed0 = rec.get("n_episodes"), rec.get("seed0")
    opponent = rec.get("opponent")
    if n != N_EPISODES:
        return "INVALID", f"n_episodes {n!r} != the pre-registered {N_EPISODES}"
    if seed0 != GATE_SEED0:
        return "INVALID", f"seed0 {seed0!r} != the pre-registered {GATE_SEED0}"
    if opponent != CELLS[name]["opponent"]:
        return "INVALID", f"opponent {opponent!r} != the cell's declared {CELLS[name]['opponent']}"
    return "MEASURABLE", ""


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    import traceback

    return {
        "name": name,
        "verdict": S1.UNDETERMINED,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "n_episodes": 0,
        "seed0": GATE_SEED0,
        "ratio": CELLS[name]["ratio"],
        "opponent": CELLS[name]["opponent"],
        "note": (
            "a crashed cell is UNDETERMINED and is NEVER re-run (the stop "
            "rule, inherited). Delete this file only with an amendment that "
            "says so, appended before the number it touches exists."
        ),
        "provenance": provenance(device),
    }


# ----------------------------------------------------------------------------
# the cell's env — the only difference between cells is the evader opponent
# ----------------------------------------------------------------------------
def _install_evader(env, opponent: str, cfg, device: str) -> None:
    """The opponent dispatch: S1's own `_set_opponent` for the scripted seats
    (`TACTIC-V0` is its own no-op — a fresh `SelectorMDP` per cell already has
    the assembly's default), and `game.evader.select` for the three learned
    `EVS` seats (deliverable 1's `LearnedEvaderSelect`, over an
    `EvaderSelectorMDP` built OVER this cell's own game — never a second
    one)."""
    if opponent in EVS_OPPONENTS:
        from sb3_contrib import MaskablePPO

        sel = EvaderSelectorMDP.over_game(env.game, env.rules, cfg=cfg)
        model = MaskablePPO.load(T2.ckpt_path(opponent), device=device)
        env.game.evader.select = LearnedEvaderSelect(sel, model, name=opponent)
        return
    S1._set_opponent(env.game, opponent, cfg, device)


def build_cell_env(
    name: str,
    cfg=None,
    device: str = "cpu",
    *,
    recorder: bool = False,
    label: str = "",
):
    """This round's cell env: `SelectorMDP` (S24.1's class — every row here is
    a pursuer selector), with this cell's opponent installed and NOTHING else
    varied. `recorder=True` builds the SAME object with the recording added."""
    from pe2.selector_mdp import SelectorMDP

    spec = CELLS[name]
    cfg = S1.cell_config() if cfg is None else cfg
    env = SelectorMDP(
        cfg.evader_speed_ratio,
        memory=spec["memory"],
        recorder=recorder,
        label=label,
        cfg=cfg,
    )
    _install_evader(env, spec["opponent"], cfg, device)
    return env


# ----------------------------------------------------------------------------
# the binding onto S24.1's gate (see the module docstring)
# ----------------------------------------------------------------------------
_BOUND_ALWAYS: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "record_status",
    "provenance", "write_cell", "undetermined_record", "ROWS",
    "build_cell_env", "ckpt_path",
)
_ORIG: dict = {
    k: getattr(S1, k)
    for k in (
        *_BOUND_ALWAYS, "run_cell", "run_cells", "scan_records", "load_records",
        "read_cell", "render_cell",
    )
}


@contextlib.contextmanager
def _bound():
    """Rebind the names above on S24.1's module for one call and restore every
    one afterwards. `run_cell` is NOT rebound — S1's own is reused through
    this binding (see the module docstring)."""
    over = {
        "CELLS": CELLS, "CELL_ORDER": CELL_ORDER,
        "cells_dir": cells_dir, "cell_file": cell_file,
        "record_status": record_status, "provenance": provenance,
        "write_cell": write_cell, "undetermined_record": undetermined_record,
        "ROWS": ROWS, "build_cell_env": build_cell_env,
        "ckpt_path": T.ckpt_path,
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


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """S1's OWN `run_cell`, run through the binding: every cell here is
    `kind == "learned"`, so its learned branch (`build_cell_env` +
    `load_policy(spec["row"], ...).predict` + `env.play_episode`) is already
    this round's shape."""
    with _bound():
        return _ORIG["run_cell"](name, device, n_episodes)


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    with _bound():
        return _ORIG["run_cells"](names, device, n_episodes)


def require_ckpt(row: str) -> pathlib.Path:
    """This round's own, calling the TRAINER's `ckpt_path` directly (never a
    literal), so a precondition check outside `_bound()` (the GIFs) still
    resolves the right path."""
    path = T.ckpt_path(row)
    if not path.exists():
        raise SystemExit(
            f"missing the final checkpoint for row {row} at {path}. The budget "
            f"is fixed at {T.TOTAL} env steps; there is no substitute."
        )
    return path


def render_cell(
    row: str, opponent: str, seed: int, device: str = "cpu", every: int | None = None
):
    with _bound():
        return _ORIG["render_cell"](
            row, opponent, seed, device, S1.GIF_EVERY if every is None else every
        )


# ----------------------------------------------------------------------------
# the comparator joins — OUTSIDE any binding, by the rule that measured them
# ----------------------------------------------------------------------------
def _assert_outside_binding(what: str) -> None:
    if not (S1.CELLS is _ORIG["CELLS"] and S1.cells_dir is _ORIG["cells_dir"]):
        raise SystemExit(
            f"{what} may not run inside a binding: with S24.1's globals rebound "
            "to this round's, `record_status` and `cell_file` would be checked "
            "against the wrong rule and read from the wrong directory."
        )


def load_s1_comparators() -> dict:
    """`SEL-s_{TACTIC-V0,GREEDY,HUG}` of `s24_1_cells`, validated by S1's OWN
    `record_status`. Never re-run; REFUSES whole if any one is not
    measurable."""
    _assert_outside_binding("the s24_1_cells comparator join")
    out: dict = {}
    problems: list[str] = []
    for s in SEEDS:
        for opponent in NI_OPPONENTS:
            name = f"SEL-{s}_{opponent}"
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
            "REFUSED: the SEL-vs-scripted comparator column (s24_1_cells) is "
            "not fully measurable: " + "; ".join(problems)
        )
    return out


def load_s2_comparators() -> dict:
    """`EVS-s'_SEL-s` of `s24_2_cells` (E.2's own schema: row = `EVS-s'`,
    pursuer = `SEL-s`), validated by G2's OWN `record_status`. G2's globals
    are never mutated by this round's `_bound()` (each `_load` call makes an
    independent module object), so no guard is needed here."""
    out: dict = {}
    problems: list[str] = []
    for s in SEEDS:
        for sp in SEEDS:
            name = G2.cell_name(f"EVS-{sp}", f"SEL-{s}")
            path = G2.cell_file(name)
            if not path.exists():
                problems.append(f"{path}: absent")
                continue
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{path}: unreadable: {exc}")
                continue
            status, why = G2.record_status(rec)
            if status != "MEASURABLE":
                problems.append(f"{path}: {status}: {why}")
                continue
            out[name] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the EVS-vs-SEL comparator column (s24_2_cells) is not "
            "fully measurable: " + "; ".join(problems)
        )
    return out


# ----------------------------------------------------------------------------
# the training probes (this round's own, over its own arms)
# ----------------------------------------------------------------------------
def read_probes() -> dict:
    """This round's three arms' probe rows, read exactly as the trainer wrote
    them."""
    out: dict = {}
    for arm in T.ARM_ORDER:
        path = T._probe_file(arm)
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
# hypotheses (prereg section 3)
# ----------------------------------------------------------------------------
def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _direction_label(delta: float, a: str, b: str) -> str:
    if delta > 0:
        return f"{a} > {b}"
    if delta < 0:
        return f"{a} < {b}"
    return f"{a} == {b}"


def _up_test(row_rec: dict, comparator_rec: dict, row_name: str, comp_name: str) -> dict:
    """Reading 1: the row's indicator is the test's FIRST argument, so `pass`
    means the row out-captures the comparator — the row RAISED capture
    relative to it (the UP direction)."""
    test = {
        "test": "paired_indicator",
        **paired_indicator_test(row_rec["capture_ind"], comparator_rec["capture_ind"], P_GATE),
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


def h_alt_1_pair(records: dict, s2_comparators: dict, s: int, sp: int) -> dict:
    row_name = cell_name(f"SEL3-{s}", f"EVS-{sp}")
    comp_name = G2.cell_name(f"EVS-{sp}", f"SEL-{s}")
    a, b = records.get(row_name), s2_comparators.get(comp_name)
    if a is None or b is None:
        missing = [n for n, v in ((row_name, a), (comp_name, b)) if v is None]
        return {
            "verdict": "INCOMPLETE", "missing": missing, "row": row_name,
            "comparator": comp_name, "pursuer_seed": s, "evader_seed": sp,
        }
    return _up_test(a, b, row_name, comp_name) | {"pursuer_seed": s, "evader_seed": sp}


def h_alt_1_seed(records: dict, s2_comparators: dict, s: int) -> dict:
    """Reading 2: ALL three evader seeds s' are tested (not only the two
    held-out ones), `>= 2/3` of them must PASS for this pursuer seed."""
    per_sp = {str(sp): h_alt_1_pair(records, s2_comparators, s, sp) for sp in SEEDS}
    verdicts_ = [b["verdict"] for b in per_sp.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "pursuer_seed": s,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "evader_seeds": per_sp,
    }


def h_alt_1(records: dict, s2_comparators: dict) -> dict:
    """H-ALT-1 (gated): for each pursuer seed s, `SEL3-s` vs `EVS-s'` raises
    capture relative to `SEL-s` vs `EVS-s'` on >= 2/3 of the three evader
    seeds; round PASS iff >= 2/3 pursuer seeds."""
    per_seed = {str(s): h_alt_1_seed(records, s2_comparators, s) for s in SEEDS}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "hypothesis": "H-ALT-1",
        "gated": True,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "rule": "paired indicator test, UP direction, p < 0.05, vs the stage-0 "
        "pursuer SEL-s on the SAME evader seed s', on >= 2/3 of the three "
        "evader seeds, on >= 2/3 pursuer seeds",
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "seeds": per_seed,
    }


def h_alt_2_seed(records: dict, s1_comparators: dict, opponent: str, s: int) -> dict:
    row_name = cell_name(f"SEL3-{s}", opponent)
    comp_name = f"SEL-{s}_{opponent}"
    a, b = records.get(row_name), s1_comparators.get(comp_name)
    if a is None or b is None:
        missing = [n for n, v in ((row_name, a), (comp_name, b)) if v is None]
        return {
            "verdict": "INCOMPLETE", "missing": missing, "row": row_name,
            "comparator": comp_name, "seed": s,
        }
    # the LOSING direction: does the comparator beat the row significantly?
    losing = paired_indicator_test(b["capture_ind"], a["capture_ind"], P_GATE)
    delta = a["capture_rate"] - b["capture_rate"]  # row - comparator (reported)
    # The margin is compared on exact COUNTS (S24-OS's review defect 2's
    # discipline): a float subtraction can land exactly off the margin in
    # binary floating point.
    gap = int(sum(a["capture_ind"])) - int(sum(b["capture_ind"]))
    passed = (not losing["pass"]) or (gap >= round(NI_MARGIN * N_EPISODES))
    return {
        "verdict": "PASS" if passed else "FAIL",
        "row": row_name,
        "comparator": comp_name,
        "seed": s,
        "capture_row": a["capture_rate"],
        "capture_comparator": b["capture_rate"],
        "delta": delta,
        "margin": NI_MARGIN,
        "losing_direction_significant": losing["pass"],
        "losing_direction_test": losing,
    }


def h_alt_2_opponent(records: dict, s1_comparators: dict, opponent: str) -> dict:
    per_seed = {str(s): h_alt_2_seed(records, s1_comparators, opponent, s) for s in SEEDS}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "opponent": opponent,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "seeds": per_seed,
    }


def h_alt_2(records: dict, s1_comparators: dict) -> dict:
    """H-ALT-2 (gated, non-inferiority): on TACTIC-V0, GREEDY and HUG,
    `SEL3-s` is not worse than `SEL-s` — per opponent, >= 2/3 seeds pass; the
    round passes iff ALL THREE opponents do."""
    per = {o: h_alt_2_opponent(records, s1_comparators, o) for o in NI_OPPONENTS}
    verds = [b["verdict"] for b in per.values()]
    return {
        "hypothesis": "H-ALT-2 (non-inferiority)",
        "gated": True,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verds
        else ("PASS" if all(v == "PASS" for v in verds) else "FAIL"),
        "rule": "per opponent: NOT significant loss (p >= 0.05) OR count gap "
        ">= -0.03 * N (exact counts), on >= 2/3 seeds, for EACH of TACTIC-V0, "
        "GREEDY, HUG",
        "opponents": per,
    }


def cg_e3(records: dict) -> dict:
    """CG-E3 (reported, reading 4): the share of the most-used PURSUER
    tactic per cell; flag a seed whose pursuer uses >= 0.95 of its decisions
    on ONE tactic against EVERY one of the six opponents."""
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
        cells = [cell_name(f"SEL3-{s}", o) for o in OPPONENTS]
        if (
            all(c in per_cell for c in cells)
            and all((per_cell[c]["top_share"] or 0.0) >= 0.95 for c in cells)
            and len({per_cell[c]["top_tactic"] for c in cells}) == 1
        ):
            flagged.append(s)
    return {
        "gate": "CG-E3 (reported, not gated): the share of the most-used "
        "pursuer tactic per cell",
        "verdict": "REPORTED",
        "flag_threshold": 0.95,
        "per_cell": per_cell,
        "flagged_seeds": sorted(flagged),
    }


def h_safe(records: dict, probes: dict | None = None) -> dict:
    """H-SAFE: 0 contact steps in the 18 cells AND in training (this round's
    OWN probes, A3.1: absence = offender; the PURSUER's shield, as always)."""
    probes = read_probes() if probes is None else probes
    have = {n: records[n] for n in CELL_ORDER if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violation_steps", 0)))
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    train: dict = {}
    for arm in T.ARM_ORDER:
        block = probe_block(probes, arm)
        rows = block["rows"]
        train[arm] = {
            "n_probe_rows": block["n_probe_rows"],
            "train_violations_max": max(
                (int(r["train_violations"]) for r in rows if "train_violations" in r),
                default=None,
            ),
            "probe_violations_total": sum(int(r["violations"]) for r in rows if "violations" in r),
            "probe_violations_absent": any("violations" not in r for r in rows),
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
    """CG-T1, inherited (S24.1's rule — reading 5): 0 `inapplicable_picks`
    over this round's 18 cells AND over every probe row's
    `train_inapplicable_picks` / `inapplicable_picks` (A3.1: an absent
    counter is an offender)."""
    cells = {n: int(records[n].get("inapplicable_picks", 0)) for n in CELL_ORDER if n in records}
    non_zero = sorted(n for n, v in cells.items() if v)
    probes = read_probes() if probes is None else probes
    training: dict = {}
    for arm, block in probes.items():
        rows = (block or {}).get("rows", [])
        absent = not rows or any(
            ("train_inapplicable_picks" not in r) or ("inapplicable_picks" not in r)
            for r in rows
        )
        mx = max(
            (int(r.get("train_inapplicable_picks", 0)) + int(r.get("inapplicable_picks", 0)) for r in rows),
            default=0,
        )
        training[arm] = {"evidence_absent": absent, "max_inapplicable": mx}
        if absent or mx:
            non_zero.append(f"probe:{arm}")
    return {
        "gate": "CG-T1 (the mask holds), inherited from S24.1 (reading 5)",
        "verdict": "REFUSE" if non_zero else "PASS",
        "cells": cells,
        "training": training,
        "offenders": non_zero,
    }


def cg_a2_1(records: dict) -> dict:
    """A2.1's guard, inherited: with every DONE-on-entry tactic masked out, a
    learned cell can take no `(search policy)` env step. S1's own tolerance
    (<= 1 per 100 000 env steps per cell, A5)."""
    SEARCH_KEY = S1.SEARCH_KEY
    cells = {n: int(records[n].get("search_policy_steps", 0)) for n in CELL_ORDER if n in records}
    env_steps = {n: int(sum(records[n].get("steps", []))) for n in cells}
    tolerance = {n: max(1, -(-env_steps[n] // 100_000)) for n in cells}
    non_zero = sorted(n for n, v in cells.items() if v)
    offenders = sorted(n for n, v in cells.items() if v > tolerance[n])
    return {
        "gate": "A2.1 — every decision spends >= 1 tactic step, inherited "
        "(S24-OS's A5: <= 1 per 100k env steps)",
        "verdict": "REFUSE" if offenders else "PASS",
        "quantity": f"{SEARCH_KEY} env steps per cell",
        "cells": cells,
        "tolerance": tolerance,
        "non_zero": non_zero,
        "offenders": offenders,
    }


def direction(v: dict) -> str:
    """Precedence: H-SAFE, then CG-T1/A2.1, then completeness, then H-ALT-1,
    H-ALT-2, CG-E3."""
    safe, gate1, gatea21 = v["h_safe"], v["cg_t1"], v["cg_a2_1"]
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
    if gatea21["verdict"] == "REFUSE":
        return (
            f"A2.1 REFUSES: excess search-policy steps in {', '.join(gatea21['offenders'])}."
        )
    if v["missing"] or safe["verdict"] == "INCOMPLETE":
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record, or the arms' training probe evidence is absent (A3.1)."
        )
    t1, t2, e3 = v["h_alt_1"], v["h_alt_2"], v["cg_e3"]
    parts: list[str] = []
    for s in SEEDS:
        b = t1["seeds"][str(s)]
        parts.append(f"SEL3-{s}: H-ALT-1 {b['verdict']} ({b['n_pass']}/3 evader seeds).")
    parts.append(f"H-ALT-1 (round) {t1['verdict']} ({t1['n_pass']}/{len(SEEDS)} seeds).")
    for o, b in t2["opponents"].items():
        parts.append(f"H-ALT-2 vs {o}: {b['verdict']} ({b['n_pass']}/{len(SEEDS)} seeds).")
    parts.append(f"H-ALT-2 (round) {t2['verdict']}.")
    if t1["verdict"] == "PASS" and t2["verdict"] == "PASS":
        parts.append(
            "H-ALT-1 and H-ALT-2 both PASS: the pursuer's re-trained selector "
            "recovers capture against the learned evader (held-out seeds "
            "included) without losing ground on the scripted opponents of its "
            "own training mixture — the alternation answers."
        )
    elif t1["verdict"] != "PASS":
        parts.append(
            "H-ALT-1 FAILS: the re-trained pursuer does not recover capture "
            "against the learned evader; the alternation's claim is not "
            "licensed."
        )
    else:
        parts.append(
            "H-ALT-1 passes but H-ALT-2 fails: the recovery against the "
            "learned evader comes at a cost against the pursuer's own "
            "training-mixture opponents."
        )
    if e3["flagged_seeds"]:
        parts.append(
            "CG-E3: a single tactic covers >= 95% of decisions against EVERY "
            "opponent for " + ", ".join(f"SEL3-{s}" for s in e3["flagged_seeds"])
            + " (reported, not gated)."
        )
    return " ".join(parts)


def verdicts(
    records: dict, s1_comparators: dict | None = None, s2_comparators: dict | None = None,
    probes: dict | None = None,
) -> dict:
    """Every verdict of the round — a pure function of the records, the two
    comparator columns and the probe rows, so a test can state a round
    rather than run one."""
    missing = _missing(records)
    v = {
        "h_alt_1": h_alt_1(records, s2_comparators or {}),
        "h_alt_2": h_alt_2(records, s1_comparators or {}),
        "cg_e3": cg_e3(records),
        "h_safe": h_safe(records, probes),
        "cg_t1": cg_t1(records, probes),
        "cg_a2_1": cg_a2_1(records),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "rows": list(ROW_NAMES),
        "opponents": list(OPPONENTS),
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

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(OPPONENTS))
    width = 0.8 / len(ROW_NAMES)
    for i, row in enumerate(ROW_NAMES):
        vals = [
            np.nan if (r := records.get(cell_name(row, o))) is None else r["capture_rate"]
            for o in OPPONENTS
        ]
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=row)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(OPPONENTS), fontsize=8)
    ax.set_ylabel("capture rate", fontsize=8)
    ax.set_title("H-ALT-1 / H-ALT-2: capture per evader opponent", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24.2 Step E.3 — the alternation (RecoveryShieldV2, hold {HOLD}, "
        f"ratio {RATIO}; N = {N_EPISODES} seed-paired episodes from {GATE_SEED0})."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs`. H-SAFE and the
    inherited CG-T1/A2.1 REFUSE here, before any capture rate is read."""
    records = load_records()
    _, rejected, _ = scan_records()
    s1_comparators = load_s1_comparators()
    s2_comparators = load_s2_comparators()
    probes = read_probes()
    v = verdicts(records, s1_comparators, s2_comparators, probes)
    if v["cg_t1"]["verdict"] == "REFUSE":
        raise SystemExit(
            f"CG-T1 REFUSES the summary: {v['cg_t1']['offenders']}. No capture "
            "rate here is read."
        )
    if v["cg_a2_1"]["verdict"] == "REFUSE":
        raise SystemExit(
            f"A2.1 REFUSES the summary: {v['cg_a2_1']['offenders']}."
        )
    if v["h_safe"]["missing_training"]:
        raise SystemExit(
            "H-SAFE REFUSES the summary: no training evidence for "
            f"{v['h_safe']['missing_training']} (A3.1)."
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
# GIFs: the watched set, BEFORE the table
# ----------------------------------------------------------------------------
GIF_ROWS: tuple[str, ...] = ROW_NAMES  # SEL3-80/81/82
GIF_OPPONENTS: tuple[str, ...] = ("EVS-80", "HUG")  # brief: EVS-best and HUG
GIF_N = S1.GIF_N  # 3
GIF_EVERY = S1.GIF_EVERY  # 4


def gif_seeds() -> tuple[int, ...]:
    return S1.gif_seeds()  # (2 400 000, 2 400 001, 2 400 002) — a rule, not a pick


def gif_title(row: str, opponent: str, seed: int) -> str:
    return f"{row} vs {opponent} · episode {seed}"


def write_gifs(
    rows: list[str] | None = None, device: str = "cpu", every: int = GIF_EVERY, fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """SEL3-80/81/82 x {EVS-80, HUG} x the first three gate seeds — 18 GIFs.
    Every precondition — each cell's measurable record and each row's
    checkpoint — is resolved before the first frame."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown row(s): {unknown}; this gate watches {GIF_ROWS}")
    seeds = gif_seeds()
    todo = [
        (row, o, seed)
        for row in wanted
        for o in GIF_OPPONENTS
        for seed in seeds
        if not gif_path(row, o, seed).exists()
    ]
    for row, o in dict.fromkeys((r, o) for r, o, _ in todo):
        read_cell(cell_name(row, o))
        require_ckpt(row)
    n_all = len(wanted) * len(GIF_OPPONENTS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"the first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS}.",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for row, o, seed in todo:
        frames, log = render_cell(row, o, seed, device, every)
        path = gif_path(row, o, seed)
        save_gif(frames, str(path), fps=fps)
        print(
            f"wrote {path} ({len(frames)} frames; {log['outcome']} @ {log['steps']})",
            flush=True,
        )
    return todo


# ----------------------------------------------------------------------------
# CLI (S24.1's dispatch order: cells, then the blocking GIF gate, then the table)
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
        print(json.dumps({h: v[h]["verdict"] for h in ("h_alt_1", "h_alt_2")}, indent=2))
        print(json.dumps(v["cg_e3"], indent=2))
        print(json.dumps(v["h_safe"], indent=2))
        print(json.dumps(v["cg_t1"], indent=2))
        print(json.dumps(v["cg_a2_1"], indent=2))
        print(f"\n{v['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
