"""S24.3-PR Step 2 — the ratio-matched gate: 18 cells, joined to the zero-shot
records, and the pre-registered verdicts.

Pre-registration: `docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md`, all
sections. Where the brief and the prereg could be read to disagree, the
PREREGISTRATION wins and the reading is named at the line.

Evaluation only. Nothing trains, selects or tunes here, and nothing re-runs or
rewrites a comparator: `results/derisk/s24_3zs_cells/` (the zero-shot column at
0.8 and 1.0) and `results/derisk/s24_1_cells/` (the 1.2 column, for H-PR-2's
profile baseline) are READ.

A THIN REBINDING of `scripts/derisk_s24_3zs.py` (the S24.3-ZS runner, frozen at
`2d8ccbc`) and through it of `scripts/derisk_s24_1_gate.py`, loaded by file path
with that runner's own `_load` + `_bound` pattern. Three things change and
nothing else: the ROW SET (six trained arms instead of six frozen 1.2 rows),
each row's single ratio (a PR row is gated at ITS ratio only, so the table is
6 x 3 and not 6 x 3 x 2) and the artefact paths — plus this round's verdicts,
which are different hypotheses about records of the same schema.

**How the reuse works, because it is the one unusual thing in this file.**
The 3zs runner's `cg_z1`, `run_cell`, `run_cells`, `build_cell_env`,
`render_cell`, `record_status`, `scan_records`/`load_records`/`read_cell`,
`undetermined_record`, `load_base_records`, `_picks`/`_shares` and S24.1's
`_paired`/`_per_seed`/`_round`, `chi2_table`, `cg_t1`, `cg_a2_1`, `h_safe`,
`read_probes`, `load_policy` are CALLED, not copied. `_bound()` rebinds an
ENUMERATED set of globals — five on the 3zs module, four on S24.1's gate — for
the duration of a call and restores every one on exit, and it nests the 3zs
module's OWN `_bound` so that S24.1 sees this round's cells. `_ORIG_Z` /
`_ORIG_S1` hold the frozen versions, so a wrapper can never re-enter itself. The
point is not brevity: a 0.8-TRAINED record and its zero-shot twin are then
written by the SAME function, so the paired comparison is between two records of
one schema rather than two schemas that look alike.
COPIED, each saying so at its definition, only where this round's text differs:
`write_gifs` (CG-P3's watch list is not CG-Z2's, and a PR row has ONE ratio),
`write_figure`, `main`.

Readings this module fixes, because the text can be read two ways:

1. **A PR row is gated at ITS ratio only** (section 4: "Each trained arm is
   gated at ITS ratio"), so `CELLS` is 6 x 3 = 18 and every cell's ratio comes
   from its row's `ARMS` entry. Nothing sweeps here.
2. **H-PR-1's round verdict is ratio 0.8's.** Section 5 says "Reported per
   ratio; a pass at 0.8 alone is the headline the question asks for", so the
   container's verdict is the 0.8 block's and every ratio's own verdict is
   reported beside it. H-PR-3 carries no such sentence and is read as the ZS
   round's H-ZS-1 was: PASS at BOTH ratios.
3. **H-PR-2 is gated at 0.8 and reported at 1.0** (section 5: "The same for 1.0
   reported, not gated"). The pick histogram pools the three opponents per row,
   which is the 3zs runner's reading 4 and its `_picks`, called here.
4. **The comparator records are validated by the rule that MEASURED them.** The
   zero-shot column by the 3zs runner's `record_status` (which also checks the
   DECLARED RATIO, so a 0.8 record filed as a 1.0 comparator is INVALID), the
   1.2 column by `load_base_records`, which is S24.1's own rule. Both joins
   REFUSE to run inside a binding: with the globals rebound they would be
   checked against THIS round's cell table and read from the wrong directory.
5. **CG-P1 is RE-READ from the arm records, not recomputed.** Section 6 puts it
   on the built TRAINING env; that env no longer exists at gate time, so the
   gate reads the block the trainer wrote and applies S24.1's A3.1 rule —
   absent evidence is an offender, never a zero. The gate's own per-cell
   construct gate is the 3zs runner's CG-Z1, which is the same assertion about
   the EVALUATION env and is called unchanged.
6. **The inherited gates are S24.1's own functions, training half included.**
   Section 6's CG-P2 reads "0 inapplicable picks (training and gate); A2.1 at
   A5's tolerance; absent training evidence is an offender", which is exactly
   `S24.1.cg_t1` + `S24.1.cg_a2_1` + `S24.1.h_safe` over this round's cells and
   this round's probe files. The 3zs runner's cut-down `cg_inherited` (which
   drops the training half because nothing trained in that round) is NOT used.
7. **The checkpoint path is the trainer's own `ckpt_path`.** The file stem the
   frozen S24.1 trainer writes is `S24-1-{ARM}` inside this round's
   `ckpts_S24-3PR-{ARM}/` (the trainer's reading 1); binding the trainer's
   function rather than restating the name is what keeps the two from drifting.

Usage:
  ./venv/bin/python scripts/derisk_s24_3pr_gate.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_3pr_gate.py --cells SELr0.8-80_HUG_r0.8
  ./venv/bin/python scripts/derisk_s24_3pr_gate.py --gifs           (CG-P3)
  ./venv/bin/python scripts/derisk_s24_3pr_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 7, the stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # prereg section 7 (as S24.1 / S24.3-ZS); BEFORE numpy's pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib

import numpy as np

from pe2.selector_mdp import ACTIONS
from pe2.structured import HOLD

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The 3zs runner (and through it S24.1's gate, S23's and S19's) and THIS round's
# trainer, both loaded by FILE PATH. The trainer is the authority on the arm
# names, the ratios and the checkpoint paths (reading 7).
Z = _load("derisk_s24_3zs", _HERE / "derisk_s24_3zs.py")
S1, S19, GAME = Z.S1, Z.S19, Z.GAME
TR = _load("derisk_s24_3pr_train", _HERE / "derisk_s24_3pr_train.py")

# --- imported and CALLED, not reimplemented ----------------------------------
_write_json, _ci, UNDETERMINED = S1._write_json, S1._ci, S1.UNDETERMINED
load_policy, require_ckpt = S1.load_policy, S1.require_ckpt
gif_seeds = S1.gif_seeds  # the first three GATE seeds — a rule, not a pick
chi2_table, _fmt = S1.chi2_table, S1._fmt
cell_name, _r, _f32_eq = Z.cell_name, Z._r, Z._f32_eq
cell_config, config_diff = Z.cell_config, Z.config_diff
CHASE, PATIENT = Z.CHASE, Z.PATIENT

PREREG = "docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md"
ARTEFACT_PREFIX = "s24_3pr"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 4 to 7). Do not tune. ---
GATE_SEED0 = Z.GATE_SEED0  # 2 400 000 — section 4: THE SAME seeds as S24.3-ZS
N_EPISODES = Z.N_EPISODES  # 500
P_GATE = Z.P_GATE  # 0.05, H-PR-1 / H-PR-3
P_CHI2 = Z.P_CHI2  # 0.01, H-PR-2
PASS_MIN = Z.PASS_MIN  # ">= 2/3", opponents and seeds alike
RATIOS: tuple[float, ...] = TR.RATIOS  # (0.8, 1.0)
BASE_RATIO = Z.BASE_RATIO  # 1.2 — H-PR-2's profile baseline, READ not re-run
HEADLINE_RATIO = 0.8  # section 5, reading 2
SEEDS: tuple[int, ...] = TR.SEEDS  # (80, 81, 82)
OPPONENTS: dict[str, dict] = Z.OPPONENTS  # GREEDY / HUG / TACTIC-V0
BASELINE_ROW = Z.BASELINE_ROW  # "LEAD-1.0" — H-PR-3's comparator
RULES_ROW = Z.RULES_ROW  # "RULES-V0" — joined and reported

# Section 3's six arms are this round's six rows, with the TRAINER's own specs.
ROWS: dict[str, dict] = {
    arm: {
        "kind": "learned",
        "arm": arm,
        "seed": int(spec["seed"]),
        "memory": bool(spec["memory"]),
        "ratio": float(spec["ratio"]),
    }
    for arm, spec in TR.ARMS.items()
}
ROW_NAMES: tuple[str, ...] = tuple(ROWS)
# The zero-shot rows this round joins against, at the SAME ratio (section 4).
ZS_ROWS: tuple[str, ...] = (*(f"SEL-{s}" for s in SEEDS), BASELINE_ROW, RULES_ROW)

if set(ROW_NAMES) & set(ZS_ROWS):
    raise SystemExit(
        "a trained row and a joined comparator share a name; the paired "
        "comparison would then read one record as both sides of itself."
    )
if (Z.GATE_SEED0, Z.N_EPISODES, Z.PASS_MIN, Z.P_GATE, Z.P_CHI2) != (
    S1.GATE_SEED0, S1.N_EPISODES, S1.PASS_MIN, S1.P_GATE, S1.P_CHI2
):  # the reuse below is sound only while the three rounds agree on these
    raise SystemExit(
        "the 3zs runner's stream, N or thresholds differ from S24.1's; the "
        "joins and the imported verdict helpers would compare different rounds."
    )


def row_ratio(row: str) -> float:
    return float(ROWS[row]["ratio"])


def pr_row(ratio: float, seed: int) -> str:
    """This round's row for one (ratio, seed) — the TRAINER's arm name."""
    return f"SELr{float(ratio):.1f}-{seed}"


def _build_cells() -> dict[str, dict]:
    """The 18 cells (reading 1): each row at ITS ratio, against the three
    opponents. Each carries its row, opponent and RATIO stored — nothing parses
    a name back apart (`SELr0.8-80` and `r0.8` both contain a separator)."""
    return {
        cell_name(r, o, row_ratio(r)): {
            "name": cell_name(r, o, row_ratio(r)),
            "row": r,
            "opponent": o,
            "ratio": row_ratio(r),
            "in_distribution": OPPONENTS[o]["in_distribution"],
            "hold": HOLD,
            **ROWS[r],
        }
        for r in ROW_NAMES
        for o in OPPONENTS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance  (this round's prefix; the 3zs shapes)
# ----------------------------------------------------------------------------
def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


def cell_file(name: str) -> pathlib.Path:
    return cells_dir() / f"{name}.json"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_gifs"


def gif_path(row: str, opponent: str, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{cell_name(row, opponent, row_ratio(row))}_{seed}.gif"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def spec_path() -> pathlib.Path:
    return _HERE.parent / PREREG


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    _write_json(path, record)
    return path


def provenance(device: str = "cpu") -> dict:
    """S19's provenance block with THIS round's spec, stream and ratios. Built
    from `S19.provenance` directly, never from the 3zs module's or S24.1's
    (which `_bound` rebinds): a record must prove which text's thresholds it was
    measured under."""
    return S19.provenance(device) | {
        "spec": PREREG,
        "spec_sha256": S1._sha256(spec_path()),
        "cg_b": None,
        "shield": "RecoveryShieldV2 (D131), fresh per episode — the v0b assembly's",
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
        "ratios": list(RATIOS),
        "base_ratio": BASE_RATIO,
        "trained_by": TR.PREREG,
        "joined_from": [Z.PREREG, S1.PREREG],
        "ckpt_stem": "the trainer's own `ckpt_path` (reading 7)",
    }


# ----------------------------------------------------------------------------
# the binding (see the module docstring)
# ----------------------------------------------------------------------------
# Each name is a global that one of the frozen functions called here reads:
#   Z.CELLS/CELL_ORDER  cg_z1, record_status, run_cell, run_cells, scan_records
#   Z.cells_dir/cell_file  scan_records, load_records, read_cell, run_cells
#   Z.provenance  run_cell, undetermined_record  (and through Z's own `_bound`,
#                 every S24.1 function that writes a record)
#   S1.ROWS  run_cells, build_cell_env     S1.LEARNED_ROWS  read_probes, cg_t1
#   S1.ckpt_path  require_ckpt / load_policy     S1.probe_file  read_probes
_BOUND_Z: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "provenance",
)
_BOUND_S1: tuple[str, ...] = ("ROWS", "LEARNED_ROWS", "ckpt_path", "probe_file")
_BOUND_KEYS: tuple[str, ...] = (*_BOUND_Z, *_BOUND_S1)
# The frozen versions, captured at import and called THROUGH these tables.
_ORIG_Z: dict = {
    k: getattr(Z, k)
    for k in (
        *_BOUND_Z, "run_cell", "run_cells", "build_cell_env", "render_cell",
        "scan_records", "load_records", "read_cell", "undetermined_record",
        "cg_z1", "cg_z1_round", "load_base_records", "record_status", "_picks",
        "_shares", "write_cell", "ROW_NAMES",
    )
}
_ORIG_S1: dict = {
    k: getattr(S1, k)
    for k in (
        *_BOUND_S1, "_paired", "_per_seed", "_round", "cg_t1", "cg_a2_1",
        "h_safe", "read_probes", "probe_coverage",
    )
}


@contextlib.contextmanager
def _bound(ratio: float | None = None, cells: dict | None = None):
    """Rebind the names above and nest the 3zs module's OWN `_bound`, which is
    what pushes this round's cells down onto S24.1. `ratio` binds the 3zs
    module's three ratio-dependent names (S24.1's take no ratio argument), which
    is what makes S24.1's `_paired` resolve `{row}_{opponent}_r{ratio}`; `cells`
    narrows the table to the one cell a call is about."""
    over_z: dict = {
        "CELLS": CELLS if cells is None else cells,
        "CELL_ORDER": CELL_ORDER if cells is None else tuple(cells),
        "cells_dir": cells_dir,
        "cell_file": cell_file,
        "provenance": provenance,
    }
    over_s1: dict = {
        "ROWS": ROWS,
        "LEARNED_ROWS": ROW_NAMES,
        "ckpt_path": TR.ckpt_path,
        "probe_file": TR.probe_file,
    }
    saved = {Z: {k: getattr(Z, k) for k in over_z}, S1: {k: getattr(S1, k) for k in over_s1}}
    for mod, over in ((Z, over_z), (S1, over_s1)):
        for k, v in over.items():
            setattr(mod, k, v)
    try:
        # The 3zs `_bound` reads the globals just rebound, so its own push onto
        # S24.1 carries THIS round's table, directory, naming and provenance.
        with Z._bound(ratio, cells):
            yield
    finally:
        for mod, was in saved.items():
            for k, v in was.items():
                setattr(mod, k, v)


def _one(name: str):
    """`_bound` narrowed to one cell, at that cell's declared ratio."""
    return _bound(CELLS[name]["ratio"], {name: CELLS[name]})


def _outside_binding(what: str) -> None:
    """Reading 4: a join must see the FROZEN globals. Inside a binding the
    comparator records would be validated against this round's cell table and
    read from this round's directory."""
    if not (
        Z.CELLS is _ORIG_Z["CELLS"]
        and Z.cells_dir is _ORIG_Z["cells_dir"]
        and S1.CELLS is Z._ORIG["CELLS"]
    ):
        raise SystemExit(
            f"{what} may not run inside a binding: with the globals rebound to "
            "this round's, `record_status` and `cell_file` would be THIS "
            "round's and the comparator column would be validated against the "
            "wrong rule and read from the wrong directory."
        )


# ----------------------------------------------------------------------------
# this round's cells (the frozen functions, under the binding)
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    with _bound():
        return _ORIG_Z["record_status"](rec)


def scan_records() -> tuple[dict, dict, dict]:
    with _bound():
        return _ORIG_Z["scan_records"]()


def load_records(*, quiet: bool = False) -> dict:
    with _bound():
        return _ORIG_Z["load_records"](quiet=quiet)


def read_cell(name: str) -> dict:
    with _one(name):
        return _ORIG_Z["read_cell"](name)


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    with _one(name):
        return _ORIG_Z["undetermined_record"](name, exc, device)


def cg_z1(name: str, device: str = "cpu") -> dict:
    """The 3zs runner's CG-Z1 on the BUILT EVALUATION env of one cell, before
    that cell's first episode (reading 5)."""
    with _one(name):
        return _ORIG_Z["cg_z1"](name, device)


def build_cell_env(
    name: str,
    cfg=None,
    device: str = "cpu",
    *,
    recorder: bool = False,
    label: str = "",
):
    with _one(name):
        return _ORIG_Z["build_cell_env"](name, cfg, device, recorder=recorder, label=label)


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    with _one(name):
        return _ORIG_Z["run_cell"](name, device, n_episodes)


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell with no record yet (default: all 18). Literal names
    only. The 3zs runner's refusal order is kept whole: an unknown name refuses
    against THIS round's 18, CG-Z1 is swept over the cells with no measurable
    record before the first episode, and a file that is not a measurable record
    of this round is a refusal rather than an overwrite."""
    with _bound():
        return _ORIG_Z["run_cells"](names, device, n_episodes)


def read_probes() -> dict:
    """S24.1's probe reader over THIS round's arms and probe files (reading 6)."""
    with _bound():
        return _ORIG_S1["read_probes"]()


# ----------------------------------------------------------------------------
# the joined comparator columns — READ, never re-run (section 4, reading 4)
# ----------------------------------------------------------------------------
def zs_cell_file(row: str, opponent: str, ratio: float) -> pathlib.Path:
    return _ORIG_Z["cell_file"](cell_name(row, opponent, ratio))


def load_zs_records() -> dict:
    """The ZERO-SHOT column at 0.8 and 1.0 for the rows this round joins, keyed
    by the 3zs runner's own cell names (which are this round's naming too), or
    SystemExit. Validated by the 3zs runner's OWN `record_status`, which also
    checks the declared ratio. Nothing here writes, moves or re-runs them."""
    _outside_binding("the zero-shot join")
    out: dict = {}
    problems: list[str] = []
    for row in ZS_ROWS:
        for ratio in RATIOS:
            for o in OPPONENTS:
                path = zs_cell_file(row, o, ratio)
                if not path.exists():
                    problems.append(f"{path}: absent")
                    continue
                try:
                    rec = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    problems.append(f"{path}: unreadable: {exc}")
                    continue
                status, why = _ORIG_Z["record_status"](rec)
                if status != "MEASURABLE":
                    problems.append(f"{path}: {status}: {why}")
                    continue
                out[cell_name(row, o, ratio)] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the zero-shot column is S24.3-ZS's record and must be "
            "measurable by its own validity rule (N, stream AND declared ratio) "
            "for every row x opponent x ratio this round joins. "
            + "; ".join(problems)
            + ". H-PR-1 and H-PR-3 compare against it and are not computed from "
            "a partial column."
        )
    return out


def load_base_records() -> dict:
    """The 1.2 column, through the 3zs runner's OWN join (which is S24.1's own
    validity rule): H-PR-2's profile baseline is D151's record."""
    _outside_binding("the 1.2 join")
    return _ORIG_Z["load_base_records"]()


def load_joined() -> dict:
    """Both comparator columns, both outside any binding."""
    return load_zs_records() | load_base_records()


# ----------------------------------------------------------------------------
# hypotheses (prereg section 5)
# ----------------------------------------------------------------------------
def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _paired_seed(records: dict, ratio: float, seed: int, base: str, name: str) -> dict:
    """One seed of H-PR-1 / H-PR-3 at one ratio: the trained row against its
    comparator on the SAME seeds and the SAME ratio, so S24.1's `_paired` is
    CALLED and the DIRECTION is reported whether or not the test passes."""
    row = pr_row(ratio, seed)
    with _bound(ratio):
        per = _ORIG_S1["_paired"](records, row, base, OPPONENTS)
    return _ORIG_S1["_per_seed"](
        per, name, row, base, {"seed": seed, "ratio": float(ratio)}
    )


def _per_ratio(
    records: dict, base_of, name: str, ratio: float, gated: bool = True
) -> dict:
    """One ratio of a paired hypothesis: `base_of(seed)` is the comparator row,
    which is seed-dependent for H-PR-1 (the zero-shot twin) and fixed for
    H-PR-3 (the law). `gated` marks whether THIS ratio sets the round verdict
    (review item 3: H-PR-1 gates on the headline ratio only)."""
    per_seed = {
        str(s): _paired_seed(records, ratio, s, base_of(s), name) for s in SEEDS
    }
    return _ORIG_S1["_round"](per_seed, f"{name} @ ratio {ratio}", gated) | {
        "ratio": float(ratio)
    }


def h_pr_1(records: dict) -> dict:
    """H-PR-1 (section 5): per ratio, the trained `SELr-s` beats the ZERO-SHOT
    `SEL-s` at that ratio, paired by seed, p < 0.05 on >= 2/3 opponents and
    >= 2/3 seeds. Reading 2: the round's verdict is ratio 0.8's."""
    per = {
        _r(x): _per_ratio(
            records, lambda s: f"SEL-{s}", "H-PR-1", x, float(x) == HEADLINE_RATIO
        )
        for x in RATIOS
    }
    head = per[_r(HEADLINE_RATIO)]
    return {
        "hypothesis": "H-PR-1 (seeing the ratio helps)",
        "verdict": head["verdict"],
        "gated": True,
        "rule": (
            f"the round's verdict is ratio {HEADLINE_RATIO}'s (section 5: 'a "
            "pass at 0.8 alone is the headline the question asks for'); every "
            "ratio is reported"
        ),
        "headline_ratio": HEADLINE_RATIO,
        "comparator": "the ZERO-SHOT SEL-s at the same ratio (S24.3-ZS's record)",
        "pairing": "same seed, same ratio (section 4)",
        "failing_ratios": [r for r, b in per.items() if b["verdict"] != "PASS"],
        "incomplete_ratios": [r for r, b in per.items() if b["verdict"] == "INCOMPLETE"],
        "ratios": per,
    }


def h_pr_3(records: dict) -> dict:
    """H-PR-3 (section 5): the trained row still beats `LEAD-1.0` at its ratio,
    >= 2/3 opponents, >= 2/3 seeds. Reading 2: PASS at BOTH ratios."""
    per = {
        _r(x): _per_ratio(records, lambda _s: BASELINE_ROW, "H-PR-3", x)
        for x in RATIOS
    }
    got = [b["verdict"] for b in per.values()]
    return {
        "hypothesis": "H-PR-3 (still beats the law)",
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in got
        else ("PASS" if all(v == "PASS" for v in got) else "FAIL"),
        "gated": True,
        "rule": "PASS at BOTH ratios (reading 2: section 5 names no headline here)",
        "comparator": BASELINE_ROW,
        "failing_ratios": [r for r, b in per.items() if b["verdict"] != "PASS"],
        "ratios": per,
    }


def _profile(records: dict, row: str, ratio: float) -> dict | None:
    """The row's per-DECISION pick counts at one ratio, pooled over the three
    opponents — the 3zs runner's `_picks` (its reading 4), called."""
    return _ORIG_Z["_picks"](records, row, ratio)


def h_pr_2_seed(records: dict, ratio: float, seed: int) -> dict:
    """H-PR-2 for ONE seed (section 5): the trained row's chase share at its
    ratio is HIGHER than the zero-shot `SEL-s`'s chase share at 1.2 (D151's
    record) AND the chi^2 on the 5-tactic picks between the two reads p < 0.01.

    A histogram with nothing in one of the two rows is VACUOUS (never PASS):
    there is no profile to compare. Two rows that both used the same single
    tactic is a FAIL — S24.1's own vocabulary for the degenerate table."""
    row, base = pr_row(ratio, seed), f"SEL-{seed}"
    counts = _profile(records, row, ratio)
    base_counts = _profile(records, base, BASE_RATIO)
    if counts is None or base_counts is None:
        return {
            "hypothesis": "H-PR-2",
            "row": row,
            "comparator": base,
            "seed": seed,
            "ratio": float(ratio),
            "verdict": "INCOMPLETE",
            "missing": [
                cell_name(r, o, x)
                for r, x, c in ((row, ratio, counts), (base, BASE_RATIO, base_counts))
                if c is None
                for o in OPPONENTS
            ],
        }
    shares = _ORIG_Z["_shares"](counts)
    base_shares = _ORIG_Z["_shares"](base_counts)
    chi = chi2_table(
        [[counts[a] for a in ACTIONS], [base_counts[a] for a in ACTIONS]], P_CHI2
    )
    up = shares["chase"] > base_shares["chase"]
    if chi["verdict"] == "DEGENERATE":
        verdict = "VACUOUS" if chi["n_nonzero_rows"] < 2 else "FAIL"
        note = (
            f"one of {row} / {base} took no decisions, so the two profiles "
            "cannot be compared"
            if chi["n_nonzero_rows"] < 2
            else "both selectors used the same single tactic: the profiles do "
            "not differ"
        )
    else:
        verdict = "PASS" if (chi["differs"] and up) else "FAIL"
        note = None
    return {
        "hypothesis": "H-PR-2",
        "row": row,
        "comparator": base,
        "seed": seed,
        "ratio": float(ratio),
        "statistic": "chi^2 of the per-DECISION picks histogram, 5 tactics (A2.2)",
        "verdict": verdict,
        "note": note,
        "chi2": chi,
        "chase_share": shares["chase"],
        "chase_share_comparator": base_shares["chase"],
        "chase_higher": up,
        "shares": {_r(ratio): shares, _r(BASE_RATIO): base_shares},
        "chase_tactics": list(CHASE),
        "patient_tactics": list(PATIENT),
        "pooled_over": list(OPPONENTS),
        "p_chi2": P_CHI2,
    }


def h_pr_2_ratio(records: dict, ratio: float) -> dict:
    per_seed = {str(s): h_pr_2_seed(records, ratio, s) for s in SEEDS}
    gated = float(ratio) == HEADLINE_RATIO  # reading 3
    return _ORIG_S1["_round"](per_seed, f"H-PR-2 @ ratio {ratio}", gated) | {
        "ratio": float(ratio)
    }


def h_pr_2(records: dict) -> dict:
    """Reading 3: gated at 0.8, reported at 1.0."""
    per = {_r(x): h_pr_2_ratio(records, x) for x in RATIOS}
    head = per[_r(HEADLINE_RATIO)]
    return {
        "hypothesis": "H-PR-2 (the profile shifts)",
        "verdict": head["verdict"],
        "gated": True,
        "rule": (
            f"gated at ratio {HEADLINE_RATIO} only (section 5: 'The same for 1.0 "
            "reported, not gated'); chase share > the 1.2 record AND chi^2 "
            f"p < {P_CHI2}, on >= {PASS_MIN}/{len(SEEDS)} seeds"
        ),
        "headline_ratio": HEADLINE_RATIO,
        "comparator": "the zero-shot SEL-s at 1.2 (D151's record)",
        "ratios": per,
    }


def h_safe(records: dict, probes: dict | None = None) -> dict:
    """H-SAFE (section 5): 0 contact steps in the 18 gate cells AND in training
    (the probe rows). S24.1's own `h_safe`, called — including A3.1: absent
    training evidence is `missing_training` and INCOMPLETE, never a zero."""
    with _bound():
        return _ORIG_S1["h_safe"](records, probes) | {
            "scope": "the 18 gate cells and this round's six arms' probe rows"
        }


def cg_z1_round(records: dict, base: dict | None = None) -> dict:
    """The 3zs runner's CG-Z1 over the round, read off the records' own gate
    blocks (reading 5: this is the EVALUATION env's half)."""
    with _bound():
        return _ORIG_Z["cg_z1_round"](records, base or {})


def cg_inherited(records: dict, probes: dict | None = None, rejected: dict | None = None) -> dict:
    """Reading 6 / section 6's CG-P2: S24.1's OWN `cg_t1` (gate AND training
    halves, A3.1's absent-evidence rule included) and `cg_a2_1` (A5's tolerance
    of <= 1 search-policy step per 100 000 env steps per cell). Each carries its
    own verdict and the COMBINED verdict refuses the summary."""
    with _bound():
        t1 = _ORIG_S1["cg_t1"](records, probes, rejected)
        a21 = _ORIG_S1["cg_a2_1"](records)
    return {
        "gate": "CG-P2 = CG-T1 + A2.1, inherited (prereg section 6: as S24.1)",
        "verdict": "REFUSE" if "REFUSE" in (t1["verdict"], a21["verdict"]) else "PASS",
        "offenders": list(t1["offenders"]) + list(a21["offenders"]),
        "cg_t1": t1,
        "cg_a2_1": a21,
    }


def read_arm_records() -> dict:
    """The six arm records the trainer wrote, or the fact that one is absent."""
    out: dict = {}
    for arm in ROW_NAMES:
        path = TR.arm_file(arm)
        if not path.exists():
            out[arm] = {"path": str(path), "missing": True}
            continue
        try:
            out[arm] = {"path": str(path), "missing": False, "record": json.loads(path.read_text())}
        except (OSError, json.JSONDecodeError) as exc:
            out[arm] = {"path": str(path), "missing": True, "error": str(exc)}
    return out


def cg_p1_round(arms: dict | None = None) -> dict:
    """CG-P1 RE-READ from the arm records (reading 5). An arm whose record
    carries no `cg_p1` block cannot witness "only the ratio differed", and A3.1
    makes that an offender rather than a zero."""
    arms = read_arm_records() if arms is None else arms
    blocks: dict = {}
    absent: list[str] = []
    bad: list[str] = []
    for arm in ROW_NAMES:
        rec = (arms.get(arm) or {}).get("record") or {}
        block = rec.get("cg_p1")
        blocks[arm] = block
        ratio = row_ratio(arm)
        if not block:
            absent.append(arm)
            continue
        if (
            block.get("verdict") != "PASS"
            or block.get("config_diff_vs_twin") != ["evader_speed_ratio"]
            or block.get("cfg_ratio") != ratio
            or not _f32_eq(block.get("obs_ratio"), ratio)
            or rec.get("ratio") != ratio
        ):
            bad.append(arm)
    return {
        "gate": "CG-P1 (only the ratio differs from an S24.1 arm), prereg section 6",
        "verdict": "REFUSE" if (absent or bad) else "PASS",
        "offenders": bad + absent,
        "failed": bad,
        "absent_evidence": absent,  # A3.1
        "n_checked": len(ROW_NAMES),
        "blocks": blocks,
        "rule": (
            "re-read from the TRAINING env's own gate block; absent evidence is "
            "an offender, never a zero (S24.1 A3.1)"
        ),
    }


def verdicts(
    records: dict,
    joined: dict | None = None,
    probes: dict | None = None,
    rejected: dict | None = None,
    arms: dict | None = None,
) -> dict:
    """Every verdict of the round — a pure function of this round's records, the
    joined columns, the probe rows, the rejected map and the arm records, so a
    test can state a round rather than run one."""
    joined = {} if joined is None else joined
    all_records = records | joined
    missing = _missing(records)
    return {
        "h_pr_1": h_pr_1(all_records),
        "h_pr_2": h_pr_2(all_records),
        "h_pr_3": h_pr_3(all_records),
        "h_safe": h_safe(records, probes),
        "cg_p1": cg_p1_round(arms),
        "cg_z1": cg_z1_round(records, joined),
        "cg_inherited": cg_inherited(records, probes, rejected),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "n_joined": len(joined),
        "rows": list(ROW_NAMES),
        "zs_rows": list(ZS_ROWS),
        "opponents": list(OPPONENTS),
        "ratios": list(RATIOS),
        "base_ratio": BASE_RATIO,
        "seeds": list(SEEDS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": PREREG,
        "joined_from": [Z.PREREG, S1.PREREG],
    }


def direction(v: dict) -> str:
    """Section 5's refutation paragraph, in the order of precedence the round's
    logic imposes: H-SAFE, then the gates that REFUSE, then completeness, then
    H-PR-1, H-PR-2 and H-PR-3."""
    safe, p1, p2, p3 = v["h_safe"], v["h_pr_1"], v["h_pr_2"], v["h_pr_3"]
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
    for gate in (v["cg_p1"], v["cg_z1"], v["cg_inherited"]):
        if gate["verdict"] == "REFUSE":
            return (
                f"{gate['gate']} REFUSES: {gate['offenders']}. Section 6 puts the "
                "construct gates before any capture rate, so no verdict here is "
                "about seeing the ratio."
            )
    if "INCOMPLETE" in (
        safe["verdict"], p1["verdict"], p2["verdict"], p3["verdict"]
    ) or p1.get("incomplete_ratios"):  # review item 2: a non-headline ratio
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record"
            + (
                " and there is NO training evidence for "
                + ", ".join(safe["missing_training"])
                + " (A3.1)"
                if safe["missing_training"]
                else ""
            )
            + ". Every row x opponent must run on the one stream, and both "
            "comparator columns must read measurable, before a verdict is read "
            "(prereg section 7: a crashed cell is UNDETERMINED, never re-run)."
        )
    parts: list[str] = []
    for rk, block in p1["ratios"].items():
        for s in SEEDS:
            b = block["seeds"][str(s)]
            dirs = ", ".join(
                f"{o} {x['direction']} ({x['delta']:+.3f}, p = {x['p']:.3g})"
                + (" [in-distribution]" if x["in_distribution"] else "")
                for o, x in b["opponents"].items()
            )
            parts.append(
                f"{rk} seed {s}: H-PR-1 {b['verdict']} ({b['n_pass']}/"
                f"{b['n_opponents']} opponents vs the zero-shot SEL-{s}; {dirs})."
            )
    parts.append(
        f"H-PR-1 (round, ratio {HEADLINE_RATIO}) {p1['verdict']}"
        + (
            f"; it fails at {', '.join(p1['failing_ratios'])}."
            if p1["failing_ratios"]
            else ", and it holds at every ratio measured."
        )
    )
    if p1["verdict"] != "PASS":
        # Section 5's refutation, verbatim in substance:
        parts.append(
            f"H-PR-1 FAILS at {HEADLINE_RATIO}: SEEING THE RATIO DOES NOT HELP. "
            "The zero-shot selector's graceful transfer (D152) is the whole "
            "speed-agnostic story and the ratio input is inert — reported as "
            "such (section 5)."
        )
    for rk, block in p2["ratios"].items():
        for s in SEEDS:
            b = block["seeds"][str(s)]
            parts.append(
                f"{rk} seed {s}: H-PR-2 {b['verdict']} (chase share "
                f"{_fmt(b.get('chase_share'))} vs {_fmt(b.get('chase_share_comparator'))} "
                f"at {BASE_RATIO}; chi^2 p = "
                f"{_fmt((b.get('chi2') or {}).get('p'))})."
            )
    parts.append(
        f"H-PR-2 (round, ratio {HEADLINE_RATIO}) {p2['verdict']} — gated at 0.8 "
        f"only; ratio 1.0 reads {p2['ratios'][_r(1.0)]['verdict']} and gates "
        "nothing (section 5)."
    )
    if p1["verdict"] == "PASS" and p2["verdict"] != "PASS":
        parts.append(
            "H-PR-1 passes but H-PR-2 FAILS: THE GAIN COMES FROM SOMETHING "
            "OTHER THAN CHASING MORE and the mechanism stays open (section 5's "
            "refutation). The pick histograms above are the evidence."
        )
    for rk, block in p3["ratios"].items():
        parts.append(
            f"{rk}: H-PR-3 {block['verdict']} ({block['n_pass']}/{len(SEEDS)} "
            f"seeds beat {BASELINE_ROW} at that ratio)."
        )
    parts.append(
        f"H-PR-3 (round) {p3['verdict']} — the bar is PASS at BOTH ratios"
        + (
            f"; it fails at {', '.join(p3['failing_ratios'])}."
            if p3["failing_ratios"]
            else ", so the ratio-trained selector still beats the best fixed law."
        )
    )
    parts.append(
        "Scope notes: TACTIC-V0 is the TRAINING opponent and every number "
        "against it is in-distribution; the zero-shot column and the 1.2 column "
        "are S24.3-ZS's and S24.1's records, read and never re-run; H-PR-1 is "
        "paired by seed at the same ratio, so the only difference between the "
        "two sides is whether the arm trained at that ratio (section 4)."
    )
    return " ".join(parts)


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_BULK_KEYS = Z._BULK_KEYS


def write_figure(records: dict, joined: dict, path: pathlib.Path) -> None:
    """Two panels: capture per trained row against its zero-shot twin and the
    law (H-PR-1 / H-PR-3) and the chase share against the 1.2 record (H-PR-2).
    COPIED from the 3zs runner's `write_figure` because its x axis is the ratio
    and this round's is the row. Rows are NEVER parsed out of a cell name."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_records = records | joined

    def comparator(row: str, who: str) -> str:
        """The row whose cell this series reads: the trained arm itself, its
        zero-shot twin (same seed) or the law."""
        return {"trained": row, "zero-shot": f"SEL-{ROWS[row]['seed']}"}.get(
            who, BASELINE_ROW
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(ROW_NAMES))
    series = [(o, who) for o in OPPONENTS for who in ("trained", "zero-shot", BASELINE_ROW)]
    width = 0.8 / len(series)
    for i, (o, who) in enumerate(series):
        vals = [
            np.nan
            if (rec := all_records.get(cell_name(comparator(row, who), o, row_ratio(row)))) is None
            else float(rec["capture_rate"])
            for row in ROW_NAMES
        ]
        axes[0].bar(
            xs + i * width - 0.4 + width / 2, vals, width, label=f"{o} {who}"
        )
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-PR-1 / H-PR-3: capture per trained row", fontsize=9)
    for j, (label, who, ratio_of) in enumerate(
        (
            ("trained, at its ratio", "trained", row_ratio),
            (f"zero-shot, at {BASE_RATIO}", "zero-shot", lambda _row: BASE_RATIO),
        )
    ):
        vals = [
            np.nan
            if (c := _profile(all_records, comparator(row, who), ratio_of(row))) is None
            else _ORIG_Z["_shares"](c)["chase"]
            for row in ROW_NAMES
        ]
        axes[1].bar(xs + j * 0.4 - 0.2, vals, 0.4, label=label)
    axes[1].set_ylabel(f"share of DECISIONS on {'+'.join(CHASE)}", fontsize=8)
    axes[1].set_title("H-PR-2: the chase share", fontsize=9)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(list(ROW_NAMES), fontsize=7, rotation=20)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24.3-PR — per-ratio training of the tactic selector (hold {HOLD}; N = "
        f"{N_EPISODES} seed-paired episodes from {GATE_SEED0}; each row at ITS "
        "ratio). The zero-shot and 1.2 columns are S24.3-ZS's and S24.1's "
        "records. TACTIC-V0 is the training opponent (in-distribution)."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-P3).

    CG-P1, CG-Z1 and the inherited CG-T1/A2.1 REFUSE here (section 6: before any
    capture rate is read), absent training evidence refuses (A3.1), and so does
    a comparator column that is not measurable by the rule that measured it."""
    records = load_records()
    joined = load_joined()
    probes = read_probes()
    _measurable, rejected, _by_cell = scan_records()
    v = verdicts(records, joined, probes, rejected)
    for gate in (v["cg_p1"], v["cg_z1"], v["cg_inherited"]):
        if gate["verdict"] == "REFUSE":
            raise SystemExit(
                f"{gate['gate']} REFUSES the summary: {gate['offenders']}. "
                "Section 6 puts the construct gates before any capture rate, so "
                "no table is written while this stands."
            )
    if v["h_safe"]["missing_training"]:
        raise SystemExit(
            "H-SAFE REFUSES the summary: no training evidence for "
            f"{v['h_safe']['missing_training']} (probe rows absent or without "
            "`train_violations`). A3.1: absent evidence is an offender, never a "
            "zero, so H-SAFE stays INCOMPLETE and no table is written."
        )
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "rejected_records": rejected,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in (records | joined).items()
        },
        "provenance": provenance(device),
    }
    _write_json(round_file("json"), blob)
    write_figure(records, joined, round_file("png"))
    return blob


# ----------------------------------------------------------------------------
# CG-P3 (prereg section 6): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
# The seeds are a RULE, not a pick: the FIRST THREE gate seeds (`S1.gif_seeds`),
# so no scan can quietly choose which episodes are watched — and 2 400 001 is
# the episode CG-Z2 watched the zero-shot selector wait in AMBUSH for ~700
# steps. 2 rows x 2 opponents x 3 seeds = 12 GIFs. No caption carries a measured
# rate (S19 A5.1); the FRAME label is the v0b game's own
# `EVADER: ... | PURSUER: ...`.
GIF_ROWS: tuple[str, ...] = ("SELr0.8-81", "SELr1.0-81")  # section 6, CG-P3
GIF_OPPONENTS: tuple[str, ...] = ("TACTIC-V0", "GREEDY")
GIF_N = S1.GIF_N  # 3
GIF_EVERY = S1.GIF_EVERY  # 4


def gif_title(row: str, opponent: str, seed: int) -> str:
    """The file's own label: the row, the opponent, the RATIO and the episode
    SEED, and nothing else (A5.1). The 3zs runner's `gif_title`, called."""
    return Z.gif_title(row, opponent, row_ratio(row), seed)


def render_cell(
    row: str, opponent: str, seed: int, device: str = "cpu", every: int = GIF_EVERY
) -> tuple[list, dict]:
    """One gate episode of this cell, as frames — the 3zs runner's `render_cell`
    under the binding, so the selector, opponent, hold, shield and observation
    flags are `run_cell`'s and only the recording differs."""
    with _bound():
        return _ORIG_Z["render_cell"](row, opponent, row_ratio(row), seed, device, every)


def write_gifs(
    rows: list[str] | None = None,
    device: str = "cpu",
    every: int = GIF_EVERY,
    fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """CG-P3: `SELr0.8-81` and `SELr1.0-81` vs `TACTIC-V0` and `GREEDY` on the
    first three gate seeds — 12 GIFs, each row at ITS ratio.

    COPIED from the 3zs runner's `write_gifs` because the watch list is this
    round's and a row here has ONE ratio rather than two. Every precondition is
    resolved before the first frame — each cell's measurable record and the
    row's checkpoint — because a pass that renders one row and then refuses
    leaves a half-watched gate, and that is how a construct error survives
    (D125)."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown CG-P3 row(s): {unknown}; this gate watches {GIF_ROWS}")
    seeds = gif_seeds()
    todo = [
        (row, o, seed)
        for row in wanted
        for o in GIF_OPPONENTS
        for seed in seeds
        if not gif_path(row, o, seed).exists()
    ]
    # Every record first, THEN every checkpoint: a pass that refuses on the
    # second row's checkpoint while the first row's record is also missing would
    # name only one of the two reasons the gate cannot be watched.
    for row, o in dict.fromkeys((r, o) for r, o, _ in todo):
        read_cell(cell_name(row, o, row_ratio(row)))
    for row in dict.fromkeys(r for r, _o, _s in todo):
        require_ckpt_of(row)
    n_all = len(wanted) * len(GIF_OPPONENTS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), the "
        f"first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS}, each row at "
        "its own ratio. CG-P3 asks: on 2 400 001 — the episode where the "
        "ZERO-SHOT selector waited in AMBUSH for ~700 steps — does the "
        "ratio-trained selector chase the slower evader down; is there any pin.",
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


def require_ckpt_of(row: str) -> pathlib.Path:
    """S24.1's `require_ckpt` under the binding, so the path it refuses on is
    the TRAINER's (reading 7)."""
    with _bound():
        return require_ckpt(row)


# ----------------------------------------------------------------------------
# CLI  (the 3zs dispatch order, copied: cells, then the blocking GIF gate, then
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
    # CG-P3 is a BLOCKING construct gate: the episodes are watched before any
    # capture rate is read, so in a single invocation the GIFs are written
    # BEFORE the table is summarised.
    if args.gifs is not None:
        write_gifs(args.gifs or None, args.device)
    if args.summarize:
        blob = summarize(args.device)
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    h: {
                        "verdict": v[h]["verdict"],
                        "per_ratio": {r: b["verdict"] for r, b in v[h]["ratios"].items()},
                    }
                    for h in ("h_pr_1", "h_pr_2", "h_pr_3")
                },
                indent=2,
            )
        )
        for key in ("h_safe", "cg_p1", "cg_z1", "cg_inherited"):
            print(json.dumps(v[key], indent=2))
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']} (+{v['n_joined']} joined "
            f"comparator cells){'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
