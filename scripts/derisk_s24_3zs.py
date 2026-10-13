"""S24.3-ZS — the zero-shot speed sweep: 36 cells at ratio 0.8 and 1.0, joined
to S24.1's 1.2 column, and the pre-registered verdicts.

Pre-registration: `docs/pathway/tactics_prereg_s24_3zs_2026-10-13.md`, all
sections. Where the brief and the prereg could be read to disagree, the
PREREGISTRATION wins and the reading is named at the line.

Evaluation only. Nothing trains, selects or tunes here, and nothing re-runs or
rewrites a 1.2 cell: `results/derisk/s24_1_cells/` is READ.

A THIN PARAMETRISATION of `scripts/derisk_s24_1_gate.py` (S24.1's gate, frozen
at `d269737`), loaded by file path with that runner's own `_load` pattern. Four
things change and nothing else: the env's `evader_speed_ratio` (and so the
selector's `speed_ratio` observation entry), the row set, the cell naming and
the artefact paths — plus this round's verdicts, which are different hypotheses
about records of the same schema.

**How the reuse works, because it is the one unusual thing in this file.**
S24.1's `run_cell`, `build_cell_env`, `render_cell`, `run_cells`,
`scan_records`/`load_records`/`read_cell` and `undetermined_record` are CALLED,
not copied. They are not parametric in the ratio or the cell table (they read
their module's globals) and S24.1's runner is FROZEN — its records were measured
by it, so editing it would invalidate the join. `_bound()` therefore rebinds an
ENUMERATED set of S24.1 globals to this round's equivalents for the duration of
a call and restores every one on exit; `_ORIG` holds S24.1's own versions so a
delegating call never re-enters this module. The point is not brevity: a 0.8
record and its 1.2 twin are then written by the SAME function, so the join
compares two records of one schema rather than two schemas that look alike.
COPIED, each saying so at its definition, only where this round's text differs:
`record_status` (it must also check the DECLARED ratio), `write_gifs` (CG-Z2's
watch list is not CG-T3's), `write_figure` and `main`.

Readings this module fixes, because the text can be read two ways:

1. **CG-Z1's "observation's ratio entry" is read on the LEARNED rows.** Only
   `SelectorMDP`'s observation has a `speed_ratio` entry; the scripted rows
   drive the raw env, which has none. Every row is checked for the built game's
   `cfg.evader_speed_ratio` and for the config differing from the 1.2 twin
   (`EnvConfig()`) in that ONE field; the learned rows additionally for the
   observation entry. The evidence is written INTO the cell record (`cg_z1`), so
   the summary reads a gate that ran before that cell's first episode.
2. **The 1.2 column is validated by S24.1's OWN `record_status`** — the function
   that decided those records were measurable decides it again. `spec_sha256` is
   NOT gated: S24.1's prereg gained Amendment A5 after its cells ran, so the
   hash on disk legitimately differs from the file's. The 1.2 ratio is
   structural instead — S24.1's `cell_config()` refuses any other ratio, which
   this module asserts at join time.
3. **H-ZS-2's adjacent comparison is UNPAIRED.** Initial layouts are paired
   across ratios (same seeds); the dynamics are not, so a paired indicator test
   on 0.8 vs 1.0 would test episodes that are not pairs. The step test is
   Newcombe's square-and-add interval for the DIFFERENCE of the two rates,
   built from their Wilson intervals (`_newcombe` says why not disjointness),
   and the record and `direction` say so. A step passes iff it is a
   non-increase whose drop is significant, OR the rates are within 0.02 (the
   prereg's "ties allowed");
   anything else is a violation, named with its direction.
4. **H-ZS-3's histogram pools the three opponents per ratio** — the chi^2 is
   "on the 5-tactic histogram across ratios", one row per ratio. The chase and
   patient shares are complements (the five tactics partition the decisions), so
   the prereg's two direction clauses are one fact stated twice; both are
   evaluated as written and the redundancy is reported, not hidden.
5. **H-ZS-3 is quantified over the SEL seeds** ("the learned selectors");
   `NOMEM-81`'s shift is REPORTED beside them and gates nothing.
6. **`RULES-V0`'s "three numbers" are its capture against `TACTIC-V0`** — the
   configuration D150 measured (1.00 / 0.68 / 0.68) and so the only one
   comparable to it. GREEDY and HUG are reported alongside.
7. **H-ZS-3 failing dominates the decision rule.** Section 4's third branch says
   "per-ratio training regardless"; `regardless` is read as written and
   overrides the first branch. The balanced ratio is reported in every branch.
8. **S24.1's CG-T1 and A2.1 guards are inherited** (section 6: "as S24.1"). The
   mask is computed from an observation whose ratio entry is now OUT OF RANGE;
   if either broke at 0.8 no rate here would be readable, so both REFUSE the
   summary, at S24.1's own A5 tolerance.

Usage:
  ./venv/bin/python scripts/derisk_s24_3zs.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_3zs.py --cells SEL-80_HUG_r0.8
  ./venv/bin/python scripts/derisk_s24_3zs.py --gifs           (CG-Z2)
  ./venv/bin/python scripts/derisk_s24_3zs.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 6, the stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # prereg section 6 (as S24.1); BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.selector_mdp import ACTIONS, OBS_SLICES
from pe2.structured import HOLD

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's gate runner (and through it S23's and S19's), loaded by FILE PATH so
# this module imports the same way as a script and as a package module.
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
S23, S19, GAME = S1.S23, S1.S19, S1.GAME

# --- imported from S24.1 and CALLED, not reimplemented -----------------------
_write_json, _ci, UNDETERMINED = S1._write_json, S1._ci, S1.UNDETERMINED
load_policy, require_ckpt, ckpt_path = S1.load_policy, S1.require_ckpt, S1.ckpt_path
gif_seeds = S1.gif_seeds  # the first three GATE seeds — a rule, not a pick
chi2_table = S1.chi2_table

PREREG = "docs/pathway/tactics_prereg_s24_3zs_2026-10-13.md"
ARTEFACT_PREFIX = "s24_3zs"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 2 to 6). Do not tune. ---
GATE_SEED0 = 2_400_000  # section 3: THE SAME seeds as S24.1's 1.2 cells
N_EPISODES = 500  # section 3
P_GATE = 0.05  # section 4, H-ZS-1
P_CHI2 = 0.01  # section 4, H-ZS-3
PASS_MIN = 2  # ">= 2/3", opponents and seeds alike
TIE = 0.02  # section 4, H-ZS-2's "within 0.02 (ties allowed)"
RATIOS: tuple[float, ...] = (0.8, 1.0)  # section 3: the cells measured here
BASE_RATIO = 1.2  # section 3: S24.1's column, READ not re-run
ALL_RATIOS: tuple[float, ...] = (*RATIOS, BASE_RATIO)  # H-ZS-2's order
SEEDS: tuple[int, ...] = S1.SEEDS  # (80, 81, 82) — section 2's frozen arms
BALANCED_TARGET = 0.5  # section 4's decision rule
D150_SCRIPTED: dict[float, float] = {0.8: 1.00, 1.0: 0.68, 1.2: 0.68}  # section 1
# Section 4's H-ZS-3 split of the five tactics (reading 4: they partition).
CHASE: tuple[str, ...] = ("INTERCEPT", "LEAD")
PATIENT: tuple[str, ...] = ("COUNTER-ORBIT", "AMBUSH", "SWEEP")

BASELINE_ROW = S1.BASELINE_ROW  # "LEAD-1.0" — H-ZS-1's comparator
RULES_ROW = S1.RULES_ROW  # "RULES-V0" — the scripted check's row
CONTROL_ROW = "NOMEM-81"  # section 2: the strongest control
# Section 3's six rows, a SUBSET of S24.1's eight, with S24.1's own specs.
ROW_NAMES: tuple[str, ...] = (
    "SEL-80", "SEL-81", "SEL-82", CONTROL_ROW, BASELINE_ROW, RULES_ROW,
)
ROWS: dict[str, dict] = {r: S1.ROWS[r] for r in ROW_NAMES}
SEL_ROWS: tuple[str, ...] = tuple(f"SEL-{s}" for s in SEEDS)
LEARNED_ROWS: tuple[str, ...] = tuple(r for r in ROW_NAMES if ROWS[r]["kind"] == "learned")
OPPONENTS: dict[str, dict] = S1.OPPONENTS  # GREEDY / HUG / TACTIC-V0
SCRIPTED_CHECK_OPPONENT = "TACTIC-V0"  # reading 6: the cell D150 measured

if (S1.GATE_SEED0, S1.N_EPISODES, S1.PASS_MIN, S1.P_GATE, S1.P_CHI2) != (
    GATE_SEED0, N_EPISODES, PASS_MIN, P_GATE, P_CHI2
):  # the reuse below is sound only while the two rounds agree on these
    raise SystemExit(
        "S24.1's stream, N or thresholds differ from this round's; the 1.2 join "
        "and the imported verdict helpers would compare different rounds."
    )


def _r(ratio: float) -> str:
    return f"r{float(ratio):.1f}"


def _f32_eq(a, b) -> bool:
    """Equality at the observation's own precision: the selector's observation
    is a float32 array and 0.8 is not exactly representable in it."""
    return a is not None and np.float32(a) == np.float32(b)


def cell_name(row: str, opponent: str, ratio: float) -> str:
    return f"{row}_{opponent}_{_r(ratio)}"


def _build_cells() -> dict[str, dict]:
    """The 36 cells. Each carries its row, opponent and RATIO stored — nothing
    parses a name back apart (S24.1's convention; here it would also be wrong,
    because `LEAD-1.0` and `r1.0` both contain a separator)."""
    return {
        cell_name(r, o, x): {
            "name": cell_name(r, o, x),
            "row": r,
            "opponent": o,
            "ratio": float(x),
            "in_distribution": OPPONENTS[o]["in_distribution"],
            "hold": HOLD,
            **ROWS[r],
        }
        for x in RATIOS
        for r in ROW_NAMES
        for o in OPPONENTS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance  (this round's prefix; S24.1's shapes)
# ----------------------------------------------------------------------------
def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


def cell_file(name: str) -> pathlib.Path:
    return cells_dir() / f"{name}.json"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_gifs"


def gif_path(row: str, opponent: str, ratio: float, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{cell_name(row, opponent, ratio)}_{seed}.gif"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def spec_path() -> pathlib.Path:
    return _HERE.parent / PREREG


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    _write_json(path, record)
    return path


def provenance(device: str = "cpu") -> dict:
    """S19's provenance block with THIS round's spec hash, stream and ratios.
    Built from `S19.provenance` directly, never from `S1.provenance` (which
    `_bound` rebinds): a record must prove which text's thresholds it was
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
        "joined_from": S1.PREREG,
    }


# ----------------------------------------------------------------------------
# the ratio, and CG-Z1 (prereg section 5)
# ----------------------------------------------------------------------------
def config_diff(cfg: EnvConfig) -> list[str]:
    """The fields in which `cfg` differs from the 1.2 TWIN — `EnvConfig()`,
    whose `evader_speed_ratio` default IS S24.1's 1.2 (CG-Z1)."""
    twin = EnvConfig()
    return [
        f.name
        for f in dataclasses.fields(EnvConfig)
        if getattr(cfg, f.name) != getattr(twin, f.name)
    ]


def cell_config(ratio: float) -> EnvConfig:
    """`EnvConfig(evader_speed_ratio=ratio)`, with CG-Z1's config half asserted
    on the spot: the ratio is the ONLY field that may differ from the twin."""
    if EnvConfig().evader_speed_ratio != BASE_RATIO:
        raise SystemExit(
            f"EnvConfig's speed ratio moved to {EnvConfig().evader_speed_ratio}; "
            f"the twin this round joins is {BASE_RATIO} and nothing else."
        )
    cfg = EnvConfig(evader_speed_ratio=float(ratio))
    diff = config_diff(cfg)
    if diff != ["evader_speed_ratio"]:
        raise SystemExit(
            f"CG-Z1 REFUSES: a ratio-{ratio} config differs from the 1.2 twin in "
            f"{diff}, not in `evader_speed_ratio` alone, so a capture difference "
            "would not be attributable to the speed."
        )
    return cfg


def cg_z1(name: str, device: str = "cpu") -> dict:
    """CG-Z1 on the BUILT env of one cell, before that cell's first episode
    (reading 1). The block is carried in the cell record as the gate's
    evidence."""
    spec = CELLS[name]
    ratio = float(spec["ratio"])
    env = build_cell_env(name, cell_config(ratio), device)
    game = env.game if spec["kind"] == "learned" else env
    learned = spec["kind"] == "learned"
    block = {
        "gate": "CG-Z1 (only the ratio differs), prereg section 5",
        "cell": name,
        "ratio": ratio,
        "twin_ratio": BASE_RATIO,
        "config_diff_vs_twin": config_diff(game.cfg),
        "cfg_ratio": float(game.cfg.evader_speed_ratio),
        "obs_ratio": None,
        "obs_checked": learned,
    }
    if learned:
        obs, _info = env.reset(seed=GATE_SEED0)
        block["obs_ratio"] = float(np.asarray(obs)[OBS_SLICES["speed_ratio"]][0])
    why = [
        f"the {what} is {got}, not {ratio}"
        for what, got, check, same in (
            ("built env's cfg ratio", block["cfg_ratio"], True, block["cfg_ratio"] == ratio),
            # The observation is a float32 array, so the entry must equal the
            # declared ratio AS REPRESENTED IN float32: 0.8 is not exactly
            # representable and a bare `!=` would refuse every 0.8 learned cell.
            ("observation's ratio entry", block["obs_ratio"], learned, _f32_eq(block["obs_ratio"], ratio)),
        )
        if check and not same
    ]
    if block["config_diff_vs_twin"] != ["evader_speed_ratio"]:
        why.append(f"the built config differs from the twin in {block['config_diff_vs_twin']}")
    if why:
        raise SystemExit(
            f"CG-Z1 REFUSES {name} before any episode ran: {'; '.join(why)}. "
            "Section 5 puts this gate before any rate, because a cell whose env "
            "or policy input is not at the declared ratio measures nothing."
        )
    return block | {"verdict": "PASS"}


# ----------------------------------------------------------------------------
# the binding of S24.1's machinery to this round (see the module docstring)
# ----------------------------------------------------------------------------
# Each name is a global that one of the S24.1 functions called here reads:
#   CELLS/CELL_ORDER  run_cell, build_cell_env, render_cell, run_cells, scan_records
#   cells_dir/cell_file  scan_records, load_records, read_cell, run_cells
#   record_status  scan_records, read_cell      provenance  run_cell, undetermined_record
#   run_cell/write_cell  run_cells (so a chain writes THIS round's records)
#   cell_config  run_cell, build_cell_env, render_cell
#   cell_name  render_cell, _rec (and so _paired)   gif_title  render_cell
_BOUND_ALWAYS: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "record_status",
    "provenance", "run_cell", "write_cell",
)
_BOUND_PER_RATIO: tuple[str, ...] = ("cell_config", "cell_name", "gif_title")
_BOUND_KEYS: tuple[str, ...] = (*_BOUND_ALWAYS, *_BOUND_PER_RATIO)
# S24.1's own versions, captured at import and called THROUGH this table: a
# delegating call made inside a binding would otherwise re-enter this module.
_ORIG: dict = {
    k: getattr(S1, k)
    for k in (
        *_BOUND_KEYS, "build_cell_env", "render_cell", "run_cells", "scan_records",
        "load_records", "read_cell", "undetermined_record", "_paired",
    )
}


@contextlib.contextmanager
def _bound(ratio: float | None = None, cells: dict | None = None):
    """Rebind the names above on S24.1's module for one call and restore every
    one afterwards. `ratio` binds the three ratio-dependent names (S24.1's are
    1.2-only and take no ratio argument); `cells` narrows the table to the one
    cell a call is about, so S24.1's `run_cell`/`render_cell` resolve this
    round's spec."""
    over: dict = {
        "CELLS": CELLS if cells is None else cells,
        "CELL_ORDER": CELL_ORDER if cells is None else tuple(cells),
        "cells_dir": cells_dir,
        "cell_file": cell_file,
        "record_status": record_status,
        "provenance": provenance,
        "run_cell": run_cell,
        "write_cell": write_cell,
    }
    if ratio is not None:
        over |= {
            "cell_config": lambda: cell_config(ratio),
            "cell_name": lambda r, o: cell_name(r, o, ratio),
            "gif_title": lambda r, o, seed: gif_title(r, o, ratio, seed),
        }
    saved = {k: getattr(S1, k) for k in over}
    for k, v in over.items():
        setattr(S1, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(S1, k, v)


def _one(name: str):
    """`_bound` narrowed to one cell, at that cell's declared ratio."""
    return _bound(CELLS[name]["ratio"], {name: CELLS[name]})


# ----------------------------------------------------------------------------
# record validity — COPIED from S24.1 because this round's records must also
# carry the DECLARED RATIO (section 6: "the declared ratio")
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    """`(status, why)` — `"MEASURABLE"`, `"UNDETERMINED"` or `"INVALID"`."""
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == UNDETERMINED:
        return UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    n, seed0, ratio = rec.get("n_episodes"), rec.get("seed0"), rec.get("ratio")
    if n != N_EPISODES:
        return "INVALID", f"n_episodes {n!r} != the pre-registered {N_EPISODES}"
    if seed0 != GATE_SEED0:
        return "INVALID", f"seed0 {seed0!r} != the pre-registered {GATE_SEED0}"
    if ratio != CELLS[name]["ratio"]:
        return "INVALID", f"ratio {ratio!r} != the cell's declared {CELLS[name]['ratio']}"
    return "MEASURABLE", ""


def scan_records() -> tuple[dict, dict, dict]:
    with _bound():
        return _ORIG["scan_records"]()


def load_records(*, quiet: bool = False) -> dict:
    with _bound():
        return _ORIG["load_records"](quiet=quiet)


def read_cell(name: str) -> dict:
    with _one(name):
        return _ORIG["read_cell"](name)


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    with _one(name):
        return _ORIG["undetermined_record"](name, exc, device) | {
            "ratio": CELLS[name]["ratio"]
        }


# ----------------------------------------------------------------------------
# the 1.2 column — READ from S24.1's records, never re-run (section 3)
# ----------------------------------------------------------------------------
def s24_1_cell_file(row: str, opponent: str) -> pathlib.Path:
    return _ORIG["cell_file"](_ORIG["cell_name"](row, opponent))


def load_base_records() -> dict:
    """S24.1's 1.2 records for this round's rows, keyed by THIS round's cell
    names, or SystemExit (reading 2). Validated by S24.1's OWN `record_status`.
    Nothing here writes, moves or re-runs them."""
    if _ORIG["cell_config"]().evader_speed_ratio != BASE_RATIO:
        raise SystemExit(
            "S24.1's gate refuses to run at anything but 1.2, so its records "
            f"cannot be the {BASE_RATIO} column unless that holds; it does not."
        )
    out: dict = {}
    problems: list[str] = []
    for row in ROW_NAMES:
        for o in OPPONENTS:
            path = s24_1_cell_file(row, o)
            if not path.exists():
                problems.append(f"{path}: absent")
                continue
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{path}: unreadable: {exc}")
                continue
            status, why = _ORIG["record_status"](rec)
            if status != "MEASURABLE":
                problems.append(f"{path}: {status}: {why}")
                continue
            out[cell_name(row, o, BASE_RATIO)] = rec | {"ratio": BASE_RATIO}
    if problems:
        raise SystemExit(
            "REFUSED: the 1.2 column is S24.1's record and must be measurable by "
            "S24.1's own validity rule for every row x opponent this round joins. "
            + "; ".join(problems)
            + ". H-ZS-2 and H-ZS-3 compare against it and are not computed from a "
            "partial column."
        )
    return out


# ----------------------------------------------------------------------------
# the cell's env and the cell  (S24.1's, under the binding)
# ----------------------------------------------------------------------------
def build_cell_env(
    name: str,
    cfg: EnvConfig | None = None,
    device: str = "cpu",
    *,
    recorder: bool = False,
    label: str = "",
):
    """S24.1's `build_cell_env` at this cell's ratio: `SelectorMDP` for a
    learned row, the v0b `TacticGameMDP` for a scripted one, with NOTHING but
    the ratio, the row's `select` and the cell's opponent varied."""
    with _one(name):
        return _ORIG["build_cell_env"](name, cfg, device, recorder=recorder, label=label)


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """One cell: N paired episodes on the gate stream at this cell's ratio.
    CG-Z1 runs FIRST and refuses before the first episode; its block and the
    declared ratio (section 6's validity rule) travel in the record."""
    gate = cg_z1(name, device)
    with _one(name):
        rec = _ORIG["run_cell"](name, device, n_episodes)
    return rec | {"ratio": CELLS[name]["ratio"], "cg_z1": gate}


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell with no record yet (default: all 36). Literal names only.

    S24.1's refusal order unchanged — unknown names, then a file that is not a
    measurable record of THIS round, then every learned row's checkpoint, all
    before the first episode. CG-Z1 is then swept over the whole request, so a
    chain refuses WHOLE rather than after its first cells have been measured."""
    with _bound():
        wanted = list(CELL_ORDER) if names is None else list(names)
        if all(n in CELLS for n in wanted):
            for n in wanted:
                cg_z1(n, device)
        return _ORIG["run_cells"](names, device, n_episodes)


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4)
# ----------------------------------------------------------------------------
def _rec(records: dict, row: str, opponent: str, ratio: float) -> dict | None:
    return records.get(cell_name(row, opponent, ratio))


def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def h_zs_1_seed(records: dict, ratio: float, seed: int) -> dict:
    """H-ZS-1 for ONE seed at ONE ratio: `SEL-s` > `LEAD-1.0`, paired indicator
    test p < 0.05, on >= 2/3 opponents. This is H-TAC-1's shape at another
    ratio, so S24.1's `_paired` / `_per_seed` are CALLED — and so the direction
    is reported whether or not the test passes."""
    row = f"SEL-{seed}"
    with _bound(ratio):
        per = _ORIG["_paired"](records, row, BASELINE_ROW, OPPONENTS)
    return S1._per_seed(per, "H-ZS-1", row, BASELINE_ROW, {"seed": seed, "ratio": float(ratio)})


def h_zs_1_ratio(records: dict, ratio: float) -> dict:
    per_seed = {str(s): h_zs_1_seed(records, ratio, s) for s in SEEDS}
    return S1._round(per_seed, f"H-ZS-1 @ ratio {ratio}") | {"ratio": float(ratio)}


def h_zs_1(records: dict) -> dict:
    """Section 4: reported per ratio; the ROUND passes iff it holds at BOTH."""
    per = {_r(x): h_zs_1_ratio(records, x) for x in RATIOS}
    got = [b["verdict"] for b in per.values()]
    return {
        "hypothesis": "H-ZS-1 (graceful transfer)",
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in got
        else ("PASS" if all(v == "PASS" for v in got) else "FAIL"),
        "gated": True,
        "rule": "PASS at BOTH ratios (section 4)",
        "comparator": BASELINE_ROW,
        "failing_ratios": [r for r, b in per.items() if b["verdict"] != "PASS"],
        "ratios": per,
    }


def _wilson(rec: dict) -> list[float]:
    block = rec.get("wilson_capture") or _ci(
        int(sum(rec["capture_ind"])), int(rec["n_episodes"])
    )
    return [float(x) for x in block["wilson"]]


def _newcombe(a: dict, b: dict) -> tuple[float, list[float], list[float], list[float]]:
    """Newcombe's square-and-add (MOVER-R, "method 10") interval for the
    DIFFERENCE of two independent capture rates, built from the two Wilson
    intervals.

    Reading 3's unpaired "Wilson comparison" is THIS, not disjoint Wilson
    intervals: disjointness is markedly conservative — at N = 500 it calls a
    0.600 vs 0.520 difference non-significant where a two-proportion test
    resolves it at p = 0.01 — and a conservative test here would manufacture
    H-ZS-2 violations out of correctly ordered steps."""
    p1, p2 = float(a["capture_rate"]), float(b["capture_rate"])
    (l1, u1), (l2, u2) = _wilson(a), _wilson(b)
    d = p1 - p2
    lo = d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
    hi = d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
    return d, [lo, hi], [l1, u1], [l2, u2]


def _step(a: dict, b: dict, r_a: float, r_b: float) -> dict:
    """One adjacent step of H-ZS-2: capture at the slower-evader ratio `r_a`
    against the next ratio `r_b`. UNPAIRED, because the dynamics are not paired
    across ratios (reading 3)."""
    delta, ci, w_a, w_b = _newcombe(a, b)
    significant = bool(ci[0] > 0)
    tie = bool(abs(delta) <= TIE)
    return {
        "from_ratio": float(r_a),
        "to_ratio": float(r_b),
        "delta": delta,
        "direction": S1._direction_label(delta, _r(r_a), _r(r_b)),
        "wilson_from": w_a,
        "wilson_to": w_b,
        "delta_ci": ci,
        "test": "unpaired Newcombe square-and-add on the two Wilson intervals (reading 3)",
        "significant_drop": significant,
        "tie_within": tie,
        "tie_tolerance": TIE,
        "verdict": "PASS" if ((delta >= 0 and significant) or tie) else "FAIL",
    }


def h_zs_2(records: dict, base: dict) -> dict:
    """H-ZS-2 (section 4): for every row and opponent, capture is non-increasing
    in the ratio — 0.8 >= 1.0 >= 1.2 — with each adjacent drop significant or
    within 0.02. Every violation is reported."""
    all_records = records | base
    per: dict = {}
    violations: list[dict] = []
    incomplete: list[str] = []
    for row in ROW_NAMES:
        for o in OPPONENTS:
            got = [_rec(all_records, row, o, x) for x in ALL_RATIOS]
            key = f"{row}_{o}"
            if any(g is None for g in got):
                incomplete.append(key)
                per[key] = {
                    "verdict": "INCOMPLETE",
                    "missing": [
                        cell_name(row, o, x)
                        for x, g in zip(ALL_RATIOS, got, strict=True)
                        if g is None
                    ],
                }
                continue
            steps = [
                _step(got[i], got[i + 1], ALL_RATIOS[i], ALL_RATIOS[i + 1])
                for i in range(len(ALL_RATIOS) - 1)
            ]
            bad = [s for s in steps if s["verdict"] == "FAIL"]
            violations += [dict(s, row=row, opponent=o) for s in bad]
            per[key] = {
                "row": row,
                "opponent": o,
                "verdict": "FAIL" if bad else "PASS",
                "captures": {
                    _r(x): float(g["capture_rate"])
                    for x, g in zip(ALL_RATIOS, got, strict=True)
                },
                "steps": steps,
            }
    return {
        "hypothesis": "H-ZS-2 (ordering)",
        "verdict": "INCOMPLETE" if incomplete else ("FAIL" if violations else "PASS"),
        "gated": True,
        "rule": f"capture non-increasing over {list(ALL_RATIOS)}; each drop "
        f"significant (unpaired Wilson) or within {TIE}",
        "n_violations": len(violations),
        "violations": violations,
        "incomplete": incomplete,
        "cells": per,
    }


def _picks(records: dict, row: str, ratio: float) -> dict | None:
    """The row's per-DECISION pick counts at one ratio, pooled over the three
    opponents (reading 4). `None` if any of the three cells is absent."""
    got = [_rec(records, row, o, ratio) for o in OPPONENTS]
    if any(g is None for g in got):
        return None
    return {a: int(sum(int(g["picks"][a]) for g in got)) for a in ACTIONS}


def _shares(counts: dict) -> dict:
    total = max(1, sum(counts.values()))
    return {
        "n": int(sum(counts.values())),
        "chase": sum(counts[a] for a in CHASE) / total,
        "patient": sum(counts[a] for a in PATIENT) / total,
        "per_tactic": {a: counts[a] / total for a in ACTIONS},
    }


def h_zs_3_row(records: dict, row: str) -> dict:
    """H-ZS-3 for ONE learned row (section 4): chi^2 of the 5-tactic pick
    histogram across the three ratios p < 0.01 AND the chase share higher at 0.8
    than at 1.2 AND the patient share lower (reading 4: those two clauses are
    one fact stated twice, and both are evaluated as written)."""
    counts = {x: _picks(records, row, x) for x in ALL_RATIOS}
    missing = [x for x, c in counts.items() if c is None]
    if missing:
        return {
            "hypothesis": "H-ZS-3",
            "row": row,
            "verdict": "INCOMPLETE",
            "missing_ratios": [_r(x) for x in missing],
        }
    shares = {x: _shares(counts[x]) for x in ALL_RATIOS}
    chi = chi2_table([[counts[x][a] for a in ACTIONS] for x in ALL_RATIOS], P_CHI2)
    up = shares[0.8]["chase"] > shares[BASE_RATIO]["chase"]
    down = shares[0.8]["patient"] < shares[BASE_RATIO]["patient"]
    return {
        "hypothesis": "H-ZS-3",
        "row": row,
        "verdict": "PASS"
        if (chi["verdict"] == "OK" and chi["differs"] and up and down)
        else "FAIL",
        "chi2": chi,
        "shares": {_r(x): shares[x] for x in ALL_RATIOS},
        "chase_tactics": list(CHASE),
        "patient_tactics": list(PATIENT),
        "chase_higher_at_0.8": up,
        "patient_lower_at_0.8": down,
        "pooled_over": list(OPPONENTS),
    }


def h_zs_3(records: dict, base: dict) -> dict:
    """Section 4's wording: a pass here is SUGGESTIVE (the ratio input is out of
    the training range), a fail is the trigger for per-ratio training and not
    evidence against speed-agnosticism. Reading 5: over the SEL seeds, with
    `NOMEM-81` reported beside them."""
    all_records = records | base
    per = {row: h_zs_3_row(all_records, row) for row in SEL_ROWS}
    passing = [r for r, b in per.items() if b["verdict"] == "PASS"]
    return {
        "hypothesis": "H-ZS-3 (profile shift, zero-shot)",
        "verdict": "INCOMPLETE"
        if any(b["verdict"] == "INCOMPLETE" for b in per.values())
        else (
            "SUGGESTIVE-PASS" if len(passing) >= PASS_MIN else "FAIL→PER-RATIO-TRAINING"
        ),
        "gated": True,
        "rule": f">= {PASS_MIN}/{len(SEL_ROWS)} seeds",
        "passing_rows": passing,
        "n_pass": len(passing),
        "rows": per,
        "control": h_zs_3_row(all_records, CONTROL_ROW),  # reported, not gated
        "weak_by_construction": (
            "section 2: no tactic reads the ratio and the selector's ratio entry "
            "is OUT OF ITS TRAINING RANGE at 0.8 and 1.0"
        ),
    }


def h_safe(records: dict) -> dict:
    """H-SAFE (section 4): 0 contact steps in all 36 cells. Nothing trains here,
    so there is no training-evidence clause: the arms' training contact was
    gated at 0 by S24.1's own H-SAFE (D151) and the checkpoints are frozen."""
    have = {n: records[n] for n in CELL_ORDER if n in records}
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    missing = _missing(records)
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": sorted(n for n, r in have.items() if int(r.get("violation_steps", 0))),
        "gated_cells": list(CELL_ORDER),
        "missing": missing,
        "training": (
            "not re-measured: evaluation only, on S24.1's frozen checkpoints "
            "whose training contact was gated at 0 by D151"
        ),
    }


def cg_z1_round(records: dict, base: dict) -> dict:
    """CG-Z1 over the round, read off the records' own gate blocks. The 1.2
    column carries none — it was measured by a runner that refuses any other
    ratio, which is reading 2's structural guarantee and is stated."""
    blocks = {n: records[n].get("cg_z1") for n in CELL_ORDER if n in records}
    absent = sorted(n for n, b in blocks.items() if not b)
    bad = sorted(
        n
        for n, b in blocks.items()
        if b
        and (
            b.get("verdict") != "PASS"
            or b.get("config_diff_vs_twin") != ["evader_speed_ratio"]
            or b.get("cfg_ratio") != CELLS[n]["ratio"]
            or (b.get("obs_checked") and not _f32_eq(b.get("obs_ratio"), CELLS[n]["ratio"]))
            or records[n].get("ratio") != CELLS[n]["ratio"]
        )
    )
    return {
        "gate": "CG-Z1 (only the ratio differs), prereg section 5",
        "verdict": "REFUSE" if (absent or bad) else "PASS",
        "offenders": bad + absent,
        "failed": bad,
        "absent_evidence": absent,
        "n_checked": len(blocks),
        "base_column": {
            "n_records": len(base),
            "ratio": BASE_RATIO,
            "note": "S24.1's cells carry no CG-Z1 block: its `cell_config` "
            "refuses to run under any ratio but 1.2 (reading 2)",
        },
    }


def cg_inherited(records: dict) -> dict:
    """Reading 8 / section 6 ("as S24.1"): CG-T1's mask counter and A2.1's
    search-policy guard on THIS round's learned cells, at S24.1's own A5
    tolerance of <= 1 search-policy step per 100 000 env steps. Both REFUSE the
    summary — the mask is computed from an observation whose ratio entry is out
    of range, so if either broke here no rate in this round is readable."""
    learned = {
        n: records[n]
        for n in CELL_ORDER
        if n in records and CELLS[n]["kind"] == "learned"
    }
    picks = {n: int(r.get("inapplicable_picks", 0)) for n, r in learned.items()}
    search = {n: int(r.get("search_policy_steps", 0)) for n, r in learned.items()}
    tol = {n: max(1, -(-int(sum(r.get("steps", []))) // 100_000)) for n, r in learned.items()}
    bad = sorted(n for n, v in picks.items() if v) + sorted(
        n for n, v in search.items() if v > tol[n]
    )
    return {
        "gate": "CG-T1 + A2.1, inherited (prereg section 6: as S24.1)",
        "verdict": "REFUSE" if bad else "PASS",
        "offenders": bad,
        "inapplicable_picks": picks,
        "search_policy_steps": search,
        "tolerance": tol,
        "non_zero_search": sorted(n for n, v in search.items() if v),
        "n_learned_cells": len(learned),
    }


def scripted_check(records: dict, base: dict) -> dict:
    """Section 4, REPORTED: does `RULES-V0` reproduce D150's direction on the
    same seeds? Reading 6: the three numbers are its capture against
    `TACTIC-V0`; the other opponents are reported beside them. This block also
    carries the balanced ratio the decision rule reads."""
    all_records = records | base
    o = SCRIPTED_CHECK_OPPONENT
    got = {x: _rec(all_records, RULES_ROW, o, x) for x in ALL_RATIOS}
    if any(g is None for g in got.values()):
        return {
            "check": "scripted (reported)",
            "verdict": "INCOMPLETE",
            "missing_ratios": [_r(x) for x, g in got.items() if g is None],
            "balanced_ratio": None,
        }
    ours = {x: float(got[x]["capture_rate"]) for x in ALL_RATIOS}
    matches = bool(ours[0.8] > ours[1.0] and abs(ours[1.0] - ours[BASE_RATIO]) <= TIE)
    balanced = min(ALL_RATIOS, key=lambda x: abs(ours[x] - BALANCED_TARGET))
    return {
        "check": "scripted (reported)",
        "row": RULES_ROW,
        "opponent": o,
        "verdict": "REPRODUCES" if matches else "DIFFERS",
        "capture": {_r(x): ours[x] for x in ALL_RATIOS},
        "d150": {_r(x): D150_SCRIPTED[x] for x in ALL_RATIOS},
        "d150_direction": "0.8 > 1.0, 1.0 == 1.2 (1.00 / 0.68 / 0.68)",
        "direction_matches": matches,
        "per_opponent": {
            opp: {
                _r(x): (
                    None
                    if (g := _rec(all_records, RULES_ROW, opp, x)) is None
                    else float(g["capture_rate"])
                )
                for x in ALL_RATIOS
            }
            for opp in OPPONENTS
        },
        # Section 4: "the ratio whose scripted capture is nearest 0.5 (balanced)".
        "balanced_ratio": float(balanced),
        "balanced_capture": ours[balanced],
        "balanced_target": BALANCED_TARGET,
    }


def decision(v: dict) -> dict:
    """Section 4's decision rule, written before any number existed: H-ZS-1
    passes at both ratios -> per-ratio training is NOT needed to claim graceful
    transfer and S24.2 runs at the ratio whose scripted capture is nearest 0.5;
    H-ZS-1 fails at a ratio -> per-ratio training at that ratio (3 seeds) before
    S24.2; H-ZS-3 fails -> per-ratio training regardless, as the profile
    question's proper test (reading 7: it overrides the first branch)."""
    zs1, zs3 = v["h_zs_1"], v["h_zs_3"]
    balanced = v["scripted_check"].get("balanced_ratio")
    tail = (
        f"The balanced ratio for S24.2 is {balanced} (scripted capture nearest "
        f"{BALANCED_TARGET})."
    )
    if "INCOMPLETE" in (zs1["verdict"], zs3["verdict"]) or balanced is None:
        out = (
            "INCOMPLETE",
            None,
            (
                "the decision rule is not read from a partial round: H-ZS-1 or "
                "H-ZS-3 is INCOMPLETE, or the scripted row has no three numbers."
            ),
        )
    elif zs3["verdict"] != "SUGGESTIVE-PASS":
        out = (
            "PER-RATIO TRAINING REGARDLESS",
            "REGARDLESS",
            (
                "H-ZS-3 fails -> per-ratio training regardless, as the profile "
                f"question's proper test (section 4). {tail} Reading 7 makes "
                "this branch override H-ZS-1's."
            ),
        )
    elif zs1["verdict"] != "PASS":
        out = (
            "PER-RATIO TRAINING AT " + ", ".join(zs1["failing_ratios"]),
            [float(r.removeprefix("r")) for r in zs1["failing_ratios"]],
            "H-ZS-1 fails at "
            + ", ".join(zs1["failing_ratios"])
            + f" -> per-ratio training at that ratio (3 seeds) before S24.2. {tail}",
        )
    else:
        out = (
            "NO PER-RATIO TRAINING",
            [],
            (
                "H-ZS-1 passes at both ratios -> per-ratio training is NOT "
                f"needed to claim graceful transfer; S24.2 runs at ratio "
                f"{balanced}, whose scripted capture is nearest "
                f"{BALANCED_TARGET} (balanced)."
            ),
        )
    return {
        "rule": "section 4, verbatim",
        "verdict": out[0],
        "per_ratio_training": out[1],
        "balanced_ratio": balanced,
        "statement": out[2],
    }


def direction(v: dict) -> str:
    """Section 4's refutation paragraph, in the order of precedence the round's
    logic imposes: H-SAFE, then the gates that REFUSE, then completeness, then
    H-ZS-1, H-ZS-2, H-ZS-3, the scripted check and the decision."""
    safe, z1, z2, z3 = v["h_safe"], v["h_zs_1"], v["h_zs_2"], v["h_zs_3"]
    if safe["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {safe['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(safe['offenders'])}. Stop — the filter's guarantee is the "
            "precondition of the whole round and nothing else is read."
        )
    for gate in (v["cg_z1"], v["cg_inherited"]):
        if gate["verdict"] == "REFUSE":
            return (
                f"{gate['gate']} REFUSES: {gate['offenders']}. Section 5 puts the "
                "construct gates before any capture rate, so no verdict here is "
                "about the speed."
            )
    if "INCOMPLETE" in (safe["verdict"], z1["verdict"], z2["verdict"], z3["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record. Every row x opponent x ratio must run on the one stream, and "
            "the 1.2 column must read measurable, before a verdict is read (prereg "
            "section 6: a crashed cell is UNDETERMINED, never re-run)."
        )
    parts = []
    for rk, block in z1["ratios"].items():
        for s in SEEDS:
            b = block["seeds"][str(s)]
            dirs = ", ".join(
                f"{o} {x['direction']} ({x['delta']:+.3f}, p = {x['p']:.3g})"
                + (" [in-distribution]" if x["in_distribution"] else "")
                for o, x in b["opponents"].items()
            )
            parts.append(
                f"{rk} SEL-{s}: H-ZS-1 {b['verdict']} ({b['n_pass']}/"
                f"{b['n_opponents']} opponents vs {BASELINE_ROW}; {dirs})."
            )
    parts.append(
        f"H-ZS-1 (round) {z1['verdict']} — the bar is PASS at BOTH ratios"
        + (
            f"; it fails at {', '.join(z1['failing_ratios'])}."
            if z1["failing_ratios"]
            else ", and it holds at both: the 1.2-trained selector still beats the "
            "best fixed law where the evader is slower and where it is equal, with "
            "its ratio input out of range."
        )
    )
    parts.append(
        f"H-ZS-2 {z2['verdict']} ({z2['n_violations']} violation(s) of 0.8 >= 1.0 "
        ">= 1.2"
        + (
            ": "
            + "; ".join(
                f"{b['row']} vs {b['opponent']} {b['direction']} ({b['delta']:+.3f}, "
                f"{'significant' if b['significant_drop'] else 'n.s.'})"
                for b in z2["violations"]
            )
            if z2["violations"]
            else ""
        )
        + "). The adjacent test is UNPAIRED (disjoint Wilson intervals): the "
        "initial layouts are paired across ratios, the dynamics are not (reading 3)."
    )
    for row, b in z3["rows"].items():
        parts.append(
            f"{row}: H-ZS-3 {b['verdict']} (chi^2 p = {S1._fmt(b['chi2']['p'])}; chase "
            f"share {S1._fmt(b['shares']['r0.8']['chase'])} at 0.8 vs "
            f"{S1._fmt(b['shares']['r1.2']['chase'])} at 1.2)."
        )
    parts.append(
        f"H-ZS-3 (round) {z3['verdict']}. A pass here is SUGGESTIVE only and a fail "
        "is NOT evidence against speed-agnosticism — no tactic reads the ratio and "
        "the selector's ratio entry is out of its training range at 0.8 and 1.0 "
        f"(section 2/4). The control {CONTROL_ROW} reads {z3['control']['verdict']}."
    )
    sc = v["scripted_check"]
    parts.append(
        f"Scripted check (reported): {RULES_ROW} vs {sc['opponent']} captures "
        + " / ".join(f"{sc['capture'][_r(x)]:.3f}" for x in ALL_RATIOS)
        + " at 0.8 / 1.0 / 1.2 against D150's "
        + " / ".join(f"{D150_SCRIPTED[x]:.2f}" for x in ALL_RATIOS)
        + f" — {sc['verdict']}."
    )
    parts.append(f"DECISION: {v['decision']['statement']}")
    parts.append(
        "Scope notes: TACTIC-V0 is S24.1's TRAINING opponent and every number "
        "against it is in-distribution; the 1.2 column is S24.1's record, read and "
        "never re-run; nothing here trains, so the selectors are evaluated out of "
        "their training range by construction (section 2)."
    )
    return " ".join(parts)


def verdicts(records: dict, base: dict | None = None) -> dict:
    """Every verdict of the round — a pure function of this round's records and
    the joined 1.2 column, so a test can state a round rather than run one."""
    base = {} if base is None else base
    missing = _missing(records)
    v = {
        "h_zs_1": h_zs_1(records),
        "h_zs_2": h_zs_2(records, base),
        "h_zs_3": h_zs_3(records, base),
        "h_safe": h_safe(records),
        "cg_z1": cg_z1_round(records, base),
        "cg_inherited": cg_inherited(records),
        "scripted_check": scripted_check(records, base),
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "n_base_records": len(base),
        "rows": list(ROW_NAMES),
        "opponents": list(OPPONENTS),
        "ratios": list(RATIOS),
        "base_ratio": BASE_RATIO,
        "seeds": list(SEEDS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": PREREG,
        "joined_from": S1.PREREG,
    }
    return v | {"decision": decision(v)}


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_BULK_KEYS = (
    "capture_ind", "steps", "violations", "n_switches", "n_zero_length_spans",
    "n_decisions", "inapplicable_picks_per_episode", "tactics", "branch",
)


def write_figure(records: dict, base: dict, path: pathlib.Path) -> None:
    """Two panels: capture against the ratio per row (H-ZS-2's ordering) and the
    learned rows' chase share against the ratio (H-ZS-3's quantity). Rows are
    NEVER parsed out of a cell name."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_records = records | base
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(ALL_RATIOS))
    for row in ROW_NAMES:
        for o, style in zip(OPPONENTS, ("-", "--", ":"), strict=True):
            axes[0].plot(
                xs,
                [
                    np.nan if (g := _rec(all_records, row, o, x)) is None else g["capture_rate"]
                    for x in ALL_RATIOS
                ],
                style,
                marker="o",
                label=f"{row} vs {o}",
            )
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-ZS-2: capture against the evader speed ratio", fontsize=9)
    for row in (*SEL_ROWS, CONTROL_ROW):
        axes[1].plot(
            xs,
            [
                np.nan if (c := _picks(all_records, row, x)) is None else _shares(c)["chase"]
                for x in ALL_RATIOS
            ],
            marker="s",
            label=row,
        )
    axes[1].set_ylabel(f"share of DECISIONS on {'+'.join(CHASE)}", fontsize=8)
    axes[1].set_title("H-ZS-3: the chase share against the ratio", fontsize=9)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([_r(x) for x in ALL_RATIOS], fontsize=8)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24.3-ZS — zero-shot speed sweep of S24.1's selectors (hold {HOLD}; N = "
        f"{N_EPISODES} episodes from {GATE_SEED0}; the 1.2 column is S24.1's "
        "record). TACTIC-V0 is the training opponent (in-distribution)."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-Z2).

    CG-Z1 and the inherited CG-T1/A2.1 REFUSE here (section 5: before any
    capture rate is read), and so does a 1.2 column that is not measurable by
    S24.1's own rule (`load_base_records` raises)."""
    records = load_records()
    base = load_base_records()
    v = verdicts(records, base)
    for gate in (v["cg_z1"], v["cg_inherited"]):
        if gate["verdict"] == "REFUSE":
            raise SystemExit(
                f"{gate['gate']} REFUSES the summary: {gate['offenders']}. Section "
                "5 puts the construct gates before any capture rate, so no table "
                "is written while this stands."
            )
    _measurable, rejected, _by_cell = scan_records()
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "decision": v["decision"],
        "rejected_records": rejected,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in (records | base).items()
        },
        "provenance": provenance(device),
    }
    _write_json(round_file("json"), blob)
    write_figure(records, base, round_file("png"))
    return blob


# ----------------------------------------------------------------------------
# CG-Z2 (prereg section 5): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
# The seeds are a RULE, not a pick: the FIRST THREE gate seeds (`S1.gif_seeds`),
# so no scan can quietly choose which episodes are watched. 1 row x 2 opponents
# x 2 ratios x 3 seeds = 12 GIFs. No caption carries a measured rate (S19 A5.1);
# the FRAME label is the v0b game's own `EVADER: ... | PURSUER: ...`.
GIF_ROWS: tuple[str, ...] = ("SEL-81",)  # section 5, CG-Z2
GIF_OPPONENTS: tuple[str, ...] = ("TACTIC-V0", "GREEDY")
GIF_N = S1.GIF_N  # 3
GIF_EVERY = S1.GIF_EVERY  # 4


def gif_title(row: str, opponent: str, ratio: float, seed: int) -> str:
    """The file's own label: the row, the opponent, the RATIO and the episode
    SEED, and nothing else (A5.1)."""
    return f"{row} vs {opponent} · ratio {float(ratio):.1f} · episode {seed}"


def render_cell(
    row: str,
    opponent: str,
    ratio: float,
    seed: int,
    device: str = "cpu",
    every: int = GIF_EVERY,
) -> tuple[list, dict]:
    """One gate episode of this cell, as frames — S24.1's `render_cell` under
    the binding, so the selector, opponent, hold, shield and observation flags
    are `run_cell`'s and only the recording differs."""
    with _one(cell_name(row, opponent, ratio)):
        return _ORIG["render_cell"](row, opponent, seed, device, every)


def write_gifs(
    rows: list[str] | None = None,
    device: str = "cpu",
    every: int = GIF_EVERY,
    fps: float = 15.0,
) -> list[tuple[str, str, float, int]]:
    """CG-Z2: `SEL-81` vs `TACTIC-V0` and `GREEDY`, the first three gate seeds,
    at 0.8 and at 1.0 — 12 GIFs.

    COPIED from S24.1's `write_gifs` because the watch list is this round's and
    the cell has a fourth coordinate. Every precondition is resolved before the
    first frame — each cell's measurable record and the row's checkpoint —
    because a pass that renders one ratio and then refuses leaves a half-watched
    gate, and that is how a construct error survives (D125)."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown CG-Z2 row(s): {unknown}; this gate watches {GIF_ROWS}")
    seeds = gif_seeds()
    todo = [
        (row, o, x, seed)
        for row in wanted
        for o in GIF_OPPONENTS
        for x in RATIOS
        for seed in seeds
        if not gif_path(row, o, x, seed).exists()
    ]
    for row, o, x in dict.fromkeys((r, o, y) for r, o, y, _ in todo):
        read_cell(cell_name(row, o, x))
        if ROWS[row]["kind"] == "learned":
            require_ckpt(row)
    n_all = len(wanted) * len(GIF_OPPONENTS) * len(RATIOS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), the "
        f"first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS} at ratios "
        f"{list(RATIOS)}. CG-Z2 asks: does the SLOWER evader get caught by direct "
        "chase; does the pursuer still hop-ambush when it no longer needs to; is "
        "there any pin.",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for row, o, x, seed in todo:
        frames, log = render_cell(row, o, x, seed, device, every)
        path = gif_path(row, o, x, seed)
        save_gif(frames, str(path), fps=fps)
        print(
            f"wrote {path} ({len(frames)} frames; {log['outcome']} @ {log['steps']})",
            flush=True,
        )
    return todo


# ----------------------------------------------------------------------------
# CLI  (S24.1's dispatch order, copied: cells, then the blocking GIF gate, then
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
    # CG-Z2 is a BLOCKING construct gate: the episodes are watched before any
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
                    "h_zs_1": {
                        "verdict": v["h_zs_1"]["verdict"],
                        "per_ratio": {r: b["verdict"] for r, b in v["h_zs_1"]["ratios"].items()},
                    },
                    "h_zs_2": {
                        "verdict": v["h_zs_2"]["verdict"],
                        "n_violations": v["h_zs_2"]["n_violations"],
                    },
                    "h_zs_3": {
                        "verdict": v["h_zs_3"]["verdict"],
                        "passing_rows": v["h_zs_3"]["passing_rows"],
                    },
                },
                indent=2,
            )
        )
        for key in ("h_safe", "cg_z1", "cg_inherited", "scripted_check", "decision"):
            print(json.dumps(v[key], indent=2))
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']} (+{v['n_base_records']} joined "
            f"at {BASE_RATIO}){'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
