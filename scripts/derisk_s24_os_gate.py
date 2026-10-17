"""S24-OS Step O.2 — the gate: 12 cells, the verdicts, the construct gates.

Pre-registration: `docs/pathway/tactics_prereg_s24_os_2026-10-17.md` sections 2
to 5. Where the brief and the prereg could be read to disagree, the
PREREGISTRATION wins and the reading is named at the line.

Evaluation only. Nothing here trains, selects or tunes anything, and nothing
re-runs or rewrites a comparator: `results/derisk/s24_1_cells/` (the 1.2
column) and `results/derisk/s24_3zs_cells/` (the zero-shot 0.8 column) are
READ.

A rebinding derivative of `scripts/derisk_s24_1_gate.py` (S24.1's gate),
loaded by file path with that runner's own `_load` pattern and
`scripts/derisk_s24_3zs.py`'s `_bound` technique, in the one place it is sound:
the ORCHESTRATION layer — `run_cells`, `scan_records`, `load_records`,
`read_cell`, `render_cell` — is generic over a cell table and is CALLED, not
copied, with `CELLS`, `cells_dir`/`cell_file`, `record_status`, `provenance`,
`run_cell`, `write_cell`, `undetermined_record`, `ROWS`, `build_cell_env` and
`ckpt_path` rebound for the duration of a call (`_bound`/`_ORIG`,
`_BOUND_ALWAYS` — the same names S24.3-ZS rebinds, `ROWS`/`build_cell_env`/
`ckpt_path` added because this round's rows are not S24.1's and its
checkpoints are not at S24.1's path). Every HYPOTHESIS, the GIF list, the
figure and `main` are this round's own — S24.1's and S24.3-ZS's ask different
questions of a different row set, exactly as S24.3-ZS writes its own against
S24.1's.

**What is new rather than bound, and why (finding (a) of the brief).**
`build_cell_env` must pass `library="v1"` to `SelectorMDP`, which S24.1's
version never does, and `run_cell`'s own `picks` projection must read
`env.actions` — THIS round's six-name library — rather than S24.1's
module-level v0 `ACTIONS` (five names): `derisk_s24_1_gate.py::run_cell`
projects onto that module-level v0 `ACTIONS`, so calling it unchanged on a
library-v1 row would silently drop `ORBIT-SEARCH`'s share from every cell.
Both are therefore this file's own `build_cell_env`/`run_cell`, bound onto
S24.1's module so `run_cells` (reused unchanged) calls them.

**The checkpoint path is the trainer's own.** `ckpt_path` is bound to
`derisk_s24_os_train.py`'s `ckpt_path` (loaded by file path as `TR`), so
S24.1's `require_ckpt`/`load_policy` — reused unchanged — resolve `S24-OS-{ARM}`
checkpoints without this file ever repeating that stem as a literal.

**Comparators are read OUTSIDE any binding, by the rule that measured them.**
`SEL-{s}_{opp}` from `results/derisk/s24_1_cells/`, validated by S24.1's OWN
`record_status`/`cell_file`; `SEL-{s}_GREEDY_r0.8` from
`results/derisk/s24_3zs_cells/`, validated by S24.3-ZS's OWN `record_status`/
`cell_file`/`cell_name`. Both joins assert `S1.CELLS is` this file's captured
original before reading a single file: inside a binding, `record_status` and
`cell_file` would be THIS round's and the comparator column would be checked
against the wrong rule and read from the wrong directory (the lesson
S24.3-PR's own join states).

Readings this module fixes, because the text can be read two ways:

1. **Cell naming follows S24.1's convention at 1.2 and S24.3-ZS's at 0.8**
   (brief): `{ROW}_{OPP}` with no suffix at the round's own ratio, 1.2, and
   `{ROW}_{OPP}_r0.8` at the off-training ratio — never a bare `_r1.2` suffix,
   so the nine ratio-1.2 cell names are the exact strings S24.1's own
   `cell_name` would produce for these rows.
2. **The 0.8 cells are built exactly as `derisk_s24_3zs.py` builds its learned
   rows** (brief): `cell_config` reuses `derisk_s24_3zs.py::cell_config`
   (which asserts the built config differs from the 1.2 twin in
   `evader_speed_ratio` alone) for 0.8, and plain `EnvConfig()` for 1.2 — and
   `CG-ratio` below replicates that runner's CG-Z1 check (reading 1 there: the
   observation's own ratio entry, not just the config) for every cell of this
   round, learned-only by construction.
3. **CG-T1/A2.1 are inherited at the GATE level only** (brief): 0
   `inapplicable_picks` and 0 `(search policy)` usage (at S24.1's own A5
   tolerance) over this round's 12 cells. Nothing here re-reads the arms'
   training probes for those two counters; H-SAFE is the hypothesis that reads
   them, separately, for the 0-contact-steps claim (A3.1: absent evidence is
   an offender, never a zero).
4. **H-OS-1's non-inferiority test is unpaired-direction, not "the same test as
   H-TAC-1 run backwards"**: `paired_indicator_test(comparator, row, 0.05)`
   is run once per seed per opponent to get the LOSING direction's
   significance (comparator beats row); a seed passes iff that is NOT
   significant, OR the rate `row - comparator` is >= -0.03 regardless of
   significance (the prereg's "OR").

Usage:
  ./venv/bin/python scripts/derisk_s24_os_gate.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_os_gate.py --cells SEL2-80_HUG
  ./venv/bin/python scripts/derisk_s24_os_gate.py --gifs           (CG-O3)
  ./venv/bin/python scripts/derisk_s24_os_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 2, the stop rule, as S24.1's).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # as S24.1 / S24.3-ZS; BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.selector_mdp import SelectorMDP
from pe2.structured import HOLD

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's gate (and through it S23's and S19's), S24.3-ZS's gate (the 0.8
# comparator column and its `cell_config`/`config_diff`/`_f32_eq`), and this
# round's OWN trainer (the authority on the checkpoint path) — all loaded by
# FILE PATH.
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
Z = _load("derisk_s24_3zs", _HERE / "derisk_s24_3zs.py")
TR = _load("derisk_s24_os_train", _HERE / "derisk_s24_os_train.py")
GAME = S1.GAME

PREREG = "docs/pathway/tactics_prereg_s24_os_2026-10-17.md"
ARTEFACT_PREFIX = "s24_os"
OUT = pathlib.Path("results/derisk")
LIBRARY = "v1"
SEARCH_KEY = GAME.SEARCH_KEY  # "(search policy)" — A2.1's guard reads this

# --- pre-registered. Do not tune. ---
GATE_SEED0 = 2_400_000  # S24.1's GATE_SEED0, cross-checked at import below
N_EPISODES = 500
P_GATE = 0.05  # H-OS-1 (non-inferiority), H-OS-2
NI_MARGIN = -0.03  # H-OS-1's "delta (row - comparator) >= -0.03"
PASS_MIN = 2  # ">= 2/3", opponents and seeds alike
RATIO_1_2 = 1.2
RATIO_0_8 = 0.8
SEEDS: tuple[int, ...] = TR.SEEDS  # (80, 81, 82)
OPPONENTS: tuple[str, ...] = ("GREEDY", "HUG", "TACTIC-V0")
H_OS_1_OPPONENTS: tuple[str, ...] = ("GREEDY", "TACTIC-V0")  # prereg section 4

if S1.GATE_SEED0 != GATE_SEED0:
    raise SystemExit(
        f"S24.1's GATE_SEED0 is {S1.GATE_SEED0}, not {GATE_SEED0}; this round's "
        "gate stream must be the same one (brief: cross-checked at import)."
    )
if (
    S1.N_EPISODES != N_EPISODES
    or S1.P_GATE != P_GATE
    or Z.GATE_SEED0 != GATE_SEED0
    or Z.N_EPISODES != N_EPISODES
):
    raise SystemExit(
        "S24.1's or S24.3-ZS's N, p-gate or stream differ from this round's; the "
        "comparator joins and the reused orchestration would disagree."
    )

ROW_NAMES: tuple[str, ...] = tuple(f"SEL2-{s}" for s in SEEDS)
ROWS: dict[str, dict] = {
    row: {"kind": "learned", "arm": row, "seed": s, "memory": True, "library": LIBRARY}
    for row, s in zip(ROW_NAMES, SEEDS, strict=True)
}


def cell_name(row: str, opponent: str, ratio: float) -> str:
    """Reading 1: S24.1's bare `{row}_{opponent}` at this round's own ratio,
    1.2, and S24.3-ZS's `_r{ratio}` suffix at the off-training ratio, 0.8."""
    return f"{row}_{opponent}" if ratio == RATIO_1_2 else f"{row}_{opponent}_r{ratio:.1f}"


def _build_cells() -> dict[str, dict]:
    """The 12 cells (prereg section 3): nine at 1.2 (every row x opponent) and
    three at 0.8 (every row vs GREEDY only). Each carries its row, opponent and
    ratio stored — nothing parses a name back apart."""
    cells: dict[str, dict] = {}
    for r in ROW_NAMES:
        for o in OPPONENTS:
            name = cell_name(r, o, RATIO_1_2)
            cells[name] = {
                "name": name, "row": r, "opponent": o, "ratio": RATIO_1_2,
                "in_distribution": o == "TACTIC-V0", "hold": HOLD, **ROWS[r],
            }
    for r in ROW_NAMES:
        name = cell_name(r, "GREEDY", RATIO_0_8)
        cells[name] = {
            "name": name, "row": r, "opponent": "GREEDY", "ratio": RATIO_0_8,
            "in_distribution": False, "hold": HOLD, **ROWS[r],
        }
    return cells


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
        "cg_b": None,
        "shield": "RecoveryShieldV2 (D131), fresh per episode — the v0b assembly's",
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
        "library": LIBRARY,
        "trained_by": TR.PREREG,
        "joined_from": [S1.PREREG, Z.PREREG],
        "ckpt_stem": "the trainer's own `ckpt_path` (never a literal)",
    }


# ----------------------------------------------------------------------------
# record validity (S24.1's / S24.3-ZS's shape: this round's own n/seed0/ratio)
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == S1.UNDETERMINED:
        return S1.UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    n, seed0, ratio = rec.get("n_episodes"), rec.get("seed0"), rec.get("ratio")
    if n != N_EPISODES:
        return "INVALID", f"n_episodes {n!r} != the pre-registered {N_EPISODES}"
    if seed0 != GATE_SEED0:
        return "INVALID", f"seed0 {seed0!r} != the pre-registered {GATE_SEED0}"
    if ratio != CELLS[name]["ratio"]:
        return "INVALID", f"ratio {ratio!r} != the cell's declared {CELLS[name]['ratio']}"
    return "MEASURABLE", ""


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    # `_ORIG`, never the bare `S1.undetermined_record`: while `_bound()` is
    # active that name IS this function, and calling it would recurse forever.
    return _ORIG["undetermined_record"](name, exc, device) | {"ratio": CELLS[name]["ratio"]}


# ----------------------------------------------------------------------------
# the cell's env: library v1, memory on, only the opponent and the ratio vary
# ----------------------------------------------------------------------------
def cell_config(ratio: float) -> EnvConfig:
    """Reading 2: S24.1's own config at 1.2 (this round's own training ratio);
    S24.3-ZS's `cell_config` at 0.8 (asserts the built config differs from the
    1.2 twin in `evader_speed_ratio` alone, and refuses any other call)."""
    if ratio == RATIO_1_2:
        return EnvConfig()
    return Z.cell_config(ratio)


def cg_ratio(name: str, device: str = "cpu") -> dict:
    """S24.3-ZS's CG-Z1, replicated for this round's learned-only cells
    (reading 2; brief: "build the 0.8 env exactly as derisk_s24_3zs.py builds
    its learned rows"): only `evader_speed_ratio` may differ from the 1.2
    twin, and the observation's own ratio entry must equal the declared ratio
    at float32 precision. Runs BEFORE the cell's first episode."""
    spec = CELLS[name]
    ratio = float(spec["ratio"])
    cfg = cell_config(ratio)
    env = build_cell_env(name, cfg, device)
    obs, _info = env.reset(seed=GATE_SEED0)
    obs_ratio = float(np.asarray(obs)[env.obs_slices["speed_ratio"]][0])
    diff = Z.config_diff(env.cfg)
    expected_diff = [] if ratio == RATIO_1_2 else ["evader_speed_ratio"]
    block = {
        "gate": "CG-ratio (only the ratio differs), as derisk_s24_3zs.py's CG-Z1 "
        "builds its learned rows",
        "cell": name,
        "ratio": ratio,
        "cfg_ratio": float(env.cfg.evader_speed_ratio),
        "obs_ratio": obs_ratio,
        "config_diff_vs_twin": diff,
    }
    why = []
    if block["cfg_ratio"] != ratio:
        why.append(f"built env's cfg ratio is {block['cfg_ratio']}, not {ratio}")
    if not Z._f32_eq(obs_ratio, ratio):
        why.append(f"observation's ratio entry is {obs_ratio}, not {ratio}")
    if diff != expected_diff:
        why.append(f"the built config differs from the twin in {diff}, not {expected_diff}")
    if why:
        raise SystemExit(
            f"CG-ratio REFUSES {name} before any episode ran: {'; '.join(why)}."
        )
    return block | {"verdict": "PASS"}


def build_cell_env(
    name: str, cfg: EnvConfig | None = None, device: str = "cpu",
    *, recorder: bool = False, label: str = "",
) -> SelectorMDP:
    """Finding (a)'s other half: `library="v1"`, which S24.1's `build_cell_env`
    never passes."""
    spec = CELLS[name]
    cfg = cell_config(spec["ratio"]) if cfg is None else cfg
    env = SelectorMDP(
        cfg.evader_speed_ratio, memory=True, library=LIBRARY,
        recorder=recorder, label=label, cfg=cfg,
    )
    S1._set_opponent(env.game, spec["opponent"], cfg, device)
    return env


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """One cell: N paired episodes of the SEL2 row under the recovery shield
    against this opponent, at this cell's ratio. CG-ratio runs first and
    refuses before the first episode. Finding (a): `picks` is projected onto
    `env.actions` (this cell's own six-name library), not S24.1's five."""
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    ratio = float(spec["ratio"])
    gate = cg_ratio(name, device)
    env = build_cell_env(name, cell_config(ratio), device)
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
    actions = env.actions  # finding (a)
    total_picks = max(1, sum(picks.values()))
    return {
        "name": name,
        "row": spec["row"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": "learned",
        "memory": True,
        "library": LIBRARY,
        "ratio": ratio,
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
        "picks": {a: int(picks.get(a, 0)) for a in actions},  # finding (a)
        "usage_steps": dict(usage),
        "search_policy_steps": int(usage.get(SEARCH_KEY, 0)),  # A2.1's guard
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
            "picks_share": {a: picks.get(a, 0) / total_picks for a in actions},
            "orbit_search_share": picks.get("ORBIT-SEARCH", 0) / total_picks,
            "evader_tactics_recorded": spec["opponent"] == "TACTIC-V0",
        },
        "cg_ratio": gate,
        "provenance": provenance(device),
    }


# ----------------------------------------------------------------------------
# the binding onto S24.1's gate (see the module docstring)
# ----------------------------------------------------------------------------
# Each name is a global that one of S24.1's reused functions reads:
#   CELLS/CELL_ORDER      scan_records, run_cells, read_cell (via record_status)
#   cells_dir/cell_file   scan_records, load_records, read_cell, run_cells
#   record_status         scan_records, read_cell
#   provenance            run_cell, undetermined_record (called through this binding)
#   run_cell/write_cell    run_cells (so a chain writes THIS round's records)
#   undetermined_record   run_cells (on a crashed cell)
#   ROWS                  run_cells ("learned" rows get `require_ckpt` before running)
#   build_cell_env         render_cell (CG-O3's GIFs)
#   ckpt_path              require_ckpt / load_policy (so S24.1's unchanged
#                          functions resolve THIS round's `S24-OS-{ARM}` path)
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
        "ckpt_path": TR.ckpt_path,
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
    """Every named cell with no record yet (default: all 12). Literal names
    only — S24.1's own refusal order (unknown names, blocked cells, missing
    checkpoints, all before the first episode) runs unchanged."""
    with _bound():
        return _ORIG["run_cells"](names, device, n_episodes)


def require_ckpt(row: str) -> pathlib.Path:
    """This round's own, calling the TRAINER's `ckpt_path` directly (never a
    literal), so a precondition check outside `_bound()` (CG-O3's GIFs) still
    resolves the right path."""
    path = TR.ckpt_path(row)
    if not path.exists():
        raise SystemExit(
            f"missing the final checkpoint for row {row} at {path}. Section 3 "
            f"fixes the budget at {TR.TOTAL} env steps; there is no substitute."
        )
    return path


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


def load_comparators_1_2() -> dict:
    """`SEL-{s}_{opp}` from S24.1's OWN cells, validated by S24.1's OWN
    `record_status` (brief: "via S24.1's record_status")."""
    _assert_outside_binding("the 1.2 comparator join")
    out: dict = {}
    problems: list[str] = []
    for s in SEEDS:
        for o in OPPONENTS:
            name = f"SEL-{s}_{o}"
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
            out[(f"SEL-{s}", o)] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the 1.2 comparator column (S24.1's SEL rows) is not fully "
            "measurable: " + "; ".join(problems)
        )
    return out


def load_comparators_0_8() -> dict:
    """`SEL-{s}_GREEDY_r0.8` from S24.3-ZS's OWN cells, validated by its OWN
    `record_status` (brief: "via S24.3-ZS's record_status")."""
    _assert_outside_binding("the 0.8 comparator join")
    out: dict = {}
    problems: list[str] = []
    for s in SEEDS:
        name = Z.cell_name(f"SEL-{s}", "GREEDY", RATIO_0_8)
        path = Z.cell_file(name)
        if not path.exists():
            problems.append(f"{path}: absent")
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path}: unreadable: {exc}")
            continue
        status, why = Z.record_status(rec)
        if status != "MEASURABLE":
            problems.append(f"{path}: {status}: {why}")
            continue
        out[(f"SEL-{s}", "GREEDY")] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the 0.8 comparator column (S24.3-ZS's SEL rows) is not "
            "fully measurable: " + "; ".join(problems)
        )
    return out


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4)
# ----------------------------------------------------------------------------
def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def h_os_1_seed(records: dict, comparators: dict, opponent: str, seed: int) -> dict:
    row, base = f"SEL2-{seed}", f"SEL-{seed}"
    a = records.get(cell_name(row, opponent, RATIO_1_2))
    b = comparators.get((base, opponent))
    if a is None or b is None:
        missing = [n for n, v in ((cell_name(row, opponent, RATIO_1_2), a), (f"{base}_{opponent}", b)) if v is None]
        return {"verdict": "INCOMPLETE", "missing": missing, "row": row, "comparator": base, "opponent": opponent, "seed": seed}
    # the LOSING direction: does the comparator beat the row significantly?
    losing = paired_indicator_test(b["capture_ind"], a["capture_ind"], P_GATE)
    delta = a["capture_rate"] - b["capture_rate"]  # row - comparator (reported)
    # The margin is compared on exact COUNTS: 175/500 - 190/500 is
    # -0.030000000000000027 in binary floating point and would score a seed
    # sitting exactly on the pre-registered margin as FAIL (review defect 2).
    gap = int(sum(a["capture_ind"])) - int(sum(b["capture_ind"]))
    passed = (not losing["pass"]) or (gap >= round(NI_MARGIN * N_EPISODES))
    return {
        "verdict": "PASS" if passed else "FAIL",
        "row": row, "comparator": base, "opponent": opponent, "seed": seed,
        "capture_row": a["capture_rate"], "capture_comparator": b["capture_rate"],
        "delta": delta, "margin": NI_MARGIN,
        "losing_direction_significant": losing["pass"],
        "losing_direction_test": losing,
    }


def h_os_1_opponent(records: dict, comparators: dict, opponent: str) -> dict:
    per_seed = {str(s): h_os_1_seed(records, comparators, opponent, s) for s in SEEDS}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "opponent": opponent,
        "verdict": "INCOMPLETE" if "INCOMPLETE" in verdicts_ else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass, "pass_min": PASS_MIN, "seeds": per_seed,
    }


def h_os_1(records: dict, comparators: dict) -> dict:
    """H-OS-1 (non-inferiority, gated): on GREEDY and TACTIC-V0 at 1.2,
    `SEL2-s` is not worse than `SEL-s` — per opponent, >= 2/3 seeds pass; the
    round passes iff BOTH opponents do."""
    per = {o: h_os_1_opponent(records, comparators, o) for o in H_OS_1_OPPONENTS}
    verds = [b["verdict"] for b in per.values()]
    return {
        "hypothesis": "H-OS-1 (non-inferiority)",
        "verdict": "INCOMPLETE" if "INCOMPLETE" in verds else ("PASS" if all(v == "PASS" for v in verds) else "FAIL"),
        "gated": True,
        "rule": "per opponent: NOT significant loss (p >= 0.05) OR delta >= -0.03, on >= 2/3 seeds, for EACH of GREEDY and TACTIC-V0",
        "opponents": per,
    }


def h_os_2_seed(records: dict, comparators: dict, seed: int) -> dict:
    row, base = f"SEL2-{seed}", f"SEL-{seed}"
    a = records.get(cell_name(row, "HUG", RATIO_1_2))
    b = comparators.get((base, "HUG"))
    if a is None or b is None:
        missing = [n for n, v in ((cell_name(row, "HUG", RATIO_1_2), a), (f"{base}_HUG", b)) if v is None]
        return {"verdict": "INCOMPLETE", "missing": missing, "row": row, "comparator": base, "seed": seed}
    winning = paired_indicator_test(a["capture_ind"], b["capture_ind"], P_GATE)
    delta = a["capture_rate"] - b["capture_rate"]
    return {
        "verdict": "PASS" if winning["pass"] else "FAIL",
        "row": row, "comparator": base, "seed": seed,
        "capture_row": a["capture_rate"], "capture_comparator": b["capture_rate"],
        "delta": delta, "test": winning,
    }


def h_os_2(records: dict, comparators: dict) -> dict:
    """H-OS-2 (the claim, gated): `SEL2-s` > `SEL-s` vs HUG at 1.2, paired
    p < 0.05 in the winning direction, >= 2/3 seeds."""
    per_seed = {str(s): h_os_2_seed(records, comparators, s) for s in SEEDS}
    verdicts_ = [b["verdict"] for b in per_seed.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "hypothesis": "H-OS-2 (the claim)",
        "verdict": "INCOMPLETE" if "INCOMPLETE" in verdicts_ else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "gated": True,
        "rule": ">= 2/3 seeds, p < 0.05 paired, winning direction",
        "opponent": "HUG", "n_pass": n_pass, "pass_min": PASS_MIN, "seeds": per_seed,
    }


def h_os_3(records: dict, comparators_08: dict) -> dict:
    """H-OS-3 (reported): zero-shot 0.8 vs GREEDY, `SEL2-s` vs `SEL-s`
    (D153's failure case); the ORBIT-SEARCH usage share per cell; decisions
    per episode."""
    per_seed: dict = {}
    for s in SEEDS:
        row = f"SEL2-{s}"
        a = records.get(cell_name(row, "GREEDY", RATIO_0_8))
        b = comparators_08.get((f"SEL-{s}", "GREEDY"))
        if a is None or b is None:
            per_seed[str(s)] = {"verdict": "INCOMPLETE"}
            continue
        test = paired_indicator_test(a["capture_ind"], b["capture_ind"], P_GATE)
        delta = a["capture_rate"] - b["capture_rate"]
        per_seed[str(s)] = {
            "verdict": "REPORTED",
            "capture_row": a["capture_rate"], "capture_comparator": b["capture_rate"],
            "delta": delta, "test": test,
            "orbit_search_share": a["aggregate"]["orbit_search_share"],
            "mean_decisions": a["aggregate"]["mean_decisions"],
        }
    usage = {
        n: {
            "orbit_search_share": records[n]["aggregate"]["orbit_search_share"],
            "mean_decisions": records[n]["aggregate"]["mean_decisions"],
        }
        for n in CELL_ORDER
        if n in records
    }
    return {
        "hypothesis": "H-OS-3 (reported)",
        "gated": False,
        "zero_shot_0.8_vs_greedy": per_seed,
        "orbit_search_usage_per_cell": usage,
    }


def _probe_training_evidence() -> dict:
    """The three arms' probe rows, read exactly as the trainer wrote them —
    A3.1: absent evidence is an offender, never a zero."""
    train: dict = {}
    for arm in TR.ARM_ORDER:
        path = TR._probe_file(arm)
        rows: list[dict] = []
        absent = True
        if path.exists():
            try:
                rows = json.loads(path.read_text()).get("checkpoints", [])
                absent = not rows
            except (OSError, json.JSONDecodeError):
                absent = True
        train[arm] = {
            "path": str(path),
            "n_probe_rows": len(rows),
            "evidence_absent": absent,
            "train_violations_max": max(
                (int(r["train_violations"]) for r in rows if "train_violations" in r),
                default=None,
            ),
            "probe_violations_total": sum(int(r["violations"]) for r in rows if "violations" in r),
            # A3.1: a probe row without the counter is absent evidence (review note 5)
            "probe_violations_absent": any("violations" not in r for r in rows),
        }
    offenders = sorted(
        a for a, b in train.items()
        if (b["train_violations_max"] or 0) or b["probe_violations_total"]
    )
    missing = sorted(
        a for a, b in train.items()
        if b["evidence_absent"] or b["train_violations_max"] is None or b["probe_violations_absent"]
    )
    return {"arms": train, "offenders": offenders, "missing": missing}


def h_safe(records: dict) -> dict:
    """H-SAFE: 0 contact steps in the 12 cells AND the three arms' probe rows
    (A3.1: absence = offender)."""
    have = {n: records[n] for n in CELL_ORDER if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violation_steps", 0)))
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    training = _probe_training_evidence()
    missing = _missing(records)
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS"
        if (steps or training["offenders"])
        else ("INCOMPLETE" if (missing or training["missing"]) else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": offenders,
        "training": training,
        "gated_cells": list(CELL_ORDER),
        "missing": missing,
    }


def cg_t1(records: dict) -> dict:
    """CG-T1, inherited at the gate level (brief): 0 `inapplicable_picks` over
    this round's 12 cells."""
    cells = {n: int(records[n].get("inapplicable_picks", 0)) for n in CELL_ORDER if n in records}
    non_zero = sorted(n for n, v in cells.items() if v)
    return {
        "gate": "CG-T1 (the mask holds), inherited from S24.1 prereg section 5",
        "verdict": "REFUSE" if non_zero else "PASS",
        "cells": cells,
        "offenders": non_zero,
    }


def cg_a2_1(records: dict) -> dict:
    """A2.1, inherited at the gate level (brief): 0 `(search policy)` usage per
    cell, at S24.1's own A5 tolerance (<= 1 per 100 000 env steps)."""
    cells = {n: int(records[n].get("search_policy_steps", 0)) for n in CELL_ORDER if n in records}
    env_steps = {n: int(sum(records[n].get("steps", []))) for n in cells}
    tolerance = {n: max(1, -(-env_steps[n] // 100_000)) for n in cells}
    non_zero = sorted(n for n, v in cells.items() if v)
    offenders = sorted(n for n, v in cells.items() if v > tolerance[n])
    return {
        "gate": "A2.1 — every decision spends >= 1 tactic step, inherited (A5 tolerance)",
        "verdict": "REFUSE" if offenders else "PASS",
        "cells": cells,
        "tolerance": tolerance,
        "non_zero": non_zero,
        "offenders": offenders,
    }


def direction(v: dict) -> str:
    """Section 4's refutation paragraph, in the order of precedence the
    round's logic imposes: H-SAFE, then the construct gates that REFUSE, then
    completeness, then H-OS-1 and H-OS-2."""
    safe, gate1, a21 = v["h_safe"], v["cg_t1"], v["cg_a2_1"]
    if safe["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {safe['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(safe['offenders']) or 'the gate'}"
            + (
                f" and training contact in {', '.join(safe['training']['offenders'])}"
                if safe["training"]["offenders"]
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
    if a21["verdict"] == "REFUSE":
        return (
            f"A2.1 REFUSES: {SEARCH_KEY} env steps in {', '.join(a21['offenders'])}. "
            "No verdict here is about choosing."
        )
    if v["missing"] or safe["verdict"] == "INCOMPLETE":
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record, or the arms' training probe evidence is absent (A3.1). "
            "Nothing is re-run after a number is seen (the stop rule)."
        )
    t1, t2, t3 = v["h_os_1"], v["h_os_2"], v["h_os_3"]
    parts = []
    for o, b in t1["opponents"].items():
        parts.append(f"H-OS-1 vs {o}: {b['verdict']} ({b['n_pass']}/{len(SEEDS)} seeds).")
    parts.append(f"H-OS-1 (round) {t1['verdict']}.")
    for s, b in t2["seeds"].items():
        parts.append(
            f"SEL2-{s}: H-OS-2 {b['verdict']} vs SEL-{s} on HUG "
            f"(delta {b.get('delta', float('nan')):+.3f})."
        )
    parts.append(f"H-OS-2 (round) {t2['verdict']} ({t2['n_pass']}/{len(SEEDS)} seeds).")
    if t1["verdict"] != "PASS":
        parts.append(
            "H-OS-1 FAILS: ORBIT-SEARCH is dropped and S24.2 uses SEL "
            "(section 4's refutation)."
        )
    elif t2["verdict"] != "PASS":
        parts.append(
            "H-OS-1 passes and H-OS-2 fails: the library addition is harmless "
            "but does not reach the hiding evader at this ratio (section 4's "
            "refutation)."
        )
    else:
        parts.append(
            "H-OS-1 and H-OS-2 both pass: the library gap of D153 is closed by a "
            "memory-based tactic (section 4's refutation)."
        )
    for s, b in t3["zero_shot_0.8_vs_greedy"].items():
        if b.get("verdict") == "REPORTED":
            parts.append(
                f"SEL2-{s} vs GREEDY at 0.8 (reported): delta {b['delta']:+.3f}, "
                f"ORBIT-SEARCH share {b['orbit_search_share']:.3f}."
            )
    return " ".join(parts)


def verdicts(records: dict, comparators_1_2: dict, comparators_0_8: dict) -> dict:
    """Every verdict of the round — a pure function of the records and the two
    comparator columns, so a test can state a round rather than run one."""
    missing = _missing(records)
    v = {
        "h_os_1": h_os_1(records, comparators_1_2),
        "h_os_2": h_os_2(records, comparators_1_2),
        "h_os_3": h_os_3(records, comparators_0_8),
        "h_safe": h_safe(records),
        "cg_t1": cg_t1(records),
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.arange(len(OPPONENTS))
    width = 0.8 / len(ROW_NAMES)
    for i, row in enumerate(ROW_NAMES):
        vals = [
            np.nan if (r := records.get(cell_name(row, o, RATIO_1_2))) is None else r["capture_rate"]
            for o in OPPONENTS
        ]
        axes[0].bar(xs + i * width - 0.4 + width / 2, vals, width, label=row)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(list(OPPONENTS), fontsize=8)
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-OS-1 / H-OS-2: capture at ratio 1.2", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    shares = [
        (records.get(cell_name(row, "GREEDY", RATIO_0_8)) or {}).get("aggregate", {}).get(
            "orbit_search_share"
        )
        for row in ROW_NAMES
    ]
    axes[1].bar(list(ROW_NAMES), [np.nan if s is None else s for s in shares])
    axes[1].set_ylabel("ORBIT-SEARCH share of decisions", fontsize=8)
    axes[1].set_title("H-OS-3: zero-shot 0.8 vs GREEDY", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24-OS — library v1 selector (RecoveryShieldV2, hold {HOLD}; N = "
        f"{N_EPISODES} seed-paired episodes from {GATE_SEED0})."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-O3). H-SAFE
    and the inherited CG-T1/A2.1 REFUSE here, before any capture rate is
    read."""
    records = load_records()
    comparators_1_2 = load_comparators_1_2()
    comparators_0_8 = load_comparators_0_8()
    v = verdicts(records, comparators_1_2, comparators_0_8)
    for gate in (v["cg_t1"], v["cg_a2_1"]):
        if gate["verdict"] == "REFUSE":
            raise SystemExit(
                f"{gate['gate']} REFUSES the summary: {gate['offenders']}. No "
                "table is written while this stands."
            )
    if v["h_safe"]["training"]["missing"]:
        raise SystemExit(
            "H-SAFE REFUSES the summary: no training evidence for "
            f"{v['h_safe']['training']['missing']} (A3.1: absent evidence is an "
            "offender, never a zero)."
        )
    if v["h_safe"]["verdict"] == "FAILS":
        raise SystemExit(
            "H-SAFE FAILS the summary: "
            f"{v['h_safe']['offenders'] + v['h_safe']['training']['offenders']}."
        )
    blob = {
        "verdicts": v,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in records.items()
        },
        "provenance": provenance(device),
    }
    S1._write_json(round_file("json"), blob)
    write_figure(records, round_file("png"))
    return blob


# ----------------------------------------------------------------------------
# CG-O3 (prereg section 2): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
GIF_ROWS: tuple[str, ...] = ROW_NAMES  # SEL2-80/81/82
GIF_OPPONENTS: tuple[str, ...] = ("HUG", "GREEDY")
GIF_N = S1.GIF_N  # 3
GIF_EVERY = S1.GIF_EVERY  # 4


def gif_seeds() -> tuple[int, ...]:
    return S1.gif_seeds()  # (2 400 000, 2 400 001, 2 400 002) — a rule, not a pick


def gif_title(row: str, opponent: str, seed: int) -> str:
    return f"{row} vs {opponent} · episode {seed}"


def render_cell(row: str, opponent: str, seed: int, device: str = "cpu", every: int = GIF_EVERY):
    with _bound():
        return _ORIG["render_cell"](row, opponent, seed, device, every)


def write_gifs(
    rows: list[str] | None = None, device: str = "cpu", every: int = GIF_EVERY, fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """CG-O3: SEL2-80/81/82 x {HUG, GREEDY} x the first three gate seeds, at
    ratio 1.2 — 18 GIFs. Every precondition — each cell's measurable record and
    each row's checkpoint — is resolved before the first frame."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown CG-O3 row(s): {unknown}; this gate watches {GIF_ROWS}")
    seeds = gif_seeds()
    todo = [
        (row, o, seed)
        for row in wanted
        for o in GIF_OPPONENTS
        for seed in seeds
        if not gif_path(row, o, seed).exists()
    ]
    for row, o in dict.fromkeys((r, o) for r, o, _ in todo):
        read_cell(cell_name(row, o, RATIO_1_2))
        require_ckpt(row)
    n_all = len(wanted) * len(GIF_OPPONENTS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"the first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS} at ratio "
        f"{RATIO_1_2}. Watch for the pursuer regaining line of sight via "
        "ORBIT-SEARCH where SWEEP would stall, and for any pin.",
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
                {h: v[h]["verdict"] for h in ("h_os_1", "h_os_2")},
                indent=2,
            )
        )
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
