"""S24-FLAT Step F.3 — the gate: 27 cells (FLAT/FLAT40/STRUCT x3 seeds x3
opponents), joined to the S24.1 selector records, and the pre-registered
verdicts.

Pre-registration: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md`,
sections 2 to 6 INCLUDING Amendment A1 (2026-10-16), which adds the `FLAT40`
family and binds the cell count to 27 (9 rows x 3 opponents). Where a brief
and the prereg could be read to disagree, the PREREGISTRATION (+A1) wins and
the reading is named at the line.

Evaluation only. Nothing here trains, selects or tunes anything, and nothing
re-runs or rewrites an S24.1 cell: `results/derisk/s24_1_cells/` is READ.

A REBINDING DERIVATIVE of `scripts/derisk_s24_1_gate.py` (S24.1's gate),
loaded by file path with that runner's own `_load` pattern, in the style of
`scripts/derisk_s24_3zs.py` / `scripts/derisk_s24_3pr_gate.py` (`_load`,
`_bound`, `_ORIG`, `_BOUND_ALWAYS`). It differs from those two in WHAT is
reused, because of one structural fact: this round's two env classes
(`pe2.flat_mdp.FlatMDP` and `pe2.structured.SearchMDP`) are not a `select`
swap inside S24.1's own assembly — FLAT has a continuous action and STRUCT is
a different gym.Env entirely, with a different `play_episode` log schema
(`SearchMDP`'s carries no `n_switches`, `n_decisions`, `picks`, `tactics` or
`branch` key at all; `FlatMDP`'s carries them but `n_switches` is `None`,
which S24.1's own `run_cell` would crash on with `int(None)`). So:

**CALLED, via `_bound()` (this round's cells/paths/provenance rebound onto
S24.1 for one call, `_ORIG` holding the frozen originals so a wrapper cannot
re-enter itself): `scan_records`, `load_records`, `read_cell`,
`undetermined_record`, `h_safe` (A3.1's absent-evidence rule included),
`read_probes`.** These only ever read `CELLS`/`cells_dir`/`cell_file`/
`provenance`/`LEARNED_ROWS`/`probe_file` as module globals, so rebinding them
is enough; `S1.record_status` is ALSO sound unbound except for its `CELLS`
check, which the same binding covers.

**CALLED DIRECTLY, no binding (pure functions of their arguments, and
`cell_name` here is `f"{row}_{opponent}"`, bit-for-bit S24.1's own, so the
keys already line up): `_paired`, `_per_seed`, `_round`, `cell_config`,
`_set_opponent`, `scripted_opponent`, `gif_seeds`, `gif_title`, `_ci`,
`_write_json`, `_sha256`, `_cell_line`, `UNDETERMINED`.**

**OWN implementation, NOT called from S24.1 (the log schemas differ too much
to share): `build_cell_env`, `run_cell`, `render_cell`, `run_cells`,
`write_gifs`, every hypothesis (H-FLAT-1/2/3), `cg_f1_summary`, `cg_f3`,
`direction`, `summarize`, `main`.**

**The join to S24.1's 1.2 column (`load_s24_1_comparators`) runs OUTSIDE any
binding and asserts it** (`_assert_outside_binding`): inside a binding,
`S1.record_status`/`S1.cell_file` would be THIS round's and the comparator
column would be validated against the wrong rule and read from the wrong
directory — exactly the discipline `derisk_s24_3zs.py::load_base_records` and
`derisk_s24_3pr_gate.py::load_zs_records` state for their own joins.

Readings this module fixes, because the text can be read two ways:

1. **CG-F1 for `FLAT40` is read literally off `FlatMDP.config_diff_vs_selector()`
   (F.1, not touched by this round), with no augmentation of `assembly_diff`**
   (the A1.2 correction, 2026-10-16). That frozen method compares
   `_assembly()`'s `"hold"` field, which reads `env.game.hold` — the
   underlying `TacticGameMDP`'s BUILD-TIME constant (10), identical for
   `FLAT` and `FLAT40` — so `assembly_diff == []` for BOTH families
   (verified empirically:
   `FlatMDP(1.2, hold=40).config_diff_vs_selector()["assembly_diff"] == []`).
   `FLAT40`'s 40-step span is the ENV's own `.hold` attribute, not the
   game's, so this gate's own `flat_cg_f1` passes `allowed_assembly=()` for
   both families and records `flat_hold`/`game_hold`/`own_hold_differs`
   beside the frozen method's own output. (The first draft of this module,
   before the correction, augmented `assembly_diff` with `"hold"` instead;
   that reading is superseded and removed.)
2. **CG-F2's captions are forced to the literal `PURSUER: FLAT` / `PURSUER:
   STRUCT`** (the brief's exact wording), not S24.1's default `_name(law)`
   (which would print `UNKNOWN` for FLAT, whose law is never consulted, or
   the SearchMDP learner's current tactic, which does not exist). The EVADER
   half keeps S24.1's own convention (`_name(evader)` for `TACTIC-V0`, the
   literal opponent name for a scripted evader that is not a tactic adapter).
3. **STRUCT's gate/probe shield is fresh per episode** (F.2's reading 4,
   A1.3): `STRUCT_TR.make_gate_env()` builds with `shield=None` and this
   module assigns `env.shield = GAME.build_shield(cfg)` before every episode
   — training held one persistent shield for the whole run, which is a
   different, already-stated discipline (A1.3) and not changed here.
4. **H-FLAT-3's paired comparison is literal to the prereg: `FLAT-s` only**
   (section 4 names no other row) vs `LEAD-1.0` and `RULES-V0`. Its
   immobile/pinned CHARACTERISATION additionally reports `FLAT40` and
   `STRUCT` beside `SEL-s` (the brief's own bullet: "FLAT's and STRUCT's
   immobile-step rate and pinned-run episode fraction ... beside the
   selector's") — FLAT40 costs nothing extra to add and A1.2 treats it as a
   full second family throughout the rest of the round.
5. **STRUCT has no recorder (`pe2/recorder.py` is `SearchMDP`-free), so its
   "pinned-run" analogue is read from `ShieldMetrics.summary()`**
   (`pe2/metrics.py`): `longest_immobile_run >= DEADLOCK_STEPS` in place of
   the recorder's `longest_pinned_run >= DEADLOCK_STEPS`, and `n_immobile`
   in place of `immobile_by_branch`'s sum. Both are the nearest available
   quantities STRUCT's own log actually carries; the record says which
   source was used, every time.
6. **"decisions" / "env steps per decision" (A1.1) are counted here with a
   call-counting wrapper around `model.predict`**, uniformly for both env
   classes, rather than trusting either class's own internal counter: FLAT's
   `n_decisions` field would work but `SearchMDP.play_episode`'s log has no
   such field at all, and one counting method for both arms is what makes
   the two numbers comparable.
7. **`n_switches` is `None` for every row of this round, never averaged**
   (the brief): FLAT/FLAT40 run no tactics at all, and `SearchMDP`'s log
   carries no switch count either. `picks` stays `{}` and
   `inapplicable_picks` stays `0` for the same reason CG-F3 states: there is
   no mask here, so the S24.1 fields are kept only so the two round's cell
   schemas overlay for the parts that DO compare (`capture_ind`, `steps`,
   `violations`).
8. **STRUCT's GIF recorder has no `ShieldOverlay`.** That overlay needs a
   per-step `(nominal, executed)` action pair, which `SearchMDP._step_inner`
   computes and discards internally without exposing it; reproducing the
   shield call a second time here would duplicate dynamics rather than
   record them. `FovOverlay` + `Trails` are kept, which already show
   stalling, spinning and pins as position traces — CG-F2's stated purpose.
9. **Checkpoints and probes are read through each family's OWN trainer**
   (`ckpt_path`/`probe_file` dispatch to `derisk_s24_flat_train.py` or
   `derisk_s24_struct_train.py`'s own `_ckpt_dir`/`_label`/`_probe_file`),
   never re-typed: both trainers already agree on the `S24-{ARM}` stem, so
   the only difference is the probe file's `PREFIX` (`s24_flat` /
   `s24_struct`), which this module does not re-decide either.

Usage:
  ./venv/bin/python scripts/derisk_s24_flat_gate.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s24_flat_gate.py --cells FLAT-80_HUG
  ./venv/bin/python scripts/derisk_s24_flat_gate.py --gifs           (CG-F2)
  ./venv/bin/python scripts/derisk_s24_flat_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a
record is NEVER re-run (prereg section 6, the stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # prereg section 6 (as S24.1); BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

from pe2.metrics import DEADLOCK_STEPS
from pe2.flat_mdp import FlatMDP
from pe2.structured import HOLD, SearchMDP

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's gate runner (and through it S23's and S19's), and THIS round's two
# trainers, all loaded by FILE PATH so this module imports the same way as a
# script and as `scripts.derisk_s24_flat_gate`.
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
S19, GAME = S1.S19, S1.GAME
FLAT_TR = _load("derisk_s24_flat_train", _HERE / "derisk_s24_flat_train.py")
STRUCT_TR = _load("derisk_s24_struct_train", _HERE / "derisk_s24_struct_train.py")

PREREG = "docs/pathway/tactics_prereg_s24_flat_2026-10-14.md"
ARTEFACT_PREFIX = "s24_flat"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 2 to 6, Amendment A1). Do not tune. ---
GATE_SEED0 = 2_400_000  # section 4: THE SAME stream as S24.1's 1.2 cells
N_EPISODES = 500  # section 4
P_GATE = 0.05  # section 4, H-FLAT-1 / H-FLAT-2 / H-FLAT-3
PASS_MIN = 2  # ">= 2/3", opponents and seeds alike
SEEDS: tuple[int, ...] = (80, 81, 82)  # section 3
TRAIN_TOTAL = 3_000_000  # section 2's budget: the final checkpoint's mark
FLAT_FAMILIES: tuple[str, ...] = ("FLAT", "FLAT40")  # A1.2
ALL_FAMILIES: tuple[str, ...] = (*FLAT_FAMILIES, "STRUCT")  # section 3

if (S1.GATE_SEED0, S1.N_EPISODES, S1.PASS_MIN, S1.P_GATE) != (
    GATE_SEED0, N_EPISODES, PASS_MIN, P_GATE,
):  # the reuse below is sound only while the two rounds agree on these
    raise SystemExit(
        "S24.1's stream, N or thresholds differ from this round's; the 1.2 "
        "join and the imported verdict helpers would compare different rounds."
    )
if FLAT_TR.TOTAL != TRAIN_TOTAL or STRUCT_TR.TOTAL != TRAIN_TOTAL:
    raise SystemExit(
        "a trainer's env-step budget differs from this round's TRAIN_TOTAL; "
        "the checkpoint path this gate requires would not be the final one."
    )

ROWS: dict[str, dict] = {
    f"{family}-{seed}": {
        "kind": "learned",
        "arm": f"{family}-{seed}",
        "family": family,
        "seed": seed,
    }
    for family in ALL_FAMILIES
    for seed in SEEDS
}
ROW_NAMES: tuple[str, ...] = tuple(ROWS)
OPPONENTS: dict[str, dict] = S1.OPPONENTS  # GREEDY / HUG / TACTIC-V0, reused whole
SEL_ROWS: tuple[str, ...] = tuple(f"SEL-{s}" for s in SEEDS)
NOMEM_ROWS: tuple[str, ...] = tuple(f"NOMEM-{s}" for s in SEEDS)
COMPARATOR_ROWS: tuple[str, ...] = (*SEL_ROWS, *NOMEM_ROWS, S1.BASELINE_ROW, S1.RULES_ROW)
GIF_EVERY = 4  # keep every 4th env step as a frame, as S14/S18/S19/S23/S24.1


def cell_name(row: str, opponent: str) -> str:
    return f"{row}_{opponent}"  # bit-for-bit S24.1's own `cell_name`


def _build_cells() -> dict[str, dict]:
    """The 27 cells (A1.1/A1.2). Each carries its row, opponent and family
    STORED — nothing parses a name back apart."""
    return {
        cell_name(r, o): {
            "name": cell_name(r, o),
            "row": r,
            "opponent": o,
            "in_distribution": OPPONENTS[o]["in_distribution"],
            "hold": 40 if ROWS[r]["family"] == "FLAT40" else HOLD,
            **ROWS[r],
        }
        for r in ROWS
        for o in OPPONENTS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance
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


def ckpt_path(row: str) -> pathlib.Path:
    """Reading 9: each family's OWN trainer's `_ckpt_dir`/`_label` — both
    trainers agree on the `S24-{ARM}` stem."""
    tr = STRUCT_TR if ROWS[row]["family"] == "STRUCT" else FLAT_TR
    return tr._ckpt_dir(row) / f"{tr._label(row)}_{TRAIN_TOTAL}.zip"


def probe_file(row: str) -> pathlib.Path:
    tr = STRUCT_TR if ROWS[row]["family"] == "STRUCT" else FLAT_TR
    return tr._probe_file(row)


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    S1._write_json(path, record)
    return path


def provenance(device: str = "cpu") -> dict:
    return S19.provenance(device) | {
        "spec": PREREG,
        "spec_sha256": S1._sha256(spec_path()),
        "cg_b": None,
        "shield": (
            "RecoveryShieldV2 (D131), fresh per episode for FLAT/FLAT40 (the "
            "v0b assembly's own `_begin`) and for STRUCT at gate/probe time "
            "(A1.3: training held one persistent shield for the run; the "
            "gate and the probe rebuild it per episode, S17's convention)"
        ),
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
        "families": list(ALL_FAMILIES),
        "trained_by": [FLAT_TR.PREREG, STRUCT_TR.PREREG],
        "joined_from": S1.PREREG,
    }


# ----------------------------------------------------------------------------
# the rebinding onto S24.1 (see the module docstring)
# ----------------------------------------------------------------------------
_BOUND_ALWAYS: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "provenance",
    "LEARNED_ROWS", "probe_file",
)
_ORIG: dict = {
    k: getattr(S1, k)
    for k in (
        *_BOUND_ALWAYS, "scan_records", "load_records", "read_cell",
        "undetermined_record", "record_status", "h_safe", "read_probes",
    )
}


@contextlib.contextmanager
def _bound():
    """Rebind S24.1's generic record-validity / H-SAFE machinery onto this
    round's cells, directory and provenance for one call, restoring every one
    on exit. `run_cell`/`build_cell_env`/`render_cell` are deliberately NOT
    in this set — see the module docstring for why S24.1's own cannot be
    called for either env class."""
    over = {
        "CELLS": CELLS,
        "CELL_ORDER": CELL_ORDER,
        "cells_dir": cells_dir,
        "cell_file": cell_file,
        "provenance": provenance,
        "LEARNED_ROWS": ROW_NAMES,
        "probe_file": probe_file,
    }
    saved = {k: getattr(S1, k) for k in over}
    for k, v in over.items():
        setattr(S1, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(S1, k, v)


def _assert_outside_binding(what: str) -> None:
    """Reading (module docstring): a join must see S24.1's FROZEN globals.
    Inside a binding the comparator records would be validated against this
    round's cell table and read from this round's directory."""
    if not (S1.CELLS is _ORIG["CELLS"] and S1.cells_dir is _ORIG["cells_dir"]):
        raise SystemExit(
            f"{what} may not run inside a binding: with S24.1's globals "
            "rebound to this round's, `record_status` and `cell_file` would "
            "be THIS round's and the comparator column would be validated "
            "against the wrong rule and read from the wrong directory."
        )


def scan_records() -> tuple[dict, dict, dict]:
    with _bound():
        return _ORIG["scan_records"]()


def load_records(*, quiet: bool = False) -> dict:
    with _bound():
        return _ORIG["load_records"](quiet=quiet)


def read_cell(name: str) -> dict:
    with _bound():
        return _ORIG["read_cell"](name)


def record_status(rec: dict) -> tuple[str, str]:
    with _bound():
        return _ORIG["record_status"](rec)


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    with _bound():
        rec = _ORIG["undetermined_record"](name, exc, device)
    return rec | {
        "row": CELLS[name]["row"],
        "family": CELLS[name]["family"],
        "opponent": CELLS[name]["opponent"],
    }


def read_probes() -> dict:
    with _bound():
        return _ORIG["read_probes"]()


def h_safe(records: dict, probes: dict | None = None) -> dict:
    with _bound():
        return _ORIG["h_safe"](records, probes)


# ----------------------------------------------------------------------------
# the S24.1 join — OUTSIDE any binding, asserted (reading, module docstring)
# ----------------------------------------------------------------------------
def load_s24_1_comparators() -> dict:
    """`SEL-s`/`NOMEM-s`/`LEAD-1.0`/`RULES-V0` at ratio 1.2, keyed by S24.1's
    OWN `cell_name` (identical to ours: `f"{row}_{opponent}"`), validated by
    S24.1's OWN `record_status`. Nothing here writes, moves or re-runs them."""
    _assert_outside_binding("the S24.1 join")
    out: dict = {}
    problems: list[str] = []
    for row in COMPARATOR_ROWS:
        for o in OPPONENTS:
            path = S1.cell_file(S1.cell_name(row, o))
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
            out[S1.cell_name(row, o)] = rec
    if problems:
        raise SystemExit(
            "REFUSED: the S24.1 comparator column (SEL/NOMEM/LEAD-1.0/"
            "RULES-V0) must be measurable by S24.1's own validity rule for "
            "every row this round joins. " + "; ".join(problems)
            + ". H-FLAT-1/2/3 compare against it and are not computed from a "
            "partial column."
        )
    return out


# ----------------------------------------------------------------------------
# checkpoints and policies
# ----------------------------------------------------------------------------
_POLICIES: dict = {}


def require_ckpt(row: str) -> pathlib.Path:
    path = ckpt_path(row)
    if not path.exists():
        raise SystemExit(
            f"missing the final checkpoint for row {row} at {path}. Section "
            f"2/3 fix the budget at {TRAIN_TOTAL} env steps; there is no "
            "substitute and the round refuses to run."
        )
    return path


def load_policy(row: str, device: str = "cpu"):
    """CG-F3: no mask — both trainers used plain `stable_baselines3.PPO`, not
    `MaskablePPO`."""
    from stable_baselines3 import PPO

    path = require_ckpt(row)
    key = (str(path), device)
    if key not in _POLICIES:
        _POLICIES[key] = PPO.load(path, device=device)
    return _POLICIES[key]


# ----------------------------------------------------------------------------
# CG-F1 — before any capture rate is read (prereg section 5)
# ----------------------------------------------------------------------------
def _flat_caption(game, opponent: str) -> None:
    """Reading 2: force the literal `PURSUER: FLAT`."""
    if not hasattr(game, "frames"):
        return
    if opponent == "TACTIC-V0":
        game.caption = lambda _tag: GAME.frame_label(GAME._name(game.evader), "FLAT")
    else:
        game.caption = lambda _tag: GAME.frame_label(opponent, "FLAT")


def flat_cg_f1(family: str) -> dict:
    """Reading 1 (A1.2 correction): `FlatMDP.config_diff_vs_selector()`'s own
    output, unaugmented — `allowed_assembly=()` for both families, since the
    frozen method already reads `assembly_diff == []` for `FLAT40` (it
    compares the GAME's `hold`, identical for every family). `FLAT40`'s own
    40-step span is recorded separately as `flat_hold`/`game_hold`/
    `own_hold_differs`, not folded into `assembly_diff`."""
    hold = 40 if family == "FLAT40" else None
    env = FlatMDP(1.2, hold=hold)
    cg = env.config_diff_vs_selector(allowed_assembly=())
    return cg | {
        "family": family,
        "flat_hold": int(env.hold),
        "game_hold": int(env.game.hold),
        "own_hold_differs": bool(int(env.hold) != int(env.game.hold)),
    }


def struct_cg_f1() -> dict:
    """F.2's own `assert_cg_f1`, on the gate's env with a freshly built shield
    assigned (reading 3: `make_gate_env()` builds `shield=None`)."""
    env = STRUCT_TR.make_gate_env()
    env.shield = GAME.build_shield(env.cfg)
    return STRUCT_TR.assert_cg_f1(env) | {"family": "STRUCT"}


_CG_F1_CACHE: dict = {}


def cg_f1(family: str) -> dict:
    if family not in _CG_F1_CACHE:
        _CG_F1_CACHE[family] = struct_cg_f1() if family == "STRUCT" else flat_cg_f1(family)
    return _CG_F1_CACHE[family]


# ----------------------------------------------------------------------------
# the cell's env (two classes; see the module docstring for why they are not
# S24.1's `build_cell_env`)
# ----------------------------------------------------------------------------
class _StructRecorder(SearchMDP):
    """CG-F2's STRUCT episodes: `SearchMDP` plus frame capture — S19's
    `CellRecorder` pattern, adapted for plain `SearchMDP` (reading 8: no
    `ShieldOverlay`, which needs a per-step nominal/executed pair this class
    does not expose outside `_step_inner`'s own shield call)."""

    def __init__(
        self, cfg, opponent_factory, *, label: str, opponent_label: str | None,
        every: int = GIF_EVERY, **kw,
    ):
        from pe2.render import FovOverlay, Renderer, Trails

        super().__init__(cfg, opponent_factory, **kw)
        self.label = label
        self._opponent_label = opponent_label  # None -> _name(live evader)
        self.every = int(every)
        self.frames: list = []
        self._r = Renderer(cfg, dpi=80, figsize=(5.5, 5.0))
        self._t = Trails()
        self._ov = [FovOverlay("pursuer")]

    def _draw(self, title: str) -> None:
        self.frames.append(
            self._r.draw(self.inner.env, trails=self._t, overlays=self._ov, title_prefix=title)
        )

    def caption(self, tag: str) -> str:
        ev = (
            GAME._name(self.inner._opponent)
            if self._opponent_label is None
            else self._opponent_label
        )
        return GAME.frame_label(ev, "STRUCT")

    def _step_inner(self, action, branch: str = "law"):
        env = self.inner.env
        self._t.append("pursuer", *env.pursuer_pose[:2])
        self._t.append("evader", *env.evader_pose[:2])
        out = super()._step_inner(action, branch)
        if env.current_step % self.every == 0:
            self._draw(self.caption(branch))
        return out

    def finish(self, outcome: str, step: int, hold_frames: int = 10) -> None:
        for _ in range(hold_frames):
            self._draw(self.caption(f"{outcome.upper()} @ {step}"))
        self._r.close()


def build_cell_env(
    name: str, cfg=None, device: str = "cpu", *, recorder: bool = False, label: str = "",
):
    """`FlatMDP` for a `FLAT`/`FLAT40` row, `SearchMDP` (via
    `STRUCT_TR.make_gate_env()`) for `STRUCT`, with ONLY the opponent and
    (FLAT40) the hold varied — everything else is the family's own frozen
    construction."""
    spec = CELLS[name]
    cfg = S1.cell_config() if cfg is None else cfg
    family, opponent = spec["family"], spec["opponent"]
    if family in FLAT_FAMILIES:
        hold = 40 if family == "FLAT40" else None
        env = FlatMDP(cfg.evader_speed_ratio, memory=True, recorder=recorder, label=label, cfg=cfg, hold=hold)
        S1._set_opponent(env.game, opponent, cfg, device)
        _flat_caption(env.game, opponent)
        return env
    # STRUCT
    if recorder:
        opponent_label = None if opponent == "TACTIC-V0" else opponent
        factory = (
            STRUCT_TR.evader_factory_for(cfg)
            if opponent == "TACTIC-V0"
            else S1.scripted_opponent(opponent, cfg, device)
        )
        return _StructRecorder(
            cfg, factory, label=label, opponent_label=opponent_label, every=GIF_EVERY,
            hold=HOLD, r_acq=0.0, shield=GAME.build_shield(cfg),
            shield_obs=True, escape_obs=True, reward_wrapper=None,
        )
    env = STRUCT_TR.make_gate_env()
    if opponent != "TACTIC-V0":
        env.inner.opponent_factory = S1.scripted_opponent(opponent, cfg, device)
    return env


# ----------------------------------------------------------------------------
# one episode, one cell
# ----------------------------------------------------------------------------
def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """N paired episodes of this row's policy under the recovery shield
    against this opponent, on the gate stream.

    `n_switches` is always `None` (reading 7); the decision count comes from
    a call-counting wrapper around `model.predict` (reading 6), not from
    either env class's own internal counter."""
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    family = spec["family"]
    cfg = S1.cell_config()
    env = build_cell_env(name, cfg, device)
    model = load_policy(spec["row"], device)

    capture_ind: list[bool] = []
    ep_steps: list[int] = []
    violations: list[int] = []
    decisions: list[int] = []
    branch_rows: list[dict] = []
    shield_rows: list[dict] = []
    outcomes: Counter = Counter()

    for i in range(n):
        seed = GATE_SEED0 + i
        if family == "STRUCT":
            env.shield = GAME.build_shield(cfg)  # reading 3 (F.2): fresh per episode
        counter = [0]

        def counted_predict(obs, deterministic: bool = True):
            counter[0] += 1
            return model.predict(obs, deterministic=deterministic)

        log = env.play_episode(counted_predict, seed)
        capture_ind.append(log["outcome"] == "capture")
        outcomes[log["outcome"]] += 1
        ep_steps.append(int(log["steps"]))
        violations.append(int(log["violations"]))
        decisions.append(int(counter[0]))
        if "branch" in log:
            branch_rows.append(log["branch"])
        if "shield" in log:
            shield_rows.append(log["shield"])

    k = sum(capture_ind)
    n_dec = sum(decisions)
    return {
        "name": name,
        "row": spec["row"],
        "family": family,
        "seed": spec["seed"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": "learned",
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
        # Reading 7: never averaged (the brief). Kept as a field, `None`
        # throughout, so a reader does not have to know the family to know
        # there is nothing to average.
        "n_switches": [None] * n,
        "picks": {},  # CG-F3: no tactics
        "inapplicable_picks": 0,  # CG-F3: no mask
        "n_decisions": decisions,
        "branch": branch_rows,  # [] for STRUCT (SearchMDP has no recorder)
        "shield": shield_rows,
        "cg_f1": cg_f1(family),
        "aggregate": {
            "capture_rate": k / n if n else float("nan"),
            "outcomes": dict(outcomes),
            # The env terminates on capture and truncates on timeout and
            # nothing else (S24.1's own reading), so the evader escapes
            # exactly when it times out.
            "escape_rate": outcomes["timeout"] / n if n else float("nan"),
            "mean_steps": float(statistics.fmean(ep_steps)) if ep_steps else float("nan"),
            "mean_decisions": float(statistics.fmean(decisions)) if decisions else float("nan"),
            "env_steps_per_decision": (sum(ep_steps) / n_dec) if n_dec else float("nan"),  # A1.1
        },
        "provenance": provenance(device),
    }


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell with no record yet (default: all 27). Literal names
    only. Refusal order, all of it before the first episode: an unknown name;
    a file that is not a measurable record of THIS round; a missing final
    checkpoint for any requested row; CG-F1 for any family about to run its
    first cell (raises on failure)."""
    wanted = list(CELL_ORDER) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cell(s): {unknown}; this round's cells are {CELL_ORDER}")
    cells_dir().mkdir(parents=True, exist_ok=True)
    with _bound():
        measurable, rejected, by_cell = _ORIG["scan_records"]()
    blocked = {
        n: "; ".join(f"{cells_dir() / f} — {rejected[f]}" for f in by_cell[n])
        for n in wanted
        if n in by_cell
    }
    if blocked:
        raise SystemExit(
            "REFUSED before any episode ran — "
            f"{len(blocked)} requested cell(s) already have a file that is not a "
            "measurable record of this round: "
            + "; ".join(f"{n}: {why}" for n, why in sorted(blocked.items()))
            + f". A measurable record has n_episodes == {N_EPISODES} and seed0 "
            f"== {GATE_SEED0}. Nothing here overwrites or re-runs it."
        )
    todo = [n for n in wanted if n not in measurable]
    for row in dict.fromkeys(CELLS[n]["row"] for n in todo):
        require_ckpt(row)
    for family in dict.fromkeys(CELLS[n]["family"] for n in todo):
        block = cg_f1(family)  # construct gate, before the first episode of that family
        # STRUCT's own `assert_cg_f1` already raises on failure (reading 3);
        # `flat_cg_f1` never does (it is a pure function of its argument), so
        # this gate raises here for FLAT/FLAT40 instead.
        if family in FLAT_FAMILIES and not block["ok"]:
            raise SystemExit(
                f"CG-F1 FAILED for {family}: FLAT's env differs from the "
                "selector's in more than the action and the tactic block — "
                f"config_diff {block['config_diff']}, assembly_diff "
                f"{block['assembly_diff']}, observation entries differing on "
                f"an identical reset {block['obs_diff_indices_on_identical_reset']}, "
                f"FLAT tactic entries non-zero {block['flat_tactic_entries_nonzero']}, "
                f"selector tactic entries non-zero {block['selector_tactic_entries_nonzero']}"
            )
    print(
        f"{len(todo)}/{len(wanted)} cells to run ({len(wanted) - len(todo)} already "
        f"have a measurable record; stop rule: never re-run), N = "
        f"{N_EPISODES if n_episodes is None else n_episodes} from seed "
        f"{GATE_SEED0}. No rate is printed here — read --summarize, after --gifs.",
        flush=True,
    )
    for name in todo:
        try:
            rec = run_cell(name, device, n_episodes)
        except Exception as exc:  # noqa: BLE001 — record it, do not re-run
            rec = undetermined_record(name, exc, device)
            write_cell(rec)
            print(f"[{name}] {S1.UNDETERMINED}: {rec['error']}", flush=True)
            continue
        write_cell(rec)
        print(S1._cell_line(rec), flush=True)
    return todo


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4, Amendment A1)
# ----------------------------------------------------------------------------
def h_flat_1_family(records: dict, family: str) -> dict:
    """H-FLAT-1 for one flat family: `SEL-s` > `{family}-s` (same seed,
    paired), >= 2/3 opponents, >= 2/3 seeds, each seed.

    `row` is `SEL-s` and `base` is `{family}-s`: `S1._paired`'s PASS means
    `row` beats `base` (S24.1's own `h_tac_1` passes `row=SEL-s`,
    `base=LEAD-1.0` for exactly this reason), so the claim under test —
    the selector beats flat — is `row=SEL-s`, `base={family}-s`, not the
    other way round."""
    per_seed = {}
    for s in SEEDS:
        row, base = f"SEL-{s}", f"{family}-{s}"
        per = S1._paired(records, row, base, OPPONENTS)
        per_seed[str(s)] = S1._per_seed(per, "H-FLAT-1", row, base, {"seed": s, "family": family})
    return S1._round(per_seed, f"H-FLAT-1 ({family})")


def h_flat_1(records: dict) -> dict:
    """H-FLAT-1 (the claim): PASSES only if the selector beats BOTH flat
    families (A1.2)."""
    families = {f: h_flat_1_family(records, f) for f in FLAT_FAMILIES}
    verdicts_ = [b["verdict"] for b in families.values()]
    verdict = (
        "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if all(v == "PASS" for v in verdicts_) else "FAIL")
    )
    return {
        "hypothesis": "H-FLAT-1 (the claim)",
        "verdict": verdict,
        "gated": True,
        "rule": "SEL-s beats BOTH flat families (A1.2): each >= 2/3 opponents, >= 2/3 seeds, p < 0.05",
        "comparator": "SEL-s (same seed)",
        "families": families,
        "passing_families": sorted(f for f, b in families.items() if b["verdict"] == "PASS"),
    }


def h_flat_2(records: dict) -> dict:
    """H-FLAT-2: `SEL-s` > `STRUCT-s` likewise (see `h_flat_1_family` for the
    row/base convention)."""
    per_seed = {}
    for s in SEEDS:
        row, base = f"SEL-{s}", f"STRUCT-{s}"
        per = S1._paired(records, row, base, OPPONENTS)
        per_seed[str(s)] = S1._per_seed(per, "H-FLAT-2", row, base, {"seed": s})
    return S1._round(per_seed, "H-FLAT-2")


def pooled_immobile_and_pinned(records: dict, row: str) -> dict | None:
    """Reading 4/5: `row`'s immobile-step rate and pinned-run episode
    fraction, pooled over its three opponent cells, from the recorder's
    `branch` summary where present (FLAT/FLAT40/SEL/NOMEM/LEAD-1.0/RULES-V0)
    or from `ShieldMetrics.summary()` where it is not (STRUCT)."""
    cells = [records[cell_name(row, o)] for o in OPPONENTS if cell_name(row, o) in records]
    if not cells:
        return None
    total_steps = sum(s for c in cells for s in c["steps"])
    branch_rows = [b for c in cells for b in c.get("branch", [])]
    if branch_rows:
        n_immobile = sum(sum(b["immobile_by_branch"].values()) for b in branch_rows)
        n_pinned = sum(b["longest_pinned_run"] >= DEADLOCK_STEPS for b in branch_rows)
        return {
            "source": "recorder branch_summary (pe2/recorder.py)",
            "n_episodes": len(branch_rows),
            "immobile_step_rate": n_immobile / total_steps if total_steps else None,
            "pinned_run_episode_fraction": n_pinned / len(branch_rows),
        }
    shield_rows = [s for c in cells for s in c.get("shield", [])]
    if shield_rows:
        n_immobile = sum(s["n_immobile"] for s in shield_rows)
        n_pinned = sum(s["longest_immobile_run"] >= DEADLOCK_STEPS for s in shield_rows)
        return {
            "source": (
                "ShieldMetrics.summary() (pe2/metrics.py) — this row's env "
                "class has no recorder/branch_summary; longest_immobile_run "
                "is the nearest available analogue of the recorder's "
                "longest_pinned_run (reading 5, module docstring)"
            ),
            "n_episodes": len(shield_rows),
            "immobile_step_rate": n_immobile / total_steps if total_steps else None,
            "pinned_run_episode_fraction": n_pinned / len(shield_rows),
        }
    return None


def h_flat_3(records: dict) -> dict:
    """H-FLAT-3 (characterise, do not assume; reported, not gated). Reading
    4: the paired comparison is literal to the prereg — `FLAT-s` only, vs
    `LEAD-1.0` and `RULES-V0`. The immobile/pinned table additionally reports
    `FLAT40` and `STRUCT` beside `SEL-s` (the brief's own bullet)."""
    vs_comparators = {}
    for s in SEEDS:
        row = f"FLAT-{s}"
        for base in (S1.BASELINE_ROW, S1.RULES_ROW):
            per = S1._paired(records, row, base, OPPONENTS)
            vs_comparators[f"{row}_vs_{base}"] = S1._per_seed(
                per, "H-FLAT-3", row, base, {"seed": s, "family": "FLAT"}
            ) | {"gated": False}
    rows_characterised = (
        *(f"{f}-{s}" for f in ALL_FAMILIES for s in SEEDS),
        *SEL_ROWS,
    )
    characterisation = {row: pooled_immobile_and_pinned(records, row) for row in rows_characterised}
    return {
        "hypothesis": "H-FLAT-3 (characterise, do not assume)",
        "verdict": "REPORTED",
        "gated": False,
        "vs_comparators": vs_comparators,
        "immobile_and_pinned": characterisation,
    }


def _cadence(records: dict, row: str) -> dict | None:
    """A1.1: decisions and env steps per decision, pooled over `row`'s three
    opponent cells, from THIS round's own records only (not the comparators)."""
    cells = [records[cell_name(row, o)] for o in OPPONENTS if cell_name(row, o) in records]
    if not cells:
        return None
    decisions_all = [d for c in cells for d in c["n_decisions"]]
    steps_all = [s for c in cells for s in c["steps"]]
    n_dec = sum(decisions_all)
    return {
        "mean_decisions_per_episode": statistics.fmean(decisions_all) if decisions_all else None,
        "env_steps_per_decision": (sum(steps_all) / n_dec) if n_dec else None,
    }


def _check_arm_record(row: str) -> tuple[bool, str]:
    """Fix round 1, item 1: the nine ARM records (one per row, written by
    each family's OWN trainer — `FLAT_TR`/`STRUCT_TR`'s `_arm_file`) are
    re-read here, not just the per-cell `cg_f1` block this gate itself
    wrote. An offender is a missing/unreadable arm file, a `verdict` other
    than `MEASURED`, a `resumed_from` that is not `None` (A1.3: a crashed
    arm is skipped, never resumed), an absent/not-`ok` `cg_f1` block, or
    (`FLAT40` only) a declared `hold` other than 40."""
    family = ROWS[row]["family"]
    tr = STRUCT_TR if family == "STRUCT" else FLAT_TR
    path = tr._arm_file(row)
    if not path.exists():
        return False, f"{path}: absent"
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{path}: unreadable: {exc}"
    if rec.get("verdict") != "MEASURED":
        return False, f"verdict = {rec.get('verdict')!r}, not MEASURED"
    if rec.get("resumed_from") is not None:
        return False, f"resumed_from = {rec.get('resumed_from')!r}, not None"
    cg = rec.get("cg_f1")
    if not (cg and cg.get("ok")):
        return False, "cg_f1 absent or not ok"
    if family == "FLAT40" and rec.get("hold") != 40:
        return False, f"hold = {rec.get('hold')!r}, not 40"
    return True, ""


def cg_f1_summary(records: dict) -> dict:
    """Section 5: CG-F1 is re-read from every existing record's own `cg_f1`
    block (not recomputed) — a resumed run must still refuse before any
    capture rate is read if a prior arm's block does not read `ok`. Every
    cell's block is compared against the first one seen for its family (not
    just stored once and never checked again) — a mismatch within a family
    is itself an offender. The nine arm JSONs are also re-read (item 1) and
    reported under their own key, `arm_records`."""
    by_family: dict = {}
    offenders: list[str] = []
    for name in CELL_ORDER:
        if name not in records:
            continue
        rec = records[name]
        family = rec["family"]
        block = rec.get("cg_f1")
        if not (block and block.get("ok")):
            offenders.append(name)
            continue
        if family in by_family and block != by_family[family]:
            offenders.append(name)  # mismatch within the family
            continue
        by_family.setdefault(family, block)
    arm_records: dict = {}
    for row in ROW_NAMES:
        ok, why = _check_arm_record(row)
        arm_records[row] = {"ok": ok, "why": why}
        if not ok:
            offenders.append(f"arm:{row}")
    return {
        "gate": (
            "CG-F1 (prereg section 5): FLAT's env differs from the selector's "
            "only in the action space and the zeroed tactic block (FLAT40: "
            "additionally `hold`, A1.2); STRUCT's only in the law/learner split."
        ),
        "verdict": "REFUSE" if offenders else "PASS",
        "families": by_family,
        "arm_records": arm_records,
        "offenders": offenders,
    }


def cg_f3() -> dict:
    """CG-F3 (prereg section 5): documentary, not computed — there is
    nothing to gate."""
    return {
        "gate": "CG-F3 (prereg section 5)",
        "verdict": "N/A (documentary)",
        "note": (
            "S24.1's CG-T1 and A2.1 (the mask / search-policy guards) do not "
            "apply here: FLAT/FLAT40 are continuous (v, omega) with no action "
            "mask, and STRUCT's learner is also a plain continuous PPO "
            "(SearchMDP, S17 BASE) with no mask. The A3.1 absent-evidence "
            "rule DOES apply and is enforced by H-SAFE below. S24.1's CG-T2 "
            "labels (mean switches per gate episode, 'it switches') also do "
            "not apply to FLAT/FLAT40/STRUCT rows: none of the three ever "
            "logs a tactic switch (`n_switches` is `None` throughout, never "
            "averaged), so no CG-T2 label is computed or carried for any row "
            "of this round."
        ),
    }


def direction(v: dict) -> str:
    """Section 4's refutation, in the order of precedence the round's logic
    imposes: H-SAFE, then CG-F1, then completeness, then H-FLAT-1/2/3."""
    safe, cgf1 = v["h_safe"], v["cg_f1"]
    f1, f2 = v["h_flat_1"], v["h_flat_2"]
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
    if cgf1["verdict"] == "REFUSE":
        return (
            f"CG-F1 REFUSES: {', '.join(cgf1['offenders'])}. FLAT's env must "
            "differ from the selector's only in the action space and the "
            "zeroed tactic block (FLAT40: additionally `hold`), and STRUCT's "
            "only in the law/learner split; until that holds no capture rate "
            "here is read."
        )
    if v["missing"] or safe["verdict"] == "INCOMPLETE":
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record"
            + (
                f" and there is NO training evidence for {', '.join(safe['missing_training'])} (A3.1)"
                if safe["missing_training"]
                else ""
            )
            + ". Every row x opponent must run on the one stream, and every "
            "arm's probe rows must be read, before a verdict is read (prereg "
            "section 6: a crashed cell is UNDETERMINED, never re-run)."
        )
    parts = []
    for family, block in f1["families"].items():
        parts.append(
            f"H-FLAT-1 ({family}): {block['verdict']} "
            f"({block['n_pass']}/{len(SEEDS)} seeds passed vs SEL-s)."
        )
    if f1["verdict"] == "PASS":
        parts.append(
            "H-FLAT-1 PASSES on BOTH flat families (A1.2): the selector beats "
            "flat learning at its own 10-step cadence AND at the selector's "
            "own 40-step decision cap. D151's claim stands."
        )
    elif not f1["passing_families"]:
        parts.append(
            "H-FLAT-1 FAILS on BOTH flat families: flat PPO on today's game "
            "learns as well as the selector; D151's claim is withdrawn to 'a "
            "selector matches flat learning with legible tactics' and the "
            "paper's framing changes (section 4's refutation)."
        )
    else:
        parts.append(
            "H-FLAT-1 fails the 'BOTH families' bar (A1.2) — passes only on "
            f"{', '.join(f1['passing_families'])}; S24.2 is not fully "
            "licensed by this round alone."
        )
    if f1["verdict"] == "PASS" and f2["verdict"] != "PASS":
        parts.append(
            "H-FLAT-2 FAILS where H-FLAT-1 passes: the selector beats flat "
            "but not the structured learner — the honest form of the claim "
            "(section 4)."
        )
    elif f2["verdict"] == "PASS":
        parts.append("H-FLAT-2 PASSES: the selector also beats the structured (SearchMDP) learner.")
    else:
        parts.append(f"H-FLAT-2: {f2['verdict']} ({f2['n_pass']}/{len(SEEDS)} seeds passed vs SEL-s).")
    parts.append(
        "H-FLAT-3 (characterise, do not assume): reported, not gated — "
        "FLAT-s vs LEAD-1.0/RULES-V0, and the immobile-step-rate / "
        "pinned-run-episode-fraction table beside the selector's."
    )
    cadence_bits = [
        f"{row} {d['mean_decisions_per_episode']:.1f} decisions/ep, "
        f"{d['env_steps_per_decision']:.2f} env steps/decision"
        for row in (*ROW_NAMES, *SEL_ROWS)
        if (d := v["cadence"].get(row))
    ]
    if cadence_bits:
        parts.append("A1.1's cadence, reported beside H-FLAT-1: " + "; ".join(cadence_bits) + ".")
    return " ".join(parts)


def verdicts(records: dict, combined: dict, probes: dict | None = None) -> dict:
    """Every verdict of the round — a pure function of this round's own
    records, the records JOINED with S24.1's comparators, and the probe
    rows."""
    missing = [n for n in CELL_ORDER if n not in records]
    return {
        "h_flat_1": h_flat_1(combined),
        "h_flat_2": h_flat_2(combined),
        "h_flat_3": h_flat_3(combined),
        "h_safe": h_safe(records, probes),
        "cg_f1": cg_f1_summary(records),
        "cg_f3": cg_f3(),
        # A1.1: the selector's OWN decisions / env-steps-per-decision, read
        # from the JOINED S24.1 records (`SEL-80/81/82`), beside this round's
        # own nine rows, so the cadence ratio can be read off one artefact.
        "cadence": {row: _cadence(combined, row) for row in (*ROW_NAMES, *SEL_ROWS)},
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "rows": list(ROWS),
        "opponents": list(OPPONENTS),
        "seeds": list(SEEDS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": PREREG,
    }


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
_BULK_KEYS = ("capture_ind", "steps", "violations", "n_switches", "n_decisions", "branch", "shield")


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
            np.nan if cell_name(row, o) not in records else records[cell_name(row, o)]["capture_rate"]
            for o in OPPONENTS
        ]
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=row)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(OPPONENTS), fontsize=8)
    ax.set_ylabel("capture rate", fontsize=9)
    ax.set_title(
        f"S24-FLAT — flat (hold 10/40) and structured baselines, ratio 1.2; "
        f"N = {N_EPISODES} seed-paired episodes from {GATE_SEED0}"
    )
    ax.legend(fontsize=6, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-F2)."""
    records = load_records()
    _measurable, rejected, _by_cell = scan_records()
    comparators = load_s24_1_comparators()
    combined = {**records, **comparators}
    probes = read_probes()
    v = verdicts(records, combined, probes)
    if v["cg_f1"]["verdict"] == "REFUSE":
        raise SystemExit(
            f"CG-F1 REFUSES the summary: {v['cg_f1']['offenders']}. Section 5 "
            "puts the construct gate before any capture rate, so no table is "
            "written while this stands."
        )
    if v["h_safe"]["missing_training"]:
        raise SystemExit(
            "H-SAFE REFUSES the summary: no training evidence for "
            f"{v['h_safe']['missing_training']} (probe rows absent or without "
            "`train_violations`). A3.1: absent evidence is an offender, never "
            "a zero."
        )
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "rejected_records": rejected,  # as S24.1's own summarize does
        "cells": {n: {k: val for k, val in r.items() if k not in _BULK_KEYS} for n, r in records.items()},
        "provenance": provenance(device),
    }
    S1._write_json(round_file("json"), blob)
    write_figure(records, round_file("png"))
    return blob


# ----------------------------------------------------------------------------
# CG-F2 (prereg section 5): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
gif_seeds = S1.gif_seeds  # the first three GATE seeds — a rule, not a pick
gif_title = S1.gif_title  # pure: f"{row} vs {opponent} · episode {seed}"
GIF_ROWS: tuple[str, ...] = (
    "FLAT-80", "FLAT-81", "FLAT-82",
    "FLAT40-80", "FLAT40-81", "FLAT40-82",  # A1.4: FLAT40 joins the watched rows
    "STRUCT-81",
)
GIF_OPPONENTS: tuple[str, ...] = ("TACTIC-V0", "GREEDY")
GIF_N = 3


def render_cell(row: str, opponent: str, seed: int, device: str = "cpu", every: int = GIF_EVERY):
    name = cell_name(row, opponent)
    env = build_cell_env(name, S1.cell_config(), device, recorder=True, label=gif_title(row, opponent, seed))
    family = CELLS[name]["family"]
    if family in FLAT_FAMILIES:
        game = env.game
        game.every = int(every)
        log = env.play_episode(load_policy(row, device).predict, seed)
        game.finish(log["outcome"], log["steps"])
        return game.frames, log
    env.every = int(every)
    log = env.play_episode(load_policy(row, device).predict, seed)
    env.finish(log["outcome"], log["steps"])
    return env.frames, log


def write_gifs(
    rows: list[str] | None = None, device: str = "cpu", every: int = GIF_EVERY, fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """CG-F2: the first three gate seeds per row vs TACTIC-V0 and GREEDY —
    (3 FLAT seeds + 3 FLAT40 seeds + STRUCT-81) x 2 opponents x 3 seeds = 42
    GIFs (A1.4: FLAT40 joins the watched rows; the prereg's own count, "36",
    does not match 7 x 2 x 3 — flagged, not silently matched)."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(f"unknown CG-F2 row(s): {unknown}; this gate watches {GIF_ROWS}")
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
        f"the first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS}. Watch "
        "for stalling, spinning and pins.",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for row, o, seed in todo:
        frames, log = render_cell(row, o, seed, device, every)
        path = gif_path(row, o, seed)
        save_gif(frames, str(path), fps=fps)
        print(f"wrote {path} ({len(frames)} frames; {log['outcome']} @ {log['steps']})", flush=True)
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
    # CG-F2 is a BLOCKING construct gate: the episodes are watched before any
    # capture rate is read, so in a single invocation the GIFs are written
    # BEFORE the table is summarised.
    if args.gifs is not None:
        write_gifs(args.gifs or None, args.device)
    if args.summarize:
        blob = summarize(args.device)
        v = blob["verdicts"]
        print(json.dumps(
            {
                "h_flat_1": {"verdict": v["h_flat_1"]["verdict"], "passing_families": v["h_flat_1"]["passing_families"]},
                "h_flat_2": {"verdict": v["h_flat_2"]["verdict"]},
            },
            indent=2,
        ))
        print(json.dumps(v["h_safe"], indent=2))
        print(json.dumps(v["cg_f1"], indent=2))
        print(f"\n{blob['direction']}")
        print(f"\ncells {v['n_records']}/{v['n_cells']}" + ("" if v["complete"] else " — INCOMPLETE"))


if __name__ == "__main__":
    main()
