"""S24.1 Step 1.2 — the gate: 24 cells, the verdicts, the construct gates.

Pre-registration: `docs/pathway/tactics_prereg_s24_1_2026-10-13.md` sections 4
to 6 INCLUDING Amendments A1 and A2. Where the brief and the prereg could be
read to disagree, the PREREGISTRATION wins and the reading is named at the line.

Evaluation only. Nothing here trains, selects or tunes anything.

A THIN derivative of `scripts/derisk_s23_strategy.py` (S23). IMPORTED and
CALLED: `_write_json`, `_sha256`, `_ci`, `_q`, `_cell_line`, `UNDETERMINED`,
`S19.provenance` (extended here), `S19.opponent_factory` (GREEDY / HUG only).
COPIED because S23's version closes over S23's own cells, prefix or stream —
each says so at its definition: the artefact paths, the record-validity trio,
`read_cell`, `undetermined_record`, `run_cells`'s refusal order, `write_gifs`'s
shape and `main`'s dispatch order.

Twenty-four cells, 8 x 3, N = 500 paired episodes on one stream (section 4):

    rows       SEL-80/81/82      the learned selector (memory ON)
               NOMEM-80/81/82    the negative control (A2.3: the selector's
                                 memory INPUTS zeroed, nothing else)
               LEAD-1.0          `Lead` = `LeadPlannerInterceptLaw(cfg, 1.0)`,
                                 S23's best law, as the pursuer's only tactic
               RULES-V0          the scripted pursuer selector `PursuerRulesV0`
    opponents  GREEDY / HUG / TACTIC-V0 (the scripted tactical evader — the
               TRAINING opponent, in-distribution and labelled so)

The ONLY difference between the rows is the pursuer's `select`. The v0b game
assembly is not rebuilt here: `scripts/s24_tactic_game.py::build_game` is
called and ONE field is replaced, exactly as `pe2/selector_mdp.py` does it, so
the shield (`RecoveryShieldV2`, fresh per episode), the hold, the observation
flags, the tactic library, the memories and the evader seat are bit-for-bit the
baseline's; the table is asserted on the BUILT env in
`tests/pe2/test_derisk_s24_1_round.py`.

Readings this module fixes, because the text can be read two ways:

1. **`LEAD-1.0` is the v0b assembly with `select = fixed(pursuer.lead)`.** The
   brief says "`LeadPlannerInterceptLaw(cfg, tau=1.0)` as the pursuer law in the
   SAME v0b game assembly"; `pe2.tactics.Lead` IS `LeadPlannerInterceptLaw(cfg,
   1.0)` (its docstring names S23's `LEAD-1.0`), so fixing the selector on that
   tactic installs the comparator law through the game's own law seat and keeps
   one variable — the selector — between this row and the learned ones. Driving
   it as a bare `law=` object instead would also have replaced the span log and
   the usage accounting the record reads.
2. **The GREEDY / HUG opponents replace `inner.opponent_factory`,** which
   `SingleAgentWrapper.reset` reads on every episode; everything else in the
   assembly, the evader-side memory included, is untouched. The evader's tactic
   history is then EMPTY and its usage reads `UNKNOWN` on every step — the
   scripted evaders are not tactic adapters — and that is recorded as the fact
   it is rather than hidden. `TACTIC-V0` keeps the assembly's own evader.
3. **The scripted rows' per-decision `picks` come from the span log.** A2.2
   fixes H-TAC-2 on `picks`, which only `SelectorMDP` counts. `TacticLaw`
   appends exactly one span per `select` call (`_TacticHolder._enter`), so the
   span families ARE the per-decision histogram for `LEAD-1.0` / `RULES-V0`,
   zero-length spans included: they were picks.
4. **CG-T2's mean is over all of a row's gate episodes** (the three opponents
   pooled, as "on the gate" reads), and the per-opponent means are reported
   beside it. The label is attached to every row — `LEAD-1.0` is law-like by
   construction — and gates nothing.
5. **A2.1's `(search policy)` guard is read per LEARNED cell.** The comparator
   rows reach the search policy whenever their law is DONE on entry (S23's
   `LEAD-1.0` did too), which is the comparator working as measured; only the
   learned cells carry A2.1's invariant and `summarize` refuses without it.

Usage:
  ./venv/bin/python scripts/derisk_s24_1_gate.py --cells         (all missing)
  ./venv/bin/python scripts/derisk_s24_1_gate.py --cells SEL-80_HUG
  ./venv/bin/python scripts/derisk_s24_1_gate.py --gifs          (CG-T3)
  ./venv/bin/python scripts/derisk_s24_1_gate.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 6, the stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # prereg section 6 (as S23); BEFORE numpy / torch build pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.recogniser import tactic_family
from pe2.selector_mdp import ACTIONS, SelectorMDP, count_switches, spans_with_length
from pe2.structured import HOLD, OrbitSearch
from pe2.tactics import fixed

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S23's runner (and through it S19's), loaded by FILE PATH so this module
# imports the same way as a script and as `scripts.derisk_s24_1_gate`.
S23 = _load("derisk_s23_strategy", _HERE / "derisk_s23_strategy.py")
S19 = S23.S19
# The v0b assembly, through Step 1.1's OWN loader, so this module and
# `SelectorMDP` hold the SAME module object (and so the same `build_game`,
# recorder class and `label_branch` guard) rather than two copies of it.
from pe2.selector_mdp import _game_module  # loaded after S23, by design

GAME = _game_module()

# --- imported from S23/S19 and CALLED, not reimplemented ---------------------
_write_json = S23._write_json
_sha256 = S23._sha256
_ci = S23._ci
_q = S23._q
_cell_line = S23._cell_line  # A4.6: no rate while the cells run
UNDETERMINED = S23.UNDETERMINED
scripted_opponent = S19.opponent_factory  # GREEDY / HUG; never called with ATK

PREREG = "docs/pathway/tactics_prereg_s24_1_2026-10-13.md"
ARTEFACT_PREFIX = "s24_1"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 3 to 6). Do not tune. ---
GATE_SEED0 = 2_400_000  # section 4, the gate stream
N_EPISODES = 500  # section 4
P_GATE = 0.05  # section 4, H-TAC-1 / H-TAC-3
P_CHI2 = 0.01  # section 4, H-TAC-2
PASS_MIN = 2  # ">= 2/3", opponents and seeds alike
SEEDS: tuple[int, ...] = (80, 81, 82)  # section 3
TRAIN_TOTAL = 3_000_000  # section 3's budget: the final checkpoint's mark
CG_T2_MIN_SWITCHES = 2.0  # section 5, CG-T2
LAW_LIKE = "LAW-LIKE"  # section 5's label for a selector that does not switch
SEARCH_KEY = GAME.SEARCH_KEY  # "(search policy)" — A2.1's guard reads this

BASELINE_ROW = "LEAD-1.0"  # H-TAC-1's comparator (section 4)
RULES_ROW = "RULES-V0"  # H-TAC-1b's and H-TAC-2(b)'s comparator
ROWS: dict[str, dict] = {
    f"{family}-{seed}": {
        "kind": "learned",
        "arm": f"{family}-{seed}",
        "seed": seed,
        "memory": family == "SEL",
    }
    for family in ("SEL", "NOMEM")
    for seed in SEEDS
} | {
    BASELINE_ROW: {"kind": "scripted", "select": "fixed(Lead) = LEAD-1.0"},
    RULES_ROW: {"kind": "scripted", "select": "PursuerRulesV0"},
}
LEARNED_ROWS: tuple[str, ...] = tuple(r for r, s in ROWS.items() if s["kind"] == "learned")
SEL_ROWS: tuple[str, ...] = tuple(f"SEL-{s}" for s in SEEDS)
# Section 4: the training opponent is in-distribution and said so.
OPPONENTS: dict[str, dict] = {
    "GREEDY": {"in_distribution": False},
    "HUG": {"in_distribution": False},
    "TACTIC-V0": {"in_distribution": True},
}


def cell_name(row: str, opponent: str) -> str:
    return f"{row}_{opponent}"


def _build_cells() -> dict[str, dict]:
    """The 24 cells. Each carries its row and opponent STORED — nothing parses
    a name back apart (S17's convention; here it would also be wrong, because
    `LEAD-1.0` and `SEL-80` both contain a separator)."""
    return {
        cell_name(r, o): {
            "name": cell_name(r, o),
            "row": r,
            "opponent": o,
            "in_distribution": OPPONENTS[o]["in_distribution"],
            "hold": HOLD,
            **ROWS[r],
        }
        for r in ROWS
        for o in OPPONENTS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance  (copied from S23: each closes over S23's own
# ARTEFACT_PREFIX / SPEC / OUT)
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


def ckpt_dir(row: str) -> pathlib.Path:
    return OUT / f"ckpts_S24-1-{ROWS[row]['arm']}"


def ckpt_path(row: str) -> pathlib.Path:
    """Step 1.1's own layout (`derisk_s24_1_train.py::_Probe`):
    `ckpts_S24-1-{ARM}/S24-1-{ARM}_{mark}.zip`. The brief's
    `{ARM}_3000000.zip` omits the `S24-1-` stem; the implemented interface
    wins, because that is the file on disk."""
    arm = ROWS[row]["arm"]
    return ckpt_dir(row) / f"S24-1-{arm}_{TRAIN_TOTAL}.zip"


def probe_file(row: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_probe_{ROWS[row]['arm']}.json"


def spec_path() -> pathlib.Path:
    return _HERE.parent / PREREG


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    _write_json(path, record)
    return path


def provenance(device: str = "cpu") -> dict:
    """S19's provenance block with THIS round's spec hash and gate names.

    `S19.provenance` is CALLED for the commit, the `pe2` package, the device,
    the library versions and the thread setting; only the fields naming this
    round's text are replaced. The spec hash is the point: a cell record proves
    which text's thresholds it was measured under."""
    return S19.provenance(device) | {
        "spec": PREREG,
        "spec_sha256": _sha256(spec_path()),
        "cg_b": None,
        "shield": "RecoveryShieldV2 (D131), fresh per episode — the v0b assembly's",
        "label_branch": GAME.label_branch(),
        "gate_stream": GATE_SEED0,
    }


# ----------------------------------------------------------------------------
# record validity (S19 A4.4 / A4.5 / A5.2), copied: S23's versions close over
# its own CELLS, N_EPISODES, GATE_SEED0 and cells_dir
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    """`(status, why)` — `"MEASURABLE"`, `"UNDETERMINED"` or `"INVALID"`.
    Section 6: records are measurable only at N = 500 on seed0 2 400 000."""
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == UNDETERMINED:
        return UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    n, seed0 = rec.get("n_episodes"), rec.get("seed0")
    if n != N_EPISODES:
        return "INVALID", f"n_episodes {n!r} != the pre-registered {N_EPISODES}"
    if seed0 != GATE_SEED0:
        return "INVALID", f"seed0 {seed0!r} != the pre-registered {GATE_SEED0}"
    return "MEASURABLE", ""


def scan_records() -> tuple[dict, dict, dict]:
    """`(measurable, rejected, rejected_by_cell)`. A cell claimed by more than
    one file is poisoned and every file claiming it is rejected (A5.2:
    duplicates reject BOTH, however they are named)."""
    d = cells_dir()
    measurable: dict = {}
    source: dict = {}
    poisoned: set = set()
    rejected: dict = {}
    by_cell: dict = {}
    if not d.exists():
        return measurable, rejected, by_cell

    def reject(fname: str, cell: str | None, why: str) -> None:
        rejected[fname] = why
        if cell in CELLS:
            by_cell.setdefault(cell, []).append(fname)

    for p in sorted(d.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            reject(p.name, None, f"unreadable: {exc}")
            continue
        status, why = record_status(rec)
        name = rec.get("name")
        if status != "MEASURABLE":
            reject(p.name, name, f"{status}: {why}")
            continue
        if name in poisoned:
            reject(p.name, name, f"duplicate record for cell {name}")
            continue
        if name in measurable:
            poisoned.add(name)
            del measurable[name]
            reject(source.pop(name), name, f"duplicate record for cell {name}")
            reject(p.name, name, f"duplicate record for cell {name}")
            continue
        measurable[name] = rec
        source[name] = p.name
    return measurable, rejected, by_cell


def load_records(*, quiet: bool = False) -> dict:
    """Only the MEASURABLE records; anything else is dropped and listed."""
    measurable, rejected, _by_cell = scan_records()
    if rejected and not quiet:
        print(
            f"REFUSED {len(rejected)} record(s) in {cells_dir()} — not measurable "
            f"(n_episodes == {N_EPISODES} and seed0 == {GATE_SEED0} are required):",
            flush=True,
        )
        for fname, why in sorted(rejected.items()):
            print(f"  {fname}: {why}", flush=True)
    return measurable


def read_cell(name: str) -> dict:
    """The cell's MEASURABLE record, or SystemExit. CG-T3 is watched per
    MEASURED cell: existence of the file is not enough."""
    path = cell_file(name)
    if not path.exists():
        raise SystemExit(
            f"no cell record for {name} at {path}: run --cells {name} first. "
            "CG-T3 is watched per measured cell (prereg section 5)."
        )
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cell record {path} is unreadable: {exc}") from exc
    status, why = record_status(rec)
    if status != "MEASURABLE":
        raise SystemExit(
            f"cell record {path} is not measurable ({status}: {why}), so there is "
            f"nothing to watch for {name}. An UNDETERMINED cell stays "
            "UNDETERMINED under the stop rule and is never rendered from."
        )
    return rec


# ----------------------------------------------------------------------------
# the cell's env — the only difference between rows is the pursuer's `select`
# ----------------------------------------------------------------------------
def cell_config() -> EnvConfig:
    """`EnvConfig()` — whose `evader_speed_ratio` default IS the round's 1.2
    (prereg section 2). Nothing sweeps here."""
    cfg = EnvConfig()
    if cfg.evader_speed_ratio != 1.2:
        raise SystemExit(
            f"EnvConfig's speed ratio moved to {cfg.evader_speed_ratio}; this "
            "round is pre-registered at 1.2 and may not run under another."
        )
    return cfg


def _set_opponent(game, opponent: str, cfg: EnvConfig, device: str) -> None:
    """Reading 2: the opponent seat, and nothing else in the assembly.

    A recorded cell against a scripted evader also has its frame label's EVADER
    half named: the game's own caption reads the tactic adapter, which those
    cells do not drive, so it would print `UNKNOWN` on every frame of a CG-T3
    GIF. The FORMAT is the game's, unchanged, and the PURSUER half is still read
    off the law's current tactic inside the step the frame shows (A2)."""
    if opponent == "TACTIC-V0":
        return
    game.inner.opponent_factory = scripted_opponent(opponent, cfg, device)
    if hasattr(game, "frames"):
        game.caption = lambda _tag: GAME.frame_label(opponent, GAME._name(game.law))


def build_cell_env(
    name: str,
    cfg: EnvConfig | None = None,
    device: str = "cpu",
    *,
    recorder: bool = False,
    label: str = "",
):
    """The cell's env: `SelectorMDP` for a learned row, the v0b `TacticGameMDP`
    for a scripted one, with this row's `select` and this cell's opponent and
    NOTHING else varied. `recorder=True` builds the SAME object with the
    recording added (CG-C's discipline, through the game's own class)."""
    spec = CELLS[name]
    cfg = cell_config() if cfg is None else cfg
    if spec["kind"] == "learned":
        env = SelectorMDP(
            cfg.evader_speed_ratio,
            memory=spec["memory"],
            recorder=recorder,
            label=label,
            cfg=cfg,
        )
        _set_opponent(env.game, spec["opponent"], cfg, device)
        return env
    game, rules = GAME.build_game(cfg, recorder=recorder, label=label)
    if spec["row"] == BASELINE_ROW:
        game.law.select = fixed(rules["pursuer"].lead)  # reading 1
    _set_opponent(game, spec["opponent"], cfg, device)
    return game


_POLICIES: dict = {}


def require_ckpt(row: str) -> pathlib.Path:
    """A learned row REFUSES to run without its frozen final checkpoint. Not
    defensive coding: a cell quietly run from an earlier mark would be a
    different arm wearing this one's name (section 6's stop rule)."""
    path = ckpt_path(row)
    if not path.exists():
        raise SystemExit(
            f"missing the final checkpoint for row {row} at {path}. Section 3 "
            f"fixes the budget at {TRAIN_TOTAL} env steps; there is no "
            "substitute and no default, so the round refuses to run."
        )
    return path


def load_policy(row: str, device: str = "cpu"):
    """The row's final checkpoint as a `MaskablePPO`, cached per path."""
    from sb3_contrib import MaskablePPO

    path = require_ckpt(row)
    key = (str(path), device)
    if key not in _POLICIES:
        _POLICIES[key] = MaskablePPO.load(path, device=device)
    return _POLICIES[key]


# ----------------------------------------------------------------------------
# one episode, one cell
# ----------------------------------------------------------------------------
def scripted_log(game, cfg: EnvConfig, seed: int) -> dict:
    """One scripted-row episode, in `SelectorMDP.play_episode`'s log schema.

    The search policy is `OrbitSearch` (`SWEEP`'s own primitive, as S19/S23 and
    the v0b game), reached whenever the pursuer's law is DONE on entry. `picks`
    and `n_switches` come from the span log (readings 3 and A1.2's own
    `count_switches`), so the two row types are counted by the same functions.
    """
    log = game.play_episode(OrbitSearch(cfg).predict, seed)
    spans = spans_with_length(game.law.history, game.law.step_index)
    picks = Counter(tactic_family(s["name"]) for s in spans)
    return log | {
        "tactics": {
            "evader": GAME._history(game.evader),
            "pursuer": GAME._history(game.law),
        },
        "usage": {role: dict(counts) for role, counts in game.usage.items()},
        "branch": game.branch_summary(),
        "n_decisions": len(spans),
        "picks": {name: int(picks.get(name, 0)) for name in ACTIONS},
        "pursuer_spans": spans,
        "n_zero_length_spans": sum(s["zero_length"] for s in spans),
        "n_switches": int(count_switches(spans)),
        "inapplicable_picks": 0,  # a fixed or scripted selector cannot mis-pick
        "recovery": {
            "n_recoveries": int(game.shield.n_recoveries),
            "n_forward": int(game.shield.n_recoveries_forward),
            "n_rotation": int(game.shield.n_recoveries_rotation),
        },
    }


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """One cell: N paired episodes of this row's selector under the recovery
    shield against this opponent, on the gate stream. The shield is rebuilt per
    episode by the assembly's own `_begin` (D149/S17), so the paired-seed
    boundary carries no shield state across it."""
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    cfg = cell_config()
    env = build_cell_env(name, cfg, device)
    learned = spec["kind"] == "learned"
    predict = load_policy(spec["row"], device).predict if learned else None

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
        log = (
            env.play_episode(predict, GATE_SEED0 + i)
            if learned
            else scripted_log(env, cfg, GATE_SEED0 + i)
        )
        capture_ind.append(log["outcome"] == "capture")
        outcomes[log["outcome"]] += 1
        ep_steps.append(int(log["steps"]))
        violations.append(int(log["violations"]))
        switches.append(int(log["n_switches"]))
        zero_spans.append(int(log["n_zero_length_spans"]))
        decisions.append(int(log["n_decisions"]))
        # `inapplicable_picks` is CUMULATIVE on the env (CG-T1's counter has no
        # episode hook), so the per-episode figure is its increment.
        total_bad = int(log["inapplicable_picks"])
        bad_picks.append(total_bad - seen_bad)
        seen_bad = total_bad
        tactics.append(log["tactics"])
        branch.append(log["branch"])
        picks.update(log["picks"])
        usage.update(log["usage"]["pursuer"])

    k = sum(capture_ind)
    return {
        "name": name,
        "row": spec["row"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": spec["kind"],
        "memory": spec.get("memory"),
        "select": spec.get("select"),
        "hold": spec["hold"],
        "n_episodes": n,
        "seed0": GATE_SEED0,
        "device": device,
        "capture_ind": capture_ind,
        "capture_rate": k / n if n else float("nan"),
        "wilson_capture": _ci(k, n),
        "steps": ep_steps,
        "violations": violations,
        "violation_steps": int(sum(violations)),
        "n_switches": switches,
        "n_zero_length_spans": zero_spans,
        "n_decisions": decisions,
        "inapplicable_picks_per_episode": bad_picks,
        "inapplicable_picks": int(seen_bad),  # CG-T1
        # A2.2: the per-DECISION histogram over the five tactics, never
        # `usage_steps` (which weighs a span by its length and carries a sixth
        # key). `usage_steps` is recorded beside it, as the v0b table's unit.
        "picks": {a: int(picks.get(a, 0)) for a in ACTIONS},
        "usage_steps": dict(usage),
        "search_policy_steps": int(usage.get(SEARCH_KEY, 0)),  # A2.1's guard
        "tactics": tactics,
        "branch": branch,
        "aggregate": {
            "capture_rate": k / n if n else float("nan"),
            "outcomes": dict(outcomes),
            # The env terminates on capture and truncates on timeout and
            # nothing else, so the evader escapes exactly when it times out.
            "escape_rate": outcomes["timeout"] / n if n else float("nan"),
            "mean_steps": float(statistics.fmean(ep_steps)) if ep_steps else float("nan"),
            "mean_switches": float(statistics.fmean(switches)) if switches else float("nan"),
            "switches_p90": _q(switches, 0.90),
            "mean_decisions": float(statistics.fmean(decisions)) if decisions else float("nan"),
            "mean_zero_length_spans": float(statistics.fmean(zero_spans))
            if zero_spans
            else float("nan"),
            "picks_share": {
                a: picks.get(a, 0) / max(1, sum(picks.values())) for a in ACTIONS
            },
            "evader_tactics_recorded": spec["opponent"] == "TACTIC-V0",
        },
        "provenance": provenance(device),
    }


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    """A4.5 / section 6, copied (S23's closes over its own stream): a crashed
    cell is `UNDETERMINED`, written as a record rather than left absent so the
    stop rule can be enforced mechanically. `n_episodes` is 0 so it can never
    read as a measurement."""
    import traceback

    return {
        "name": name,
        "verdict": UNDETERMINED,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "n_episodes": 0,
        "seed0": GATE_SEED0,
        "note": (
            "a crashed cell is UNDETERMINED and is NEVER re-run (prereg section "
            "6). Delete this file only with an amendment that says so, appended "
            "before the number it touches exists."
        ),
        "provenance": provenance(device),
    }


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell with no record yet (default: all 24). Literal names only.

    Everything that can refuse refuses BEFORE the first episode: every learned
    row's checkpoint, and every existing file's validity. A file that is not a
    measurable record of THIS round is a REFUSAL, not a skip and not an
    overwrite. Copied from S23 because its version closes over S23's cells."""
    wanted = list(CELL_ORDER) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(
            f"unknown cell(s): {unknown}; this round's cells are {CELL_ORDER}"
        )
    cells_dir().mkdir(parents=True, exist_ok=True)
    measurable, rejected, by_cell = scan_records()
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
        if ROWS[row]["kind"] == "learned":
            require_ckpt(row)
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
            print(f"[{name}] {UNDETERMINED}: {rec['error']}", flush=True)
            continue
        write_cell(rec)
        print(_cell_line(rec), flush=True)
    return todo


# ----------------------------------------------------------------------------
# the training probes (A2.4: CG-T1 and H-SAFE evidence survive a crashed arm)
# ----------------------------------------------------------------------------
def read_probes() -> dict:
    """`{row: {"rows": [...], "missing": bool}}` for the learned rows, read off
    Step 1.1's probe JSONs. A2.4 writes `train_inapplicable_picks` and
    `train_violations` into every probe row, so a crashed arm still carries
    them; `inapplicable_picks` and `violations` in the same row are the PROBE
    env's, and CG-T1 reads both."""
    out: dict = {}
    for row in LEARNED_ROWS:
        path = probe_file(row)
        if not path.exists():
            out[row] = {"path": str(path), "missing": True, "rows": []}
            continue
        try:
            rows = json.loads(path.read_text()).get("checkpoints", [])
        except (OSError, json.JSONDecodeError) as exc:
            out[row] = {"path": str(path), "missing": True, "error": str(exc), "rows": []}
            continue
        out[row] = {"path": str(path), "missing": False, "rows": rows}
    return out


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4)
# ----------------------------------------------------------------------------
def _rec(records: dict, row: str, opponent: str) -> dict | None:
    return records.get(cell_name(row, opponent))


def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _direction_label(delta: float, a: str, b: str) -> str:
    if delta > 0:
        return f"{a} > {b}"
    if delta < 0:
        return f"{a} < {b}"
    return f"{a} == {b}"


def _paired(records: dict, row: str, base: str, opponents) -> dict:
    """`{opponent: block}` of the paired capture test `row` vs `base`. The
    DIRECTION is reported whether or not the test passes: section 4's
    refutation paragraph turns on the sign, so a failure that does not say
    which way it went is not a usable refutation."""
    per: dict = {}
    for o in opponents:
        a, b = _rec(records, row, o), _rec(records, base, o)
        if a is None or b is None:
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": [
                    cell_name(r, o) for r, v in ((row, a), (base, b)) if v is None
                ],
            }
            continue
        test = {
            "test": "paired_indicator",
            **paired_indicator_test(a["capture_ind"], b["capture_ind"], P_GATE),
        }
        delta = a["capture_rate"] - b["capture_rate"]
        per[o] = {
            "verdict": "PASS" if test["pass"] else "FAIL",
            "capture_row": a["capture_rate"],
            "capture_comparator": b["capture_rate"],
            "delta": delta,
            "direction": _direction_label(delta, row, base),
            "p": test["p"],
            "in_distribution": a["in_distribution"],
            "test": test,
        }
    return per


def _per_seed(per: dict, name: str, row: str, base: str, extra: dict) -> dict:
    verdicts_ = [b["verdict"] for b in per.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    return {
        "hypothesis": name,
        "row": row,
        "comparator": base,
        "verdict": "INCOMPLETE"
        if "INCOMPLETE" in verdicts_
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "n_opponents": len(per),
        "p_gate": P_GATE,
        "opponents": per,
        **extra,
    }


def h_tac_1(records: dict, seed: int, *, base: str = BASELINE_ROW) -> dict:
    """H-TAC-1 for ONE seed: shielded capture, `SEL-s` > `LEAD-1.0`,
    `paired_indicator_test` p < 0.05, on >= 2/3 opponents."""
    row = f"SEL-{seed}"
    name = "H-TAC-1" if base == BASELINE_ROW else "H-TAC-1b"
    return _per_seed(
        _paired(records, row, base, OPPONENTS), name, row, base, {"seed": seed}
    )


def h_tac_1b(records: dict, seed: int) -> dict:
    """H-TAC-1b (section 4: reported, NOT gated): the same test vs `RULES-V0`."""
    return h_tac_1(records, seed, base=RULES_ROW) | {"gated": False}


def _round(per_seed: dict, name: str, gated: bool = True) -> dict:
    passing = [s for s, b in per_seed.items() if b["verdict"] == "PASS"]
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in per_seed.values())
    return {
        "hypothesis": f"{name} (round)",
        "verdict": "INCOMPLETE"
        if incomplete
        else ("PASS" if len(passing) >= PASS_MIN else "FAIL"),
        "gated": gated,
        "rule": f">= {PASS_MIN}/{len(per_seed)} seeds",
        "passing_seeds": passing,
        "n_pass": len(passing),
        "seeds": per_seed,
    }


def h_tac_1_round(records: dict) -> dict:
    return _round({str(s): h_tac_1(records, s) for s in SEEDS}, "H-TAC-1")


def h_tac_1b_round(records: dict) -> dict:
    """Reported, not gated (section 4): the container's verdict says only what
    the same bar would have said against the scripted selector."""
    return _round({str(s): h_tac_1b(records, s) for s in SEEDS}, "H-TAC-1b", False)


def chi2_table(table, p_gate: float = P_CHI2) -> dict:
    """A chi^2 test of independence on a counts table, with the DEGENERATE case
    named rather than crashed on.

    `scipy.stats.chi2_contingency` is used (scipy is in the venv; the test
    suite also checks it against a hand-computed table). All-zero rows and
    columns are dropped first — a tactic neither side ever picked carries no
    information and would give the test a zero expected frequency — and a table
    left with fewer than two rows or two columns is DEGENERATE: there is nothing
    to be independent of. The CALLER decides what that means, because an empty
    table is missing data while a one-column table is a selector that used one
    tactic (section 4's "single-tactic default"), and those are not the same.
    """
    from scipy.stats import chi2_contingency

    a = np.asarray(table, dtype=float)
    rows = [i for i in range(a.shape[0]) if a[i, :].sum() > 0]
    cols = [j for j in range(a.shape[1]) if a[:, j].sum() > 0]
    out = {
        "table": [[int(x) for x in r] for r in a.tolist()],
        "n": int(a.sum()),
        "n_nonzero_rows": len(rows),
        "n_nonzero_cols": len(cols),
        "p_gate": p_gate,
    }
    if len(rows) < 2 or len(cols) < 2:
        return out | {
            "verdict": "DEGENERATE",
            "p": None,
            "chi2": None,
            "dof": None,
            "differs": False,
        }
    chi2, p, dof, _exp = chi2_contingency(a[np.ix_(rows, cols)])
    return out | {
        "verdict": "OK",
        "p": float(p),
        "chi2": float(chi2),
        "dof": int(dof),
        "differs": bool(p < p_gate),
    }


def _picks_row(rec: dict | None) -> list[int]:
    return [0] * len(ACTIONS) if rec is None else [int(rec["picks"][a]) for a in ACTIONS]


def h_tac_2(records: dict, seed: int) -> dict:
    """H-TAC-2 for ONE seed (section 4, A2.2): the learned selector's
    per-DECISION pick distribution (a) differs across the three opponents
    (chi^2 over the 5 tactics, p < 0.01) AND (b) differs from `RULES-V0`'s
    profile on the same opponent, on >= 2/3 opponents.

    A one-tactic pick histogram makes both tables DEGENERATE and is a FAIL, not
    a vacuum: "a selector that passes H-TAC-1 by always picking `LEAD` has not
    chosen" is the case this hypothesis exists to catch. VACUOUS (never PASS) is
    kept for the only reading under which there is nothing to test at all — a
    cell with no decisions in it."""
    row = f"SEL-{seed}"
    cells = {o: _rec(records, row, o) for o in OPPONENTS}
    missing = [cell_name(row, o) for o, r in cells.items() if r is None] + [
        cell_name(RULES_ROW, o) for o in OPPONENTS if _rec(records, RULES_ROW, o) is None
    ]
    if missing:
        return {
            "hypothesis": "H-TAC-2",
            "seed": seed,
            "row": row,
            "verdict": "INCOMPLETE",
            "missing": missing,
        }
    across = chi2_table([_picks_row(cells[o]) for o in OPPONENTS])
    a_block = dict(across, opponents=list(OPPONENTS), tactics=list(ACTIONS))
    if across["verdict"] == "DEGENERATE":
        a_block["verdict"] = "VACUOUS" if across["n"] == 0 else "FAIL"
        a_block["note"] = (
            "no decisions were taken in these cells, so there is no profile"
            if across["n"] == 0
            else "the selector used fewer than two tactics in total: a "
            "single-tactic default, which section 4 names as a failure of this "
            "hypothesis and not as an absent phenomenon"
        )
    else:
        a_block["verdict"] = "PASS" if across["differs"] else "FAIL"
    per: dict = {}
    for o in OPPONENTS:
        block = chi2_table([_picks_row(cells[o]), _picks_row(_rec(records, RULES_ROW, o))])
        block["comparator"] = RULES_ROW
        if block["verdict"] == "DEGENERATE":
            block["verdict"] = "VACUOUS" if block["n_nonzero_rows"] < 2 else "FAIL"
            block["note"] = (
                f"one of {row} / {RULES_ROW} took no decisions here, so the two "
                "profiles cannot be compared"
                if block["n_nonzero_rows"] < 2
                else "both selectors used the same single tactic: the profiles do "
                "not differ"
            )
        else:
            block["verdict"] = "PASS" if block["differs"] else "FAIL"
        per[o] = block
    n_pass = sum(b["verdict"] == "PASS" for b in per.values())
    b_pass = n_pass >= PASS_MIN
    return {
        "hypothesis": "H-TAC-2",
        "seed": seed,
        "row": row,
        "statistic": "chi^2 of the per-DECISION picks histogram (A2.2), 5 tactics",
        "verdict": "PASS" if (a_block["verdict"] == "PASS" and b_pass) else "FAIL",
        "across_opponents": a_block,
        "vs_rules_v0": per,
        "n_pass_vs_rules": n_pass,
        "pass_min": PASS_MIN,
        "p_chi2": P_CHI2,
    }


def h_tac_2_round(records: dict) -> dict:
    return _round({str(s): h_tac_2(records, s) for s in SEEDS}, "H-TAC-2")


def h_tac_3(records: dict, seed: int) -> dict:
    """H-TAC-3 for ONE seed (section 4, A2.3): `SEL-s` > `NOMEM-s` against
    `TACTIC-V0`, p < 0.05. `NOMEM` ablates the selector's memory INPUTS only —
    the system's dead reckoning still drives `SWEEP` and resolves
    `COUNTER-ORBIT`/`AMBUSH`'s obstacle in both arms."""
    row, base = f"SEL-{seed}", f"NOMEM-{seed}"
    per = _paired(records, row, base, ("TACTIC-V0",))
    return _per_seed(
        per,
        "H-TAC-3",
        row,
        base,
        {
            "seed": seed,
            "pass_min": 1,  # one opponent: TACTIC-V0, the training opponent
            "verdict": per["TACTIC-V0"]["verdict"],
            "ablation": "the selector's memory INPUTS only (A2.3)",
        },
    )


def h_tac_3_round(records: dict) -> dict:
    return _round({str(s): h_tac_3(records, s) for s in SEEDS}, "H-TAC-3")


def h_safe(records: dict, probes: dict | None = None) -> dict:
    """H-SAFE: 0 contact steps in all 24 cells AND 0 in training (A2.4: every
    probe row carries the training env's `train_violations`, so a crashed arm
    still carries the evidence)."""
    have = {n: records[n] for n in CELL_ORDER if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violation_steps", 0)))
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    train: dict = {}
    for row, block in (probes or {}).items():
        rows = block.get("rows") or []
        train[row] = {
            "n_probe_rows": len(rows),
            "train_violations_max": max(
                (int(r.get("train_violations", 0)) for r in rows), default=None
            ),
            "probe_violations_total": sum(int(r.get("violations", 0)) for r in rows),
            "missing": bool(block.get("missing")),
        }
    train_offenders = sorted(
        r for r, b in train.items() if (b["train_violations_max"] or 0) or b["probe_violations_total"]
    )
    missing = _missing(records)
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS"
        if (steps or train_offenders)
        else ("INCOMPLETE" if missing else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": offenders,
        "training": train,
        "training_offenders": train_offenders,
        "gated_cells": list(CELL_ORDER),
        "missing": missing,
    }


def cg_t1(records: dict, probes: dict | None = None) -> dict:
    """CG-T1 (section 5): 0 inapplicable picks over all learned cells and all
    training probes. A failure REFUSES the summary — the mask is what makes an
    inapplicable pick impossible, so a non-zero count means the action space the
    round measured is not the one it registered."""
    cells = {
        n: int(records[n].get("inapplicable_picks", 0))
        for n in CELL_ORDER
        if n in records and CELLS[n]["kind"] == "learned"
    }
    train = {}
    for row, block in (probes or {}).items():
        rows = block.get("rows") or []
        train[row] = {
            "train_inapplicable_picks_max": max(
                (int(r.get("train_inapplicable_picks", 0)) for r in rows), default=None
            ),
            "probe_inapplicable_picks_max": max(
                (int(r.get("inapplicable_picks", 0)) for r in rows), default=None
            ),
            "n_probe_rows": len(rows),
            "missing": bool(block.get("missing")),
        }
    offenders = sorted(n for n, v in cells.items() if v) + sorted(
        row
        for row, b in train.items()
        if (b["train_inapplicable_picks_max"] or 0)
        or (b["probe_inapplicable_picks_max"] or 0)
    )
    return {
        "gate": "CG-T1 (the mask holds), prereg section 5",
        "verdict": "REFUSE" if offenders else "PASS",
        "cells": cells,
        "training": train,
        "offenders": offenders,
    }


def cg_a2_1(records: dict) -> dict:
    """A2.1's guard: with every DONE-on-entry tactic masked out, a learned cell
    can take no `(search policy)` env step. A non-zero count REFUSES the
    summary: the zero-length-span invariant the horizon argument rests on would
    not hold."""
    cells = {
        n: int(records[n].get("search_policy_steps", 0))
        for n in CELL_ORDER
        if n in records and CELLS[n]["kind"] == "learned"
    }
    offenders = sorted(n for n, v in cells.items() if v)
    return {
        "gate": "A2.1 — every decision spends >= 1 tactic step",
        "verdict": "REFUSE" if offenders else "PASS",
        "quantity": f"{SEARCH_KEY} env steps per learned cell",
        "cells": cells,
        "offenders": offenders,
        "scope": (
            "learned cells only (reading 5): the comparator rows reach the search "
            "policy whenever their law is DONE on entry, as S23's LEAD-1.0 did"
        ),
    }


def cg_t2(records: dict) -> dict:
    """CG-T2 (section 5): per row, the mean switches per episode on the gate.
    `>= 2` or the row is labelled `LAW-LIKE` in every table — reported, not
    gated, and the label travels with the number (reading 4: the mean pools the
    row's three opponents; A1.2: spans of >= 1 env step only)."""
    per: dict = {}
    for row in ROWS:
        got = [records[cell_name(row, o)] for o in OPPONENTS if cell_name(row, o) in records]
        pooled = [s for r in got for s in r["n_switches"]]
        mean = float(statistics.fmean(pooled)) if pooled else None
        per[row] = {
            "n_cells": len(got),
            "n_episodes": len(pooled),
            "mean_switches": mean,
            "per_opponent": {
                o: (
                    None
                    if cell_name(row, o) not in records
                    else records[cell_name(row, o)]["aggregate"]["mean_switches"]
                )
                for o in OPPONENTS
            },
            "label": None
            if mean is None
            else (LAW_LIKE if mean < CG_T2_MIN_SWITCHES else "SWITCHES"),
            "kind": ROWS[row]["kind"],
        }
    return {
        "gate": "CG-T2 (it switches), prereg section 5",
        "verdict": "REPORTED",
        "min_mean_switches": CG_T2_MIN_SWITCHES,
        "law_like_rows": sorted(r for r, b in per.items() if b["label"] == LAW_LIKE),
        "rows": per,
        "note": (
            "not gated: a selector that never switches is a law and is REPORTED "
            f"as one. {BASELINE_ROW} is {LAW_LIKE} by construction."
        ),
    }


def row_label(cg: dict, row: str) -> str:
    """The row's name as every table must print it (section 5)."""
    block = cg.get("rows", {}).get(row) or {}
    return f"{row} [{LAW_LIKE}]" if block.get("label") == LAW_LIKE else row


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def direction(v: dict) -> str:
    """Section 4's refutation paragraph, in the order of precedence the round's
    logic imposes: H-SAFE (the filter's guarantee is the precondition of every
    other number), then the construct gates that REFUSE (CG-T1, A2.1), then
    completeness, then H-TAC-1, H-TAC-2 and H-TAC-3."""
    safe, t1, t2, t3 = v["h_safe"], v["h_tac_1"], v["h_tac_2"], v["h_tac_3"]
    gate1, a21, gate2 = v["cg_t1"], v["cg_a2_1"], v["cg_t2"]
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
            "The mask is what makes an inapplicable pick impossible (prereg "
            "section 5), so the action space measured is not the one registered "
            "and no capture rate here is read."
        )
    if a21["verdict"] == "REFUSE":
        return (
            f"A2.1 REFUSES: {SEARCH_KEY} env steps in "
            f"{', '.join(a21['offenders'])}. Every decision must spend at least "
            "one tactic step; a zero-length span means the selector was "
            "consulted on a state it could not act from (D128's horizon failure) "
            "and no verdict here is about choosing."
        )
    if "INCOMPLETE" in (t1["verdict"], t2["verdict"], t3["verdict"], safe["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record. Every row x opponent must run on the one stream before a "
            "verdict is read (prereg section 6: a crashed cell is UNDETERMINED, "
            "never re-run after a number is seen)."
        )
    parts = []
    for s in SEEDS:
        b = t1["seeds"][str(s)]
        dirs = ", ".join(
            f"{o} {x['direction']} ({x['delta']:+.3f}, p = {x['p']:.3g})"
            + (" [in-distribution]" if x["in_distribution"] else "")
            for o, x in b["opponents"].items()
        )
        parts.append(
            f"SEL-{s}: H-TAC-1 {b['verdict']} "
            f"({b['n_pass']}/{b['n_opponents']} opponents vs {BASELINE_ROW}; {dirs})."
        )
    if t1["verdict"] != "PASS":
        parts.append(
            f"H-TAC-1 FAILS ({t1['n_pass']}/{len(SEEDS)} seeds): the discrete "
            "semi-MDP selector does not beat the best fixed law under the shield. "
            "Reported, and S24.2 IS NOT LICENSED (section 4's refutation)."
        )
    else:
        parts.append(
            f"H-TAC-1 PASSES on seeds {', '.join(t1['passing_seeds'])}: choosing "
            "among the five tactics captures more under the shield than the best "
            "fixed law, so the selector was a binding constraint."
        )
    for s in SEEDS:
        b = t2["seeds"][str(s)]
        a = b["across_opponents"]
        rules = ", ".join(
            f"{o} {x['verdict']} (p = {_fmt(x['p']) if x['p'] is not None else 'n/a'})"
            for o, x in b["vs_rules_v0"].items()
        )
        parts.append(
            f"SEL-{s}: H-TAC-2 {b['verdict']} (across opponents {a['verdict']}, "
            f"p = {_fmt(a['p']) if a['p'] is not None else 'n/a'}; vs {RULES_ROW} "
            f"{rules})."
        )
    if t1["verdict"] == "PASS" and t2["verdict"] != "PASS":
        parts.append(
            "H-TAC-1 passes but H-TAC-2 FAILS: THE GAIN IS A SINGLE-TACTIC "
            f"DEFAULT (probably {BASELINE_ROW.split('-')[0]}), reported as such — "
            "the selector is not the contribution (section 4's refutation). The "
            "pick histograms above are the evidence."
        )
    for s in SEEDS:
        b = t3["seeds"][str(s)]
        x = b["opponents"]["TACTIC-V0"]
        parts.append(
            f"SEL-{s}: H-TAC-3 {b['verdict']} vs NOMEM-{s} on TACTIC-V0 "
            f"({x.get('direction', '?')}, {x.get('delta', float('nan')):+.3f}, "
            f"p = {x.get('p', float('nan')):.3g})."
        )
    if t3["verdict"] != "PASS":
        parts.append(
            f"H-TAC-3 FAILS ({t3['n_pass']}/{len(SEEDS)} seeds): memory is INERT "
            "at this ratio — as an INPUT to the selector (A2.3; the system's dead "
            "reckoning still drives SWEEP in both arms). The speed sweep is still "
            "run, because memory may matter where the evader is slower, but with "
            "that stated (section 4's refutation)."
        )
    if gate2["law_like_rows"]:
        parts.append(
            "CG-T2: mean switches per gate episode below "
            f"{CG_T2_MIN_SWITCHES:g} on {', '.join(gate2['law_like_rows'])} — those "
            f"rows are labelled {LAW_LIKE} in every table (reported, not gated; "
            f"{BASELINE_ROW} is law-like by construction)."
        )
    parts.append(
        "Section 2's pre-registered EXPECTATION (not a bar), attached to every "
        "pick histogram: AMBUSH appears in no v0b top-3, and TACTIC-V0's hop "
        "alternation is where a selector that really chooses should show it — "
        "SEL's AMBUSH share vs TACTIC-V0 is "
        + ", ".join(
            f"SEL-{s} {_fmt((_rec(v['records_picks'], f'SEL-{s}', 'TACTIC-V0') or {}).get('AMBUSH'))}"
            for s in SEEDS
        )
        + " of its decisions."
    )
    parts.append(
        "Scope notes: TACTIC-V0 is the TRAINING opponent and every number "
        "against it is in-distribution (section 4); H-TAC-1b vs RULES-V0 is "
        "REPORTED and gates nothing; the per-decision picks histogram (A2.2), "
        "not usage_steps, is what H-TAC-2 reads."
    )
    return " ".join(parts)


def verdicts(records: dict, probes: dict | None = None) -> dict:
    """Every verdict of the round — a pure function of the records and the probe
    rows, so a test can state a round rather than run one."""
    missing = _missing(records)
    cg2 = cg_t2(records)
    return {
        "h_tac_1": h_tac_1_round(records),
        "h_tac_1b": h_tac_1b_round(records),
        "h_tac_2": h_tac_2_round(records),
        "h_tac_3": h_tac_3_round(records),
        "h_safe": h_safe(records, probes),
        "cg_t1": cg_t1(records, probes),
        "cg_a2_1": cg_a2_1(records),
        "cg_t2": cg2,
        "labels": {r: row_label(cg2, r) for r in ROWS},
        # The pick SHARES only, for `direction`'s AMBUSH sentence: the round
        # blob never carries a second copy of the 500-long arrays.
        "records_picks": {
            n: records[n]["aggregate"]["picks_share"] for n in records
        },
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
_BULK_KEYS = (
    "capture_ind",
    "steps",
    "violations",
    "n_switches",
    "n_zero_length_spans",
    "n_decisions",
    "inapplicable_picks_per_episode",
    "tactics",
    "branch",
)


def write_figure(records: dict, path: pathlib.Path, cg2: dict | None = None) -> None:
    """Two panels: the capture rate per row, and the per-decision pick shares of
    the SEL rows per opponent (H-TAC-2's quantity). Rows are NEVER parsed out of
    a cell name, so the bars are built from `CELLS`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cg2 = cg_t2(records) if cg2 is None else cg2
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(len(OPPONENTS))
    width = 0.8 / len(ROWS)
    for i, row in enumerate(ROWS):
        vals = [
            np.nan if (r := _rec(records, row, o)) is None else r["capture_rate"]
            for o in OPPONENTS
        ]
        axes[0].bar(
            xs + i * width - 0.4 + width / 2, vals, width, label=row_label(cg2, row)
        )
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-TAC-1: shielded capture per row", fontsize=9)
    width2 = 0.8 / (len(SEL_ROWS) * len(ACTIONS))
    for i, row in enumerate(SEL_ROWS):
        for j, tactic in enumerate(ACTIONS):
            vals = [
                np.nan
                if (r := _rec(records, row, o)) is None
                else r["aggregate"]["picks_share"][tactic]
                for o in OPPONENTS
            ]
            axes[1].bar(
                xs + (len(ACTIONS) * i + j) * width2 - 0.4 + width2 / 2,
                vals,
                width2,
                label=f"{row} {tactic}",
            )
    axes[1].set_ylabel("share of DECISIONS (A2.2)", fontsize=8)
    axes[1].set_title("H-TAC-2: the learned pick profile per opponent", fontsize=9)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(list(OPPONENTS), fontsize=8)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"S24.1 — learned tactic selector (RecoveryShieldV2, hold {HOLD}, ratio "
        f"1.2; N = {N_EPISODES} seed-paired episodes from {GATE_SEED0}). "
        "TACTIC-V0 is the training opponent (in-distribution)."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-T3).

    CG-T1 and A2.1 REFUSE here (section 5: before any capture rate is read)."""
    records = load_records()
    _measurable, rejected, _by_cell = scan_records()
    probes = read_probes()
    v = verdicts(records, probes)
    for gate in (v["cg_t1"], v["cg_a2_1"]):
        if gate["verdict"] == "REFUSE":
            raise SystemExit(
                f"{gate['gate']} REFUSES the summary: {gate['offenders']}. "
                "Section 5 puts the construct gates before any capture rate, so "
                "no table is written while this stands."
            )
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "rejected_records": rejected,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in records.items()
        },
        "provenance": provenance(device),
    }
    _write_json(round_file("json"), blob)
    write_figure(records, round_file("png"), v["cg_t2"])
    return blob


# ----------------------------------------------------------------------------
# CG-T3 (prereg section 5): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
# The seeds are a RULE, not a pick: the FIRST THREE gate seeds, so no scan can
# quietly choose which episodes are watched. 5 rows x 2 opponents x 3 seeds = 30
# GIFs. No caption carries a measured rate (S19 A5.1): the frame label is
# `EVADER: ... | PURSUER: ...` with the belief marker, which is the v0b game's
# OWN recorder — reused, not reimplemented, which is also how A2's ruling is
# honoured (the label is read off the law's CURRENT tactic inside the step the
# frame shows, never by indexing frames with a span's law-call boundaries).
GIF_ROWS: tuple[str, ...] = (*SEL_ROWS, "NOMEM-80", RULES_ROW)
GIF_OPPONENTS: tuple[str, ...] = ("TACTIC-V0", "GREEDY")
GIF_N = 3
GIF_EVERY = 4  # keep every 4th env step as a frame, as S14/S18/S19/S23


def gif_seeds() -> tuple[int, ...]:
    return tuple(GATE_SEED0 + i for i in range(GIF_N))


def gif_title(row: str, opponent: str, seed: int) -> str:
    """The file's own label: the row, the opponent and the episode SEED, and
    nothing else (A5.1). The FRAME caption is the game's `EVADER: ... |
    PURSUER: ...`."""
    return f"{row} vs {opponent} · episode {seed}"


def render_cell(
    row: str, opponent: str, seed: int, device: str = "cpu", every: int = GIF_EVERY
) -> tuple[list, dict]:
    """One gate episode of this cell, as frames. Same selector, opponent, hold,
    shield and observation flags as `run_cell` — only the recording differs."""
    name = cell_name(row, opponent)
    cfg = cell_config()
    env = build_cell_env(
        name, cfg, device, recorder=True, label=gif_title(row, opponent, seed)
    )
    game = env.game if CELLS[name]["kind"] == "learned" else env
    game.every = int(every)
    log = (
        env.play_episode(load_policy(row, device).predict, seed)
        if CELLS[name]["kind"] == "learned"
        else scripted_log(env, cfg, seed)
    )
    game.finish(log["outcome"], log["steps"])
    return game.frames, log


def write_gifs(
    rows: list[str] | None = None,
    device: str = "cpu",
    every: int = GIF_EVERY,
    fps: float = 15.0,
) -> list[tuple[str, str, int]]:
    """CG-T3: the first three gate seeds per row per opponent — 30 GIFs.

    Every precondition is resolved before the first frame — each cell's
    measurable record and each learned row's checkpoint — because a pass that
    renders two rows and then refuses leaves a half-watched gate, and that is
    how a construct error survives (D125). Copied from S23's shape; its version
    closes over S23's frozen-seed artefact, which this round does not have."""
    from pe2.render import save_gif

    wanted = list(GIF_ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in GIF_ROWS]
    if unknown:
        raise SystemExit(
            f"unknown CG-T3 row(s): {unknown}; this gate watches {GIF_ROWS}"
        )
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
        if ROWS[row]["kind"] == "learned":
            require_ckpt(row)
    n_all = len(wanted) * len(GIF_OPPONENTS) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"the first {GIF_N} gate seeds {seeds} per row vs {GIF_OPPONENTS}. Watch "
        "for the pursuer choosing COUNTER-ORBIT/AMBUSH where a human would, for "
        "pins (there should be none under the recovery shield), and for whether "
        "NOMEM visibly behaves like a fixed law.",
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
# CLI  (S23's dispatch order, copied: cells, then the blocking GIF gate, then
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
    # CG-T3 is a BLOCKING construct gate: the episodes are watched before any
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
                        "passing_seeds": v[h]["passing_seeds"],
                        "gated": v[h]["gated"],
                    }
                    for h in ("h_tac_1", "h_tac_1b", "h_tac_2", "h_tac_3")
                },
                indent=2,
            )
        )
        print(json.dumps(v["h_safe"], indent=2))
        print(json.dumps(v["cg_t1"], indent=2))
        print(json.dumps(v["cg_a2_1"], indent=2))
        print(json.dumps({r: b["label"] for r, b in v["cg_t2"]["rows"].items()}, indent=2))
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
