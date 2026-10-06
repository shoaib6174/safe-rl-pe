"""S19 — the planner intercept: is the shield's cost a property of the nominal
controller?

Spec: `docs/pathway/planner_intercept_prereg_2026-10-06.md` INCLUDING amendments
A1 and A2. Where this module and the prereg+amendments could be read to
disagree, the prereg+amendments win; every such reading is named in a comment.

Evaluation only. Nothing here trains, selects or tunes anything.

Twelve cells, 2 x 2 x 3, N = 500 paired episodes each on one stream:

    rows     LINE  (`pe2.structured.InterceptLaw`, the law every "cost of the
                    shield" this project has reported was measured on)
             PLAN  (`pe2.planner.PlannerInterceptLaw`, bit-identical to LINE
                    whenever the straight line is clear)
    columns  OPEN  (no shield)
             SHIELD(`pe2.shield.CBFShield(cfg)` at its defaults, as S18)
    columns  GREEDY / HUG / ATK (frozen structured attacker ATK4-45)

The ONLY differences between the rows are the law object and between the
columns the shield: hold, `shield_obs`, `escape_obs`, config, search policy,
opponent and seed stream are identical everywhere, so a capture difference is
attributable to the law or the filter and to nothing else. The cell table is
asserted on the BUILT env in `tests/pe2/test_derisk_s19_round.py`, not on this
docstring.

Three things this module is careful about, each because the prereg's amendments
say the obvious version was wrong:

- **Intervention scope is the all-step scope (A1.2).** S18's
  `intervention.total` began at the first learner decision; S19 reads the
  recorder's per-episode `intervention_total`, which covers EVERY env step of
  the episode including the law steps `play_episode` plays before the first
  decision. Any comparison to an S18 number must say which scope it uses.
- **H-PLAN-3 is a tail statistic (A1.1).** The median longest law pin is 0 on
  the reference cell, so "median falls by >= 50 %" would read 0 against 0. The
  gated quantity is the FRACTION of episodes carrying a law-branch pinned run
  >= `DEADLOCK_STEPS`; the median and the 90th percentile are reported beside
  it and gate nothing.
- **The NULL is the obstacle-free bit-identity alone (A2.2).** The OPEN column
  is UNSHIELDED, not obstacle-free: obstacles are there, and a law that plans
  around them legitimately differs from one that drives into them even with no
  filter. The OPEN-column PLAN - LINE difference is REPORTED, and H-PLAN-2 has
  no OPEN-band conjunct.

Usage:
  ./venv/bin/python scripts/derisk_s19_planner.py --cells          (all missing)
  ./venv/bin/python scripts/derisk_s19_planner.py --cells PLAN-SHIELD_HUG
  ./venv/bin/python scripts/derisk_s19_planner.py --gifs           (CG-C)
  ./venv/bin/python scripts/derisk_s19_planner.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 7, the stop rule). There is no `--workers`: the
cells are independent and already addressable by literal name, so the plan's
section 7 discipline (literal arm lists, `OMP_NUM_THREADS` <= 4) is served by
launching several processes with disjoint `--cells` lists rather than by a pool
inside this module.
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 7; set BEFORE numpy / torch build their pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import dataclasses
import hashlib
import json
import pathlib
import statistics
from collections import Counter

import numpy as np

import pe2
from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.eval import wilson_ci
from pe2.metrics import DEADLOCK_STEPS
from pe2.planner import PlannerInterceptLaw
from pe2.recorder import BranchTaggedSearchMDP
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD, OrbitSearch

SPEC = "docs/pathway/planner_intercept_prereg_2026-10-06.md"
ARTEFACT_PREFIX = "s19"
OUT = pathlib.Path("results/derisk")
CKPT_ROOT = pathlib.Path("results/derisk")  # read-only input

# --- pre-registered (prereg sections 2-5 + A1, A2). Do not tune. ---
GATE_SEED0 = 2_000_000  # section 3, the gate stream
CONSTRUCT_SEED0 = 1_990_000  # section 3, the construct stream
N_EPISODES = 500  # section 3
P_GATE = 0.05  # section 4
PASS_MIN = 2  # ">= 2/3 opponents", every hypothesis
# The S18 shield configuration (section 3): hold 10, `shield_obs`, `escape_obs`.
SHIELD_OBS = True
ESCAPE_OBS = True
# A1.1's bar is the recorder's own pin threshold, imported rather than written
# as a literal so "a law pin >= 40 steps" cannot drift from `ShieldMetrics`'
# deadlock length.
LAW_PIN_STEPS = DEADLOCK_STEPS  # 40
H3_RELATIVE_DROP = 0.50  # A1.1: ">= 50 % relative"
# A2.2's null: 50 obstacle-free episodes from the construct stream.
NULL_N = 50
# `pe2.evade.StructuredEvaderPolicy`'s hold, as in S6/S7/S9/S14/S17.
EVADER_HOLD = 10
ATK_CKPT = CKPT_ROOT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"

ROWS: dict[str, dict] = {
    # `law` is the class NAME the record carries; `row_law` builds the object.
    # `None` means "the `SearchMDP` default", i.e. a fresh `InterceptLaw`.
    "LINE": {"law": None},
    "PLAN": {"law": "PlannerInterceptLaw"},
}
COLS: dict[str, dict] = {"OPEN": {"shielded": False}, "SHIELD": {"shielded": True}}
OPPONENTS: dict[str, dict] = {
    "GREEDY": {"scripted": GreedyEvader},
    "HUG": {"scripted": ObstacleHuggingEvader},
    "ATK": {"ckpt": ATK_CKPT},  # the frozen structured attacker ATK4-45
}
PANELS: tuple[str, ...] = tuple(f"{r}-{c}" for r in ROWS for c in COLS)

_MODELS: dict = {}


def cell_name(row: str, col: str, opponent: str) -> str:
    return f"{row}-{col}_{opponent}"


def _build_cells() -> dict[str, dict]:
    """The 12 cells. The cell carries its row, column and opponent STORED —
    nothing parses the name back apart (S17's convention, after `RECF-BASE-60`
    broke the parsing route)."""
    return {
        cell_name(r, c, o): {
            "name": cell_name(r, c, o),
            "row": r,
            "col": c,
            "panel": f"{r}-{c}",
            "opponent": o,
            "law": ROWS[r]["law"],
            "shielded": COLS[c]["shielded"],
            "hold": HOLD,
        }
        for r in ROWS
        for c in COLS
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


def gif_path(panel: str, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{panel}_{seed}.gif"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def spec_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / SPEC


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson": [lo, hi]}


def provenance() -> dict:
    """Commit, the `pe2` package this ran against, and the sha256 of the
    pre-registration. The spec hash is the point: a cell record proves which
    text's thresholds it was measured under, so an amendment appended after a
    number existed is detectable rather than arguable."""
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
        "pe2_file": pe2.__file__,
        "spec": SPEC,
        "spec_sha256": _sha256(spec_path()),
        "sb3_version": stable_baselines3.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    _write_json(path, record)
    return path


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


def read_cell(name: str) -> dict:
    """The cell's record, or SystemExit. A CG-C caption quotes that cell's
    MEASURED capture and law-pin fraction and is never fabricated."""
    path = cell_file(name)
    if not path.exists():
        raise SystemExit(
            f"no cell record for {name} at {path}: run --cells {name} first. A "
            "CG-C caption carries that cell's MEASURED capture rate and law-pin "
            "fraction and is never fabricated (prereg section 5)."
        )
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------
# the cell's env — the only differences are the law and the shield
# ----------------------------------------------------------------------------
def cell_config() -> EnvConfig:
    """`EnvConfig()` defaults in every cell. Nothing sweeps in this round."""
    return EnvConfig()


def _model(path, device: str):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


def require_atk_ckpt() -> pathlib.Path:
    """The ATK column REFUSES to run without its frozen checkpoint. Not
    defensive coding: `ATK4-45` is a specific frozen attacker (D120/D124) and a
    cell quietly run against some other evader would be a different experiment
    wearing this one's name."""
    path = pathlib.Path(ATK_CKPT)
    if not path.exists():
        raise SystemExit(
            f"missing the ATK opponent checkpoint at {path}. The ATK column is "
            "the frozen structured attacker ATK4-45 (prereg section 3); there is "
            "no substitute and no default, so the round refuses to run."
        )
    return path


def row_law(row: str, cfg: EnvConfig):
    """The row's nominal controller, or None for the `SearchMDP` default.

    LINE passes `law=None` and gets `InterceptLaw(cfg)` built by `SearchMDP`
    itself — the bit-identical default path (prereg section 2 step 3), so the
    LINE row is the law this project has always measured and not a re-wiring of
    it."""
    if ROWS[row]["law"] is None:
        return None
    return PlannerInterceptLaw(cfg)


def opponent_factory(name: str, cfg: EnvConfig, device: str):
    """The column's evader as an opponent factory for
    `SingleAgentWrapper(role="pursuer")`.

    ATK occupies the same seat as every learned attacker in this project:
    `StructuredEvaderPolicy` at hold 10 with the hug law inside `D_POS`
    (`derisk_s17_fixes_b.gate_opponent_factory`), so nothing but the model
    differs from S17's gate."""
    spec = OPPONENTS[name]
    if "scripted" in spec:
        return spec["scripted"]
    from pe2.evade import D_POS, StructuredEvaderPolicy

    model = _model(require_atk_ckpt(), device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=EVADER_HOLD, d_pos=D_POS)


def build_shield(cfg: EnvConfig):
    """`CBFShield(cfg)` at its defaults, exactly as S18 (prereg section 3).

    The defaults are ASSERTED rather than restated as keyword arguments: the
    prereg names "margin 0.0, gamma 0.2" and the round must fail loudly if the
    library's defaults ever move, instead of silently pinning values the spec
    did not fix here."""
    from pe2.shield import CBFShield

    shield = CBFShield(cfg)
    if (shield.margin, shield.gamma) != (0.0, 0.2):
        raise SystemExit(
            f"CBFShield defaults moved to margin {shield.margin}, gamma "
            f"{shield.gamma}; the prereg (section 3) fixes the S18 configuration "
            "at margin 0.0, gamma 0.2 and this round may not run under another."
        )
    return shield


def build_cell_env(
    name: str, cfg: EnvConfig, device: str = "cpu"
) -> BranchTaggedSearchMDP:
    """The cell's env: the branch-tagged recorder (prereg section 3, gated by
    CG-B), with this cell's law and this cell's shield and NOTHING else varied.

    `cfg` is a parameter rather than `cell_config()` so the NULL can build the
    SAME construction on an obstacle-free arena; that is what makes A2.2's
    bit-identity a statement about the round's instrument rather than about a
    look-alike.
    """
    spec = CELLS[name]
    return BranchTaggedSearchMDP(
        cfg,
        opponent_factory(spec["opponent"], cfg, device),
        hold=spec["hold"],
        shield=build_shield(cfg) if spec["shielded"] else None,
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,  # evaluation only; `play_episode` discards rewards
        law=row_law(spec["row"], cfg),
    )


# ----------------------------------------------------------------------------
# one cell
# ----------------------------------------------------------------------------
def _q(xs: list[float], q: float) -> float:
    """The q-quantile, reported only. `statistics.quantiles` needs n >= 2."""
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return float(xs[0])
    return float(
        statistics.quantiles(xs, n=100, method="inclusive")[round(q * 100) - 1]
    )


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """One cell: N paired episodes of this row's law, under this column's
    shield, against this opponent, on the gate stream.

    The episodes are driven here rather than through `evaluate_structured` for
    two reasons: the record needs the recorder's per-episode `branch` block and
    `episode_violations` (which `play_episode` only puts in the log when a
    shield is present, so the OPEN column would otherwise have no contact
    count at all), and the shield is rebuilt per episode exactly as S18's gate
    rebuilt it. `CBFShield` is memoryless so the rebuild is equivalent to one
    shield per call, but the equivalence is a property of the shield rather
    than of the protocol.
    """
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    cfg = cell_config()
    mdp = build_cell_env(name, cfg, device)
    policy = OrbitSearch(cfg)  # the teacher's search, in every cell

    capture_ind: list[bool] = []
    violations: list[int] = []
    intervention_total: list[float] = []
    longest_law_run: list[int] = []
    branch: list[dict] = []
    outcomes: Counter = Counter()
    for i in range(n):
        if spec["shielded"]:
            mdp.shield = build_shield(cfg)
        log = mdp.play_episode(policy.predict, GATE_SEED0 + i)
        b = log["branch"]
        capture_ind.append(log["outcome"] == "capture")
        outcomes[log["outcome"]] += 1
        # Authoritative contact count: `SearchMDP._step_inner` increments it on
        # every step from the env's own contact predicates, shield or not.
        violations.append(int(mdp.episode_violations))
        intervention_total.append(float(b["intervention_total"]))  # A1.2 scope
        longest_law_run.append(int(b["longest_pinned_run_by_branch"]["law"]))
        branch.append(b)

    law_pin = [r >= LAW_PIN_STEPS for r in longest_law_run]
    k = sum(capture_ind)
    return {
        "name": name,
        "row": spec["row"],
        "col": spec["col"],
        "panel": spec["panel"],
        "opponent": spec["opponent"],
        "law": spec["law"] or "InterceptLaw",
        "shielded": spec["shielded"],
        "hold": spec["hold"],
        "shield_obs": SHIELD_OBS,
        "escape_obs": ESCAPE_OBS,
        "n_episodes": n,
        "seed0": GATE_SEED0,
        "capture_ind": capture_ind,
        "capture_rate": k / n if n else float("nan"),
        "wilson_capture": _ci(k, n),
        "violations": int(sum(violations)),
        "intervention_total": intervention_total,
        "longest_law_run": longest_law_run,
        "law_pin_ge_40": law_pin,
        # A2.1's residual: the planner calls that still fell back to direct
        # steering. `InterceptLaw` has no such counter, hence None on LINE.
        "n_unreachable": getattr(mdp.law, "n_unreachable", None),
        "branch": branch,
        "aggregate": {
            "capture_rate": k / n if n else float("nan"),
            "outcomes": dict(outcomes),
            # The env terminates on capture and truncates on timeout and nothing
            # else (pe2/env.py OUTCOME_*), so the evader escapes exactly when the
            # episode times out.
            "escape_rate": outcomes["timeout"] / n if n else float("nan"),
            # A1.1's gated statistic.
            "episode_frac_law_pin_ge_40": (sum(law_pin) / n) if n else float("nan"),
            "law_pin_steps": LAW_PIN_STEPS,
            # Reported beside it, gated by nothing (A1.1).
            "longest_law_run_median": float(statistics.median(longest_law_run))
            if longest_law_run
            else float("nan"),
            "longest_law_run_p90": _q(longest_law_run, 0.90),
            "longest_law_run_max": max(longest_law_run, default=0),
            "intervention_total_mean": float(np.mean(intervention_total))
            if intervention_total
            else float("nan"),
            "intervention_scope": "all steps (amendment A1.2)",
        },
        "provenance": provenance(),
    }


def _cell_line(rec: dict) -> str:
    a = rec["aggregate"]
    return (
        f"[{rec['name']}] capture {rec['capture_rate']:.3f} "
        f"law_pin>={LAW_PIN_STEPS} {a['episode_frac_law_pin_ge_40']:.3f} "
        f"int {a['intervention_total_mean']:.2f} "
        f"unreach {rec['n_unreachable']} viol {rec['violations']}"
    )


def run_cells(
    names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None
) -> list[str]:
    """Every named cell that has no record yet (default: all 12).

    Literal names only — there is no pattern, no "all shielded" shorthand and no
    way to ask for "the cells that failed", which is the plan's section 7
    discipline. The ATK checkpoint is resolved BEFORE the first episode so a
    pass cannot run three cells and then refuse on the fourth."""
    wanted = list(CELL_ORDER) if names is None else list(names)
    unknown = [n for n in wanted if n not in CELLS]
    if unknown:
        raise SystemExit(
            f"unknown cell(s): {unknown}; this round's cells are {CELL_ORDER}"
        )
    cells_dir().mkdir(parents=True, exist_ok=True)
    todo = [n for n in wanted if not cell_file(n).exists()]
    if any(CELLS[n]["opponent"] == "ATK" for n in todo):
        require_atk_ckpt()
    print(
        f"{len(todo)}/{len(wanted)} cells to run "
        f"({len(wanted) - len(todo)} already have a record; stop rule: never "
        f"re-run), N = {N_EPISODES if n_episodes is None else n_episodes} from "
        f"seed {GATE_SEED0}",
        flush=True,
    )
    for name in todo:
        rec = run_cell(name, device, n_episodes)
        write_cell(rec)
        print(_cell_line(rec), flush=True)
    return todo


# ----------------------------------------------------------------------------
# the NULL (amendment A2.2): obstacle-free bit-identity
# ----------------------------------------------------------------------------
def null_config() -> EnvConfig:
    """The obstacle-free arena. `n_obstacles` still owns the observation layout
    (`pe2/obs.py` reserves 2 slots per obstacle), so only the PLACED obstacles
    go away and the law sees an input of the round's width."""
    return dataclasses.replace(cell_config(), n_physical_obstacles=0)


def null_open(
    n: int | None = None, seed0: int | None = None, device: str = "cpu"
) -> dict:
    """A2.2's NULL: PLAN and LINE produce bit-identical `play_episode` logs on
    an OBSTACLE-FREE arena over `n` seeds of the construct stream.

    "OPEN" here means obstacle-free, NOT the OPEN column. A2.2 corrected the
    prereg on exactly this point: the OPEN column is unshielded with obstacles
    present, and a law that plans around them legitimately differs from one
    that drives into them even with no filter, so a capture band in that column
    would fail for the right reason. The OPEN-column PLAN - LINE difference is a
    REPORTED quantity (`h_plan_2`), never a null.

    The prereg's section 4 also said "the open-arena scenarios of
    `pe2/scenarios.py`". Those are NOT used, and could not be: `close_open`,
    `mid_open` and `far_open` are canonical start geometries on the canonical
    TWO-OBSTACLE layout — "open" there means the start has clear line of sight,
    not that the arena is empty. A2.2's restatement ("the obstacle-free
    bit-identity alone") is the binding text and is what runs here.

    This duplicates `tests/pe2/test_planner.py::TestInvariant` on purpose: the
    round record must carry its own null, not merely cite a test suite. It runs
    on the round's OWN instrument (the branch-tagged recorder at the cells'
    hold and observation flags, shield off) so the comparison covers the branch
    attribution too, and not only the outcome.
    """
    n = NULL_N if n is None else int(n)
    seed0 = CONSTRUCT_SEED0 if seed0 is None else int(seed0)
    cfg = null_config()
    # GREEDY's cells: the opponent must be identical on both sides and is not a
    # free choice — it is the primary metric's opponent (prereg section 5).
    plan = build_cell_env("PLAN-OPEN_GREEDY", cfg, device)
    line = build_cell_env("LINE-OPEN_GREEDY", cfg, device)
    policy = OrbitSearch(cfg)
    differing: list[int] = []
    for i in range(n):
        seed = seed0 + i
        if plan.play_episode(policy.predict, seed) != line.play_episode(
            policy.predict, seed
        ):
            differing.append(seed)
    return {
        "hypothesis": "NULL (amendment A2.2)",
        "verdict": "HOLDS" if not differing else "FAILS",
        "scenario": (
            "obstacle-free arena, EnvConfig(n_physical_obstacles=0); NOT the "
            "OPEN column and NOT pe2/scenarios.py's *_open scenarios (those "
            "sit on the canonical two-obstacle layout)"
        ),
        "n_episodes": n,
        "seed0": seed0,
        "stream": "construct",
        "pre_registered": bool(n == NULL_N and seed0 == CONSTRUCT_SEED0),
        "n_differing": len(differing),
        "differing_seeds": differing[:10],
        "n_unreachable": getattr(plan.law, "n_unreachable", None),
        "comparison": "== on the full play_episode log, including log['branch']",
    }


# A1/A2 renamed this check's subject twice; `null_identity` is the prereg
# section 4 name and `null_open` the name the round's artefact uses. One
# function, two names, so neither reader is wrong.
null_identity = null_open


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4, as amended by A1 and A2)
# ----------------------------------------------------------------------------
def _rec(records: dict, row: str, col: str, opponent: str) -> dict | None:
    return records.get(cell_name(row, col, opponent))


def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _direction_label(delta: float) -> str:
    if delta > 0:
        return "PLAN > LINE"
    if delta < 0:
        return "PLAN < LINE"
    return "PLAN == LINE"


def h_plan_1(records: dict) -> dict:
    """H-PLAN-1: SHIELD capture is higher under PLAN than LINE, on >= 2/3
    opponents, `paired_indicator_test` p < P_GATE.

    The per-opponent DIRECTION is reported whether or not the test passes: a
    refutation that does not say which way it went is not a usable refutation
    (prereg section 4's refutation paragraph sends S20 back to CG-C's episodes,
    and the sign is the first thing it needs).
    """
    per: dict = {}
    for o in OPPONENTS:
        plan, line = (
            _rec(records, "PLAN", "SHIELD", o),
            _rec(records, "LINE", "SHIELD", o),
        )
        if plan is None or line is None:
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": [
                    n
                    for n, r in (
                        (cell_name("PLAN", "SHIELD", o), plan),
                        (cell_name("LINE", "SHIELD", o), line),
                    )
                    if r is None
                ],
            }
            continue
        test = {
            "test": "paired_indicator",
            **paired_indicator_test(plan["capture_ind"], line["capture_ind"], P_GATE),
        }
        delta = plan["capture_rate"] - line["capture_rate"]
        per[o] = {
            "verdict": "PASS" if test["pass"] else "FAIL",
            "capture_plan": plan["capture_rate"],
            "capture_line": line["capture_rate"],
            "delta": delta,
            "direction": _direction_label(delta),
            "p": test["p"],
            "test": test,
        }
    n_pass = sum(b["verdict"] == "PASS" for b in per.values())
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in per.values())
    return {
        "hypothesis": "H-PLAN-1",
        "verdict": "INCOMPLETE"
        if incomplete
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "n_opponents": len(OPPONENTS),
        "p_gate": P_GATE,
        "opponents": per,
    }


def h_plan_2(records: dict) -> dict:
    """H-PLAN-2 (the paper's number), as amended by A2.2: the shield's cost
    `OPEN capture - SHIELD capture` is SMALLER under PLAN than LINE on
    >= 2/3 opponents, AND the SHIELD gain is significant (H-PLAN-1's test).

    A2.2 DROPPED the third conjunct ("the OPEN difference within the null
    band"). The difference-in-differences already isolates the shield's
    marginal cost per law regardless of the OPEN level, and the OPEN column is
    unshielded rather than obstacle-free, so an OPEN band would have failed for
    the right reason. The OPEN-column PLAN - LINE difference is reported here as
    a quantity, with no bar attached to it.
    """
    gain = h_plan_1(records)
    per: dict = {}
    for o in OPPONENTS:
        cells = {
            (r, c): _rec(records, r, c, o)
            for r in ("LINE", "PLAN")
            for c in ("OPEN", "SHIELD")
        }
        if any(v is None for v in cells.values()):
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": [
                    cell_name(r, c, o) for (r, c), v in cells.items() if v is None
                ],
            }
            continue
        cost = {
            r: cells[(r, "OPEN")]["capture_rate"] - cells[(r, "SHIELD")]["capture_rate"]
            for r in ("LINE", "PLAN")
        }
        per[o] = {
            "verdict": "PASS" if cost["PLAN"] < cost["LINE"] else "FAIL",
            "cost_line": cost["LINE"],
            "cost_plan": cost["PLAN"],
            # Positive = the planner carries the SMALLER shield cost.
            "cost_reduction": cost["LINE"] - cost["PLAN"],
            "capture": {f"{r}-{c}": cells[(r, c)]["capture_rate"] for (r, c) in cells},
            # Reported, never gated (A2.2).
            "open_plan_minus_line": (
                cells[("PLAN", "OPEN")]["capture_rate"]
                - cells[("LINE", "OPEN")]["capture_rate"]
            ),
        }
    n_pass = sum(b["verdict"] == "PASS" for b in per.values())
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in per.values())
    cost_part = n_pass >= PASS_MIN
    gain_part = gain["verdict"] == "PASS"
    return {
        "hypothesis": "H-PLAN-2",
        "verdict": "INCOMPLETE"
        if incomplete or gain["verdict"] == "INCOMPLETE"
        else ("PASS" if (cost_part and gain_part) else "FAIL"),
        "cost_part": bool(cost_part),
        "gain_part": bool(gain_part),
        "gain_verdict": gain["verdict"],
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "n_opponents": len(OPPONENTS),
        "open_band_conjunct": "dropped by amendment A2.2",
        "opponents": per,
    }


def h_plan_3(records: dict) -> dict:
    """H-PLAN-3 as restated by A1.1: under the SHIELD, the FRACTION of episodes
    carrying a law-branch pinned run >= `LAW_PIN_STEPS` falls by >= 50 %
    relative, PLAN vs LINE, on >= 2/3 opponents.

    A1.1 replaced the median because the median had no power: the longest law
    pin is 0 on a typical episode of the reference cell (85/500 carry any law
    pin at all), so "median falls by >= 50 %" would read 0 against 0 — S17's
    uninformative H3 shape (D131) again. The median and the 90th percentile of
    the longest law run are reported HERE but gate nothing.

    The zero guard: a relative fall is undefined when LINE's fraction is 0, and
    "0 to 0" is not a 50 % reduction. Such an opponent is VACUOUS — never PASS,
    never FAIL. Overall, PASS needs >= PASS_MIN opponents that actually passed;
    when the remaining non-vacuous opponents cannot reach that bar the
    hypothesis is VACUOUS rather than FAIL, because the shortfall is an absent
    phenomenon and not evidence against the planner.
    """
    per: dict = {}
    for o in OPPONENTS:
        plan, line = (
            _rec(records, "PLAN", "SHIELD", o),
            _rec(records, "LINE", "SHIELD", o),
        )
        if plan is None or line is None:
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": [
                    n
                    for n, r in (
                        (cell_name("PLAN", "SHIELD", o), plan),
                        (cell_name("LINE", "SHIELD", o), line),
                    )
                    if r is None
                ],
            }
            continue
        f_line = line["aggregate"]["episode_frac_law_pin_ge_40"]
        f_plan = plan["aggregate"]["episode_frac_law_pin_ge_40"]
        block = {
            "frac_line": f_line,
            "frac_plan": f_plan,
            "law_pin_steps": LAW_PIN_STEPS,
            "required_relative_drop": H3_RELATIVE_DROP,
            # Reported, not gated (A1.1).
            "longest_law_run_median": {
                "LINE": line["aggregate"]["longest_law_run_median"],
                "PLAN": plan["aggregate"]["longest_law_run_median"],
            },
            "longest_law_run_p90": {
                "LINE": line["aggregate"]["longest_law_run_p90"],
                "PLAN": plan["aggregate"]["longest_law_run_p90"],
            },
        }
        if f_line == 0.0:
            per[o] = block | {
                "verdict": "VACUOUS",
                "relative_drop": None,
                "note": (
                    "LINE's law-pin fraction is 0: there is no phenomenon to "
                    "remove, so a relative fall is undefined and this opponent "
                    "can never PASS (A1.1's zero guard)"
                ),
            }
            continue
        drop = (f_line - f_plan) / f_line
        per[o] = block | {
            "verdict": "PASS" if drop >= H3_RELATIVE_DROP else "FAIL",
            "relative_drop": drop,
        }
    verdicts_ = [b["verdict"] for b in per.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    n_vacuous = sum(v == "VACUOUS" for v in verdicts_)
    if "INCOMPLETE" in verdicts_:
        overall = "INCOMPLETE"
    elif n_pass >= PASS_MIN:
        overall = "PASS"
    elif n_pass + n_vacuous >= PASS_MIN:
        # The bar is out of reach only because some opponents have nothing to
        # fix. Not a failure of the planner; not a pass either.
        overall = "VACUOUS"
    else:
        overall = "FAIL"
    return {
        "hypothesis": "H-PLAN-3",
        "verdict": overall,
        "statistic": "fraction of SHIELD episodes with a law pinned run >= "
        f"{LAW_PIN_STEPS} steps (amendment A1.1)",
        "n_pass": n_pass,
        "n_vacuous": n_vacuous,
        "pass_min": PASS_MIN,
        "n_opponents": len(OPPONENTS),
        "opponents": per,
    }


def h_safe(records: dict) -> dict:
    """H-SAFE: 0 pursuer contact steps in BOTH SHIELD cells, every opponent.

    The OPEN column is not gated and must not be: an unshielded pursuer is
    allowed to touch things, and that is the point of the column. Its contact
    counts are reported separately so H-SAFE cannot be read as covering them.
    """
    gated = [cell_name(r, "SHIELD", o) for r in ROWS for o in OPPONENTS]
    have = {n: records[n] for n in gated if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violations", 0)))
    steps = sum(int(r.get("violations", 0)) for r in have.values())
    missing = [n for n in gated if n not in records]
    unshielded = {
        cell_name(r, "OPEN", o): int(records[cell_name(r, "OPEN", o)]["violations"])
        for r in ROWS
        for o in OPPONENTS
        if cell_name(r, "OPEN", o) in records
    }
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": offenders,
        "gated_cells": gated,
        "missing": missing,
        "unshielded_contact_steps_reported_not_gated": unshielded,
    }


def direction(v: dict) -> str:
    """Prereg section 4's refutation paragraph, in the order of precedence the
    round's logic imposes: H-SAFE (the filter's guarantee is the precondition of
    every other number), then the NULL (A2.2: if it fails, PLAN is a different
    law in open space and nothing else here is about obstruction), then
    completeness, then H-PLAN-1, H-PLAN-2 and H-PLAN-3."""
    safe, null = v["h_safe"], v["null_open"]
    h1, h2, h3 = v["h_plan_1"], v["h_plan_2"], v["h_plan_3"]
    if safe["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {safe['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(safe['offenders'])}. Stop — the filter's guarantee is the "
            "precondition of the whole round and nothing else is read."
        )
    if null is not None and null["verdict"] == "FAILS":
        return (
            f"NULL FAILS ({null['n_differing']}/{null['n_episodes']} obstacle-free "
            f"episodes differ; first {null['differing_seeds']}): PLAN is a "
            "DIFFERENT law in open space, so no verdict in this round may be read "
            "as being about obstruction. Report and stop (prereg section 4, as "
            "restated by A2.2); fix the law's clear-line path before any capture "
            "number here means anything."
        )
    if "INCOMPLETE" in (h1["verdict"], h2["verdict"], h3["verdict"], safe["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record. Every row x column x opponent must run on the one stream "
            "before a verdict is read (prereg section 7: a crashed cell is "
            "UNDETERMINED, never re-run after a number is seen)."
        )
    if null is None:
        return (
            "the NULL has not been evaluated in this summary, so nothing below is "
            "yet licensed to be about obstruction (A2.2). Re-run --summarize."
        )
    parts = []
    dirs = ", ".join(
        f"{o} {b['direction']} ({b['delta']:+.3f}, p = {b['p']:.3g})"
        for o, b in h1["opponents"].items()
    )
    if h1["verdict"] == "PASS":
        parts.append(
            f"H-PLAN-1 PASS ({h1['n_pass']}/{h1['n_opponents']} opponents; {dirs}): "
            "under the shield the planning law captures more than the straight-line "
            "law, so the nominal controller WAS a binding constraint on the filtered "
            "pursuer and Hypothesis P's clause (i) holds."
        )
    else:
        parts.append(
            f"H-PLAN-1 FAILS ({h1['n_pass']}/{h1['n_opponents']} opponents; {dirs}): "
            "the law was NOT the binding constraint under the shield. Per prereg "
            "section 4, re-diagnose from CG-C's episodes (results/derisk/"
            f"{ARTEFACT_PREFIX}_gifs/) before building anything else — that is the "
            "plan's gate to S20, and no further upgrade is licensed by this round."
        )
    costs = ", ".join(
        f"{o} LINE {b['cost_line']:+.3f} vs PLAN {b['cost_plan']:+.3f}"
        for o, b in h2["opponents"].items()
    )
    opens = ", ".join(
        f"{o} {b['open_plan_minus_line']:+.3f}" for o, b in h2["opponents"].items()
    )
    if h2["verdict"] == "PASS":
        parts.append(
            f"H-PLAN-2 PASS ({h2['n_pass']}/{h2['n_opponents']} opponents carry a "
            f"smaller shield cost under PLAN, and the SHIELD gain is significant; "
            f"{costs}): the shield's task cost is substantially a property of the "
            "NOMINAL CONTROLLER, not of the filter — the paper's number."
        )
    else:
        why = []
        if not h2["cost_part"]:
            why.append(
                f"the difference-in-differences clears {h2['pass_min']}/"
                f"{h2['n_opponents']} on only {h2['n_pass']}"
            )
        if not h2["gain_part"]:
            why.append(
                f"the SHIELD gain is not significant (H-PLAN-1 {h2['gain_verdict']})"
            )
        parts.append(
            f"H-PLAN-2 FAILS ({'; '.join(why)}; {costs}): the shield's cost on this "
            "controller is not shown to be the controller's."
        )
    parts.append(
        "Reported, not gated (A2.2): the OPEN-column PLAN - LINE capture "
        f"difference is {opens}. The OPEN column is UNSHIELDED, not "
        "obstacle-free, so this difference is a legitimate property of the two "
        "laws and carries no bar."
    )
    frac = ", ".join(
        f"{o} {b.get('frac_line', float('nan')):.3f} -> {b.get('frac_plan', float('nan')):.3f}"
        + (
            ""
            if b.get("relative_drop") is None
            else f" (relative drop {b['relative_drop']:+.0%})"
        )
        for o, b in h3["opponents"].items()
    )
    if h3["verdict"] == "PASS":
        parts.append(
            f"H-PLAN-3 PASS ({h3['n_pass']}/{h3['n_opponents']}; law-pin fraction "
            f"{frac}): the law-pinned runs the shield used to produce are largely "
            "gone, which is Hypothesis P's clause (iii) and the mechanism behind "
            "the other two."
        )
    elif h3["verdict"] == "VACUOUS":
        parts.append(
            f"H-PLAN-3 VACUOUS ({h3['n_vacuous']}/{h3['n_opponents']} opponents have "
            f"a LINE law-pin fraction of 0; {frac}): there was no law pinning to "
            "remove on those opponents, so A1.1's statistic says nothing here and "
            "no mechanism claim may rest on it — exactly the uninformative shape "
            "A1.1 restated H-PLAN-3 to avoid, now arising from the other side."
        )
    else:
        parts.append(
            f"H-PLAN-3 FAILS ({h3['n_pass']}/{h3['n_opponents']}; law-pin fraction "
            f"{frac}): the planner did not remove the law's pinned runs, so any "
            "capture change above is not explained by the mechanism Hypothesis P "
            "proposes. Watch CG-C for the pins the planner did not remove."
        )
    parts.append(
        "Scope note (A1.2): every intervention number in this round is the "
        "ALL-STEP scope (the recorder's `intervention_total`), not S18's "
        "first-decision-onward `intervention.total`; the two are not "
        "interchangeable."
    )
    return " ".join(parts)


def verdicts(records: dict, null: dict | None = None) -> dict:
    """Every verdict of the round. A pure function of the records and the NULL
    block, so a test can state a round rather than run one; `summarize` is what
    actually evaluates the NULL."""
    missing = _missing(records)
    return {
        "h_plan_1": h_plan_1(records),
        "h_plan_2": h_plan_2(records),
        "h_plan_3": h_plan_3(records),
        "h_safe": h_safe(records),
        "null_open": null,
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "rows": list(ROWS),
        "cols": list(COLS),
        "opponents": list(OPPONENTS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": SPEC,
    }


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
def _bars(ax, records: dict, key, title: str, ylabel: str, panels) -> None:
    xs = np.arange(len(OPPONENTS))
    width = 0.8 / len(panels)
    for i, panel in enumerate(panels):
        vals = []
        for o in OPPONENTS:
            row, col = panel.split("-")
            rec = _rec(records, row, col, o)
            vals.append(np.nan if rec is None else key(rec))
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=panel)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(OPPONENTS), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def write_figure(records: dict, path: pathlib.Path, null: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _bars(
        axes[0],
        records,
        lambda r: r["capture_rate"],
        "capture rate (all four cells)",
        "capture rate",
        PANELS,
    )
    shield_panels = tuple(p for p in PANELS if p.endswith("-SHIELD"))
    _bars(
        axes[1],
        records,
        lambda r: r["aggregate"]["episode_frac_law_pin_ge_40"],
        f"H-PLAN-3 (A1.1): episodes with a law pin >= {LAW_PIN_STEPS} steps",
        "episode fraction",
        shield_panels,
    )
    # H-PLAN-2's quantity, per row: OPEN capture - SHIELD capture.
    xs = np.arange(len(OPPONENTS))
    width = 0.35
    for i, row in enumerate(ROWS):
        vals = []
        for o in OPPONENTS:
            op, sh = _rec(records, row, "OPEN", o), _rec(records, row, "SHIELD", o)
            vals.append(
                np.nan
                if op is None or sh is None
                else op["capture_rate"] - sh["capture_rate"]
            )
        axes[2].bar(xs + i * width - width / 2, vals, width, label=row)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(list(OPPONENTS), fontsize=8)
    axes[2].set_ylabel("OPEN capture - SHIELD capture", fontsize=8)
    axes[2].set_title("H-PLAN-2: the shield's cost, per law", fontsize=9)
    axes[2].axhline(0.0, color="0.3", lw=0.8)
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="y", alpha=0.3)
    null_line = "NULL not evaluated" if null is None else f"NULL {null['verdict']}"
    fig.suptitle(
        f"S19 — the planner intercept (CBFShield defaults, hold {HOLD}, "
        f"shield_obs/escape_obs on; N = {N_EPISODES} seed-paired episodes from "
        f"{GATE_SEED0}). Intervention scope: all steps (A1.2). {null_line}."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Per-episode arrays stay in the cell records; the round blob keeps the
# aggregates so it is readable.
_BULK_KEYS = (
    "capture_ind",
    "intervention_total",
    "longest_law_run",
    "law_pin_ge_40",
    "branch",
)


def summarize(device: str = "cpu") -> dict:
    records = load_records()
    null = null_open(device=device)
    v = verdicts(records, null)
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in records.items()
        },
        "provenance": provenance(),
    }
    _write_json(round_file("json"), blob)
    write_figure(records, round_file("png"), null)
    return blob


# ----------------------------------------------------------------------------
# CG-C (prereg section 5): the behavioural gate, watched BEFORE the table
# ----------------------------------------------------------------------------
# "3 unselected episodes per cell on construct seeds 1 990 000-1 990 002 against
# GREEDY, watched before the table, specifically for route oscillation around
# obstacles (an A*-replanning pathology) and for any pinned run the planner did
# not remove."
#
# Three properties make these GIFs evidence rather than decoration, and all
# three are the retrospective's rule (docs/retrospective_2026-09-23.md section
# 4) made mandatory after D125 — the error that cost a year was caught by a
# human watching a rendered episode, not by a number:
#
#   - UNSELECTED. The seeds are the first three CONSECUTIVE seeds of the
#     construct stream, fixed in the prereg before any outcome existed. There is
#     no knob to choose an episode with.
#   - THE CELL'S OWN EPISODE. The recorder IS the cell's `BranchTaggedSearchMDP`
#     at the cell's law, hold, shield and observation flags, so what is watched
#     is the behaviour the table summarises — and the frame label carries the
#     BRANCH, so "the law drove into that wall" is readable off the frame rather
#     than inferred.
#   - NO NUMBER IS INVENTED. The caption quotes the cell's MEASURED capture rate
#     and law-pin fraction out of its record; a panel with no record stops the
#     pass before a single frame is drawn.
#
# Visuals carry NO verdict: nothing in this section writes a cell record, a
# round artefact or a verdict.

GIF_N = 3
GIF_OPPONENT = "GREEDY"  # the primary metric's opponent (prereg section 5)
GIF_EVERY = 4  # keep every 4th env step as a frame, as S14/S18


def construct_seeds() -> tuple[int, ...]:
    """CG-C's episode seeds: 1 990 000, 1 990 001, 1 990 002."""
    return tuple(CONSTRUCT_SEED0 + i for i in range(GIF_N))


def gif_cell(panel: str) -> str:
    """The record a panel's caption is quoted from: its GREEDY cell."""
    return cell_name(*panel.split("-"), GIF_OPPONENT)


def gif_title(panel: str, rec: dict, seed: int) -> str:
    """The caption: the panel, the episode seed and the cell's two MEASURED
    numbers. No verdict.

    Two lines, as in S14/S18: the recorder prefixes the branch / outcome tag to
    the first and `Renderer.draw` appends the step counter to the second, so
    neither line grows past the frame width."""
    return (
        f"{panel} vs {GIF_OPPONENT} · episode {seed}\n"
        f"measured capture {rec['capture_rate']:.3f} · law pin >= "
        f"{LAW_PIN_STEPS} {rec['aggregate']['episode_frac_law_pin_ge_40']:.3f} "
        f"(N={rec['n_episodes']})"
    )


class CellRecorder(BranchTaggedSearchMDP):
    """The cell's env, keeping every `every`-th env step as an RGB frame.

    It subclasses `BranchTaggedSearchMDP` and not plain `SearchMDP` for CG-C's
    reason: what must be watched is the episode the CELL measures, so the class,
    the law, the hold, the shield and the observation flags are the cell's and
    only the recording differs. Recording is passive — the recorder adds
    counters and touches no dynamics — so the frames are of the measured
    trajectory.

    The overlay is S4's, unchanged: the FOV cone plus nominal-vs-executed arcs
    from the shield info that was actually logged, so the difference between
    what the controller ASKED for and what the filter let through is visible in
    every frame.
    """

    def __init__(
        self, cfg, opponent_factory, *, label: str, every: int = GIF_EVERY, **kw
    ):
        # pe2.render selects the Agg backend and must be imported before
        # matplotlib.pyplot, hence the import order here (and the lazy import:
        # the cell runs must not pay for matplotlib).
        from pe2.render import FovOverlay, Renderer, ShieldOverlay, Trails

        super().__init__(cfg, opponent_factory, **kw)
        self.label = label
        self.every = int(every)
        self.frames: list = []
        self._r = Renderer(cfg, dpi=80, figsize=(5.5, 5.0))
        self._t = Trails()
        self._live = {"nominal": None, "executed": None, "info": None}
        self._ov = [
            FovOverlay("pursuer"),
            ShieldOverlay("pursuer", self.shield, self._live),
        ]

    def _draw(self, title: str) -> None:
        self.frames.append(
            self._r.draw(
                self.inner.env, trails=self._t, overlays=self._ov, title_prefix=title
            )
        )

    def _step_inner(self, action, branch: str = "law"):
        cfg, env = self.cfg, self.inner.env
        self._t.append("pursuer", *env.pursuer_pose[:2])
        self._t.append("evader", *env.evader_pose[:2])
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        nominal = ((float(a[0]) + 1.0) / 2.0 * cfg.v_max, float(a[1]) * cfg.omega_max)
        out = super()._step_inner(action, branch)
        v, w = self.trace[-1][2], self.trace[-1][3]
        self._live.update(
            nominal=nominal,
            executed=((v + 1.0) / 2.0 * cfg.v_max, w * cfg.omega_max),
            info=self._last_sinfo,
        )
        if env.current_step % self.every == 0:
            self._draw(f"[{branch}] {self.label}")
        return out

    def finish(self, outcome: str, step: int, hold_frames: int = 10) -> None:
        """Freeze the last state so the outcome is readable, then close the
        figure (one figure per episode, 12 of them)."""
        for _ in range(hold_frames):
            self._draw(f"[{outcome.upper()} @ {step}] {self.label}")
        self._r.close()


def render_panel(
    panel: str, seed: int, title: str, device: str = "cpu", every: int = GIF_EVERY
) -> tuple[list, dict]:
    """One construct-stream episode of this panel's GREEDY cell, as frames.
    Same law, opponent, hold, shield and observation flags as `run_cell` — only
    the recording differs."""
    spec = CELLS[gif_cell(panel)]
    cfg = cell_config()
    rec = CellRecorder(
        cfg,
        opponent_factory(spec["opponent"], cfg, device),
        label=title,
        every=every,
        hold=spec["hold"],
        shield=build_shield(cfg) if spec["shielded"] else None,
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,
        law=row_law(spec["row"], cfg),
    )
    log = rec.play_episode(OrbitSearch(cfg).predict, seed)
    rec.finish(log["outcome"], log["steps"])
    return rec.frames, log


def write_gifs(
    names: list[str] | None = None,
    device: str = "cpu",
    every: int = GIF_EVERY,
    fps: float = 15.0,
) -> list[tuple[str, int]]:
    """CG-C: `GIF_N` unselected episodes per panel, vs GREEDY only — 12 GIFs.

    `names` are PANEL names (`LINE-OPEN`, `PLAN-SHIELD`, ...), which is what a
    GIF's filename is keyed on. An episode whose GIF exists is skipped; a panel
    with no cell record stops the pass BEFORE anything renders rather than being
    captioned with a guess. Nothing here writes a record or a verdict."""
    from pe2.render import save_gif

    wanted = list(PANELS) if not names else list(names)
    unknown = [n for n in wanted if n not in PANELS]
    if unknown:
        raise SystemExit(
            f"unknown panel(s): {unknown}; this round's panels are {PANELS}"
        )
    todo = [
        (panel, seed)
        for panel in wanted
        for seed in construct_seeds()
        if not gif_path(panel, seed).exists()
    ]
    # Every precondition of every remaining episode resolved before the first
    # frame: a pass that renders two panels and then refuses on the third leaves
    # a half-watched gate, and a half-watched gate is how a construct error
    # survives (D125).
    prepared = {p: read_cell(gif_cell(p)) for p in dict.fromkeys(p for p, _ in todo)}
    n_all = len(wanted) * GIF_N
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"{GIF_N} unselected episodes per panel on seeds {construct_seeds()} "
        f"vs {GIF_OPPONENT} (CG-C). Watch for route oscillation around obstacles "
        "and for any pinned run the planner did not remove.",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for panel, seed in todo:
        frames, log = render_panel(
            panel, seed, gif_title(panel, prepared[panel], seed), device, every
        )
        path = gif_path(panel, seed)
        save_gif(frames, str(path), fps=fps)
        print(
            f"wrote {path} ({len(frames)} frames; {log['outcome']} @ {log['steps']})",
            flush=True,
        )
    return todo


# ----------------------------------------------------------------------------
# CLI
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
    # CG-C is a BLOCKING construct gate (prereg section 5): the episodes are
    # watched before any capture rate is read, so in a single invocation the
    # GIFs are written BEFORE the table is summarised.
    if args.gifs is not None:
        write_gifs(args.gifs or None, args.device)
    if args.summarize:
        blob = summarize(args.device)
        v = blob["verdicts"]
        print(
            json.dumps(
                {
                    k: {
                        kk: v[k][kk]
                        for kk in ("verdict", "n_pass", "n_vacuous", "pass_min")
                        if kk in v[k]
                    }
                    for k in ("h_plan_1", "h_plan_2", "h_plan_3")
                },
                indent=2,
            )
        )
        print(json.dumps(v["h_safe"], indent=2))
        print(json.dumps(v["null_open"], indent=2))
        print(
            "H-PLAN-3 (A1.1) is the FRACTION of SHIELD episodes with a law pin "
            f">= {LAW_PIN_STEPS} steps; the median and p90 of the longest law run "
            "are reported beside it and gate nothing."
        )
        print(
            "H-PLAN-2 (A2.2) has NO OPEN-band conjunct, and the NULL is the "
            "obstacle-free bit-identity alone — the OPEN column is unshielded, "
            "not obstacle-free."
        )
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
