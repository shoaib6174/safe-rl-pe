"""S23 — pursuit strategy against a faster, obstacle-hugging evader.

Spec: `docs/pathway/pursuit_strategy_prereg_2026-10-12.md`, sections 3 to 7.
Where this module and the pre-registration could be read to disagree, the
PREREGISTRATION wins and the reading is named in a comment.

Evaluation only. Nothing here trains, selects or tunes anything.

A THIN derivative of `scripts/derisk_s19_planner.py` (S19). IMPORTED and
called: `_write_json`, `_sha256`, `_ci`, `_q`, `cell_name`, `opponent_factory`,
`require_atk_ckpt`, `build_shield`, `provenance` (extended here), `CellRecorder`,
`OPPONENTS`, `ATK_CKPT`, `SHIELD_OBS`, `ESCAPE_OBS`. COPIED only where S19's
version closes over S19's own globals and so cannot be called with S23's cells:
the artefact paths, the record-validity trio (A4.4/A5.2), `undetermined_record`
(A4.5), `_cell_line` (A4.6) and `write_gifs`'s shape (A5.1) — each says so.

Twelve cells, 4 x 1 x 3, N = 500 paired episodes on one stream:

    rows       PLAN      (`pe2.planner.PlannerInterceptLaw`, the S19 reference;
                          re-run on THIS stream — S19's numbers are not reused)
               LEAD-0.5  (`LeadPlannerInterceptLaw`, tau = 0.5 s)
               LEAD-1.0  (`LeadPlannerInterceptLaw`, tau = 1.0 s)
               MPC-2s    (`MpcInterceptLaw` at its binding defaults)
    column     SHIELD    (`pe2.shield.CBFShield(cfg)` at its defaults)
    opponents  GREEDY / HUG / ATK (the frozen structured attacker ATK4-45)

The ONLY difference between the rows is the law object — the aim point when the
evader is visible. Hold, `shield_obs`, `escape_obs`, config, search policy,
shield, opponent and seed stream are identical in every cell, so a capture
difference is attributable to the aim point and to nothing else; the table is
asserted on the BUILT env in `tests/pe2/test_derisk_s23_round.py`.

Four readings this module fixes, because the text can be read two ways:

- **Per-step visibility was missing.** `pe2.recorder` now records `visible`
  additively: the visibility of the OBSERVATION the step's action was computed
  from, because "a law step with the evader visible" is about what the law was
  acting on. The run is computed HERE from `branch` + `visible`.
- **CG-B is the additive-field check, not S19's 36/36 reproduction** (section
  5): 20 replayed S18 episodes must carry EXACTLY their recorded per-episode
  all-step `intervention_total`, enforced in `run_cells` before the first cell.
- **The NULL runs on 2 090 000+**, the round's construct stream (section 4),
  not the 1 990 000+ of `tests/pe2/test_{lead,mpc}_law.py`.
- **The NULL excludes the law-EXPOSURE counters** — see `NULL_EXPOSURE_KEYS`,
  where the reading is argued in full.

Usage:
  ./venv/bin/python scripts/derisk_s23_strategy.py --cells        (all missing)
  ./venv/bin/python scripts/derisk_s23_strategy.py --cells MPC-2s-SHIELD_HUG
  ./venv/bin/python scripts/derisk_s23_strategy.py --gifs         (CG-S3)
  ./venv/bin/python scripts/derisk_s23_strategy.py --summarize

Cells are resumable and atomic: one JSON per cell, and a cell that has a record
is NEVER re-run (prereg section 6, the stop rule).
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # prereg section 7; set BEFORE numpy / torch build their pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.planner import (
    MPC_HOLD,
    MPC_K,
    LeadPlannerInterceptLaw,
    MpcInterceptLaw,
    PlannerInterceptLaw,
)
from pe2.recorder import BranchTaggedSearchMDP
from pe2.scripted import StationaryEvader
from pe2.structured import HOLD, OrbitSearch

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S19's runner, loaded by FILE PATH (the `derisk_s6b_attacker` precedent) so
# this module imports the same way whether it is run as a script or imported as
# `scripts.derisk_s23_strategy` by the tests.
S19 = _load("derisk_s19_planner", _HERE / "derisk_s19_planner.py")
REPLAY = _load("replay_s18_branch_split", _HERE / "replay_s18_branch_split.py")

# --- imported from S19 and CALLED, not reimplemented -------------------------
_write_json = S19._write_json
_sha256 = S19._sha256
_ci = S19._ci
_q = S19._q
cell_name = S19.cell_name
opponent_factory = S19.opponent_factory  # reads S19.OPPONENTS / S19.ATK_CKPT
require_atk_ckpt = S19.require_atk_ckpt  # the frozen ATK4-45 refusal
build_shield = S19.build_shield  # CBFShield(cfg), defaults ASSERTED at 0.0/0.2
OPPONENTS = S19.OPPONENTS  # GREEDY / HUG / ATK, "exactly as S19"
ATK_CKPT = S19.ATK_CKPT
SHIELD_OBS = S19.SHIELD_OBS
ESCAPE_OBS = S19.ESCAPE_OBS
UNDETERMINED = S19.UNDETERMINED

SPEC = "docs/pathway/pursuit_strategy_prereg_2026-10-12.md"
ARTEFACT_PREFIX = "s23"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 3 to 6). Do not tune. ---
GATE_SEED0 = 2_100_000  # section 3, the gate stream
CONSTRUCT_SEED0 = 2_090_000  # sections 4 and 5, the construct stream
N_EPISODES = 500  # section 3
P_GATE = 0.05  # section 4
PASS_MIN = 2  # ">= 2/3 opponents"
NULL_N = 50  # section 4's NULL
# Section 4: an ORBIT episode carries >= 100 consecutive law-branch steps with
# the evader VISIBLE and no capture.
ORBIT_RUN_STEPS = 100
# Section 4: fewer than this many orbit episodes makes the opponent VACUOUS for
# H-STRAT-2 (never PASS) — there is no concentration to test.
ORBIT_MIN_EPISODES = 20
# Section 2: `MPC-2s` runs at its BINDING defaults. Asserted rather than passed
# as keyword arguments, exactly as `build_shield` asserts the shield's, so the
# round fails loudly if the library's defaults move instead of silently pinning
# values this script chose.
MPC_BINDING = {"h_steps": 40, "n_seg": 4, "hold": MPC_HOLD, "k": MPC_K}

COL = "SHIELD"  # section 3: the SHIELD column only
BASELINE_ROW = "PLAN"
ROWS: dict[str, dict] = {
    # `law` is the class NAME the record carries; `row_law` builds the object.
    "PLAN": {"law": "PlannerInterceptLaw", "tau": None},
    "LEAD-0.5": {"law": "LeadPlannerInterceptLaw", "tau": 0.5},
    "LEAD-1.0": {"law": "LeadPlannerInterceptLaw", "tau": 1.0},
    "MPC-2s": {"law": "MpcInterceptLaw", "tau": None},
}
TREATED: tuple[str, ...] = tuple(r for r in ROWS if r != BASELINE_ROW)


def _build_cells() -> dict[str, dict]:
    """The 12 cells. Each carries its row and opponent STORED — nothing parses
    the name back apart (S17's convention; here it would also be wrong, because
    `LEAD-0.5` contains the separator)."""
    return {
        cell_name(r, COL, o): {
            "name": cell_name(r, COL, o),
            "row": r,
            "col": COL,
            "opponent": o,
            "law": ROWS[r]["law"],
            "tau": ROWS[r]["tau"],
            "shielded": True,
            "hold": HOLD,
        }
        for r in ROWS
        for o in OPPONENTS
    }


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)


# ----------------------------------------------------------------------------
# artefact paths, io, provenance  (copied from S19: each closes over S19's own
# ARTEFACT_PREFIX / SPEC / OUT)
# ----------------------------------------------------------------------------
def cells_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cells"


def cell_file(name: str) -> pathlib.Path:
    return cells_dir() / f"{name}.json"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_gifs"


def gif_path(row: str, seed: int) -> pathlib.Path:
    return gifs_dir() / f"{row}_{seed}.gif"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def null_file() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_null.json"


def cg_s3_file() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_cg_s3_seeds.json"


def spec_path() -> pathlib.Path:
    return _HERE.parent / SPEC


def write_cell(record: dict) -> pathlib.Path:
    path = cell_file(record["name"])
    _write_json(path, record)
    return path


def provenance(device: str = "cpu", cgb: dict | None = None) -> dict:
    """S19's provenance block with S23's spec hash, CG-B tally and CG-S3 seeds.

    `S19.provenance` is CALLED for the commit, the `pe2` package, the device,
    the library versions and the thread setting; only the fields that name THIS
    round's text and gates are replaced. The spec hash is the point: a cell
    record proves which text's thresholds it was measured under."""
    seeds = read_cg_s3_seeds()
    return S19.provenance(device, cgb) | {
        "spec": SPEC,
        "spec_sha256": _sha256(spec_path()),
        "cg_b": cgb,
        "cg_b_scope": (
            "CG-B is the ADDITIVE-FIELD check of prereg section 5 (the recorder "
            f"reproduces {CGB_N} S18 episodes' per-episode all-step "
            "intervention_total exactly), NOT S19's 36/36 replay tally"
        ),
        "cg_s3_seeds": None if seeds is None else list(seeds),
    }


# ----------------------------------------------------------------------------
# CG-B (prereg section 5): the additive fields do not alter intervention_total
# ----------------------------------------------------------------------------
# Section 5 does not re-run S19's 36/36 reproduction gate — the recorder is
# unchanged except for additive fields — but it does make the runner assert that
# those additive fields changed nothing, on 20 replayed S18 episodes, BEFORE the
# first cell. The comparison is `==` on the per-episode all-step
# `intervention_total` and there is no tolerance to fall back to: a mismatch
# means the instrument moved and nothing measured with it may be read.
CGB_ARM = "SPARSE-PEN-70"
CGB_OPPONENT = "GREEDY"
CGB_N = 20
CGB_SEED0 = 1_900_000  # S18's gate stream, asserted against the S18 runner's


def cgb_artefact() -> pathlib.Path:
    return OUT / f"s18_branch_split_{CGB_ARM}_{CGB_OPPONENT}.json"


def cgb_stored() -> dict[int, float]:
    """`{seed: intervention_total}` from the recorded S18 replay artefact."""
    path = cgb_artefact()
    if not path.exists():
        raise SystemExit(
            f"CG-B REFUSES: no S18 replay artefact at {path}. Section 5 requires "
            f"the runner to assert the recorder's additive fields leave "
            "`intervention_total` unchanged on 20 replayed S18 episodes, and "
            "there is nothing to compare against."
        )
    rec = json.loads(path.read_text())
    # Keyed by SEED over every recorded episode, never sliced by position: the
    # comparison must not depend on the artefact's episode ORDER.
    return {int(e["seed"]): float(e["intervention_total"]) for e in rec["episodes"]}


def cgb_replay(device: str = "cpu") -> dict[int, float]:
    """`{seed: intervention_total}` from `CGB_N` S18 episodes replayed NOW,
    through the updated recorder. The env is built by
    `scripts/replay_s18_branch_split.py`'s own `_trees` and `paths` — the S18
    runner's opponent factory, hold, observation flags, a FRESH `CBFShield` per
    episode and its gate stream — so this is the S18 cell, not a look-alike.
    """
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield

    runner, recorder, _tree = REPLAY._trees()
    if int(runner.GATE_SEED0) != CGB_SEED0:
        raise SystemExit(
            f"the S18 runner's gate stream is {runner.GATE_SEED0}, not the "
            f"{CGB_SEED0} this check's stored episodes were measured on."
        )
    p = REPLAY.paths(CGB_ARM, CGB_OPPONENT, runner)
    if not p["ckpt"].exists():
        raise SystemExit(f"CG-B REFUSES: no S18 checkpoint at {p['ckpt']}.")
    cfg = EnvConfig()
    model = PPO.load(p["ckpt"], device=device)
    mdp = recorder.BranchTaggedSearchMDP(
        cfg,
        runner.gate_opponent_factory(CGB_OPPONENT, device),
        hold=runner.HOLD,
        shield=None,
        shield_obs=runner.SHIELD_OBS,
        escape_obs=runner.ESCAPE_OBS,
        reward_wrapper=None,
    )
    out: dict[int, float] = {}
    for i in range(CGB_N):
        mdp.shield = CBFShield(cfg)
        seed = CGB_SEED0 + i
        log = mdp.play_episode(model.predict, seed)
        out[seed] = float(log["branch"]["intervention_total"])
    return out


def cgb_status(replayed: dict, stored: dict) -> dict:
    """The tally, as a pure function of the two readings. PASS iff all `CGB_N`
    episodes are present on both sides and every one is EXACTLY equal."""
    checks = {
        int(seed): {
            "replayed": got,
            "recorded": stored.get(int(seed)),
            "exact": int(seed) in stored and got == stored[int(seed)],
        }
        for seed, got in sorted(replayed.items())
    }
    bad = sorted(s for s, c in checks.items() if not c["exact"])
    ok = len(checks) == CGB_N and not bad
    return {
        "gate": "CG-B, the additive-field check (prereg section 5)",
        "arm": CGB_ARM,
        "opponent": CGB_OPPONENT,
        "required": CGB_N,
        "n_episodes": len(checks),
        "n_exact": len(checks) - len(bad),
        "verdict": "PASS" if ok else "REFUSE",
        "mismatching_seeds": bad,
        "quantity": "per-episode all-step intervention_total (S19 A1.2 scope)",
        "artefact": str(cgb_artefact()),
    }


def require_cgb_additive(device: str = "cpu") -> dict:
    """SystemExit unless the additive fields changed nothing. Returns the tally
    so it lands in provenance."""
    status = cgb_status(cgb_replay(device), cgb_stored())
    if status["verdict"] != "PASS":
        raise SystemExit(
            f"CG-B REFUSES: {status['n_exact']}/{status['n_episodes']} replayed "
            f"S18 episodes reproduce their recorded intervention_total exactly "
            f"and {CGB_N}/{CGB_N} are required; mismatching seeds "
            f"{status['mismatching_seeds']}. The recorder's new per-step fields "
            "(aim_point, v_est, feasible_count, chose_direct, visible) must be "
            "ADDITIVE (prereg section 5): a recorder that changed an old number "
            "reads nothing new."
        )
    return status


# ----------------------------------------------------------------------------
# record validity (S19 A4.4 / A4.5 / A5.2), copied: S19's versions close over
# its own CELLS, N_EPISODES, GATE_SEED0 and cells_dir
# ----------------------------------------------------------------------------
def record_status(rec: dict) -> tuple[str, str]:
    """`(status, why)` — `"MEASURABLE"`, `"UNDETERMINED"` or `"INVALID"`.
    Prereg section 6: records are measurable only at N = 500 on seed0
    2 100 000."""
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
    """`(measurable, rejected, rejected_by_cell)`, keyed by FILE NAME for the
    reasons and by CELL for the duplicates. A cell claimed by more than one
    file is poisoned and every file claiming it is rejected (A5.2: duplicates
    reject BOTH, however they are named)."""
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
            f"(n_episodes == {N_EPISODES} and seed0 == {GATE_SEED0} are "
            "required):",
            flush=True,
        )
        for fname, why in sorted(rejected.items()):
            print(f"  {fname}: {why}", flush=True)
    return measurable


def read_cell(name: str) -> dict:
    """The cell's MEASURABLE record, or SystemExit. CG-S3 is watched per
    MEASURED row (A5.1): existence of the file is not enough."""
    path = cell_file(name)
    if not path.exists():
        raise SystemExit(
            f"no cell record for {name} at {path}: run --cells {name} first. "
            "CG-S3 is watched per measured row (prereg section 5)."
        )
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cell record {path} is unreadable: {exc}") from exc
    status, why = record_status(rec)
    if status != "MEASURABLE":
        raise SystemExit(
            f"cell record {path} is not measurable ({status}: {why}), so there "
            f"is nothing to watch for {name}. An UNDETERMINED cell stays "
            "UNDETERMINED under the stop rule and is never rendered from."
        )
    return rec


# ----------------------------------------------------------------------------
# the cell's env — the only difference between rows is the aim point
# ----------------------------------------------------------------------------
def cell_config() -> EnvConfig:
    """`EnvConfig()` defaults in every cell. Nothing sweeps in this round."""
    return EnvConfig()


def null_config() -> EnvConfig:
    """The obstacle-free arena of the NULL. `n_obstacles` still owns the
    observation layout, so only the PLACED obstacles go away."""
    return dataclasses.replace(cell_config(), n_physical_obstacles=0)


def row_law(row: str, cfg: EnvConfig):
    """The row's nominal controller — the ONE variable of the round.

    `PLAN` is passed EXPLICITLY (not `law=None`): prereg section 2 names
    `PlannerInterceptLaw` as the reference, which is also what the treated laws
    delegate to when the velocity estimate reads zero, so the baseline is that
    object and not `SearchMDP`'s default `InterceptLaw`.
    """
    spec = ROWS[row]
    if spec["law"] == "PlannerInterceptLaw":
        return PlannerInterceptLaw(cfg)
    if spec["law"] == "LeadPlannerInterceptLaw":
        return LeadPlannerInterceptLaw(cfg, spec["tau"])
    law = MpcInterceptLaw(cfg)  # section 2: the binding defaults
    got = {k: getattr(law, k) for k in MPC_BINDING}
    if got != MPC_BINDING:
        raise SystemExit(
            f"MpcInterceptLaw's defaults moved to {got}; prereg section 2 fixes "
            f"{MPC_BINDING} (K = 64 sampled 40-step sequences, 4 segments x 10 "
            "steps) and this round may not run under another."
        )
    return law


def build_cell_env(
    name: str, cfg: EnvConfig, device: str = "cpu", opponent=None
) -> BranchTaggedSearchMDP:
    """The cell's env: the branch-tagged recorder with this row's law, the
    round's shield and NOTHING else varied. `cfg` and `opponent` are parameters
    so the NULL can build the SAME construction on an obstacle-free arena
    against `StationaryEvader`.
    """
    spec = CELLS[name]
    return BranchTaggedSearchMDP(
        cfg,
        opponent_factory(spec["opponent"], cfg, device)
        if opponent is None
        else opponent,
        hold=spec["hold"],
        shield=build_shield(cfg),
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,  # evaluation only; `play_episode` discards rewards
        law=row_law(spec["row"], cfg),
    )


# ----------------------------------------------------------------------------
# one cell
# ----------------------------------------------------------------------------
def longest_visible_law_run(steps: list[dict]) -> int:
    """The longest run of CONSECUTIVE law-branch steps with the evader visible
    (prereg section 4's orbit statistic).

    A branch change or a lost sighting ENDS the run: the law also drives on a
    fresh last-seen point with the evader out of sight, and those steps are law
    steps but not visible ones.
    """
    best = cur = 0
    for s in steps:
        if s["branch"] == "law" and s.get("visible"):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def exposure_counts(steps: list[dict]) -> tuple[int, int]:
    """`(n_aimed, n_zero_feasible_exact)` for one episode, from the per-step
    records (the same source `longest_visible_law_run` reads).

    Prereg section 2 reports the treated laws' EXPOSURE per episode, and
    section 5 fixes the exact 0/64 count as "`feasible_count == 0` and
    `aim_point is not None`" — neither is recoverable from the recorder's own
    summary. `n_planned_calls` counts the steps whose law EXPOSES the field at
    all (every step of an MPC episode, whether or not the sampling ran), and
    `n_zero_feasible` counts `feasible_count == 0`, which includes the CG-S1
    delegation and is therefore only an upper bound. These two are the
    quantities the prereg actually names:

    - `n_aimed` — steps on which the law aimed somewhere of its own (LEAD's
      lead point, MPC's predicted evader position), i.e. the planned calls;
    - `n_zero_feasible_exact` — steps on which MPC sampled and 0 of K rows were
      feasible, the wall-adjacent case of A2 note (i).
    """
    n_aimed = sum(s["aim_point"] is not None for s in steps)
    n_zero = sum(s["feasible_count"] == 0 and s["aim_point"] is not None for s in steps)
    return n_aimed, n_zero


def run_cell(
    name: str,
    device: str = "cpu",
    n_episodes: int | None = None,
    *,
    cgb: dict | None = None,
) -> dict:
    """One cell: N paired episodes of this row's law under the shield against
    this opponent, on the gate stream. The shield is rebuilt per episode exactly
    as S18's and S19's gates rebuilt it (`CBFShield` is memoryless, so the
    rebuild is equivalent — but that is a property of the shield, not of the
    protocol).
    """
    spec = CELLS[name]
    n = N_EPISODES if n_episodes is None else int(n_episodes)
    cfg = cell_config()
    mdp = build_cell_env(name, cfg, device)
    policy = OrbitSearch(cfg)  # the teacher's search, in every cell

    capture_ind: list[bool] = []
    ep_steps: list[int] = []
    violations: list[int] = []
    intervention_total: list[float] = []
    vis_runs: list[int] = []
    aimed: list[int] = []
    zero_feasible_exact: list[int] = []
    law_exposed: list[int] = []
    zero_feasible: list[int] = []
    chose_direct: list[int] = []
    branch: list[dict] = []
    outcomes: Counter = Counter()
    for i in range(n):
        mdp.shield = build_shield(cfg)
        log = mdp.play_episode(policy.predict, GATE_SEED0 + i)
        b = log["branch"]
        capture_ind.append(log["outcome"] == "capture")
        outcomes[log["outcome"]] += 1
        ep_steps.append(int(log["steps"]))
        # Authoritative contact count: `SearchMDP._step_inner` increments it on
        # every step from the env's own contact predicates.
        violations.append(int(mdp.episode_violations))
        intervention_total.append(float(b["intervention_total"]))  # all-step scope
        vis_runs.append(longest_visible_law_run(mdp.step_records))
        n_aimed, n_zero_exact = exposure_counts(mdp.step_records)
        aimed.append(n_aimed)
        zero_feasible_exact.append(n_zero_exact)
        # The recorder's own counter, kept under the name of what it counts:
        # steps whose law EXPOSED `feasible_count` at all, not planned calls.
        law_exposed.append(int(b["n_planned_calls"]))
        zero_feasible.append(int(b["n_zero_feasible"]))
        chose_direct.append(int(b["n_chose_direct"]))
        branch.append(b)

    orbit = [r >= ORBIT_RUN_STEPS and not c for r, c in zip(vis_runs, capture_ind)]
    k = sum(capture_ind)
    return {
        "name": name,
        "row": spec["row"],
        "col": spec["col"],
        "opponent": spec["opponent"],
        "law": spec["law"],
        "tau": spec["tau"],
        "shielded": True,
        "hold": spec["hold"],
        "shield_obs": SHIELD_OBS,
        "escape_obs": ESCAPE_OBS,
        "n_episodes": n,
        "seed0": GATE_SEED0,
        "device": device,
        "capture_ind": capture_ind,
        "capture_rate": k / n if n else float("nan"),
        "wilson_capture": _ci(k, n),
        "steps": ep_steps,
        # Per EPISODE (the prereg's per-episode field); `violation_steps` is the
        # pooled total H-SAFE is read off.
        "violations": violations,
        "violation_steps": int(sum(violations)),
        "intervention_total": intervention_total,
        "longest_visible_law_run": vis_runs,
        "orbit_ind": orbit,
        # Section 2's exposure, per episode, from the per-step records.
        "n_aimed": aimed,
        "n_zero_feasible_exact": zero_feasible_exact,
        # The recorder's summary counters, kept as they are measured.
        "n_law_exposed_steps": law_exposed,
        "n_zero_feasible": zero_feasible,
        "n_chose_direct": chose_direct,
        "branch": branch,
        "aggregate": {
            "capture_rate": k / n if n else float("nan"),
            "outcomes": dict(outcomes),
            # The env terminates on capture and truncates on timeout and nothing
            # else, so the evader escapes exactly when the episode times out.
            "escape_rate": outcomes["timeout"] / n if n else float("nan"),
            "n_orbit": sum(orbit),
            "orbit_run_steps": ORBIT_RUN_STEPS,
            "visible_law_run_median": float(statistics.median(vis_runs))
            if vis_runs
            else float("nan"),
            "visible_law_run_p90": _q(vis_runs, 0.90),
            "visible_law_run_max": max(vis_runs, default=0),
            "intervention_total_mean": float(np.mean(intervention_total))
            if intervention_total
            else float("nan"),
            "intervention_scope": "all steps (S19 A1.2)",
            # Section 2's stated limitation: the treated laws' exposure varies
            # per episode, so the planned-call count is reported beside the rate.
            "n_aimed_total": sum(aimed),
            "n_zero_feasible_exact_total": sum(zero_feasible_exact),
            "n_law_exposed_steps_total": sum(law_exposed),
            "n_zero_feasible_total": sum(zero_feasible),
            "n_chose_direct_total": sum(chose_direct),
        },
        "provenance": provenance(device, cgb),
    }


def _cell_line(rec: dict) -> str:
    """A4.6, copied: NO capture rate or any other measured number is printed
    while the cells run. `--summarize` is the only reader, and it runs after
    `--gifs` (CG-S3)."""
    return f"[{rec['name']}] recorded, n_episodes {rec['n_episodes']}"


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    """A4.5 / prereg section 6: a crashed cell is `UNDETERMINED`, written as a
    record rather than left absent so the stop rule can be enforced
    mechanically. `n_episodes` is 0 so it can never read as a measurement."""
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
    """Every named cell with no record yet (default: all 12). Literal names only.

    Everything that can refuse refuses BEFORE the first episode: CG-B, the ATK
    checkpoint, and every existing file's validity. A file that is not a
    measurable record of THIS round is a REFUSAL, not a skip and not an
    overwrite."""
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
    cgb = require_cgb_additive(device)  # prereg section 5, before the first cell
    if any(CELLS[n]["opponent"] == "ATK" for n in todo):
        require_atk_ckpt()
    print(
        f"CG-B {cgb['n_exact']}/{cgb['required']} exact per-episode "
        f"intervention_total reproductions: PASS. {len(todo)}/{len(wanted)} "
        f"cells to run ({len(wanted) - len(todo)} already have a measurable "
        "record; stop rule: never re-run), N = "
        f"{N_EPISODES if n_episodes is None else n_episodes} from seed "
        f"{GATE_SEED0}. No rate is printed here — read --summarize, after "
        "--gifs.",
        flush=True,
    )
    for name in todo:
        try:
            rec = run_cell(name, device, n_episodes, cgb=cgb)
        except Exception as exc:  # noqa: BLE001 — record it, do not re-run
            rec = undetermined_record(name, exc, device)
            write_cell(rec)
            print(f"[{name}] {UNDETERMINED}: {rec['error']}", flush=True)
            continue
        write_cell(rec)
        print(_cell_line(rec), flush=True)
    return todo


# ----------------------------------------------------------------------------
# the NULL (prereg sections 4 and 5, CG-S1)
# ----------------------------------------------------------------------------
# READING, stated because the prereg and this code could be read to disagree.
# Section 4 says "`play_episode` logs and per-step trace `==`". `MpcInterceptLaw`
# always exposes `last_feasible_count` / `last_chose_direct` / `last_v_est`, so
# the recorder reads `feasible_count == 0` and `n_planned_calls == n_steps` on
# every step of an MPC episode and `None` on every step of a `PLAN` one EVEN WHEN
# THE TWO EPISODES ARE THE SAME EPISODE. Read to include those counters the
# comparison is UNSATISFIABLE by construction for `MPC-2s`: it would fail on
# 50/50 seeds whose trajectories match to the last ulp, the opposite of what
# CG-S1 is for (section 2's claim, "bit-identical to `PLAN` when v-hat reads
# zero", is about the EPISODE). Steps 1 and 3 ran the identity through plain
# `SearchMDP`, whose log carries no branch block, so the question never arose.
# What runs here is STRICTLY STRONGER on everything that is about the episode:
# the log, the recorder's full PER-STEP records and the executed trace must all
# be equal, and the treated law must additionally have aimed NOWHERE.
NULL_EXPOSURE_KEYS = ("aim_point", "v_est", "feasible_count", "chose_direct")
NULL_EXEMPT_SUMMARY = ("n_planned_calls", "n_zero_feasible", "n_chose_direct")


def _comparable(log: dict, steps: list[dict]) -> tuple:
    """The episode as the NULL compares it: the log and every per-step record,
    minus the law-exposure fields named above."""
    pruned_log = dict(log) | {
        "branch": {
            k: v for k, v in log["branch"].items() if k not in NULL_EXEMPT_SUMMARY
        }
    }
    return (
        pruned_log,
        [{k: v for k, v in s.items() if k not in NULL_EXPOSURE_KEYS} for s in steps],
    )


def _null_row(row: str, n: int, seed0: int, device: str) -> dict:
    """The obstacle-free bit-identity for one treated law against `PLAN`.

    `StationaryEvader` is the opponent the prereg names: with the evader frozen
    the velocity estimate reads zero on every call, so every treated law must
    delegate to `PlannerInterceptLaw` and the episode must be `PLAN`'s bit for
    bit — the log, the per-step records and the executed trace alike."""
    cfg = null_config()
    base = build_cell_env(
        cell_name(BASELINE_ROW, COL, "GREEDY"), cfg, device, opponent=StationaryEvader
    )
    treated = build_cell_env(
        cell_name(row, COL, "GREEDY"), cfg, device, opponent=StationaryEvader
    )
    policy = OrbitSearch(cfg)
    differing: list[int] = []
    aimed: list[int] = []
    for i in range(n):
        seed = seed0 + i
        want = _comparable(base.play_episode(policy.predict, seed), base.step_records)
        want_trace = list(base.trace)
        got = _comparable(
            treated.play_episode(policy.predict, seed), treated.step_records
        )
        if got != want or list(treated.trace) != want_trace:
            differing.append(seed)
        if any(s["aim_point"] is not None for s in treated.step_records):
            aimed.append(seed)
    return {
        "row": row,
        "verdict": "HOLDS" if not (differing or aimed) else "FAILS",
        "n_differing": len(differing),
        "differing_seeds": differing[:10],
        # The delegation itself: the treated law must not have aimed anywhere.
        "n_aimed": len(aimed),
        "aimed_seeds": aimed[:10],
    }


def null_identity(
    n: int | None = None, seed0: int | None = None, device: str = "cpu"
) -> dict:
    """CG-S1 carried into the round record (prereg section 4): on an
    obstacle-free arena against a stationary evader, each of the three treated
    laws is `PLAN` bit for bit over `n` seeds of the CONSTRUCT stream, 2 090 000+
    (NOT the 1 990 000+ the unit tests use for the same identity).

    It runs on the round's OWN instrument — the branch-tagged recorder at the
    cells' hold, shield and observation flags — so the comparison covers the
    branch attribution too, and the record carries its own null rather than
    citing a test suite.
    """
    n = NULL_N if n is None else int(n)
    seed0 = CONSTRUCT_SEED0 if seed0 is None else int(seed0)
    rows = {row: _null_row(row, n, seed0, device) for row in TREATED}
    failing = [r for r, b in rows.items() if b["verdict"] == "FAILS"]
    return {
        "hypothesis": "NULL (CG-S1, prereg sections 4 and 5)",
        "verdict": "HOLDS" if not failing else "FAILS",
        "rows": rows,
        "failing_rows": failing,
        "scenario": (
            "obstacle-free arena, EnvConfig(n_physical_obstacles=0), "
            "StationaryEvader, the SHIELD column (the round's only column)"
        ),
        "comparison": (
            "== on the play_episode log, on every per-step record and on the "
            "executed trace, with the law-EXPOSURE fields "
            f"{NULL_EXPOSURE_KEYS} / {NULL_EXEMPT_SUMMARY} excluded (they are a "
            "property of which law object is installed, not of the episode), "
            "plus aim_point is None on every step"
        ),
        "n_episodes": n,
        "seed0": seed0,
        "stream": "construct",
        "pre_registered": bool(n == NULL_N and seed0 == CONSTRUCT_SEED0),
        "n_differing": sum(b["n_differing"] for b in rows.values()),
        "differing_seeds": sorted(
            {s for b in rows.values() for s in b["differing_seeds"]}
        )[:10],
    }


# ----------------------------------------------------------------------------
# hypotheses (prereg section 4)
# ----------------------------------------------------------------------------
def _rec(records: dict, row: str, opponent: str) -> dict | None:
    return records.get(cell_name(row, COL, opponent))


def _missing(records: dict) -> list[str]:
    return [n for n in CELL_ORDER if n not in records]


def _direction_label(delta: float, row: str) -> str:
    if delta > 0:
        return f"{row} > {BASELINE_ROW}"
    if delta < 0:
        return f"{row} < {BASELINE_ROW}"
    return f"{row} == {BASELINE_ROW}"


def h_strat_1(records: dict, row: str) -> dict:
    """H-STRAT-1 for ONE law: shielded capture, law > `PLAN`, per opponent
    `paired_indicator_test` p < P_GATE, on >= 2/3 opponents.

    The per-opponent DIRECTION is reported whether or not the test passes: the
    refutation paragraph turns on the sign, so a failure that does not say which
    way it went is not a usable refutation."""
    per: dict = {}
    for o in OPPONENTS:
        law, base = _rec(records, row, o), _rec(records, BASELINE_ROW, o)
        if law is None or base is None:
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": [
                    cell_name(r, COL, o)
                    for r, v in ((row, law), (BASELINE_ROW, base))
                    if v is None
                ],
            }
            continue
        test = {
            "test": "paired_indicator",
            **paired_indicator_test(law["capture_ind"], base["capture_ind"], P_GATE),
        }
        delta = law["capture_rate"] - base["capture_rate"]
        per[o] = {
            "verdict": "PASS" if test["pass"] else "FAIL",
            "capture_law": law["capture_rate"],
            "capture_plan": base["capture_rate"],
            "delta": delta,
            "direction": _direction_label(delta, row),
            "p": test["p"],
            "test": test,
        }
    n_pass = sum(b["verdict"] == "PASS" for b in per.values())
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in per.values())
    return {
        "hypothesis": "H-STRAT-1",
        "row": row,
        "verdict": "INCOMPLETE"
        if incomplete
        else ("PASS" if n_pass >= PASS_MIN else "FAIL"),
        "n_pass": n_pass,
        "pass_min": PASS_MIN,
        "n_opponents": len(OPPONENTS),
        "p_gate": P_GATE,
        "opponents": per,
    }


def h_strat_1_round(records: dict) -> dict:
    """Every law's H-STRAT-1 verdict. "Any law passing passes the round; each
    law's verdict is reported" (prereg section 4)."""
    laws = {row: h_strat_1(records, row) for row in TREATED}
    passing = [r for r, b in laws.items() if b["verdict"] == "PASS"]
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in laws.values())
    return {
        "hypothesis": "H-STRAT-1 (round)",
        "verdict": "INCOMPLETE" if incomplete else ("PASS" if passing else "FAIL"),
        "passing_laws": passing,
        "rule": "any law passing passes the round",
        "laws": laws,
    }


def orbit_mask(records: dict, opponent: str) -> dict:
    """The ORBIT episodes of the `PLAN-SHIELD` baseline against `opponent`: at
    least `ORBIT_RUN_STEPS` consecutive law-branch steps with the evader visible
    and NO capture (prereg section 4).

    The mask is a property of the BASELINE alone — the treated law never enters
    it — so "the gain is concentrated where the baseline was orbiting" is a
    statement about a set fixed before any treated capture is read."""
    base = _rec(records, BASELINE_ROW, opponent)
    if base is None:
        return {
            "opponent": opponent,
            "verdict": "INCOMPLETE",
            "missing": [cell_name(BASELINE_ROW, COL, opponent)],
        }
    runs, caps = base["longest_visible_law_run"], base["capture_ind"]
    mask = [r >= ORBIT_RUN_STEPS and not c for r, c in zip(runs, caps, strict=True)]
    lost = [not c for c in caps]
    return {
        "opponent": opponent,
        "verdict": "OK",
        "orbit_run_steps": ORBIT_RUN_STEPS,
        "n_episodes": len(mask),
        "n_orbit": sum(mask),
        "n_baseline_lost": sum(lost),
        "n_non_orbit_lost": sum(l and not m for l, m in zip(lost, mask, strict=True)),
        "mask": mask,
        "lost": lost,
        "baseline": base["name"],
    }


def _rate(ind: list[bool], idx: list[int]) -> float | None:
    return None if not idx else sum(ind[i] for i in idx) / len(idx)


def h_strat_2(records: dict, row: str) -> dict:
    """H-STRAT-2 for ONE law (prereg section 4): paired by seed, the conversion
    rate on the baseline's ORBIT episodes must EXCEED the rate on its non-orbit
    LOST episodes, per opponent, on >= 2/3 opponents. Conversion = the fraction
    of those seeds the law captured; orbit episodes are baseline-lost by
    definition, so the two sets partition the baseline's losses and the question
    is where the gain landed.

    VACUOUS, never PASS, when the baseline carries fewer than
    `ORBIT_MIN_EPISODES` orbit episodes (the prereg's guard) and — the reading
    taken here — when the non-orbit lost set is EMPTY, because then there is no
    comparator and "exceeds" is undefined rather than satisfied.
    """
    per: dict = {}
    for o in OPPONENTS:
        om = orbit_mask(records, o)
        law = _rec(records, row, o)
        if om.get("verdict") == "INCOMPLETE" or law is None:
            per[o] = {
                "verdict": "INCOMPLETE",
                "missing": (om.get("missing") or [])
                + ([cell_name(row, COL, o)] if law is None else []),
            }
            continue
        mask, lost = om["mask"], om["lost"]
        orbit_idx = [i for i, m in enumerate(mask) if m]
        non_idx = [
            i for i, (m, lo) in enumerate(zip(mask, lost, strict=True)) if lo and not m
        ]
        conv_orbit = _rate(law["capture_ind"], orbit_idx)
        conv_non = _rate(law["capture_ind"], non_idx)
        block = {
            "n_orbit": len(orbit_idx),
            "n_non_orbit_lost": len(non_idx),
            "conversion_orbit": conv_orbit,
            "conversion_non_orbit": conv_non,
            "orbit_min_episodes": ORBIT_MIN_EPISODES,
        }
        if len(orbit_idx) < ORBIT_MIN_EPISODES:
            per[o] = block | {
                "verdict": "VACUOUS",
                "note": (
                    f"the PLAN-SHIELD baseline carries {len(orbit_idx)} orbit "
                    f"episode(s), fewer than the {ORBIT_MIN_EPISODES} the prereg "
                    "requires: there is no concentration to test and this "
                    "opponent can never PASS"
                ),
            }
            continue
        if not non_idx:
            per[o] = block | {
                "verdict": "VACUOUS",
                "note": (
                    "every episode the baseline lost was an orbit episode, so "
                    "there is no non-orbit comparator and 'exceeds' is undefined"
                ),
            }
            continue
        per[o] = block | {
            "verdict": "PASS" if conv_orbit > conv_non else "FAIL",
            "margin": conv_orbit - conv_non,
        }
    verdicts_ = [b["verdict"] for b in per.values()]
    n_pass = sum(v == "PASS" for v in verdicts_)
    n_vacuous = sum(v == "VACUOUS" for v in verdicts_)
    if "INCOMPLETE" in verdicts_:
        overall = "INCOMPLETE"
    elif n_pass >= PASS_MIN:
        overall = "PASS"
    elif "FAIL" not in verdicts_ and n_pass + n_vacuous >= PASS_MIN:
        # The bar is out of reach ONLY because some opponents have no orbit
        # phenomenon. Not a failure of the law; not a pass either. A single
        # opponent that actually FAILED is evidence and keeps the verdict FAIL:
        # 1 PASS + 1 VACUOUS + 1 FAIL is a FAIL, not a vacuum.
        overall = "VACUOUS"
    else:
        overall = "FAIL"
    return {
        "hypothesis": "H-STRAT-2",
        "row": row,
        "verdict": overall,
        "statistic": "conversion (baseline lost -> law captured), orbit vs "
        f"non-orbit, orbit = >= {ORBIT_RUN_STEPS} visible law steps and no capture",
        "n_pass": n_pass,
        "n_vacuous": n_vacuous,
        "pass_min": PASS_MIN,
        "n_opponents": len(OPPONENTS),
        "opponents": per,
    }


def h_strat_2_round(records: dict) -> dict:
    """Every law's H-STRAT-2 verdict.

    There is no round-level PASS for H-STRAT-2: prereg section 4 states it "per
    law that passes H-STRAT-1, and reported for all", so the container's own
    verdict says only whether the laws could be evaluated. `direction()` reads
    the per-law verdicts and names the laws where H-STRAT-1 passed and this did
    not."""
    laws = {row: h_strat_2(records, row) for row in TREATED}
    incomplete = any(b["verdict"] == "INCOMPLETE" for b in laws.values())
    return {
        "hypothesis": "H-STRAT-2",
        "verdict": "INCOMPLETE" if incomplete else "REPORTED",
        "rule": (
            "gating only the laws that pass H-STRAT-1; reported for all three "
            "(prereg section 4)"
        ),
        "laws": laws,
    }


def h_safe(records: dict) -> dict:
    """H-SAFE: 0 pursuer contact steps in all 12 cells. Every cell of this round
    is shielded, so there is no unshielded column to exclude."""
    have = {n: records[n] for n in CELL_ORDER if n in records}
    offenders = sorted(n for n, r in have.items() if int(r.get("violation_steps", 0)))
    steps = sum(int(r.get("violation_steps", 0)) for r in have.values())
    missing = _missing(records)
    return {
        "hypothesis": "H-SAFE",
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "violation_steps": int(steps),
        "offenders": offenders,
        "gated_cells": list(CELL_ORDER),
        "missing": missing,
    }


def direction(v: dict) -> str:
    """Prereg section 4's refutation paragraph, in the order of precedence the
    round's logic imposes: H-SAFE (the filter's guarantee is the precondition of
    every other number), then the NULL (if the laws are not `PLAN` in open
    space, no verdict here is about the aim point), then completeness, then
    H-STRAT-1 and H-STRAT-2."""
    safe, null = v["h_safe"], v["null_identity"]
    h1, h2 = v["h_strat_1"], v["h_strat_2"]
    if safe["verdict"] == "FAILS":
        return (
            f"H-SAFE FAILS: {safe['violation_steps']} pursuer contact step(s) in "
            f"{', '.join(safe['offenders'])}. Stop — the filter's guarantee is the "
            "precondition of the whole round and nothing else is read."
        )
    if null is not None and null["verdict"] == "FAILS":
        rows = ", ".join(null.get("failing_rows") or ["?"])
        return (
            f"NULL FAILS on {rows}: {null['n_differing']} obstacle-free "
            f"episode(s) of {null['n_episodes']} differ from PLAN; first "
            f"{null['differing_seeds']}. The treated law is a DIFFERENT law "
            "against a stationary evader, so NO VERDICT in this round is about "
            "the aim point (prereg section 4). Report and stop; fix the "
            "delegation before any capture number here means anything."
        )
    if "INCOMPLETE" in (h1["verdict"], h2["verdict"], safe["verdict"]):
        return (
            f"incomplete: {len(v['missing'])} of {v['n_cells']} cells have no "
            "record. Every row x opponent must run on the one stream before a "
            "verdict is read (prereg section 6: a crashed cell is UNDETERMINED, "
            "never re-run after a number is seen)."
        )
    if null is None:
        return (
            "the NULL has not been evaluated in this summary, so nothing below "
            "is yet licensed to be about the aim point. Re-run --summarize."
        )
    parts = []
    for row in TREATED:
        b1 = h1["laws"][row]
        dirs = ", ".join(
            f"{o} {x['direction']} ({x['delta']:+.3f}, p = {x['p']:.3g})"
            for o, x in b1["opponents"].items()
        )
        parts.append(
            f"{row}: H-STRAT-1 {b1['verdict']} "
            f"({b1['n_pass']}/{b1['n_opponents']} opponents; {dirs})."
        )
    if h1["verdict"] != "PASS":
        parts.append(
            "ALL THREE LAWS FAIL H-STRAT-1: pursuit strategy is not the "
            "constraint either. With contact forbidden and a 1.2x faster evader, "
            "the shield's task cost is IRREDUCIBLE BY THE NOMINAL CONTROLLER at "
            "this horizon and these strategies — reported as a bound, not as a "
            "null result about the filter; the agent-intelligence line stops "
            "here (prereg section 4's refutation)."
        )
    else:
        parts.append(
            f"The round PASSES on {', '.join(h1['passing_laws'])}: aiming where "
            "the evader WILL be captures more under the shield than aiming where "
            "it is, so the nominal controller's aim point WAS a binding "
            "constraint on the filtered pursuer."
        )
    for row in TREATED:
        b2 = h2["laws"][row]
        rates = ", ".join(
            f"{o} orbit {_fmt(x.get('conversion_orbit'))} vs non-orbit "
            f"{_fmt(x.get('conversion_non_orbit'))} "
            f"(n {x.get('n_orbit')}/{x.get('n_non_orbit_lost')})"
            for o, x in b2["opponents"].items()
        )
        parts.append(f"{row}: H-STRAT-2 {b2['verdict']} ({rates}).")
    broken = [row for row in h1["passing_laws"] if h2["laws"][row]["verdict"] != "PASS"]
    if broken:
        parts.append(
            f"H-STRAT-1 passes but H-STRAT-2 fails on {', '.join(broken)}: the "
            "gain is REAL BUT NOT THE ORBIT MECHANISM — it is a gain from "
            "something else, reported as such, and the mechanism question stays "
            "open (prereg section 4)."
        )
    parts.append(
        "CG-S2's pre-registered expectation, attached to every MPC-2s number: "
        "the constant-velocity 2-s prediction error's median is 2.48 m on HUG "
        "and 2.43 m on ATK against H*dt*v = 2.40 m, so inertia carries no "
        "information at 2 s there; a gain on GREEDY (median 0.82 m) is expected "
        "and a gain on HUG or ATK would be DESPITE the model and needs "
        "explaining, not celebrating (prereg section 5)."
    )
    parts.append(
        "Scope note (S19 A1.2): every intervention number in this round is the "
        "ALL-STEP scope (the recorder's `intervention_total`), not S18's "
        "first-decision-onward `intervention.total`."
    )
    return " ".join(parts)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def verdicts(records: dict, null: dict | None = None) -> dict:
    """Every verdict of the round — a pure function of the records and the NULL
    block, so a test can state a round rather than run one."""
    missing = _missing(records)
    return {
        "h_strat_1": h_strat_1_round(records),
        "h_strat_2": h_strat_2_round(records),
        "h_safe": h_safe(records),
        # The counts, not the 500-long masks: `h_strat_2` recomputes the mask
        # from the baseline record, so the round blob never carries a second
        # copy of it that could drift.
        "orbit": {
            o: {
                k: val
                for k, val in orbit_mask(records, o).items()
                if k not in ("mask", "lost")
            }
            for o in OPPONENTS
        },
        "null_identity": null,
        "complete": not missing,
        "missing": missing,
        "n_cells": len(CELLS),
        "n_records": len(records),
        "rows": list(ROWS),
        "col": COL,
        "opponents": list(OPPONENTS),
        "seed0": GATE_SEED0,
        "n_episodes": N_EPISODES,
        "spec": SPEC,
    }


# ----------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------
def write_figure(records: dict, path: pathlib.Path, null: dict | None = None) -> None:
    """Two panels: the capture rate per row, and H-STRAT-2's two conversion
    rates per treated law. Rows are NEVER parsed out of a cell name (`LEAD-0.5`
    contains the separator), so the bars are built from `CELLS`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(OPPONENTS))
    width = 0.8 / len(ROWS)
    for i, row in enumerate(ROWS):
        vals = [
            np.nan if (r := _rec(records, row, o)) is None else r["capture_rate"]
            for o in OPPONENTS
        ]
        axes[0].bar(xs + i * width - 0.4 + width / 2, vals, width, label=row)
    axes[0].set_ylabel("capture rate", fontsize=8)
    axes[0].set_title("H-STRAT-1: shielded capture per law", fontsize=9)
    h2 = {row: h_strat_2(records, row) for row in TREATED}
    width2 = 0.8 / (2 * len(TREATED))
    for i, row in enumerate(TREATED):
        for j, key in enumerate(("conversion_orbit", "conversion_non_orbit")):
            vals = [
                (
                    h2[row]["opponents"][o].get(key)
                    if o in h2[row]["opponents"]
                    else None
                )
                for o in OPPONENTS
            ]
            axes[1].bar(
                xs + (2 * i + j) * width2 - 0.4 + width2 / 2,
                [np.nan if v is None else v for v in vals],
                width2,
                label=f"{row} {'orbit' if j == 0 else 'non-orbit'}",
            )
    axes[1].set_ylabel("conversion (baseline lost -> law captured)", fontsize=8)
    axes[1].set_title(
        f"H-STRAT-2: orbit (>= {ORBIT_RUN_STEPS} visible law steps) vs not",
        fontsize=9,
    )
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(list(OPPONENTS), fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(axis="y", alpha=0.3)
    null_line = "NULL not evaluated" if null is None else f"NULL {null['verdict']}"
    fig.suptitle(
        f"S23 — pursuit strategy (CBFShield defaults, hold {HOLD}, "
        f"shield_obs/escape_obs on; N = {N_EPISODES} seed-paired episodes from "
        f"{GATE_SEED0}). Intervention scope: all steps. {null_line}."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Per-episode arrays stay in the cell records; the round blob keeps the
# aggregates so it is readable.
_BULK_KEYS = (
    "capture_ind",
    "steps",
    "violations",
    "intervention_total",
    "longest_visible_law_run",
    "orbit_ind",
    "n_aimed",
    "n_zero_feasible_exact",
    "n_law_exposed_steps",
    "n_zero_feasible",
    "n_chose_direct",
    "branch",
)


def recorded_cgb(records: dict) -> dict | None:
    """CG-B's tally as the CELLS recorded it.

    The gate runs in `run_cells`, before the first episode, and its tally is
    stamped into every cell record's provenance. `summarize` does not re-run it
    (that would replay 20 S18 episodes to write a summary) and must not report
    `null` either, so it reads the tally the measured cells were taken under.
    """
    for rec in records.values():
        cgb = (rec.get("provenance") or {}).get("cg_b")
        if cgb:
            return cgb
    return None


def summarize(device: str = "cpu") -> dict:
    """The round's only reader of numbers, run after `--gifs` (CG-S3)."""
    records = load_records()
    _measurable, rejected, _by_cell = scan_records()
    null = null_identity(device=device)
    _write_json(null_file(), null)
    v = verdicts(records, null)
    blob = {
        "verdicts": v,
        "direction": direction(v),
        "rejected_records": rejected,
        "cells": {
            n: {k: val for k, val in r.items() if k not in _BULK_KEYS}
            for n, r in records.items()
        },
        "provenance": provenance(device, recorded_cgb(records)),
    }
    _write_json(round_file("json"), blob)
    write_figure(records, round_file("png"), null)
    return blob


# ----------------------------------------------------------------------------
# CG-S3 (prereg section 5): the watched episodes, BEFORE the table
# ----------------------------------------------------------------------------
# The seeds are a RULE, not a choice: the first three construct seeds on which
# `PLAN-SHIELD`'s law branch took its A* path for >= 1 step vs GREEDY, frozen by
# `scripts/s23_cg_s3_seeds.py` BEFORE any treated law ran. They are READ from
# that artefact and never recomputed, so `--gifs` cannot quietly choose another
# set; a missing artefact REFUSES. No caption carries a measured rate (A5.1):
# every number a frame shows is a property of THAT episode — the row and seed,
# the branch tag, the planner's running `n_unreachable`, the env's step and
# outcome, and for the treated laws the aim point asked for on that step.
GIF_OPPONENT = "GREEDY"  # the primary metric's opponent (prereg section 5)
GIF_EVERY = 4  # keep every 4th env step as a frame, as S14/S18/S19
GIF_N = 3  # section 5: three watched seeds


def read_cg_s3_seeds() -> tuple[int, ...] | None:
    """The frozen CG-S3 seeds, or None when the artefact is absent/unreadable."""
    path = cg_s3_file()
    if not path.exists():
        return None
    try:
        seeds = json.loads(path.read_text())["seeds"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return tuple(int(s) for s in seeds)


def require_cg_s3_seeds() -> tuple[int, ...]:
    seeds = read_cg_s3_seeds()
    if seeds is None or len(seeds) != GIF_N:
        raise SystemExit(
            f"CG-S3 REFUSES: {cg_s3_file()} must exist and carry exactly "
            f"{GIF_N} seeds (got {seeds!r}). The watched episodes are the ones "
            "the pre-registered scan selected before any treated law ran "
            "(prereg section 5); this script never chooses them."
        )
    return seeds


def gif_cell(row: str) -> str:
    """The record a row's GIFs are gated on: its GREEDY cell."""
    return cell_name(row, COL, GIF_OPPONENT)


def gif_title(row: str, seed: int) -> str:
    """The caption: the ROW and the episode SEED, and nothing else (A5.1)."""
    return f"{row} · episode {seed}"


class AimOverlay:
    """The law's current aim point as a small marker, read off the law object at
    draw time — inside the step the frame shows, so the marker is that step's
    decision (`LEAD`'s lead point, `MPC`'s predicted evader position at the
    chosen step). `PLAN` exposes none and the marker is simply absent there.
    """

    def __init__(self, law):
        self.law = law

    def __call__(self, ax, env) -> None:
        aim = getattr(self.law, "last_aim", None)
        if aim is None:
            return
        ax.plot(
            [float(aim[0])],
            [float(aim[1])],
            marker="x",
            markersize=7,
            markeredgewidth=1.6,
            color="#d62728",
            zorder=6,
        )


class CellRecorder(S19.CellRecorder):
    """S19's CG-C recorder plus the aim-point marker. Subclassed, not copied: the
    FOV cone, the nominal-vs-executed arcs, the branch-tagged caption, the frame
    cadence and the one-figure-per-episode discipline are S19's, unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ov.append(AimOverlay(self.law))


def render_row(
    row: str, seed: int, title: str, device: str = "cpu", every: int = GIF_EVERY
) -> tuple[list, dict]:
    """One construct-stream episode of this row's GREEDY cell, as frames. Same
    law, opponent, hold, shield and observation flags as `run_cell` — only the
    recording differs."""
    spec = CELLS[gif_cell(row)]
    cfg = cell_config()
    rec = CellRecorder(
        cfg,
        opponent_factory(spec["opponent"], cfg, device),
        label=title,
        every=every,
        hold=spec["hold"],
        shield=build_shield(cfg),
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,
        law=row_law(spec["row"], cfg),
    )
    log = rec.play_episode(OrbitSearch(cfg).predict, seed)
    rec.finish(log["outcome"], log["steps"])
    return rec.frames, log


def write_gifs(
    rows: list[str] | None = None,
    device: str = "cpu",
    every: int = GIF_EVERY,
    fps: float = 15.0,
) -> list[tuple[str, int]]:
    """CG-S3: the three frozen seeds per row, vs GREEDY only — 12 GIFs.

    Every precondition is resolved before the first frame — the seeds artefact
    and each row's measurable record — because a pass that renders two rows and
    then refuses leaves a half-watched gate, and that is how a construct error
    survives (D125)."""
    from pe2.render import save_gif

    wanted = list(ROWS) if not rows else list(rows)
    unknown = [n for n in wanted if n not in ROWS]
    if unknown:
        raise SystemExit(
            f"unknown row(s): {unknown}; this round's rows are {tuple(ROWS)}"
        )
    seeds = require_cg_s3_seeds()
    todo = [
        (row, seed)
        for row in wanted
        for seed in seeds
        if not gif_path(row, seed).exists()
    ]
    for row in dict.fromkeys(r for r, _ in todo):
        read_cell(gif_cell(row))
    n_all = len(wanted) * len(seeds)
    print(
        f"{len(todo)}/{n_all} GIFs to render ({n_all - len(todo)} already exist), "
        f"the {len(seeds)} frozen CG-S3 seeds {seeds} per row vs {GIF_OPPONENT}. "
        "Watch for MPC wandering, LEAD overshooting the aim into a wall, and "
        "whether any law breaks an orbit or only corners.",
        flush=True,
    )
    gifs_dir().mkdir(parents=True, exist_ok=True)
    for row, seed in todo:
        frames, log = render_row(row, seed, gif_title(row, seed), device, every)
        path = gif_path(row, seed)
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
    # CG-S3 is a BLOCKING construct gate: the episodes are watched before any
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
                    "h_strat_1": {
                        "verdict": v["h_strat_1"]["verdict"],
                        "passing_laws": v["h_strat_1"]["passing_laws"],
                        "laws": {
                            r: b["verdict"] for r, b in v["h_strat_1"]["laws"].items()
                        },
                    },
                    "h_strat_2": {
                        r: b["verdict"] for r, b in v["h_strat_2"]["laws"].items()
                    },
                },
                indent=2,
            )
        )
        print(json.dumps(v["h_safe"], indent=2))
        print(
            json.dumps({o: b.get("n_orbit") for o, b in v["orbit"].items()}, indent=2)
        )
        print(json.dumps(v["null_identity"], indent=2))
        print(f"\n{blob['direction']}")
        print(
            f"\ncells {v['n_records']}/{v['n_cells']}"
            f"{'' if v['complete'] else ' — INCOMPLETE'}"
        )


if __name__ == "__main__":
    main()
