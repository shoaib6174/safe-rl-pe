"""S23 CG-S2 — is the constant-velocity opponent model a model?

Spec: `docs/pathway/pursuit_strategy_plan_2026-10-12.md` sections 2 and 4
(CG-S2), with amendment A2 note (iii).

`MPC-H` predicts the evader by inertia alone: position plus H = 2 s of the
velocity differenced from two consecutive sightings, clipped into the inset
arena (`MpcInterceptLaw._rollout`). That model is itself a construct, so it is
measured BEFORE the round's pre-registration is written: on the S19 gate stream,
per opponent, how far the prediction is from where the evader actually is two
seconds later, in metres, beside the 0.30 m capture radius.

What is measured, and on which trajectory
-----------------------------------------
The trajectory is `PLAN-SHIELD`'s -- the round's reference cell, built by the
S19 runner's own `build_cell_env` so the law, the shield, the hold, the
observation flags, the opponent and the seed stream are that cell's and nothing
is restated here. Two things differ from the cell, both passive:

- the recorder class is a subclass that also logs the evader's TRUE world
  position at every env step index (`ModelErrorMDP`), which is the quantity the
  prediction is compared against;
- the law is `VelocityProbeLaw`, which computes `MPC`'s velocity estimate and
  then returns `PlannerInterceptLaw`'s action, unchanged, through the
  grandparent's own code path -- so the episode is `PLAN`'s bit for bit while
  the estimate the model would have used is observable
  (`tests/pe2/test_s23_cg.py` asserts that identity on paired seeds).

Measuring the model on `PLAN`'s trajectory rather than on `MPC`'s is the point:
the question is whether the model is a model of the evader, which must not be
asked on the states one particular planner steers into.

No bar. This check gates INTERPRETATION only: its reading is printed above the
table and was written before the numbers existed.

Usage:
  ./venv/bin/python scripts/s23_cg_s2_model_error.py                 (all three)
  ./venv/bin/python scripts/s23_cg_s2_model_error.py --opponent ATK   (one, for
      three background processes; each writes its own part file)
  ./venv/bin/python scripts/s23_cg_s2_model_error.py --summarize
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 7; set BEFORE numpy / torch build their pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # so `scripts.` resolves when run as a script
    sys.path.insert(0, str(_REPO))

import pe2

# The round's reference cell, its stream and its N, imported rather than
# restated: this check must be measured on the cell the round compares against.
import scripts.derisk_s19_planner as S19
from pe2.config import EnvConfig
from pe2.planner import LeadPlannerInterceptLaw, PlannerInterceptLaw
from pe2.recorder import BranchTaggedSearchMDP
from pe2.structured import OrbitSearch, opponent_world_from_obs

PLAN = "docs/pathway/pursuit_strategy_plan_2026-10-12.md"
OUT = pathlib.Path("results/derisk") / "s23_cg_s2.json"
PARTS = pathlib.Path("results/derisk") / "s23_cg_s2_parts"

# --- fixed by the plan (sections 1, 2, 4) and by S19's frozen cell. ----------
ROW, COL = "PLAN", "SHIELD"
OPPONENTS: tuple[str, ...] = ("GREEDY", "HUG", "ATK")
HORIZON_S = 2.0  # H, the plan's section 1 horizon
H_STEPS = 40  # = HORIZON_S / dt, asserted against the config in `run_opponent`

READING = """\
From docs/pathway/pursuit_strategy_plan_2026-10-12.md section 4 (CG-S2),
verbatim:

  CG-S2 (the model is a model): on the S19 gate stream, per opponent, the
  distribution of constant-velocity prediction error at H = 2 s against the
  evader's actual position. Reported as a median and p90 in metres beside the
  0.30 m capture radius. If the median exceeds the radius by an order of
  magnitude, MPC-H is planning against fiction and its result is read as such.

The reading as fixed before this ran, verbatim:

  If the median 2-s prediction error exceeds the capture radius by an order of
  magnitude (>= 3 m), MPC-H is planning against fiction and its result is read
  as such."""


# ----------------------------------------------------------------------------
# the law: PLAN's action, MPC's velocity estimate
# ----------------------------------------------------------------------------
class VelocityProbeLaw(LeadPlannerInterceptLaw):
    """`PlannerInterceptLaw` with `LEAD`/`MPC`'s velocity estimate computed
    alongside and NEVER USED.

    `LeadPlannerInterceptLaw.act` is not called. This `act` takes the inherited
    step index and velocity estimate -- the same two calls, in the same order,
    that `MpcInterceptLaw.act` makes -- and then returns
    `PlannerInterceptLaw.act(self, obs)`, the grandparent's own code path on the
    same state, so the commanded action is `PLAN`'s to the last bit.

    `last_pos` is the decoded evader position this estimate belongs to, or None
    when the evader was not visible on this call; `last_aim` stays None because
    `PLAN` aims at no point of its own, which is also what CG-S4's recorder
    should log for the `PLAN` row.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        super().__init__(cfg, tau=0.0)  # `tau` is never read: no aim point
        self.last_pos: np.ndarray | None = None

    def reset(self, env=None) -> None:
        super().reset(env)
        self.last_pos = None

    def act(self, obs: np.ndarray) -> np.ndarray | None:
        step = self._step_index()
        if obs[6] > 0.5:
            self.last_pos = opponent_world_from_obs(obs, self.cfg)
            self.last_v_est = self._velocity(self.last_pos, step)
        else:
            self.last_pos = None
            self.last_v_est = np.zeros(2)
        self.last_aim = None
        return PlannerInterceptLaw.act(self, obs)


# ----------------------------------------------------------------------------
# the env: the cell's recorder plus the evader's true trajectory
# ----------------------------------------------------------------------------
class ModelErrorMDP(BranchTaggedSearchMDP):
    """The cell's recorder, also logging the evader's TRUE world position at
    every env step index and the law steps that carry a nonzero estimate.

    `_step_inner` is the recorder's per-env-step hook and this override is
    passive: it reads `self.inner.env`'s poses before and after the base class's
    step and touches no dynamics, so the trajectory is the cell's own. The
    position is keyed by `env.current_step`, which is the index the observation
    the action was computed from describes; recording it after the step as well
    puts the episode's terminal pose in the log.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evader_xy: dict[int, list[float]] = {}
        self.samples: list[dict] = []
        self.n_visible_law_steps = 0

    def _begin(self, seed: int | None) -> np.ndarray:
        obs = super()._begin(seed)
        self.evader_xy = {}
        self.samples = []
        self.n_visible_law_steps = 0
        return obs

    def _step_inner(self, action, branch: str = "law"):
        env = self.inner.env
        law = self.law
        self.evader_xy[int(env.current_step)] = [float(x) for x in env.evader_pose[:2]]
        # The law's `last_*` describe its LAST call, which is this step exactly
        # when this step is a law step (`SearchMDP` does not consult the law
        # inside a learner's hold span).
        if branch == "law" and law.last_pos is not None:
            self.n_visible_law_steps += 1
            if np.any(law.last_v_est):
                self.samples.append(
                    {
                        "step": int(env.current_step),
                        "pos": [float(x) for x in law.last_pos],
                        "v_est": [float(x) for x in law.last_v_est],
                    }
                )
        out = super()._step_inner(action, branch)
        self.evader_xy[int(env.current_step)] = [float(x) for x in env.evader_pose[:2]]
        return out


@contextlib.contextmanager
def _recorder_class(cls):
    """`S19.build_cell_env` instantiates `derisk_s19_planner`'s
    `BranchTaggedSearchMDP` by name; inside this block that name is `cls`.

    The cell's construction then stays single-sourced in the S19 runner and only
    the recording differs -- the discipline S19's own `CellRecorder` follows for
    CG-C. Restating the constructor here instead would let this check drift away
    from the cell it claims to measure.
    """
    original = S19.BranchTaggedSearchMDP
    S19.BranchTaggedSearchMDP = cls
    try:
        yield
    finally:
        S19.BranchTaggedSearchMDP = original


def build_probe_env(
    opponent: str, cfg: EnvConfig, device: str = "cpu"
) -> ModelErrorMDP:
    """S19's `PLAN-SHIELD` cell against `opponent`, with the position log and
    the velocity probe installed and nothing else changed."""
    cell = S19.cell_name(ROW, COL, opponent)
    with _recorder_class(ModelErrorMDP):
        mdp = S19.build_cell_env(cell, cfg, device)
    if type(mdp.law) is not PlannerInterceptLaw:
        raise SystemExit(
            f"the {cell} cell's law is {type(mdp.law).__name__}, not "
            "PlannerInterceptLaw; the velocity probe is only PLAN's law when it "
            "replaces PLAN's law, so this check refuses to run."
        )
    mdp.law = VelocityProbeLaw(cfg)
    return mdp


# ----------------------------------------------------------------------------
# the model error
# ----------------------------------------------------------------------------
def predict_xy(pos, v_est, cfg: EnvConfig, h_steps: int = H_STEPS) -> np.ndarray:
    """Where the model says the evader will be `h_steps` env steps from now:
    constant velocity, clipped into the inset arena exactly as
    `MpcInterceptLaw._rollout` clips its predicted evader path."""
    lim = cfg.arena_size / 2.0 - cfg.agent_radius
    return np.clip(
        np.asarray(pos, dtype=float)
        + h_steps * cfg.dt * np.asarray(v_est, dtype=float),
        -lim,
        lim,
    )


def prediction_errors(
    samples: list[dict], evader_xy: dict, cfg: EnvConfig, h_steps: int = H_STEPS
) -> tuple[list[float], int]:
    """Per sample, the distance in metres between the prediction and the
    evader's TRUE position `h_steps` later, plus the number of samples whose
    horizon ran past the end of the episode.

    A sample with no true position to compare against yields NO error -- not a
    zero and not a clipped one; the count is reported so the distribution's
    denominator is readable.
    """
    errors: list[float] = []
    dropped = 0
    for s in samples:
        true = evader_xy.get(int(s["step"]) + h_steps)
        if true is None:
            dropped += 1
            continue
        pred = predict_xy(s["pos"], s["v_est"], cfg, h_steps)
        errors.append(float(np.hypot(*(np.asarray(true, dtype=float) - pred))))
    return errors, dropped


def error_stats(errors: list[float]) -> dict:
    """The distribution as the plan reports it (median and p90), with the max
    and the mean beside them. `S19._q` is the round's own percentile."""
    return {
        "n": len(errors),
        "median": float(statistics.median(errors)) if errors else float("nan"),
        "p90": S19._q(errors, 0.90),
        "p99": S19._q(errors, 0.99),
        "max": max(errors, default=float("nan")),
        "mean": float(np.mean(errors)) if errors else float("nan"),
    }


# ----------------------------------------------------------------------------
# one opponent
# ----------------------------------------------------------------------------
def part_file(opponent: str) -> pathlib.Path:
    return PARTS / f"{opponent}.json"


def run_opponent(opponent: str, n: int | None = None, device: str = "cpu") -> dict:
    """`PLAN-SHIELD` vs `opponent`, N paired episodes on the S19 gate stream.

    The shield is rebuilt per episode exactly as `S19.run_cell` rebuilds it.
    """
    cfg = S19.cell_config()
    if round(HORIZON_S / cfg.dt) != H_STEPS:
        raise SystemExit(
            f"H = {HORIZON_S} s is {round(HORIZON_S / cfg.dt)} steps at dt = "
            f"{cfg.dt}, not the {H_STEPS} this check and the law are written for"
        )
    n = S19.N_EPISODES if n is None else int(n)
    mdp = build_probe_env(opponent, cfg, device)
    policy = OrbitSearch(cfg)

    errors: list[float] = []
    dropped = 0
    law_steps = visible_law_steps = n_nonzero = 0
    for i in range(n):
        mdp.shield = S19.build_shield(cfg)
        log = mdp.play_episode(policy.predict, S19.GATE_SEED0 + i)
        e, d = prediction_errors(mdp.samples, mdp.evader_xy, cfg)
        errors += e
        dropped += d
        n_nonzero += len(mdp.samples)
        visible_law_steps += mdp.n_visible_law_steps
        law_steps += int(log["branch"]["steps_by_branch"]["law"])
    record = {
        "opponent": opponent,
        "cell": S19.cell_name(ROW, COL, opponent),
        "n_episodes": n,
        "seed0": S19.GATE_SEED0,
        "h_steps": H_STEPS,
        "horizon_s": HORIZON_S,
        "capture_distance": cfg.capture_distance,
        "law": "PlannerInterceptLaw (via VelocityProbeLaw)",
        "error": error_stats(errors),
        "n_law_steps": law_steps,
        "n_visible_law_steps": visible_law_steps,
        "n_v_est_nonzero": n_nonzero,
        "frac_v_est_nonzero": (n_nonzero / law_steps) if law_steps else float("nan"),
        "n_horizon_past_episode_end": dropped,
        "provenance": provenance(device),
    }
    _write_json(part_file(opponent), record)
    return record


# ----------------------------------------------------------------------------
# provenance, output, table
# ----------------------------------------------------------------------------
def plan_path() -> pathlib.Path:
    return _REPO / PLAN


def provenance(device: str = "cpu") -> dict:
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "pe2_file": pe2.__file__,
        "spec": PLAN,
        "spec_sha256": hashlib.sha256(plan_path().read_bytes()).hexdigest(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "device": device,
    }


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def merge() -> dict:
    """The three part files into the one record, or a loud refusal."""
    missing = [o for o in OPPONENTS if not part_file(o).exists()]
    if missing:
        raise SystemExit(
            f"missing CG-S2 part files for {', '.join(missing)}; run "
            f"--opponent for each of {', '.join(OPPONENTS)} before --summarize"
        )
    parts = {o: json.loads(part_file(o).read_text()) for o in OPPONENTS}
    record = {
        "spec": PLAN,
        "spec_section": "2 and 4 (CG-S2)",
        "reading": READING,
        "bar": None,  # gates interpretation only
        "row": ROW,
        "col": COL,
        "h_steps": H_STEPS,
        "horizon_s": HORIZON_S,
        "capture_distance": EnvConfig().capture_distance,
        "n_episodes": parts[OPPONENTS[0]]["n_episodes"],
        "seed0": S19.GATE_SEED0,
        "opponents": parts,
        "provenance": provenance(),
    }
    _write_json(OUT, record)
    record["path"] = str(OUT)
    return record


def table(record: dict) -> str:
    """The reading VERBATIM, then the table: the number cannot be read without
    the reading it was fixed against (plan section 4)."""
    cap = record["capture_distance"]
    head = (
        f"{'opponent':9} {'median err':>11} {'p90':>9} {'max':>9} {'n':>9} "
        f"{'frac v_est != 0':>16}"
    )
    lines = [
        READING,
        "",
        (
            f"S23 step 4 — CG-S2: constant-velocity prediction error at "
            f"H = {record['horizon_s']} s ({record['h_steps']} steps), metres, on "
            f"{record['row']}-{record['col']}, N = {record['n_episodes']} paired "
            f"per opponent, seeds {record['seed0']}+"
        ),
        f"capture distance = {cap:.2f} m",
        "",
        head,
        "-" * len(head),
    ]
    for opponent in OPPONENTS:
        part = record["opponents"][opponent]
        e = part["error"]
        lines.append(
            f"{opponent:9} {e['median']:11.3f} {e['p90']:9.3f} {e['max']:9.3f} "
            f"{e['n']:9d} {part['frac_v_est_nonzero']:16.3f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--opponent",
        action="append",
        choices=OPPONENTS,
        help="run just this opponent (repeatable); default all three",
    )
    ap.add_argument("--n-episodes", type=int, default=None, help="override N (testing)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="merge the part files and print the reading and the table",
    )
    args = ap.parse_args(argv)
    if not args.summarize:
        for opponent in args.opponent or OPPONENTS:
            record = run_opponent(opponent, args.n_episodes, args.device)
            print(f"wrote {part_file(opponent)} ({record['error']['n']} samples)")
    if args.summarize or not args.opponent:
        record = merge()
        print(f"wrote {record['path']}")
        print(table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
