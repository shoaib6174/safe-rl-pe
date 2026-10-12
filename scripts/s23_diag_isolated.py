"""S23 step 2 — the isolated-obstacle diagnostic: does ANY pursuit strategy
capture a 1.2x faster evader orbiting a single obstacle far from every wall?

Plan: `docs/pathway/pursuit_strategy_plan_2026-10-12.md` section 3 (binding).

This is a DIAGNOSTIC, not a hypothesis. Its reading was written before it ran
(section 3: "its reading is written before it runs and it moves no bar") and is
printed VERBATIM above the table by `--summarize`/the run itself, so the number
is read against the reading rather than the reading rewritten around the number.

What is varied, and what is not
-------------------------------
Three rows, one variable: the aim point the law steers at while the evader is
visible.

    PLAN      `pe2.planner.PlannerInterceptLaw`  — the evader's position
    LEAD-0.5  `pe2.planner.LeadPlannerInterceptLaw(cfg, 0.5)`
    LEAD-1.0  `pe2.planner.LeadPlannerInterceptLaw(cfg, 1.0)`

Everything else is S19's and frozen: `OrbitSearch` as the search policy,
`CBFShield(cfg)` at its defaults, `hold = 10`, `shield_obs` / `escape_obs` on,
no reward wrapper (evaluation only), `GreedyEvader` as the opponent, and the
SAME 100 construct-stream seeds (2 090 000 + i) for all three rows, so the
comparison is paired.

The arena
---------
`EnvConfig(n_physical_obstacles=1)`: `n_obstacles` (2, the observation layout)
and `n_physical_obstacles` (1, the discs actually placed) do NOT have to agree —
that field exists precisely so a frozen observation layout can be evaluated in a
denser or emptier arena (`pe2/config.py`, S9). The layout therefore stays the
11-dim vector every prior round used, with the second obstacle slot zero-filled;
nothing in this diagnostic reads it (the planner reads `env.obstacles` directly
at `reset`, and `OrbitSearch` reads only own pose), so the choice is inert here
and keeps the law objects identical to the ones S19 measured.

The single obstacle is then INSTALLED over whatever `reset` sampled: centre
(0, 0), radius 0.8 m (inside `obstacle_radius_min/max`), the same three keys
`PursuitEvasionEnv._generate_obstacles` emits.

**Where the installation happens matters.** `SearchMDP._begin` calls
`inner.reset(...)` and then `law.reset(env=...)`; the law reads the static map in
that second call, so the obstacle must be in place before it. It is installed
one level deeper still — inside the env's own `reset`, before the first
observation of the episode is built — because the observation's occlusion and
visibility are computed FROM the obstacle list, and installing it after `reset`
returned would hand the first step an observation of a map that no longer
exists. `_begin` then ASSERTS the installed map (one disc, centre, radius) and
that the law saw exactly it.

Initial poses are whatever `reset(seed)` gives (plan section 3: not hand-placed)
and no episode is skipped or reseeded, including the ones whose start happens to
hide the evader. One consequence is recorded rather than engineered away: the
poses are rejection-sampled against the obstacle `reset` DREW, not against the
one installed over it, so an agent can start inside the central disc. The env's
own position projection pushes it out on the first step; the per-episode record
carries `pursuer_start_inside` / `evader_start_inside` so the frequency is
visible in the record instead of being invisible in the rate.

Usage:
  OMP_NUM_THREADS=4 ./venv/bin/python scripts/s23_diag_isolated.py
  ./venv/bin/python scripts/s23_diag_isolated.py --summarize   (re-print only)
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 7; set BEFORE numpy / torch build their pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
from collections import Counter

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # so `scripts.` resolves when run as a script
    sys.path.insert(0, str(_REPO))

import pe2
from pe2.config import EnvConfig
from pe2.planner import LeadPlannerInterceptLaw, PlannerInterceptLaw
from pe2.recorder import BranchTaggedSearchMDP
from pe2.scripted import GreedyEvader
from pe2.structured import HOLD, OrbitSearch

# The Wilson wrapper the S19 round's records carry, imported rather than
# re-derived so this summary's interval is the same arithmetic as that one's
# (it wraps `pe2.eval.wilson_ci`, the repo's only Wilson helper).
from scripts.derisk_s19_planner import _ci, build_shield

PLAN = "docs/pathway/pursuit_strategy_plan_2026-10-12.md"
OUT = pathlib.Path("results/derisk") / "s23_diag_isolated.json"

# --- fixed by the plan (section 3) and by S19's frozen configuration. --------
DIAG_SEED0 = 2_090_000  # the construct stream
N_EPISODES = 100  # plan section 3: "100 construct-stream episodes per law"
OBSTACLE: dict = {"x": 0.0, "y": 0.0, "radius": 0.8}
SHIELD_OBS = True
ESCAPE_OBS = True
LAWS: tuple[str, ...] = ("PLAN", "LEAD-0.5", "LEAD-1.0")

# The reading, VERBATIM. The first block is the plan's own section 3; the second
# is the quantified form fixed in writing before this script was run, which is
# what makes "no more than PLAN" and "substantially more" readable off a number.
READING = """\
From docs/pathway/pursuit_strategy_plan_2026-10-12.md section 3, verbatim:

  A faster evader orbiting an isolated obstacle far from any wall can, in
  principle, orbit forever; capture without contact then comes only from the
  arena walls cutting the orbit. Before the round:

  - D-cell: a single obstacle at the arena centre, GreedyEvader, 100
    construct-stream episodes per law. If no strategy captures here, any gain
    in the full round is wall-cornering and the paper says so. If LEAD or MPC
    captures here, the orbit itself is breakable and the mechanism is
    different.
  - This is a diagnostic, not a hypothesis; its reading is written before it
    runs and it moves no bar.

The reading as fixed before this ran, verbatim:

  Reading (written now, moves no bar): if LEAD-0.5 and LEAD-1.0 capture no more
  than PLAN here, any gain in the full round is wall-cornering and the paper
  says so, and MPC-H (Step 3) is still built. If LEAD captures substantially
  more than PLAN here (>= 0.10 absolute at N = 100), the orbit itself is
  breakable by lead pursuit and the orchestrator decides whether MPC-H is
  needed."""


# ----------------------------------------------------------------------------
# the env: S19's recorder on an arena holding exactly one isolated obstacle
# ----------------------------------------------------------------------------
class IsolatedObstacleMDP(BranchTaggedSearchMDP):
    """`BranchTaggedSearchMDP` whose every episode runs on the SAME single
    obstacle, plus the three per-episode quantities the diagnostic reads.

    `min_distance` is the minimum pursuer-evader CENTRE distance over the
    episode, from the env's true poses (the capture predicate's own quantity),
    including the reset pose so it is a minimum over the whole episode.

    `visible_flags[i]` is whether the evader was visible in the observation env
    step i ACTED ON — i.e. the observation the commanded action was computed
    from, which for a law step is the observation the law chased on. The lead
    fraction is counted over exactly those law steps, because `SearchMDP` does
    not consult the law during a learner's hold span and `law.last_v_est` would
    otherwise be read stale.
    """

    def __init__(self, *args, obstacle: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle = dict(OBSTACLE if obstacle is None else obstacle)
        self._patch_env_reset()
        self.min_distance = float("inf")
        self.visible_flags: list[bool] = []
        self.lead_visible_law_steps = 0
        self.lead_v_est_nonzero = 0
        self.start_inside = (False, False)

    # -- the installation ------------------------------------------------
    def _patch_env_reset(self) -> None:
        """Install the obstacle inside `PursuitEvasionEnv.reset`, after the
        poses are sampled and BEFORE the episode's first observation is built.

        The two lines after the real reset are the two the env's own `reset`
        ends with; they are repeated here rather than left to the caller
        because an observation built from the drawn map would describe an arena
        this diagnostic does not run in.
        """
        env = self.inner.env
        real = env.reset

        def reset(*, seed: int | None = None, options: dict | None = None):
            real(seed=seed, options=options)
            self.start_inside = tuple(
                bool(
                    np.hypot(pose[0] - self.obstacle["x"], pose[1] - self.obstacle["y"])
                    < self.obstacle["radius"] + self.cfg.agent_radius
                )
                for pose in (env.pursuer_pose, env.evader_pose)
            )
            env.obstacles = [dict(self.obstacle)]
            return env._get_obs(), env._build_info()

        env.reset = reset

    # -- per-episode bookkeeping -----------------------------------------
    def _begin(self, seed: int | None) -> np.ndarray:
        obs = super()._begin(seed)
        env = self.inner.env
        want = dict(self.obstacle)
        if env.obstacles != [want]:
            raise RuntimeError(
                f"the isolated obstacle is not installed: env.obstacles = "
                f"{env.obstacles!r}, expected exactly [{want!r}]"
            )
        if self.law.obstacles != [want]:
            raise RuntimeError(
                f"the law did not read the installed map at reset: "
                f"law.obstacles = {self.law.obstacles!r}, expected [{want!r}]"
            )
        self.min_distance = self._distance()
        self.visible_flags = []
        self.lead_visible_law_steps = 0
        self.lead_v_est_nonzero = 0
        return obs

    def _distance(self) -> float:
        env = self.inner.env
        return float(np.hypot(*(env.pursuer_pose[:2] - env.evader_pose[:2])))

    def _step_inner(self, action, branch: str = "law"):
        acted_visible = self._visible  # the obs THIS action was computed from
        if (
            branch == "law"
            and acted_visible
            and isinstance(self.law, LeadPlannerInterceptLaw)
        ):
            self.lead_visible_law_steps += 1
            self.lead_v_est_nonzero += int(bool(np.any(self.law.last_v_est)))
        out = super()._step_inner(action, branch)
        self.min_distance = min(self.min_distance, self._distance())
        self.visible_flags.append(bool(acted_visible))
        return out

    # -- the episode record ----------------------------------------------
    def episode_extras(self) -> dict:
        flags = self.visible_flags
        return {
            "min_distance": float(self.min_distance),
            "visible_steps": int(sum(flags)),
            "longest_visible_run": _longest_run(flags),
            "pursuer_start_inside": bool(self.start_inside[0]),
            "evader_start_inside": bool(self.start_inside[1]),
            "lead_visible_law_steps": int(self.lead_visible_law_steps),
            # None on PLAN: the law has no velocity estimate to report.
            "lead_v_est_nonzero_frac": (
                self.lead_v_est_nonzero / self.lead_visible_law_steps
                if self.lead_visible_law_steps
                else None
            ),
        }


def _longest_run(flags: list[bool]) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


# ----------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------
def diag_config() -> EnvConfig:
    """`EnvConfig()` defaults except the number of discs PLACED (see module
    docstring): the observation layout is untouched."""
    return EnvConfig(n_physical_obstacles=1)


def build_law(name: str, cfg: EnvConfig):
    if name == "PLAN":
        return PlannerInterceptLaw(cfg)
    if name.startswith("LEAD-"):
        return LeadPlannerInterceptLaw(cfg, float(name.split("-", 1)[1]))
    raise ValueError(f"unknown law {name!r}; expected one of {LAWS}")


def build_env(name: str, cfg: EnvConfig) -> IsolatedObstacleMDP:
    """The row's env. The ONLY difference between rows is `law`."""
    return IsolatedObstacleMDP(
        cfg,
        GreedyEvader,
        hold=HOLD,
        shield=build_shield(cfg),
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,  # evaluation only; `play_episode` discards rewards
        law=build_law(name, cfg),
    )


def seeds(n: int) -> list[int]:
    return [DIAG_SEED0 + i for i in range(n)]


def run_law(name: str, n: int = N_EPISODES) -> dict:
    """One row: `n` paired episodes on the construct stream.

    The shield is rebuilt per episode exactly as S19's `run_cell` rebuilds it;
    `CBFShield` is memoryless, so this is equivalent to one shield per row, but
    the equivalence is a property of the shield and not of the protocol.
    """
    cfg = diag_config()
    mdp = build_env(name, cfg)
    policy = OrbitSearch(cfg)
    episodes: list[dict] = []
    for seed in seeds(n):
        mdp.shield = build_shield(cfg)
        log = mdp.play_episode(policy.predict, seed)
        episodes.append(
            {
                "seed": log["seed"],
                "outcome": log["outcome"],
                "steps": log["steps"],
                "final_distance": log["final_distance"],
                "violations": int(mdp.episode_violations),
                "branch": log["branch"],
                **mdp.episode_extras(),
            }
        )
    return {
        "law": name,
        "law_class": type(mdp.law).__name__,
        "tau": getattr(mdp.law, "tau", None),
        "n_episodes": n,
        "seed0": DIAG_SEED0,
        "seeds": seeds(n),
        "hold": mdp.hold,
        "shield": "CBFShield",
        "shield_obs": SHIELD_OBS,
        "escape_obs": ESCAPE_OBS,
        "opponent": "GreedyEvader",
        "search": "OrbitSearch",
        "obstacle": dict(OBSTACLE),
        "n_unreachable": getattr(mdp.law, "n_unreachable", None),
        "episodes": episodes,
        "summary": summarize_law(name, episodes),
    }


def summarize_law(name: str, episodes: list[dict]) -> dict:
    n = len(episodes)
    k = sum(e["outcome"] == "capture" for e in episodes)
    mins = [e["min_distance"] for e in episodes]
    runs = [e["longest_visible_run"] for e in episodes]
    fracs = [
        e["lead_v_est_nonzero_frac"]
        for e in episodes
        if e["lead_v_est_nonzero_frac"] is not None
    ]
    return {
        "law": name,
        "capture": _ci(k, n),
        "outcomes": dict(Counter(e["outcome"] for e in episodes)),
        "min_distance_median": float(statistics.median(mins)) if mins else float("nan"),
        "min_distance_min": min(mins, default=float("nan")),
        "longest_visible_run_mean": float(np.mean(runs)) if runs else float("nan"),
        "visible_steps_mean": float(np.mean([e["visible_steps"] for e in episodes]))
        if episodes
        else float("nan"),
        "violations": int(sum(e["violations"] for e in episodes)),
        "episodes_with_a_violation": sum(e["violations"] > 0 for e in episodes),
        "steps_mean": float(np.mean([e["steps"] for e in episodes]))
        if episodes
        else float("nan"),
        "starts_inside_obstacle": sum(
            e["pursuer_start_inside"] or e["evader_start_inside"] for e in episodes
        ),
        # None on PLAN (no velocity estimate exists there).
        "lead_v_est_nonzero_frac_mean": float(np.mean(fracs)) if fracs else None,
    }


# ----------------------------------------------------------------------------
# provenance, output, table
# ----------------------------------------------------------------------------
def plan_path() -> pathlib.Path:
    return _REPO / PLAN


def provenance() -> dict:
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
    }


def table(record: dict) -> str:
    """The reading VERBATIM, then the table. The reading is printed first so the
    number cannot be read without it (plan section 3)."""
    head = (
        f"{'law':9} {'capture [95% Wilson]':38}"
        f"{'median min_d':>12} {'mean vis.run':>12} {'viol':>5}"
    )
    lines = [
        READING,
        "",
        (
            f"S23 step 2 — isolated obstacle {OBSTACLE}, "
            f"N = {record['n_episodes']} paired, seeds {record['seed0']}+"
        ),
        "",
        head,
        "-" * len(head),
    ]
    for row in record["rows"]:
        s = row["summary"]
        c = s["capture"]
        cap = (
            f"{c['rate']:.2f} ({c['k']}/{c['n']}) "
            f"[{c['wilson'][0]:.3f}, {c['wilson'][1]:.3f}]"
        )
        lines.append(
            f"{row['law']:9} {cap:38}"
            f"{s['min_distance_median']:12.3f} "
            f"{s['longest_visible_run_mean']:12.1f} "
            f"{s['violations']:5d}"
        )
    return "\n".join(lines)


def run(n: int = N_EPISODES, out: pathlib.Path | None = None) -> dict:
    record = {
        "spec": PLAN,
        "spec_section": "3 (the D-cell diagnostic)",
        "reading": READING,
        "n_episodes": n,
        "seed0": DIAG_SEED0,
        "obstacle": dict(OBSTACLE),
        "config": {
            "n_obstacles": diag_config().n_obstacles,
            "n_physical_obstacles": diag_config().n_physical_obstacles,
        },
        "rows": [run_law(name, n) for name in LAWS],
        "provenance": provenance(),
    }
    path = OUT if out is None else pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    record["path"] = str(path)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="re-print the reading and the table from the existing record",
    )
    args = ap.parse_args(argv)
    if args.summarize:
        record = json.loads(OUT.read_text())
    else:
        record = run()
        print(f"wrote {record['path']}")
    print(table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
