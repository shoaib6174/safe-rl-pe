"""S24.0 Step 4 — the labelled tactic game: watch it before anything learns.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` sections 1 ("v0, scripted ...
exists so the tactic game can be WATCHED before anything is learned") and 2
("S24.0 — the library, the memory, the recogniser, scripted selectors (no
learning, no hypothesis) ... Output: a tactic game that visibly plays").

THE S24.0 READING, printed verbatim above any table:

    S24.0 is a construct round: no hypothesis. The scripted game is reported at
    ratios 0.8 / 1.0 / 1.2 (capture rate, mean episode length, tactic usage per
    role) so that S24.1's bars are set against a watched baseline, not an
    assumed one.

The game, one episode:
- pursuer = `TacticLaw` over its five tactics with `PursuerRulesV0`, inside
  `BranchTaggedSearchMDP` at the S18/S19 configuration — `CBFShield(cfg)` at
  its asserted defaults, `hold` 10, `shield_obs`, `escape_obs`,
  `reward_wrapper=None` — carrying `LastSeenBelief` (D146: dead reckoning is
  S24's memory) and `TacticRecogniser("evader")` as its memory;
- evader = `TacticOpponent` over its five with `EvaderRulesV0`, carrying
  `TacticRecogniser("pursuer")` with the full-state view L1.5 gives every
  scripted evader and no belief (it is never the occluded observer).

Readings this module fixes, because the brief or the plan can be read two ways:

1. **Labels: none (D147/D148).** Both recogniser gates failed, so neither rule
   table consumes a LABEL; both are written over `features()`. The verdict is
   re-read from `results/derisk/s24_recogniser_gate.json` at import and the
   printed header states the active branch, and this script REFUSES to run if
   that record ever reads PASS — the rule tables would then be out of date.
2. **The memories are driven once per env step, at the END of the step**, by a
   mixin over the MDP rather than from inside a selector (a selector is called
   only at decision points, while dead reckoning and the 40-step feature window
   need every step). Post-step poses are the pre-step poses of the next step, so
   the window a selector reads at a decision point ends exactly at the state it
   is deciding from — no staleness, and the same convention the Step 3 gate used
   (`scripts/s24_recogniser_gate.py`: observe after `env.step`, truth = the
   tactic that produced that step's action).
3. **Recogniser accuracy is reported per episode against the generating
   tactic's FAMILY**, with UNKNOWN steps excluded from the accuracy and counted
   separately — the Step 3 gate's definition, so the numbers are comparable to
   it. It gates nothing here and is consumed by no rule.
4. **`TacticLaw` can return None** — when the tactic it selects is DONE on its
   first call of a step — and `SearchMDP` then falls to the search policy
   (`OrbitSearch`, which is `SWEEP`'s own primitive, as in S19/S23). Those steps
   are driven by no tactic, so they are counted as `(search policy)` in the
   usage table and excluded from the pursuer-truth scoring rather than
   attributed to the stale `current`.
5. **Tactic usage is per ENV STEP and by FAMILY** (`ORBIT(0)` and `ORBIT(1)`
   are both `ORBIT`): the parameter is an obstacle index and the table reads as
   the plan's repertoire. The full span logs, with instance names, are in the
   JSON per episode.

Usage:
  ./venv/bin/python scripts/s24_tactic_game.py --episodes 100 --seed0 2290000
  ./venv/bin/python scripts/s24_tactic_game.py --gifs --episodes 3 --ratio 1.2
  ./venv/bin/python scripts/s24_tactic_game.py --summarize
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 4; BEFORE numpy builds its pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys
import time
from collections import Counter

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pe2
from pe2.config import EnvConfig
from pe2.recogniser import UNKNOWN, TacticRecogniser, tactic_family
from pe2.recorder import BranchTaggedSearchMDP
from pe2.selectors import EvaderRulesV0, PursuerRulesV0
from pe2.shield import CBFShield
from pe2.structured import HOLD, OrbitSearch
from pe2.tactics import BELIEF_FRESH_STEPS, TacticLaw, TacticOpponent

PLAN = "docs/pathway/tactics_plan_2026-10-13.md"
OUT = pathlib.Path("results/derisk")
RECORD = OUT / "s24_tactic_game.json"
GIFS = OUT / "s24_gifs"
GATE_RECORD = OUT / "s24_recogniser_gate.json"

# --- fixed here before the run. Do not tune. -------------------------------
CONSTRUCT_SEED0 = 2_290_000  # this step's construct stream
N_EPISODES = 100
RATIOS = (0.8, 1.0, 1.2)
GIF_RATIO = 1.2  # the ratio every S24 baseline exists at (plan section 1)
GIF_N = 3  # three watched episodes, not nine
GIF_EVERY = 4  # keep every 4th env step as a frame, as S14/S18/S19/S23
SHIELD_OBS = True  # S19's `SHIELD_OBS`
ESCAPE_OBS = True  # S19's `ESCAPE_OBS`

READING = (
    "S24.0 is a construct round: no hypothesis. The scripted game is reported "
    "at ratios 0.8 / 1.0 / 1.2 (capture rate, mean episode length, tactic "
    "usage per role) so that S24.1's bars are set against a watched baseline, "
    "not an assumed one."
)
SEARCH_KEY = "(search policy)"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# reading 1: the label branch, re-read from the gate's own record
# ----------------------------------------------------------------------------
def label_branch() -> str:
    """The active branch of the brief's conditional, read off the Step 3 gate's
    record. D147/D148: both gates FAILED, so no selector consumes labels."""
    if not GATE_RECORD.exists():
        raise SystemExit(
            f"{GATE_RECORD} is missing: this step's rule tables are written on "
            "the ruling that the recogniser gates FAILED (D147/D148) and that "
            "record is the evidence."
        )
    verdict = json.loads(GATE_RECORD.read_text())["gate_2"]["verdict"]
    passed = {role: bool(v.get("passed")) for role, v in verdict.items()}
    if any(passed.values()):
        raise SystemExit(
            f"{GATE_RECORD} now reads PASS {passed}; `pe2/selectors.py` is "
            "written over FEATURES ONLY on the D147/D148 rulings and would be "
            "out of date. Re-register the branch before running this."
        )
    return "labels: none (D147/D148)"


# ----------------------------------------------------------------------------
# the game
# ----------------------------------------------------------------------------
def build_shield(cfg: EnvConfig) -> CBFShield:
    """`CBFShield(cfg)` at its defaults, ASSERTED rather than restated as
    keyword arguments (S19's `build_shield`, same reason: the round must fail
    loudly if the library's defaults move)."""
    shield = CBFShield(cfg)
    if (shield.margin, shield.gamma) != (0.0, 0.2):
        raise SystemExit(
            f"CBFShield defaults moved to margin {shield.margin}, gamma "
            f"{shield.gamma}; S18/S19/S23 and this step run at 0.0 / 0.2."
        )
    return shield


class _TacticGameMemory:
    """Mixin over the MDP: drives both memories and scores both recognisers
    once per env step (reading 2), and keeps the per-episode record of what
    each role was doing.

    A mixin so the measured env and the RECORDED one are the same object apart
    from the recording — `TacticGameMDP` over `BranchTaggedSearchMDP` and
    `TacticGameRecorder` over S19's `CellRecorder` — which is CG-C's discipline
    and the reason the GIFs show the episodes the table counts.
    """

    def __init__(self, cfg, opponent_factory, *, evader, rules, **kw):
        super().__init__(cfg, opponent_factory, **kw)
        self.evader = evader  # the `TacticOpponent` adapter (shared instance)
        self.rules = rules  # {"evader": EvaderRulesV0, "pursuer": PursuerRulesV0}
        self._clear_record()

    def _clear_record(self) -> None:
        self.usage = {"evader": Counter(), "pursuer": Counter()}
        # (label, truth) per step, per RECOGNISED role.
        self.recognised: dict[str, list] = {"evader": [], "pursuer": []}
        self.search_steps = 0

    # -- the memory of the role whose OPPONENT is `role` ------------------
    def memory_of_watcher(self, role: str):
        """The memory that watches `role`: the pursuer's watches the evader."""
        return self.rules["pursuer" if role == "evader" else "evader"].memory

    def _begin(self, seed):
        obs = super()._begin(seed)
        for rules in self.rules.values():
            rules.reset()
        self._clear_record()
        return obs

    def _step_inner(self, action, branch: str = "law"):
        out = super()._step_inner(action, branch)
        env, joint = self.inner.env, self.inner._last_joint_obs
        # The tactic that produced THIS step's action, for each role: the law
        # was consulted before the step and the evader inside it, and neither
        # adapter switches after returning an action.
        truth = {
            "evader": _current(self.evader),
            "pursuer": SEARCH_KEY if branch != "law" else _current(self.law),
        }
        # Post-step poses and observations: reading 2.
        self.rules["pursuer"].memory.observe(
            joint["pursuer"], env.pursuer_pose, env.evader_pose, env.obstacles
        )
        self.rules["evader"].memory.observe(
            joint["evader"], env.evader_pose, env.pursuer_pose, env.obstacles
        )
        for role in ("evader", "pursuer"):
            self.usage[role][truth[role]] += 1
            if truth[role] == SEARCH_KEY:
                continue  # driven by no tactic: nothing to score against
            label = self.memory_of_watcher(role).recogniser.label
            self.recognised[role].append((label, truth[role]))
        self.search_steps += int(branch != "law")
        return out


def _current(adapter) -> str:
    """The FAMILY of the adapter's current tactic (reading 5)."""
    return UNKNOWN if adapter.current is None else tactic_family(adapter.current.name)


def _name(adapter) -> str:
    """The adapter's current tactic with its obstacle parameter, for a frame."""
    return UNKNOWN if adapter.current is None else adapter.current.name


def _history(adapter) -> list:
    """The adapter's span log with the OPEN span closed at its step count, so
    the record has no None boundary."""
    return [
        [name, int(start), int(adapter.step_index if end is None else end)]
        for name, start, end in adapter.history
    ]


def _accuracy(pairs: list) -> dict:
    """The Step 3 gate's definition (reading 3): UNKNOWN steps are excluded
    from the accuracy and reported as a fraction of the steps."""
    known = [(label, truth) for label, truth in pairs if label != UNKNOWN]
    correct = sum(int(label == truth) for label, truth in known)
    return {
        "n_steps": len(pairs),
        "n_known": len(known),
        "n_correct": correct,
        "accuracy": correct / len(known) if known else None,
        "unknown_fraction": (len(pairs) - len(known)) / len(pairs) if pairs else None,
    }


class TacticGameMDP(_TacticGameMemory, BranchTaggedSearchMDP):
    """The measured game."""


def build_game(cfg: EnvConfig, recorder: bool = False, label: str = ""):
    """The one place the game is assembled, so the recorded episode and the
    measured one differ in nothing but the recording."""
    rules = {"evader": EvaderRulesV0(cfg), "pursuer": PursuerRulesV0(cfg)}
    evader = TacticOpponent(rules["evader"].tactics, rules["evader"])
    law = TacticLaw(rules["pursuer"].tactics, rules["pursuer"])
    kw = {
        "hold": HOLD,
        "shield": build_shield(cfg),
        "shield_obs": SHIELD_OBS,
        "escape_obs": ESCAPE_OBS,
        "reward_wrapper": None,
        "law": law,
        "evader": evader,
        "rules": rules,
    }
    if not recorder:
        return TacticGameMDP(cfg, lambda: evader, **kw), rules
    return (
        _recorder_class()(cfg, lambda: evader, label=label, every=GIF_EVERY, **kw),
        rules,
    )


def play_episode(mdp, seed: int, cfg: EnvConfig) -> dict:
    """One seeded episode of the built game, as a record."""
    log = mdp.play_episode(OrbitSearch(cfg).predict, seed)
    return {
        "seed": int(seed),
        "ratio": float(cfg.evader_speed_ratio),
        "outcome": log["outcome"],
        "steps": int(log["steps"]),
        "violations": int(log.get("violations", 0)),
        "los_breaks": int(log["los_breaks"]),
        "search_steps": int(mdp.search_steps),
        "tactics": {
            "evader": _history(mdp.evader),
            "pursuer": _history(mdp.law),
        },
        "usage": {role: dict(counts) for role, counts in mdp.usage.items()},
        "recogniser": {
            role: _accuracy(pairs) for role, pairs in mdp.recognised.items()
        },
    }


def run_ratio(ratio: float, n_episodes: int, seed0: int) -> list[dict]:
    cfg = EnvConfig(evader_speed_ratio=ratio)
    mdp, _ = build_game(cfg)
    out = []
    for i in range(n_episodes):
        out.append(play_episode(mdp, seed0 + i, cfg))
    return out


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def summarise(episodes: list[dict]) -> dict:
    usage = {role: Counter() for role in ("evader", "pursuer")}
    recog = {role: {"n_known": 0, "n_correct": 0, "n_steps": 0} for role in usage}
    for episode in episodes:
        for role, counts in usage.items():
            counts.update(episode["usage"][role])
            for key in list(recog[role]):
                recog[role][key] += episode["recogniser"][role][key]
    n = len(episodes)
    return {
        "n_episodes": n,
        "capture_rate": sum(e["outcome"] == "capture" for e in episodes) / n,
        "mean_steps": statistics.fmean(e["steps"] for e in episodes),
        "violations_total": sum(e["violations"] for e in episodes),
        "episodes_with_violations": sum(bool(e["violations"]) for e in episodes),
        "search_steps_total": sum(e["search_steps"] for e in episodes),
        "usage": {role: dict(counts.most_common()) for role, counts in usage.items()},
        "usage_share": {
            role: {
                name: count / max(1, sum(counts.values()))
                for name, count in counts.most_common()
            }
            for role, counts in usage.items()
        },
        "recogniser": {
            role: dict(
                totals,
                accuracy=(
                    totals["n_correct"] / totals["n_known"] if totals["n_known"] else None
                ),
                unknown_fraction=(
                    (totals["n_steps"] - totals["n_known"]) / totals["n_steps"]
                    if totals["n_steps"]
                    else None
                ),
            )
            for role, totals in recog.items()
        },
    }


def _top3(share: dict) -> str:
    return ", ".join(f"{name} {value:.2f}" for name, value in list(share.items())[:3])


def print_reading() -> None:
    """The S24.0 reading, VERBATIM and before any number this script prints —
    the progress lines included, which is why `run` calls this too."""
    print(f"\nS24.0 READING: {READING}\n")


def print_table(record: dict) -> None:
    """The reading, then the table — never the other way round."""
    print_reading()
    print(
        f"memory: LastSeenBelief dead reckoning (D146) · {record['label_branch']} · "
        f"recogniser window {record['window']} steps · SWEEP steers at the "
        f"estimate while age < {BELIEF_FRESH_STEPS} steps"
    )
    print(
        f"scripted v0 selectors, both roles · shield CBFShield(0.0, 0.2) on the "
        f"pursuer · hold {HOLD} · seeds {record['seed0']}+ · "
        f"{record['n_episodes']} paired episodes per ratio"
    )
    for ratio in record["ratios"]:
        s = record["per_ratio"][str(ratio)]
        print(
            f"\nratio {ratio}: capture {s['capture_rate']:.3f} · mean steps "
            f"{s['mean_steps']:.1f} · violations {s['violations_total']} "
            f"({s['episodes_with_violations']} episodes) · search-policy steps "
            f"{s['search_steps_total']}"
        )
        for role in ("evader", "pursuer"):
            print(f"    {role:<8} top-3 {_top3(s['usage_share'][role])}")
        for role in ("evader", "pursuer"):
            r = s["recogniser"][role]
            acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
            print(
                f"    recogniser of the {role:<8} accuracy {acc} "
                f"(UNKNOWN {r['unknown_fraction']:.3f}, n={r['n_known']}) "
                f"— reported, consumed by no rule (D147/D148)"
            )
    print()


# ----------------------------------------------------------------------------
# GIFs
# ----------------------------------------------------------------------------
def frame_label(evader: str, pursuer: str) -> str:
    """The frame label, exact format (brief). No aggregate number ever enters
    a caption (A5.1's discipline, kept here)."""
    return f"EVADER: {evader} | PURSUER: {pursuer}"


def aim_point(tactic):
    """The world point the pursuer's current tactic is steering at: the
    tactic's own goal (`COUNTER-ORBIT`'s waypoint, `AMBUSH`'s gap, `SWEEP`'s
    dead-reckoned point) or the wrapped law's aim point (`LEAD`). `INTERCEPT`
    exposes none and the marker is simply absent there."""
    if tactic is None:
        return None
    for owner, attrs in (
        (tactic, ("last_goal", "last_gap", "goal")),
        (getattr(tactic, "law", None), ("last_aim",)),
    ):
        for attr in attrs:
            value = None if owner is None else getattr(owner, attr, None)
            if value is not None:
                return np.asarray(value, dtype=float)[:2]
    return None


class AimOverlay:
    """S23's `AimOverlay`, reading the aim off the CURRENT TACTIC."""

    def __init__(self, law) -> None:
        self.law = law

    def __call__(self, ax, env) -> None:
        aim = aim_point(getattr(self.law, "current", None))
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


class BeliefOverlay:
    """The pursuer's dead-reckoning estimate of the evader: a hollow marker
    that fades with the estimate's age, invisible once it is older than the
    horizon `SWEEP` will steer at. Drawn from the memory as of the previous
    step's update (the frame is drawn inside the step, before this step's
    observe), which is the estimate the decision in the frame was taken on."""

    def __init__(self, memory) -> None:
        self.memory = memory

    def __call__(self, ax, env) -> None:
        point, age = self.memory.point, self.memory.age
        if point is None or age is None:
            return
        alpha = 1.0 - min(1.0, age / float(BELIEF_FRESH_STEPS))
        if alpha <= 0.0:
            return
        ax.plot(
            [float(point[0])],
            [float(point[1])],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="#1f77b4",
            markeredgewidth=1.8,
            alpha=max(0.15, alpha),
            zorder=6,
        )


def _recorder_class():
    """S19's `CellRecorder` (the FOV cone, the nominal-vs-executed arcs, the
    frame cadence, one figure per episode) with this step's caption and
    overlays. Built lazily: the measured runs must not pay for matplotlib."""
    S19 = _load("derisk_s19_planner", _HERE / "derisk_s19_planner.py")

    class TacticGameRecorder(_TacticGameMemory, S19.CellRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ov.append(BeliefOverlay(self.rules["pursuer"].memory))
            self._ov.append(AimOverlay(self.law))

        def caption(self, tag: str) -> str:
            # The INSTANCE name here (`HOP(1)`, not `HOP`): the plan's own
            # example label carries the parameter ("EVADER: HOP->B"). The table
            # aggregates by family; a frame is about this episode.
            return frame_label(_name(self.evader), _name(self.law))

    return TacticGameRecorder


def render_episode(seed: int, ratio: float, every: int = GIF_EVERY):
    cfg = EnvConfig(evader_speed_ratio=ratio)
    rec, _ = build_game(cfg, recorder=True, label=f"seed {seed}")
    episode = play_episode(rec, seed, cfg)
    rec.finish(episode["outcome"], episode["steps"])
    return rec.frames, episode


def write_gifs(n: int = GIF_N, ratio: float = GIF_RATIO, seed0: int = CONSTRUCT_SEED0):
    """The first `n` construct seeds at ratio `ratio` only — three watched
    episodes, not nine (brief). Captions carry no aggregate number."""
    from pe2.render import save_gif

    GIFS.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{n} GIFs at ratio {ratio} on seeds {seed0}..{seed0 + n - 1}. Watch "
        "the tactic pair in the label: whether the evader's HOP/ORBIT switches "
        "happen where the geometry says they should, whether SWEEP walks to the "
        "hollow dead-reckoning marker, and whether either role never switches.",
        flush=True,
    )
    written = []
    for seed in range(seed0, seed0 + n):
        frames, episode = render_episode(seed, ratio)
        path = GIFS / f"tactic_game_r{ratio}_{seed}.gif"
        save_gif(frames, str(path), fps=15.0)
        written.append(str(path))
        print(
            f"wrote {path} ({len(frames)} frames; {episode['outcome']} @ "
            f"{episode['steps']}; evader spans {len(episode['tactics']['evader'])}, "
            f"pursuer spans {len(episode['tactics']['pursuer'])})",
            flush=True,
        )
    return written


# ----------------------------------------------------------------------------
# record
# ----------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance must never stop a run
        return "unknown"


def provenance(cfg: EnvConfig) -> dict:
    return {
        "plan": PLAN,
        "git_sha": _git_sha(),
        "pe2_version": getattr(pe2, "__version__", "unknown"),
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "config": dataclasses.asdict(cfg),
        "shield": "CBFShield(cfg) at margin 0.0, gamma 0.2",
        "shield_obs": SHIELD_OBS,
        "escape_obs": ESCAPE_OBS,
        "hold": HOLD,
        "search_policy": "OrbitSearch (SWEEP's own primitive), reading 4",
    }


def run(n_episodes: int, seed0: int, ratios=RATIOS) -> dict:
    branch = label_branch()
    print_reading()
    started = time.time()
    per_ratio, episodes = {}, {}
    for ratio in ratios:
        t0 = time.time()
        block = run_ratio(ratio, n_episodes, seed0)
        episodes[str(ratio)] = block
        per_ratio[str(ratio)] = summarise(block)
        # Progress only: no measured quantity before the table (and so never
        # before the reading).
        print(
            f"ran ratio {ratio}: {n_episodes} episodes in {time.time() - t0:.0f}s",
            flush=True,
        )
    return {
        "round": "S24.0 Step 4 — scripted selectors v0 + the labelled tactic game",
        "reading": READING,
        "label_branch": branch,
        "memory": "LastSeenBelief dead reckoning (D146); ParticleBelief shelved",
        "window": TacticRecogniser(EnvConfig(), "evader").window,
        "seed0": seed0,
        "n_episodes": n_episodes,
        "ratios": list(ratios),
        "paired_across_ratios": True,
        "per_ratio": per_ratio,
        "episodes": episodes,
        "wall_clock_s": time.time() - started,
        "provenance": provenance(EnvConfig()),
    }


def write_record(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RECORD.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=False))
    tmp.replace(RECORD)
    print(f"wrote {RECORD}")


def main() -> None:
    global OUT, RECORD, GIFS
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--seed0", type=int, default=CONSTRUCT_SEED0)
    p.add_argument("--ratio", type=float, default=None, help="one ratio only")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--gifs", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    if args.out != str(OUT):
        OUT = pathlib.Path(args.out)
        RECORD = OUT / "s24_tactic_game.json"
        GIFS = OUT / "s24_gifs"

    if args.summarize:
        print_table(json.loads(RECORD.read_text()))
        return
    if args.gifs:
        write_gifs(args.episodes, GIF_RATIO if args.ratio is None else args.ratio, args.seed0)
        return
    ratios = RATIOS if args.ratio is None else (args.ratio,)
    record = run(args.episodes, args.seed0, ratios)
    write_record(record)
    print_table(record)


if __name__ == "__main__":
    main()
