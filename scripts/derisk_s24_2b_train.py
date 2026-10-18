"""S24.2 Step E.3 — training the PURSUER's tactic selector (the alternation)
over a per-episode EVADER mixture: arms `SEL3-80/81/82`.

Pre-registration: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 3
and Amendments A2.4/A3. Where the brief and the prereg could be read to
disagree, the PREREGISTRATION wins and the reading is named at the line.

A rebinding derivative of `scripts/derisk_s24_1_train.py` (S24.1's pursuer
trainer — `make_learner`, the hyper-parameters, `ACTIONS`/`OBS_LEN`, `probe`'s
shape) and `scripts/derisk_s24_2_train.py` (E.2's own mixture-handling and
checkpoint-stem rewrite, done for the SAME reason this file repeats it): both
loaded by file path (`T1`, `T2`). `_write_json`, `_sha256`, `win_count` and
`train_evidence` are CALLED from S24.1's trainer through `T1`, not copied.
`make_learner` is E.2's own wrapper, repeated: S24.1's `make_learner` resolves
`ARMS[arm]["seed"]` in S24.1's OWN module (keys `SEL-80`..`NOMEM-82`), so
calling it with this round's names (`SEL3-80`) would raise `KeyError` there —
it is called with S24.1's `SEL-{seed}` name for the same seed instead.

**What could NOT be rebound, and why (E.2's own finding, repeated here).**
`_ckpt_dir`, `_Probe` and `_resume_from` bake S24.1's own file stem
`S24-1-{arm}` directly into f-strings inside the frozen trainer — not a
module global reachable by `setattr`. This round's checkpoints carry the stem
`S24-2B-{ARM}` (brief, deliverable 2): `_label`, `_ckpt_dir`, `_mark_path` and
`ckpt_path` are therefore this file's OWN functions, and `_Probe`,
`_resume_from` and `train` are rewritten against them.

**The frozen EVADER checkpoint is resolved through E.2's own trainer
function, never a literal.** `EVS-best` (D156: `EVS-80`, the lowest mean
capture over the stage-1 gate pursuers) is `T2.ckpt_path("EVS-80")` —
`derisk_s24_2_train.py::ckpt_path`, the one function of that name the E.2
round exposes.

**The per-episode evader MIXTURE is drawn where E.2 drew its pursuer
mixture, moved to this side.** `pe2.selector_mdp.SelectorMDP` has no seat-list
of its own (only `EvaderSelectorMDP` does — the OPPONENT side owns the
seat-draw in both classes, and here the opponent is the evader) and the brief
forbids touching that module, so `_EvaderMixtureSelectorMDP` below is a THIN
subclass defined in this script, overriding `reset()` to draw a seat on
`self.np_random` (gymnasium's own per-episode RNG — seeded from `seed0` when
unseeded, reseeded from the episode's own seed by `gym.Env.reset`, exactly as
`EvaderSelectorMDP.reset` already relies on) between the reseed and
`SelectorMDP.reset`'s own `game._begin(seed)` call — which is why that
method's few lines are repeated here rather than called through `super()`:
there is no seam in `SelectorMDP.reset` to inject a step between those two.

**The training env's evader is the per-episode uniform MIXTURE; the probe's
is TACTIC-V0 alone** (brief, deliverable 2) — two different envs
(`make_env` / `make_probe_env`), as E.2 kept its training and probe envs
apart.

THIS SCRIPT PRINTS NO CAPTURE RATE (S24.1's rule, inherited): every measured
quantity goes to the per-arm JSON and the gate runner reads those files.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_2b_train.py --arms SEL3-80
  ./venv/bin/python scripts/derisk_s24_2b_train.py --list
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # BEFORE numpy / torch build pools

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from collections import Counter

import gymnasium as gym

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's trainer/gate and E.2's trainer, loaded by FILE PATH (the pattern
# `derisk_s24_2_train.py` already uses on S24.1's).
T1 = _load("derisk_s24_1_train", _HERE / "derisk_s24_1_train.py")
G1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
T2 = _load("derisk_s24_2_train", _HERE / "derisk_s24_2_train.py")

from pe2.evader_selector_mdp import EvaderSelectorMDP
from pe2.learned_select import LearnedEvaderSelect
from pe2.selector_mdp import ACTIONS, OBS_LEN, SelectorMDP

# --- imported from S24.1 and CALLED, not reimplemented ----------------------
_write_json, _sha256 = T1._write_json, T1._sha256
win_count = T1.win_count
train_evidence = T1.train_evidence


def make_learner(arm: str, env, device: str = "cpu"):
    """E.2's own wrapper, repeated: S24.1's `make_learner` resolves `ARMS` in
    S24.1's OWN module, so this round's names would raise `KeyError` there.
    Same seed, same `GAMMA`, every other hyper-parameter S24.1's default."""
    return T1.make_learner(f"SEL-{ARMS[arm]['seed']}", env, device)


PREREG = "docs/pathway/tactics_prereg_s24_2_2026-10-17.md"
ROUND = "S24.2 Step E.3 — the alternation: learned pursuer tactic selector, per-episode evader mixture"
OUT = T1.OUT
PREFIX = "s24_2b"

# --- pre-registered (prereg section 3, brief's exact values). Do not tune. ---
SEEDS: tuple[int, ...] = T1.SEEDS  # (80, 81, 82)
RATIO: float = T1.RATIO  # 1.2, S24.1's own
TOTAL: int = T1.TOTAL  # 3 000 000
EVERY: int = T1.EVERY  # 250 000
PROBE_N: int = T1.PROBE_N  # 200
GAMMA: float = T1.GAMMA  # 0.99
PROBE_SEED0 = 2_650_000  # brief: "probes every 250 000 on 2 650 000+"
PROBE_OPPONENT = "TACTIC-V0"  # brief: "N = 200 vs TACTIC-V0" — the probe's ONLY evader

# D156 (binding): EVS-best = EVS-80 — resolved through E.2's own `ckpt_path`,
# never a literal.
EVS_BEST = "EVS-80"
EVS_BEST_CKPT = T2.ckpt_path(EVS_BEST)
EVS_BEST_STEM = EVS_BEST_CKPT.stem
# A2.4/A3's per-episode uniform mixture, exactly the brief's four seats.
EVADER_MIXTURE: tuple[str, ...] = ("TACTIC-V0", EVS_BEST_STEM, "GREEDY", "HUG")

ARMS: dict[str, dict] = {
    f"SEL3-{seed}": {"family": "SEL3", "seed": seed, "memory": True} for seed in SEEDS
}
ARM_ORDER: tuple[str, ...] = tuple(ARMS)

if TOTAL % EVERY:
    raise SystemExit(f"TOTAL {TOTAL} must be a multiple of EVERY {EVERY}")

# The seed streams this round touches must be pairwise disjoint (E.2's reading,
# repeated): training is independent per arm so those streams may overlap each
# other; what must not overlap is the probe stream against the gate stream
# (2 400 000+, deliverable 3's) and against S24.1's own probe stream.
_STREAMS: dict[str, set[int]] = {
    "train": {s * 1000 + i for s in SEEDS for i in range(TOTAL // 100)},
    "probe": set(range(PROBE_SEED0, PROBE_SEED0 + PROBE_N)),
    "gate": set(range(2_400_000, 2_400_000 + 500)),
    "s24_1_probe": set(range(T1.PROBE_SEED0, T1.PROBE_SEED0 + T1.PROBE_N)),
}
_CLASHES = sorted(
    f"{a} & {b}"
    for i, a in enumerate(_STREAMS)
    for b in list(_STREAMS)[i + 1 :]
    if _STREAMS[a] & _STREAMS[b]
)
if _CLASHES:
    raise SystemExit(f"the pre-registered seed streams overlap: {_CLASHES}")


# ----------------------------------------------------------------- io
def _label(arm: str) -> str:
    """This round's checkpoint stem: `S24-2B-{ARM}` (brief, deliverable 2) —
    the trainer's own, so the gate and both launchers resolve it through this
    function and never a literal (E.2's own convention)."""
    return f"S24-2B-{arm}"


def _arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_arm_{arm}.json"


def _probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_probe_{arm}.json"


def _ckpt_dir(arm: str) -> pathlib.Path:
    return OUT / f"ckpts_{_label(arm)}"


def _mark_path(arm: str, mark: int) -> pathlib.Path:
    return _ckpt_dir(arm) / f"{_label(arm)}_{mark}.zip"


def ckpt_path(arm: str) -> pathlib.Path:
    """The arm's FINAL checkpoint — the one contract the gate and the
    launchers need."""
    return _mark_path(arm, TOTAL)


def provenance() -> dict:
    """S24.1's provenance block (git commit, library versions, `label_branch`,
    the v0b baseline hash) with this round's spec and the evader mixture."""
    base = T1.provenance()
    return base | {
        "prereg": PREREG,
        "prereg_sha256": _sha256(_REPO / PREREG),
        "round": ROUND,
        "derived_from": T2.PREREG,
        "obs_len": OBS_LEN,
        "n_actions": len(ACTIONS),
        "evader_mixture": list(EVADER_MIXTURE),
    }


# -------------------------------------------------------------- the mixture env
class _EvaderMixtureSelectorMDP(SelectorMDP):
    """`SelectorMDP` (the pursuer's semi-MDP) whose EVADER seat is redrawn
    each episode from a per-episode uniform MIXTURE — see the module
    docstring for why this lives here rather than in `pe2.selector_mdp` or
    `pe2.evader_selector_mdp`.

    The four seats (brief, deliverable 2): `TACTIC-V0` restores the DEFAULT
    evader (`self.rules["evader"]`'s own select) through `game.evader.select`;
    `GREEDY` / `HUG` swap `game.inner.opponent_factory` through the S24.1
    gate's own `_set_opponent` mechanism (`G1.scripted_opponent`); the learned
    `EVS-best` seat installs `LearnedEvaderSelect` through `game.evader.select`
    — never the opponent factory, because it reuses the SAME `TacticOpponent`
    adapter TACTIC-V0 plays through. The TACTIC-V0 and learned branches both
    restore `opponent_factory` to the default EVERY time they are drawn (not
    only when it last differed), so a prior GREEDY/HUG draw never survives
    into one of them — the brief's "restoring TACTIC-V0 cleanly between
    episodes".
    """

    def __init__(self, *args, evader_seats, device: str = "cpu", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._evader_seats = tuple(evader_seats)
        if len(self._evader_seats) != 4:
            raise ValueError(
                f"the mixture must list exactly four seats, got {self._evader_seats}"
            )
        self._device = device
        # The assembly's own default (`lambda: evader`, producing `game.evader`
        # — `scripts/s24_tactic_game.py::build_game`'s own factory), read back
        # BEFORE anything here ever overwrites it.
        self._default_opponent_factory = self.game.inner.opponent_factory
        self._learned_select = None  # built lazily, once, over this shared game
        self.evader_name = ""

    def _build_learned_select(self) -> LearnedEvaderSelect:
        from sb3_contrib import MaskablePPO

        sel = EvaderSelectorMDP.over_game(self.game, self.rules, cfg=self.cfg)
        model = MaskablePPO.load(EVS_BEST_CKPT, device=self._device)
        return LearnedEvaderSelect(sel, model, name=EVS_BEST_STEM)

    def _install_evader(self, name: str) -> None:
        if name == "TACTIC-V0":
            self.game.inner.opponent_factory = self._default_opponent_factory
            self.game.evader.select = self.rules["evader"]
            return
        if name in ("GREEDY", "HUG"):
            self.game.inner.opponent_factory = G1.scripted_opponent(
                name, self.cfg, self._device
            )
            return
        if name != EVS_BEST_STEM:
            raise ValueError(f"unknown evader seat {name!r}")
        self.game.inner.opponent_factory = self._default_opponent_factory
        if self._learned_select is None:
            self._learned_select = self._build_learned_select()
        self.game.evader.select = self._learned_select
        self._learned_select.reset()

    def _draw_evader(self) -> str:
        index = int(self.np_random.integers(len(self._evader_seats)))
        return self._evader_seats[index]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # `SelectorMDP.reset`'s own body, repeated: the evader seat must be
        # drawn and installed BETWEEN gymnasium's reseed and `game._begin`,
        # and there is no seam in the inherited method to inject that step.
        gym.Env.reset(self, seed=seed)
        self.evader_name = self._draw_evader()
        self._install_evader(self.evader_name)
        self._pending = None
        self._span_steps = 0
        self.n_decisions = 0
        self.picks = Counter()
        obs = self.game._begin(seed)
        self._obs_raw = obs
        return self._observe(obs), {
            "env_steps": self.env_steps,
            "evader": self.evader_name,
        }


# -------------------------------------------------------------- the arm
def make_env(arm: str) -> _EvaderMixtureSelectorMDP:
    """The training env: the per-episode uniform MIXTURE of the four seats
    (brief, deliverable 2), never the probe's single evader."""
    spec = ARMS[arm]
    return _EvaderMixtureSelectorMDP(
        RATIO,
        memory=spec["memory"],
        seed0=spec["seed"] * 1000,
        evader_seats=EVADER_MIXTURE,
    )


def make_probe_env(arm: str) -> SelectorMDP:
    """The probe env: TACTIC-V0 alone (brief, deliverable 2), never the
    training mixture — the plain `SelectorMDP`, whose evader opponent is the
    assembly's own default."""
    spec = ARMS[arm]
    return SelectorMDP(RATIO, memory=spec["memory"], seed0=spec["seed"] * 1000)


def cg_mixture(arm: str) -> dict:
    """The construct gate on the BUILT training env, before the arm's first
    step (E.2's pattern: a configuration refusal, not a crashed arm):
    `obs_len` equals `SelectorMDP`'s declared length (55), 5 actions, the mask
    is non-empty at reset, and the mixture lists exactly the four
    pre-registered seats."""
    env = make_env(arm)
    env.reset(seed=int(ARMS[arm]["seed"]) * 1000)
    mask_nonempty = bool(env.action_masks().any())
    block = {
        "gate": "S24.2 step E.3 construct gate: obs_len, 5 actions, mask "
        "non-empty at reset, the mixture lists exactly the four seats",
        "arm": arm,
        "obs_len": int(env.obs_len),
        "n_actions": int(env.n_actions),
        "mask_nonempty_at_reset": mask_nonempty,
        "seats": list(EVADER_MIXTURE),
    }
    why = []
    if env.obs_len != OBS_LEN:
        why.append(f"obs_len is {env.obs_len}, not {OBS_LEN}")
    if env.n_actions != 5:
        why.append(f"n_actions is {env.n_actions}, not 5")
    if not mask_nonempty:
        why.append("the mask is empty at reset")
    if len(EVADER_MIXTURE) != 4:
        why.append(f"the mixture is {EVADER_MIXTURE}, not four seats")
    if why:
        raise SystemExit(
            f"S24.2's construct gate REFUSES {arm} before its first env step: "
            + "; ".join(why)
        )
    return block | {"verdict": "PASS"}


def probe(model, arm: str, n: int | None = None, seed0: int | None = None) -> dict:
    """`n` paired-seed episodes against TACTIC-V0 only (brief, deliverable 2).
    `picks` is projected onto `env.actions` — the pursuer's own fixed
    five-name library, never a module-level constant restated."""
    n = PROBE_N if n is None else int(n)
    seed0 = PROBE_SEED0 if seed0 is None else int(seed0)
    env = make_probe_env(arm)
    logs = [env.play_episode(model.predict, seed0 + i) for i in range(n)]
    k, total = win_count(logs, "pursuer")
    usage: Counter = Counter()
    picks: Counter = Counter()
    for log in logs:
        usage.update(log["usage"]["pursuer"])
        picks.update(log["picks"])
    return {
        "opponent": PROBE_OPPONENT,
        "seed0": int(seed0),
        "n": int(total),
        "k": int(k),
        "capture_rate": k / total,
        "mean_steps": sum(log["steps"] for log in logs) / total,
        "n_switches_mean": sum(log["n_switches"] for log in logs) / total,
        "n_zero_length_spans_mean": sum(
            log["n_zero_length_spans"] for log in logs
        )
        / total,
        "n_decisions_mean": sum(log["n_decisions"] for log in logs) / total,
        "usage_steps": dict(usage),
        "picks": {name: int(picks.get(name, 0)) for name in env.actions},
        "inapplicable_picks": int(env.inapplicable_picks),
        "violations": int(sum(log["violations"] for log in logs)),
        "episodes_with_violations": int(sum(bool(log["violations"]) for log in logs)),
    }


class _Probe:
    """Checkpoint + probe at every `EVERY` ENV steps, rows flushed atomically
    after each one, and `learn` STOPPED once `TOTAL` env steps are reached —
    rewritten against this round's own `_mark_path` (E.2's own finding: S24.1's
    literal stem cannot be reached by rebinding).

    `offset` is the env-step mark a resumed arm starts from.
    """

    def __init__(self, arm: str, rows: list[dict], offset: int = 0) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        ckpt_dir = _ckpt_dir(arm)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.rows = rows

        def at(model, env_steps: int, mark: int, train: dict | None = None) -> None:
            path = _mark_path(arm, mark)
            model.save(path)
            row = {
                "steps": int(env_steps),
                "mark": int(mark),
                "ckpt": str(path),
                **(train or {}),
                **probe(model, arm),
            }
            self.rows.append(row)
            _write_json(_probe_file(arm), {"arm": arm, "checkpoints": self.rows})
            print(f"[{arm}] probe at mark {mark} written ({path.name})", flush=True)

        class _CB(BaseCallback):
            def __init__(cb) -> None:
                super().__init__()
                cb.next_mark = offset + EVERY

            def _on_step(cb) -> bool:
                counts = cb.training_env.get_attr("env_steps")
                assert len(counts) == 1, "the budget is in env steps: one env only"
                env_steps = offset + int(counts[0])
                while cb.next_mark <= TOTAL and env_steps >= cb.next_mark:
                    at(cb.model, env_steps, cb.next_mark, train_evidence(cb.training_env))
                    cb.next_mark += EVERY
                return env_steps < TOTAL

        self.callback = _CB()
        self.at = at


def _resume_from(arm: str) -> tuple[pathlib.Path | None, int, list[dict]]:
    """The newest checkpoint of an unfinished arm, its mark, and the probe
    rows recorded up to it (S24.1's reading, repeated)."""
    rows = []
    path = _probe_file(arm)
    if path.exists():
        rows = json.loads(path.read_text()).get("checkpoints", [])
    stem = _label(arm)
    marks = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in _ckpt_dir(arm).glob(f"{stem}_*.zip")
        if p.stem.rsplit("_", 1)[1].isdigit()
    )
    if not marks or marks[-1] >= TOTAL:
        return None, 0, rows
    mark = marks[-1]
    return _mark_path(arm, mark), mark, [r for r in rows if r["mark"] <= mark]


def train(arm: str, device: str = "cpu") -> dict:
    gate = cg_mixture(arm)  # a configuration refusal (SystemExit), not a crashed arm
    started = time.time()
    resume_ckpt, offset, rows = _resume_from(arm)
    env = make_env(arm)
    probes = _Probe(arm, rows, offset=offset)
    if resume_ckpt is not None:  # resumable-never (A1.3's stop rule, inherited)
        raise SystemExit(f"[{arm}] a checkpoint exists ({resume_ckpt.name}); never resume an arm")
    model = make_learner(arm, env, device)
    model.learn(
        total_timesteps=TOTAL - offset,  # env steps >= decisions: an upper bound
        callback=probes.callback,  # the callback is what stops at TOTAL env steps
        reset_num_timesteps=resume_ckpt is None,
    )
    env_steps = offset + env.env_steps
    record = {
        "arm": arm,
        **ARMS[arm],
        "round": ROUND,
        "prereg": PREREG,
        "ratio": RATIO,
        "gamma": GAMMA,
        "total_env_steps_budget": TOTAL,
        "probe_every": EVERY,
        "obs_len": int(env.observation_space.shape[0]),
        "action_space": list(env.actions),
        "env_seed0": ARMS[arm]["seed"] * 1000,
        "stream_note": (
            "episode 0 runs on the PPO seed; declared stream begins at "
            "seed*1000+1"
        ),
        "resumed_from": None if resume_ckpt is None else str(resume_ckpt),
        "resume_caveat": (
            None
            if resume_ckpt is None
            else "the env seed stream restarts at seed * 1000 on resume"
        ),
        "env_steps": int(env_steps),
        "decisions": int(model.num_timesteps),
        "train_inapplicable_picks": int(env.inapplicable_picks),  # CG-T1
        "train_violations": int(env.violations_total),
        "checkpoints": probes.rows,
        "wall_clock_s": time.time() - started,
        "cg_mixture": gate,
        "provenance": provenance(),
    }
    final = ckpt_path(arm)
    if not any(r["mark"] == TOTAL for r in probes.rows) or not final.exists():
        record["verdict"] = "UNDETERMINED"
        record["reason"] = (
            f"the final probe at mark {TOTAL} never fired (env steps {env_steps})"
        )
        return record
    record["verdict"] = "MEASURED"
    record["final_ckpt"] = str(final)
    return record


def _crash_record(arm: str, exc: BaseException, started: float) -> dict:
    rows = []
    if _probe_file(arm).exists():
        rows = json.loads(_probe_file(arm).read_text()).get("checkpoints", [])
    return {
        "arm": arm,
        **ARMS[arm],
        "round": ROUND,
        "prereg": PREREG,
        "verdict": "UNDETERMINED",
        "reason": f"crashed: {type(exc).__name__}: {exc}",
        "checkpoints": rows,
        "wall_clock_s": time.time() - started,
        "provenance": provenance(),
    }


def run(arms: list[str], device: str) -> None:
    for arm in arms:
        record_path = _arm_file(arm)
        if record_path.exists() and json.loads(record_path.read_text()).get("verdict"):
            print(f"[{arm}] record exists; skipping (reading 3)", flush=True)
            continue
        started = time.time()
        print(f"[{arm}] training {TOTAL} env steps, probes every {EVERY}", flush=True)
        try:
            record = train(arm, device)
        except (KeyboardInterrupt, SystemExit):
            # `cg_mixture` raises `SystemExit`: a launch-blocking refusal about
            # the configuration, never a crashed arm, and it must abort
            # instead of becoming three UNDETERMINED records.
            raise
        except BaseException as exc:  # noqa: BLE001 — a crashed arm is UNDETERMINED
            record = _crash_record(arm, exc, started)
            print(f"[{arm}] UNDETERMINED: {record['reason']}", flush=True)
        _write_json(record_path, record)
        print(f"[{arm}] {record['verdict']}; wrote {record_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list or args.arms is None:
        print(
            f"arms: {' '.join(ARMS)}\nbudget {TOTAL} env steps, probe every {EVERY} "
            f"on {PROBE_SEED0}+ (N = {PROBE_N}) vs {PROBE_OPPONENT}; trained vs the "
            f"evader mixture {list(EVADER_MIXTURE)}; obs {OBS_LEN}, 5 actions\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
