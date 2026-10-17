"""S24-OS Step O.2 — training the pursuer's tactic selector over library v1
(the five S24.1 tactics plus `ORBIT-SEARCH`): arms `SEL2-80/81/82`.

Pre-registration: `docs/pathway/tactics_prereg_s24_os_2026-10-17.md` section 3.
Where the brief and the prereg could be read to disagree, the PREREGISTRATION
wins and the reading is named at the line.

Section 3's whole design: "`derisk_s24_1_train.py` configuration unchanged
except `library = 'v1'`". `win_count`, `_write_json`, `_sha256`, `make_learner`
(reads only `ARMS[arm]["seed"]` and `GAMMA`, neither of which the library
touches) and `_label_branch` are CALLED, not copied, from S24.1's trainer
(`scripts/derisk_s24_1_train.py`, loaded by file path with that script's own
`_load` pattern, S24.3-PR's `_bound` technique applied only where it is sound).

**What could NOT be rebound, and why.** `_ckpt_dir`, `_Probe` and
`_resume_from` bake S24.1's own file stem `S24-1-{arm}` directly into f-strings
inside the frozen trainer — not a module global reachable by `setattr`, so
rebinding `T1._ckpt_dir` alone (S24.3-PR's trick) would still leave the
checkpoint FILE named `S24-1-{arm}_{mark}.zip` inside this round's directory.
This round's checkpoints carry a new stem, `S24-OS-{arm}` (brief, deliverable
1): `_label`, `_ckpt_dir` and the mark path are therefore this file's OWN
functions, and `_Probe`, `_resume_from` and `train` are rewritten against them
rather than bound. `make_env` is also new: S24.1's does not accept `library`.

**`probe`'s one change, and why it matters (finding (a) of the brief).**
S24.1's `probe` projects the per-decision `picks` histogram onto the
MODULE-LEVEL `ACTIONS` — S24.1's own v0 library, five names — because that is
what every S24.1 arm trained over. Calling it unchanged on a v1 arm would
silently drop `ORBIT-SEARCH`'s share from every probe row. `probe` below is
this file's own, identical to S24.1's apart from projecting onto `env.actions`
(the ARM's own six-name library) instead.

Readings:

1. The probe stream is "2 750 000+" (section 3), disjoint from S24.1's own
   probe stream (2 350 000+), the gate stream both rounds share (2 400 000+)
   and the training stream `seed * 1000`+. Checked at import, as S24.3-PR
   checks its own streams.
2. The construct gate below (brief, deliverable 1) runs on the arm's BUILT
   training env before its first step and refuses with `SystemExit` — a
   configuration error, not a crashed arm (A2.4's pattern) — so `run` lets it
   propagate rather than filing it as six UNDETERMINED records.

THIS SCRIPT PRINTS NO CAPTURE RATE (S24.1's rule, inherited): every measured
quantity goes to the per-arm JSON and the gate runner reads those files.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_os_train.py --arms SEL2-80
  ./venv/bin/python scripts/derisk_s24_os_train.py --list
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

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's trainer, loaded by FILE PATH (the pattern `derisk_s24_3pr_train.py`
# already uses on this same module).
T1 = _load("derisk_s24_1_train", _HERE / "derisk_s24_1_train.py")

from pe2.selector_mdp import LIBRARIES, SelectorMDP, obs_len_for  # noqa: E402

# --- imported from S24.1 and CALLED, not reimplemented ----------------------
_write_json, _sha256 = T1._write_json, T1._sha256
win_count = T1.win_count
def make_learner(arm: str, env, device: str = "cpu"):
    """S24.1's `make_learner`, CALLED with S24.1's arm name for the same seed:
    its body resolves `ARMS` in S24.1's module (keys SEL-80..NOMEM-82), so this
    round's names would raise KeyError there (review defect 1). Same seed,
    same GAMMA, every other hyper-parameter the S24.1 default."""
    return T1.make_learner(f"SEL-{ARMS[arm]['seed']}", env, device)
_label_branch = T1._label_branch

PREREG = "docs/pathway/tactics_prereg_s24_os_2026-10-17.md"
ROUND = "S24-OS Step O.2 — learned pursuer tactic selector, library v1"
V0B_RECORD = T1.V0B_RECORD
OUT = pathlib.Path("results/derisk")
PREFIX = "s24_os"

# --- pre-registered (prereg section 3). Do not tune. ---
SEEDS: tuple[int, ...] = T1.SEEDS  # (80, 81, 82)
RATIO: float = T1.RATIO  # 1.2, S24.1's own
TOTAL: int = T1.TOTAL  # 3 000 000
EVERY: int = T1.EVERY  # 250 000
PROBE_N: int = T1.PROBE_N  # 200
GAMMA: float = T1.GAMMA  # 0.99
OPPONENT: str = T1.OPPONENT  # "TACTIC-V0"
LIBRARY = "v1"
PROBE_SEED0 = 2_750_000  # section 3: "the probe stream is 2 750 000+"

if T1.RATIO != RATIO:  # trivially true; stated so a later edit of either trips it
    raise SystemExit("S24.1's trainer ratio moved; this round reads it, not a copy")

ARMS: dict[str, dict] = {
    f"SEL2-{seed}": {
        "family": "SEL2", "seed": seed, "memory": True, "library": LIBRARY,
    }
    for seed in SEEDS
}
ARM_ORDER: tuple[str, ...] = tuple(ARMS)

if TOTAL % EVERY:
    raise SystemExit(f"TOTAL {TOTAL} must be a multiple of EVERY {EVERY}")

# Reading 1: the seed streams this round touches must be pairwise disjoint.
_STREAMS: dict[str, set[int]] = {
    # An upper bound on the episodes one arm consumes (3 M steps / a 100-step
    # floor). Arms are independent, unpaired runs, so their streams may
    # overlap EACH OTHER (review note 4); the claim tested here is that no
    # training stream touches the gate or either probe stream.
    "train": {s * 1000 + i for s in SEEDS for i in range(TOTAL // 100)},
    "s24_1_gate": set(range(2_400_000, 2_400_500)),
    "s24_1_probe": set(range(T1.PROBE_SEED0, T1.PROBE_SEED0 + T1.PROBE_N)),
    "os_probe": set(range(PROBE_SEED0, PROBE_SEED0 + PROBE_N)),
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
    """This round's checkpoint stem: `S24-OS-{ARM}` — the trainer's own, so the
    gate and both launchers resolve it through this function and never a
    literal (brief, finding (b))."""
    return f"S24-OS-{arm}"


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
    launchers need (S24.1's and S24.3-PR's `ckpt_path` have the same one-arg,
    final-mark shape)."""
    return _mark_path(arm, TOTAL)


def provenance() -> dict:
    """S24.1's provenance block (git commit, library versions, `label_branch`,
    the v0b baseline hash) with this round's spec, library and width."""
    base = T1.provenance()
    return base | {
        "prereg": PREREG,
        "prereg_sha256": _sha256(_REPO / PREREG),
        "round": ROUND,
        "derived_from": T1.PREREG,
        "library": LIBRARY,
        "obs_len": obs_len_for(LIBRARY),
        "n_actions": len(LIBRARIES[LIBRARY]),
    }


# -------------------------------------------------------------- the arm
def make_env(arm: str) -> SelectorMDP:
    spec = ARMS[arm]
    return SelectorMDP(
        RATIO, memory=spec["memory"], seed0=spec["seed"] * 1000, library=spec["library"]
    )


def cg_library(arm: str) -> dict:
    """The construct gate on the BUILT training env, before the arm's first
    step (brief, deliverable 1; A2.4's pattern: a configuration refusal, not a
    crashed arm): `obs_len == 59`, `actions[-1] == "ORBIT-SEARCH"`, and with
    the evader visible at reset the sixth mask entry is False."""
    env = make_env(arm)
    base_seed = int(ARMS[arm]["seed"]) * 1000
    visible_at = None
    mask_when_visible = None
    for i in range(50):
        obs, _info = env.reset(seed=base_seed + i)
        if bool(np.asarray(obs)[6] > 0.5):
            visible_at = base_seed + i
            if "ORBIT-SEARCH" in env.actions:
                mask_when_visible = bool(
                    env.action_masks()[env.actions.index("ORBIT-SEARCH")]
                )
            break
    block = {
        "gate": "S24-OS construct gate: obs_len 59, actions[-1] ORBIT-SEARCH, "
        "6th mask False when visible at reset",
        "arm": arm,
        "obs_len": int(env.obs_len),
        "n_actions": int(env.n_actions),
        "actions": list(env.actions),
        "visible_at_reset_seed": visible_at,
        "orbit_search_mask_when_visible": mask_when_visible,
    }
    why = []
    if env.obs_len != 59:
        why.append(f"obs_len is {env.obs_len}, not 59")
    if env.actions[-1] != "ORBIT-SEARCH":
        why.append(f"actions[-1] is {env.actions[-1]!r}, not 'ORBIT-SEARCH'")
    if visible_at is None:
        why.append(f"no visible-at-reset state found in seeds {base_seed}..{base_seed + 49}")
    elif mask_when_visible:
        why.append(
            f"the 6th mask entry is True with the evader visible at reset "
            f"(seed {visible_at})"
        )
    if why:
        raise SystemExit(
            f"S24-OS's construct gate REFUSES {arm} before its first env step: "
            + "; ".join(why)
        )
    return block | {"verdict": "PASS"}


def probe(model, arm: str, n: int | None = None, seed0: int | None = None) -> dict:
    """S24.1's `probe`, with finding (a)'s one change: `picks` is projected
    onto `env.actions` (this arm's own library, v1 — six names) rather than
    S24.1's module-level v0 `ACTIONS`, so `ORBIT-SEARCH`'s share is not
    silently dropped."""
    n = PROBE_N if n is None else int(n)
    seed0 = PROBE_SEED0 if seed0 is None else int(seed0)
    env = make_env(arm)
    logs = [env.play_episode(model.predict, seed0 + i) for i in range(n)]
    k, total = win_count(logs, "pursuer")
    usage: Counter = Counter()
    picks: Counter = Counter()
    for log in logs:
        usage.update(log["usage"]["pursuer"])
        picks.update(log["picks"])
    return {
        "opponent": OPPONENT,
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
        "picks": {name: int(picks.get(name, 0)) for name in env.actions},  # finding (a)
        "inapplicable_picks": int(env.inapplicable_picks),
        "violations": int(sum(log["violations"] for log in logs)),
        "episodes_with_violations": int(sum(bool(log["violations"]) for log in logs)),
    }


def train_evidence(vec_env) -> dict:
    return T1.train_evidence(vec_env)


class _Probe:
    """S24.1's checkpoint-and-probe callback, rewritten against this round's
    own `_mark_path` (the literal "S24-1-" stem cannot be reached by rebinding
    — see the module docstring)."""

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
    rows recorded up to it (S24.1's reading 3)."""
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
    from sb3_contrib import MaskablePPO

    gate = cg_library(arm)  # reading 2: refuses with SystemExit, not UNDETERMINED
    started = time.time()
    resume_ckpt, offset, rows = _resume_from(arm)
    env = make_env(arm)
    probes = _Probe(arm, rows, offset=offset)
    if resume_ckpt is None:
        model = make_learner(arm, env, device)
    else:
        print(f"[{arm}] resuming from {resume_ckpt.name} (mark {offset})", flush=True)
        model = MaskablePPO.load(resume_ckpt, env=env, device=device)
    model.learn(
        total_timesteps=TOTAL - offset,  # env steps >= decisions: an upper bound
        callback=probes.callback,
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
        "cg_library": gate,
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
            # The construct gate (`cg_library`) raises `SystemExit`: a
            # launch-blocking refusal about the configuration, never a crashed
            # arm, and it must abort instead of becoming three UNDETERMINED
            # records (A2.4's pattern).
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
            f"on {PROBE_SEED0}+ (N = {PROBE_N}) vs {OPPONENT}; library {LIBRARY} "
            f"(obs {obs_len_for(LIBRARY)}, {len(LIBRARIES[LIBRARY])} actions)\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
