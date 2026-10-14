"""S24.3-PR Step 1 — per-ratio training of the tactic selector at 0.8 and 1.0.

Pre-registration: `docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md`, all
sections. Where the brief and the prereg could be read to disagree, the
PREREGISTRATION wins and the reading is named at the line.

A THIN REBINDING of `scripts/derisk_s24_1_train.py` (the S24.1 trainer, frozen
at the training freeze `1f9eab2`), loaded by file path with the `_load` +
`_bound` pattern `scripts/derisk_s24_3zs.py` already uses on S24.1's gate.
Section 2 is the whole design: "Nothing else changes between a 0.8/1.0 arm and
S24.1's 1.2 arms", so `train`, `_Probe`, `_resume_from`, `make_env`,
`make_learner` and `probe` are CALLED, not copied — the 0.8 arm is trained by
the SAME code that trained the 1.2 arms, and a record of one is a record of the
other with a different `ratio`.

**What is rebound, and why each one.** Each name is a global that one of the
S24.1 functions called here reads:

    ARMS         make_env / make_learner / train / _crash_record / run
    RATIO        make_env (and `train`'s recorded `ratio`)      — PER ARM
    PROBE_SEED0  probe                                          — PER ARM
    PREREG       provenance, train, _crash_record
    _arm_file    run        _probe_file  _Probe / _resume_from / _crash_record
    _ckpt_dir    _Probe / _resume_from / train
    probe        _Probe.at      (so every probe row carries `ratio`)
    provenance   train / _crash_record
    train        run            (so the arm record carries `ratio` and CG-P1)
    _crash_record run           (the 3zs review's lesson: the UNDETERMINED
                                 record must be bound too, or a crashed arm is
                                 filed without the one field this round added)

`_ORIG` holds S24.1's own versions, captured at import and called THROUGH that
table, so a wrapper can never re-enter itself.

Readings this module fixes, because the text can be read two ways:

1. **The checkpoint FILE stem stays S24.1's `S24-1-{ARM}`; only the DIRECTORY
   is this round's.** Section 3 fixes `ckpts_S24-3PR-{ARM}/` and says nothing
   about the file name, and the stem is hardcoded in THREE places inside the
   frozen trainer (`_Probe.at`, `_resume_from`, `train`'s final-mark check) and
   is not a module global — renaming it would mean COPYING the probe loop, the
   one thing section 2 freezes. `ckpt_path` below is the single definition of
   that path and the gate runner reads it from here, so the two cannot drift.
2. **CG-P1 runs on the BUILT env before the arm's first step, and refuses with
   `SystemExit`.** Section 6 puts it "before any capture rate is read"; a
   configuration refusal is not a crash of the arm (A2.4), so `run` re-raises it
   and the launch aborts instead of writing six UNDETERMINED records. The gate
   re-reads the block from the arm record.
3. **`PROBE_SEED0` is per ARM, not per round** (section 3: 2 450 000+ for the
   0.8 arms, 2 500 000+ for the 1.0 arms), so it is bound alongside `RATIO`
   rather than once.
4. **Two arms share a seed and so a training stream.** `SELr0.8-80` and
   `SELr1.0-80` both run `seed * 1000` and PPO seed 80 — section 3's "training
   stream `seed·1000` (as S24.1)" read literally. The ratio is the only
   difference between them, which is the point of the round.

THIS SCRIPT PRINTS NO CAPTURE RATE (S24.1's rule, inherited): every measured
quantity goes to the per-arm JSON and the gate runner reads those files.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_3pr_train.py --arms SELr0.8-80
  ./venv/bin/python scripts/derisk_s24_3pr_train.py --list
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # BEFORE numpy / torch build their pools (as S24.1)
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import dataclasses
import importlib.util
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.selector_mdp import OBS_SLICES

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# S24.1's trainer, loaded by FILE PATH so this module imports the same way as a
# script and as a package module.
T1 = _load("derisk_s24_1_train", _HERE / "derisk_s24_1_train.py")

# --- imported from S24.1 and CALLED, not reimplemented -----------------------
_write_json, _sha256 = T1._write_json, T1._sha256
win_count = T1.win_count  # only so a test can see the probe's own counter

PREREG = "docs/pathway/tactics_prereg_s24_3pr_2026-10-14.md"
ROUND = "S24.3-PR Step 1 — per-ratio training of the tactic selector"
PREFIX = "s24_3pr"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (prereg sections 2 and 3). Do not tune. ---
RATIOS: tuple[float, ...] = (0.8, 1.0)  # section 3
BASE_RATIO = 1.2  # S24.1's ratio, the twin CG-P1 asserts against
SEEDS: tuple[int, ...] = T1.SEEDS  # (80, 81, 82) — section 3's three seeds
# Section 3: probes every 250 k on 2 450 000+ (0.8 arms) / 2 500 000+ (1.0 arms)
PROBE_SEED0_BY_RATIO: dict[float, int] = {0.8: 2_450_000, 1.0: 2_500_000}
# Everything else is S24.1's, READ from it rather than restated (section 2).
TOTAL, EVERY, PROBE_N = T1.TOTAL, T1.EVERY, T1.PROBE_N
GAMMA, OPPONENT = T1.GAMMA, T1.OPPONENT

ARMS: dict[str, dict] = {
    f"SELr{ratio:.1f}-{seed}": {
        "family": "SEL",
        "seed": seed,
        "memory": True,  # section 3: `memory=True` on every arm of this round
        "ratio": float(ratio),
    }
    for ratio in RATIOS
    for seed in SEEDS
}
ARM_ORDER: tuple[str, ...] = tuple(ARMS)

if T1.RATIO != BASE_RATIO:
    raise SystemExit(
        f"S24.1's trainer is at ratio {T1.RATIO}, not {BASE_RATIO}; this round "
        "parametrises exactly that constant and CG-P1 asserts against its env, "
        "so the derivation is unsound unless the twin is the 1.2 one."
    )
if set(PROBE_SEED0_BY_RATIO) != set(RATIOS):
    raise SystemExit("every ratio needs its own pre-registered probe stream")
# Section 3: "disjoint from all evaluation streams". Checked here rather than
# described, because a collision would make a probe row and a gate cell the same
# episode and nothing downstream would notice.
_STREAMS: dict[str, set[int]] = {
    "train": {s * 1000 + i for s in SEEDS for i in range(1000)},
    "gate": set(range(2_400_000, 2_400_500)),
    "s24_1_probe": set(range(T1.PROBE_SEED0, T1.PROBE_SEED0 + PROBE_N)),
    **{
        f"probe_r{x:.1f}": set(range(seed0, seed0 + PROBE_N))
        for x, seed0 in PROBE_SEED0_BY_RATIO.items()
    },
}
_CLASHES = sorted(
    f"{a} & {b}"
    for i, a in enumerate(_STREAMS)
    for b in list(_STREAMS)[i + 1 :]
    if _STREAMS[a] & _STREAMS[b]
)
if _CLASHES:
    raise SystemExit(f"the pre-registered seed streams overlap: {_CLASHES}")


# ----------------------------------------------------------------------------
# artefact paths (this round's; S24.1's shapes). The SINGLE definition of each,
# bound onto S24.1 below, so the trainer and the gate cannot disagree.
# ----------------------------------------------------------------------------
def arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_arm_{arm}.json"


def probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_probe_{arm}.json"


def ckpt_dir(arm: str) -> pathlib.Path:
    return OUT / f"ckpts_S24-3PR-{arm}"


def ckpt_path(arm: str) -> pathlib.Path:
    """The arm's FINAL checkpoint. Reading 1: the directory is this round's and
    the file stem is S24.1's `S24-1-{ARM}`, which is what `_Probe.at` writes."""
    return ckpt_dir(arm) / f"S24-1-{arm}_{TOTAL}.zip"


def spec_path() -> pathlib.Path:
    return T1._REPO / PREREG


def ratio_of(arm: str) -> float:
    return float(ARMS[arm]["ratio"])


def probe_seed0(arm: str) -> int:
    return int(PROBE_SEED0_BY_RATIO[ratio_of(arm)])


def provenance() -> dict:
    """S24.1's provenance block with THIS round's spec. Built so that the spec
    fields are right whether or not a binding is active: a record must prove
    which text's thresholds it was measured under (the 3zs module's rule)."""
    return _ORIG["provenance"]() | {
        "prereg": PREREG,
        "prereg_sha256": _sha256(spec_path()),
        "round": ROUND,
        "derived_from": T1.PREREG,
        "ratios": list(RATIOS),
        "base_ratio": BASE_RATIO,
    }


# ----------------------------------------------------------------------------
# the binding of S24.1's trainer to this round (see the module docstring)
# ----------------------------------------------------------------------------
_BOUND_ALWAYS: tuple[str, ...] = (
    "ARMS", "PREREG", "_arm_file", "_probe_file", "_ckpt_dir", "probe",
    "provenance", "train", "_crash_record",
)
_BOUND_PER_ARM: tuple[str, ...] = ("RATIO", "PROBE_SEED0")
_BOUND_KEYS: tuple[str, ...] = (*_BOUND_ALWAYS, *_BOUND_PER_ARM)
# S24.1's own versions, captured at import and called THROUGH this table.
_ORIG: dict = {
    k: getattr(T1, k)
    for k in (
        *_BOUND_KEYS, "run", "make_env", "make_learner", "_resume_from",
        "_Probe", "train_evidence",
    )
}


@contextlib.contextmanager
def _bound(arm: str | None = None):
    """Rebind the names above on S24.1's trainer for one call and restore every
    one afterwards. `arm` binds the two per-arm names (`RATIO` and the probe
    stream), which is the only thing that differs between the six arms."""
    over: dict = {
        "ARMS": ARMS,
        "PREREG": PREREG,
        "_arm_file": arm_file,
        "_probe_file": probe_file,
        "_ckpt_dir": ckpt_dir,
        "probe": probe,
        "provenance": provenance,
        "train": train,
        "_crash_record": _crash_record,
    }
    if arm is not None:
        over |= {"RATIO": ratio_of(arm), "PROBE_SEED0": probe_seed0(arm)}
    saved = {k: getattr(T1, k) for k in over}
    for k, v in over.items():
        setattr(T1, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(T1, k, v)


def _one(arm: str):
    """`_bound` at one arm's ratio and probe stream."""
    return _bound(arm)


# ----------------------------------------------------------------------------
# CG-P1 (prereg section 6): only the ratio differs from an S24.1 arm's env
# ----------------------------------------------------------------------------
def config_diff(cfg: EnvConfig) -> list[str]:
    """The fields in which `cfg` differs from an S24.1 arm's env config —
    `EnvConfig()`, whose `evader_speed_ratio` default IS S24.1's 1.2. COPIED
    from the 3zs runner (two lines) rather than imported: a TRAINER may not
    depend on a gate runner, and the gate re-reads this round's own records."""
    twin = EnvConfig()
    return [
        f.name
        for f in dataclasses.fields(EnvConfig)
        if getattr(cfg, f.name) != getattr(twin, f.name)
    ]


def _f32_eq(a, b) -> bool:
    """Equality at the observation's own precision: the selector's observation
    is a float32 array and 0.8 is not exactly representable in it."""
    return a is not None and np.float32(a) == np.float32(b)


def make_env(arm: str):
    """S24.1's `make_env` at this arm's ratio, memory and seed stream."""
    with _one(arm):
        return _ORIG["make_env"](arm)


def cg_p1(arm: str, device: str = "cpu") -> dict:
    """CG-P1 on the BUILT training env, before the arm's first step (reading 2).

    The block travels in the arm record, so the gate reads a gate that ran
    before the arm trained. It REFUSES with `SystemExit`, which S24.1's `run`
    re-raises (A2.4): an env at the wrong ratio is a configuration error, not a
    crashed arm."""
    ratio = ratio_of(arm)
    env = make_env(arm)
    obs, _info = env.reset(seed=int(ARMS[arm]["seed"]) * 1000)
    block = {
        "gate": "CG-P1 (only the ratio differs from an S24.1 arm), prereg section 6",
        "arm": arm,
        "ratio": ratio,
        "twin_ratio": BASE_RATIO,
        "config_diff_vs_twin": config_diff(env.game.cfg),
        "cfg_ratio": float(env.game.cfg.evader_speed_ratio),
        "obs_ratio": float(np.asarray(obs)[OBS_SLICES["speed_ratio"]][0]),
        "memory": bool(env.memory),
        "env_seed0": int(env.game.inner.seed0),
    }
    why = [
        f"the {what} is {got}, not {ratio}"
        for what, got, same in (
            ("built env's cfg ratio", block["cfg_ratio"], block["cfg_ratio"] == ratio),
            # float32: a bare `!=` would refuse every 0.8 arm.
            ("observation's ratio entry", block["obs_ratio"], _f32_eq(block["obs_ratio"], ratio)),
        )
        if not same
    ]
    if block["config_diff_vs_twin"] != ["evader_speed_ratio"]:
        why.append(
            f"the built config differs from the 1.2 twin in "
            f"{block['config_diff_vs_twin']}, not in `evader_speed_ratio` alone, "
            "so a capture difference would not be attributable to the speed"
        )
    if why:
        raise SystemExit(
            f"CG-P1 REFUSES {arm} before its first env step: {'; '.join(why)}. "
            "Section 2 fixes everything but the ratio, and section 6 puts this "
            "gate before any number."
        )
    return block | {"verdict": "PASS"}


# ----------------------------------------------------------------------------
# the wrappers the binding installs — each ADDS this round's fields and
# delegates through `_ORIG`, so nothing of the training loop is forked
# ----------------------------------------------------------------------------
def probe(model, arm: str, n: int | None = None, seed0: int | None = None) -> dict:
    """S24.1's probe with this round's `ratio` in the row (section 3)."""
    with _one(arm):
        row = _ORIG["probe"](model, arm, n, seed0)
    return row | {"ratio": ratio_of(arm), "probe_seed0": probe_seed0(arm)}


def train(arm: str, device: str = "cpu") -> dict:
    """S24.1's `train`, with CG-P1 before it and this round's fields after it."""
    gate = cg_p1(arm, device)
    with _one(arm):
        record = _ORIG["train"](arm, device)
    return record | {
        "round": ROUND,
        "prereg": PREREG,
        "ratio": ratio_of(arm),
        "probe_seed0": probe_seed0(arm),
        "cg_p1": gate,
    }


def _crash_record(arm: str, exc: BaseException, started: float) -> dict:
    """The 3zs review's lesson, applied: without this binding a crashed arm's
    UNDETERMINED record would carry no `ratio` and the gate could not tell which
    ratio's evidence is absent."""
    with _one(arm):
        return _ORIG["_crash_record"](arm, exc, started) | {
            "round": ROUND,
            "ratio": ratio_of(arm),
            "probe_seed0": probe_seed0(arm),
        }


def run(arms: list[str], device: str = "cpu") -> None:
    """S24.1's `run`, ONE ARM AT A TIME: the ratio and the probe stream are
    per-arm bindings, so a chain of arms cannot share one binding."""
    for arm in arms:
        with _one(arm):
            _ORIG["run"]([arm], device)


def main() -> None:
    """COPIED from S24.1's `main`: its `--list` line prints a single
    `PROBE_SEED0`, and this round has one per ratio."""
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list or args.arms is None:
        streams = ", ".join(f"r{x:.1f} on {s}+" for x, s in PROBE_SEED0_BY_RATIO.items())
        print(
            f"arms: {' '.join(ARMS)}\nbudget {TOTAL} env steps, probe every "
            f"{EVERY} ({streams}, N = {PROBE_N}) vs {OPPONENT}\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
