"""S24-FLAT Step F.1 — PPO training of the FLAT continuous-action pursuer.

Pre-registration: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md`
section 3 (arm `FLAT-s`: "continuous PPO over `(v, omega)` held for 10 env
steps (the same `hold`), through the recovery shield, at EVERY decision
regardless of visibility (no intercept law, no tactics) ... Reward = terminal
+-1 per span", 3 seeds 80/81/82, "Probes every 250 k on 2 550 000+ ... N = 200
vs `TACTIC-V0`"), section 2 (what is frozen: the S24.1 game assembly, ratio
1.2, 3 M env steps, S17 BASE hyper-parameters, gamma = 0.99) and section 5
(CG-F1, asserted on the built env before the first step; CG-F3: CG-T1 and
A2.1 do not apply — FLAT has no mask).

THIS SCRIPT PRINTS NO CAPTURE RATE. Every measured quantity goes to the
per-arm JSON and nothing else; the gate runner reads those files. The stop
rule ("nothing re-run after a number is seen") only holds if training cannot
show one.

Readings, where the brief and the prereg leave a choice:

1. **S17 BASE hyper-parameters = SB3 defaults apart from gamma** — literally
   `scripts/derisk_s17_fixes_b.py::make_learner` ("SB3 defaults apart from the
   arm's gamma") with `seed` = the arm's seed and `gamma` = `GAMMA` = 0.99.
   The head is the DEFAULT continuous one: plain `PPO`, not `MaskablePPO`,
   because there is no mask (CG-F3).
2. **The training env's seed stream is `seed * 1000`** (S17's `make_env`
   convention, S24.1's reading 2), disjoint from the probe stream 2 550 000+
   and the gate stream 2 400 000+.
3. **Resumable at ARM granularity**, exactly as S24.1: an arm whose record
   exists with a verdict is skipped; an arm with checkpoints but no record
   restarts from the newest checkpoint with the remaining budget and
   `env_steps` offset by that mark. Caveat recorded in the arm's JSON: a
   resumed arm's env seed stream restarts at `seed * 1000`.
4. **The checkpoint label is `S24-FLAT-{seed}`.** The brief writes
   `ckpts_S24-FLAT-{ARM}/S24-FLAT-{ARM}_{mark}.zip` with the arms named
   `FLAT-80/81/82`; taken with `{ARM}` = the full arm name that doubles the
   word ("S24-FLAT-FLAT-80"), so `{ARM}` is read as the arm's seed label and
   the path carries the round and the arm exactly once, as S24.1's
   `ckpts_S24-1-SEL-80/S24-1-SEL-80_{mark}.zip` does. The arm NAMES are the
   literal `FLAT-80/81/82` the brief fixes, for `--arms` and for the JSONs.
5. **"pinned-run fraction" is reported both ways**, because the phrase can be
   read as episodes or as steps: `pinned_run_episode_fraction` is the share of
   probe episodes whose longest immobile run reaches `DEADLOCK_STEPS` (40) —
   the v0b table's "pin of 40 steps" — and `pinned_step_fraction` is the share
   of env steps inside a recorded pinned run (`MIN_PINNED_RUN` = 10, the
   recorder's own threshold). H-FLAT-3 compares one of them to the selector's;
   both are recorded so the choice is not made by this script.
6. **The v0b record's sha256 is provenance, not a dependency** (S24.1's
   reading 4): the frozen baseline is `results/derisk/s24_tactic_game_v0b.json`
   (D150) and the hash pins which file the numbers came from.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_flat_train.py --arms FLAT-80
  ./venv/bin/python scripts/derisk_s24_flat_train.py --arms FLAT-80 FLAT-81
  ./venv/bin/python scripts/derisk_s24_flat_train.py --list
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # BEFORE numpy builds its pools

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pe2
from pe2.eval import win_count
from pe2.flat_mdp import FlatMDP
from pe2.metrics import DEADLOCK_STEPS
from pe2.recorder import MIN_PINNED_RUN
from pe2.selector_mdp import OBS_LEN
from pe2.structured import HOLD

PREREG = "docs/pathway/tactics_prereg_s24_flat_2026-10-14.md"
V0B_RECORD = pathlib.Path("results/derisk/s24_tactic_game_v0b.json")
OUT = pathlib.Path("results/derisk")
PREFIX = "s24_flat"
ROUND = "S24-FLAT"

# --- pre-registered (prereg sections 2 and 3). Do not tune. ---
SEEDS = (80, 81, 82)
RATIO = 1.2
TOTAL = 3_000_000  # env steps
EVERY = 250_000  # env steps between probes
PROBE_SEED0 = 2_550_000  # "probes every 250 k on 2 550 000+"
PROBE_N = 200
GAMMA = 0.99  # S17's GAMMA and the prereg's "gamma = 0.99 per decision"
OPPONENT = "TACTIC-V0"  # the scripted tactical evader; the only probe opponent

ARMS: dict[str, dict] = {
    **{f"FLAT-{seed}": {"family": "FLAT", "seed": seed, "hold": None, "probe_seed0": None} for seed in SEEDS},
    # A1.2: the selector's decision cap as a second flat family; probe stream 2 650 000+
    **{f"FLAT40-{seed}": {"family": "FLAT40", "seed": seed, "hold": 40, "probe_seed0": 2_650_000} for seed in SEEDS},
}

if TOTAL % EVERY:
    raise SystemExit(f"TOTAL {TOTAL} must be a multiple of EVERY {EVERY}")


# ----------------------------------------------------------------- io
def _label(arm: str) -> str:
    """Reading 4: `FLAT-80` -> `S24-FLAT-80`."""
    return f"S24-{arm}"


def _arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_arm_{arm}.json"


def _probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_probe_{arm}.json"


def _ckpt_dir(arm: str) -> pathlib.Path:
    return OUT / f"ckpts_{_label(arm)}"


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)  # atomic within the filesystem


def _sha256(path: pathlib.Path) -> str | None:
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _label_branch() -> str:
    """The v0b game's own guard, kept (S24.1's `_label_branch`): the EVADER's
    rule table is `EvaderRulesV0`, written on the ruling that both recogniser
    gates FAILED (D147/D148). It refuses to run if that record reads PASS."""
    import importlib.util

    module = sys.modules.get("s24_tactic_game")
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "s24_tactic_game", _HERE / "s24_tactic_game.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["s24_tactic_game"] = module
        spec.loader.exec_module(module)
    return module.label_branch()


def provenance() -> dict:
    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "prereg": PREREG,
        "prereg_sha256": _sha256(_REPO / PREREG),
        "baseline_record": str(V0B_RECORD),
        "baseline_sha256": _sha256(_REPO / V0B_RECORD),
        "git_commit": sha,
        "pe2_file": pe2.__file__,
        "pe2_version": pe2.__version__,
        "sb3_version": stable_baselines3.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "obs_len": OBS_LEN,
        "ratio": RATIO,
        "label_branch": _label_branch(),
    }


# -------------------------------------------------------------- the arm
def make_env(arm: str) -> FlatMDP:
    return FlatMDP(RATIO, seed0=ARMS[arm]["seed"] * 1000, hold=ARMS[arm].get("hold"))


def make_learner(arm: str, env, device: str = "cpu"):
    """Reading 1: S17's `make_learner` — SB3 defaults apart from gamma — with
    the DEFAULT continuous head."""
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        seed=ARMS[arm]["seed"],
        gamma=GAMMA,
        verbose=0,
        device=device,
    )


def assert_cg_f1(env: FlatMDP, allowed_assembly: tuple = ()) -> dict:
    """CG-F1 on the BUILT env, before the first step (prereg section 5). A
    failure is a refusal about the CONFIGURATION, so it aborts the launch the
    way the v0b assembly guards do (`SystemExit`), rather than becoming three
    UNDETERMINED records."""
    cg = env.config_diff_vs_selector(allowed_assembly=allowed_assembly)
    if not cg["ok"]:
        raise SystemExit(
            "CG-F1 FAILED: FLAT's env differs from the selector's in more "
            f"than the action and the tactic block — config_diff "
            f"{cg['config_diff']}, assembly_diff {cg['assembly_diff']}, "
            f"observation entries differing on an identical reset "
            f"{cg['obs_diff_indices_on_identical_reset']}, FLAT tactic "
            f"entries non-zero {cg['flat_tactic_entries_nonzero']}, selector "
            f"tactic entries non-zero {cg['selector_tactic_entries_nonzero']}"
        )
    return cg


def probe(model, arm: str, n: int | None = None, seed0: int | None = None) -> dict:
    """`n` paired-seed episodes against `TACTIC-V0` only (prereg section 3).

    Builds its own `FlatMDP` and drives the episodes one at a time through
    `play_episode`, which rebuilds the recovery shield per episode.
    """
    # The module constants are read HERE, not bound as defaults, so a test can
    # shrink the probe without the pre-registered value drifting.
    n = PROBE_N if n is None else int(n)
    seed0 = (ARMS[arm].get("probe_seed0") or PROBE_SEED0) if seed0 is None else int(seed0)  # A1.2
    env = make_env(arm)
    logs = [env.play_episode(model.predict, seed0 + i) for i in range(n)]
    k, total = win_count(logs, "pursuer")
    branches = [log["branch"] for log in logs]
    steps_total = sum(b["n_steps"] for b in branches)
    immobile = sum(sum(b["immobile_by_branch"].values()) for b in branches)
    pinned_steps = sum(r["length"] for b in branches for r in b["pinned_runs"])
    return {
        "opponent": OPPONENT,
        "seed0": int(seed0),
        "n": int(total),
        "k": int(k),
        "capture_rate": k / total,
        "mean_steps": sum(log["steps"] for log in logs) / total,
        "n_decisions_mean": sum(log["n_decisions"] for log in logs) / total,
        "mean_span_steps": sum(log["mean_span_steps"] for log in logs) / total,
        # Reading 5: both readings of "pinned-run fraction", plus the rate.
        "immobile_step_rate": immobile / steps_total if steps_total else None,
        "pinned_run_episode_fraction": (
            sum(b["longest_pinned_run"] >= DEADLOCK_STEPS for b in branches) / total
        ),
        "pinned_step_fraction": pinned_steps / steps_total if steps_total else None,
        "pinned_run_thresholds": {
            "episode_fraction_steps": DEADLOCK_STEPS,
            "pinned_run_steps": MIN_PINNED_RUN,
        },
        "longest_pinned_run_max": max(b["longest_pinned_run"] for b in branches),
        "recoveries_total": sum(log["recovery"]["n_recoveries"] for log in logs),
        "violations": int(sum(log["violations"] for log in logs)),
        "episodes_with_violations": int(sum(bool(log["violations"]) for log in logs)),
    }


def train_evidence(vec_env) -> dict:
    """H-SAFE's counter read off the TRAINING env, written into every probe row
    so a crashed arm still carries it (the arm record may never be written, but
    the probe file is flushed at every mark). CG-F3: there is no
    `inapplicable_picks` to carry — FLAT has no mask."""
    return {"train_violations": int(vec_env.get_attr("violations_total")[0])}


class _Probe:
    """Checkpoint + probe at every `EVERY` ENV steps, rows flushed atomically
    after each one, and `learn` STOPPED once `TOTAL` env steps are reached — so
    the budget is in env steps, exactly as `StructuredEvalProbe` does it.

    `offset` is the env-step mark a resumed arm starts from (reading 3).
    """

    def __init__(self, arm: str, rows: list[dict], offset: int = 0) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        ckpt_dir = _ckpt_dir(arm)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.rows = rows

        def at(model, env_steps: int, mark: int, train: dict | None = None) -> None:
            path = ckpt_dir / f"{_label(arm)}_{mark}.zip"
            model.save(path)
            row = {
                "steps": int(env_steps),
                "mark": int(mark),
                "ckpt": str(path),
                # H-SAFE evidence from the TRAINING env, so a crashed arm's
                # last probe row still carries it.
                **(train or {}),
                **probe(model, arm),
            }
            self.rows.append(row)
            _write_json(_probe_file(arm), {"arm": arm, "checkpoints": self.rows})
            # No measured quantity is printed. The JSON is the only reader.
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
    """The newest checkpoint of an unfinished arm, its mark, and the probe rows
    recorded up to it (reading 3)."""
    rows = []
    path = _probe_file(arm)
    if path.exists():
        rows = json.loads(path.read_text()).get("checkpoints", [])
    marks = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in _ckpt_dir(arm).glob(f"{_label(arm)}_*.zip")
        if p.stem.rsplit("_", 1)[1].isdigit()
    )
    if not marks or marks[-1] >= TOTAL:
        return None, 0, rows
    mark = marks[-1]
    return (
        _ckpt_dir(arm) / f"{_label(arm)}_{mark}.zip",
        mark,
        [r for r in rows if r["mark"] <= mark],
    )


def train(arm: str, device: str = "cpu") -> dict:
    from stable_baselines3 import PPO

    started = time.time()
    resume_ckpt, offset, rows = _resume_from(arm)
    env = make_env(arm)
    # A1.2: FLAT40 differs from the selector in `hold` and nothing else
    cg_f1 = assert_cg_f1(env, ("hold",) if ARMS[arm]["family"] == "FLAT40" else ())
    probes = _Probe(arm, rows, offset=offset)
    if resume_ckpt is None:
        model = make_learner(arm, env, device)
    else:
        print(f"[{arm}] resuming from {resume_ckpt.name} (mark {offset})", flush=True)
        model = PPO.load(resume_ckpt, env=env, device=device)
    model.learn(
        total_timesteps=TOTAL - offset,  # env steps >= decisions: an upper bound
        callback=probes.callback,  # the callback is what stops at TOTAL env steps
        reset_num_timesteps=resume_ckpt is None,
    )
    env_steps = offset + env.env_steps
    record = {
        "arm": arm,
        **ARMS[arm],
        "round": "S24-FLAT Step F.1 — flat continuous-action pursuer (PPO)",
        "prereg": PREREG,
        "ratio": RATIO,
        "gamma": GAMMA,
        "hold": int(env.hold),
        "total_env_steps_budget": TOTAL,
        "probe_every": EVERY,
        "probe_seed0": ARMS[arm].get("probe_seed0") or PROBE_SEED0,
        "probe_n": PROBE_N,
        "obs_len": int(env.observation_space.shape[0]),
        "action_space": str(env.action_space),
        "env_seed0": ARMS[arm]["seed"] * 1000,
        "cg_f1": cg_f1,
        # A2.4's note, recorded and not changed (S17's own behaviour):
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
        "train_violations": int(env.violations_total),  # H-SAFE
        "checkpoints": probes.rows,
        "wall_clock_s": time.time() - started,
        "provenance": provenance(),
    }
    final = _ckpt_dir(arm) / f"{_label(arm)}_{TOTAL}.zip"
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
            # The v0b assembly's guards (`build_shield`, `label_branch`) and
            # CG-F1 raise `SystemExit`. That is a launch-blocking refusal about
            # the configuration, not a crash of this arm, and must abort
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
            f"on {PROBE_SEED0}+ (N = {PROBE_N}) vs {OPPONENT}; obs {OBS_LEN}; "
            f"continuous (v, omega) held {HOLD} env steps at ratio {RATIO}\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
