"""S24.1 Step 1.1 — masked-PPO training of the pursuer's tactic selector.

Pre-registration: `docs/pathway/tactics_prereg_s24_1_2026-10-13.md` section 3
("PPO (SB3) with the S17 BASE hyper-parameters apart from the discrete head; 3
seeds (80, 81, 82); budget 3 M env steps each; probe every 250 k on 2 350 000+
(N = 200, scripted evader only)" and the `NOMEM` negative control) and section 5
(CG-T1: "0 inapplicable picks over all training decisions (logged)").

THIS SCRIPT PRINTS NO CAPTURE RATE. Every measured quantity goes to the per-arm
JSON and nothing else; the gate runner reads those files. The prereg's stop rule
("nothing re-run after a number is seen") only holds if training cannot show one.

Readings, where the brief and the prereg leave a choice:

1. **S17 BASE hyper-parameters = SB3 defaults apart from gamma** — that is
   literally what `scripts/derisk_s17_fixes_b.py::make_learner` is
   ("SB3 defaults apart from the arm's gamma"), with `seed` = the arm's seed and
   `gamma` = `GAMMA` = 0.99, which is also the prereg's "gamma = 0.99 per
   decision". The only change is the head: `MaskablePPO` over `Discrete(5)`.
2. **The training env's seed stream is `seed * 1000`** (S17's `make_env`
   convention), disjoint by construction from the probe stream 2 350 000+ and
   the prereg's gate stream 2 400 000+.
3. **Resumable at ARM granularity.** An arm whose record exists with a verdict
   is skipped; an arm with checkpoints but no record restarts from the newest
   checkpoint and the remaining budget, with `env_steps` offset by that mark.
   Caveat recorded in the arm's JSON: a resumed arm's env seed stream restarts
   at `seed * 1000`, so it replays episodes it has already seen.
4. **The v0b record's sha256 is provenance, not a dependency.** The frozen
   baseline this round is read against is
   `results/derisk/s24_tactic_game_v0b.json` (D150); the hash pins which file
   the numbers came from.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_1_train.py --arms SEL-80
  ./venv/bin/python scripts/derisk_s24_1_train.py --arms SEL-80 SEL-81 --device cpu
  ./venv/bin/python scripts/derisk_s24_1_train.py --list
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
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pe2
from pe2.eval import win_count
from pe2.selector_mdp import ACTIONS, OBS_LEN, SelectorMDP

PREREG = "docs/pathway/tactics_prereg_s24_1_2026-10-13.md"
V0B_RECORD = pathlib.Path("results/derisk/s24_tactic_game_v0b.json")
OUT = pathlib.Path("results/derisk")
PREFIX = "s24_1"

# --- pre-registered (prereg section 3). Do not tune. ---
SEEDS = (80, 81, 82)
RATIO = 1.2
TOTAL = 3_000_000  # env steps
EVERY = 250_000  # env steps between probes
PROBE_SEED0 = 2_350_000  # "probe every 250 k on 2 350 000+"
PROBE_N = 200
GAMMA = 0.99  # S17's GAMMA, and the prereg's "gamma = 0.99 per decision"
OPPONENT = "TACTIC-V0"  # the scripted tactical evader; the only probe opponent

ARMS: dict[str, dict] = {
    f"{family}-{seed}": {"family": family, "seed": seed, "memory": family == "SEL"}
    for family in ("SEL", "NOMEM")
    for seed in SEEDS
}

if TOTAL % EVERY:
    raise SystemExit(f"TOTAL {TOTAL} must be a multiple of EVERY {EVERY}")


# ----------------------------------------------------------------- io
def _arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_arm_{arm}.json"


def _probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_probe_{arm}.json"


def _ckpt_dir(arm: str) -> pathlib.Path:
    return OUT / f"ckpts_S24-1-{arm}"


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)  # atomic within the filesystem


def _sha256(path: pathlib.Path) -> str | None:
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def provenance() -> dict:
    import sb3_contrib
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
        "sb3_contrib_version": sb3_contrib.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "obs_len": OBS_LEN,
        "label_branch": _label_branch(),
    }


def _label_branch() -> str:
    """The v0b game's own guard, kept: the EVADER's rule table is still
    `EvaderRulesV0` and is written on the ruling that both recogniser gates
    FAILED (D147/D148). It refuses to run if that record ever reads PASS."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s24_tactic_game", _HERE / "s24_tactic_game.py"
    )
    module = sys.modules.get("s24_tactic_game")
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["s24_tactic_game"] = module
        spec.loader.exec_module(module)
    return module.label_branch()


# -------------------------------------------------------------- the arm
def make_env(arm: str) -> SelectorMDP:
    spec = ARMS[arm]
    return SelectorMDP(RATIO, memory=spec["memory"], seed0=spec["seed"] * 1000)


def make_learner(arm: str, env, device: str = "cpu"):
    """Reading 1: S17's `make_learner` with the masked discrete head."""
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=ARMS[arm]["seed"],
        gamma=GAMMA,
        verbose=0,
        device=device,
    )


def probe(model, arm: str, n: int = PROBE_N, seed0: int = PROBE_SEED0) -> dict:
    """`n` paired-seed episodes against `TACTIC-V0` only (prereg section 3).

    Builds its own `SelectorMDP` at the arm's own memory setting — the
    observation width must match the policy's — and drives the episodes one at a
    time through `play_episode`, which rebuilds the recovery shield per episode.
    """
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
        "n_decisions_mean": sum(log["n_decisions"] for log in logs) / total,
        "usage_steps": dict(usage),  # per env step, by family (the v0b table's unit)
        "picks": {name: int(picks.get(name, 0)) for name in ACTIONS},  # per decision
        "inapplicable_picks": int(env.inapplicable_picks),
        "violations": int(sum(log["violations"] for log in logs)),
        "episodes_with_violations": int(sum(bool(log["violations"]) for log in logs)),
    }


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

        def at(model, env_steps: int, mark: int) -> None:
            path = ckpt_dir / f"S24-1-{arm}_{mark}.zip"
            model.save(path)
            row = {
                "steps": int(env_steps),
                "mark": int(mark),
                "ckpt": str(path),
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
                    at(cb.model, env_steps, cb.next_mark)
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
        for p in _ckpt_dir(arm).glob(f"S24-1-{arm}_*.zip")
        if p.stem.rsplit("_", 1)[1].isdigit()
    )
    if not marks or marks[-1] >= TOTAL:
        return None, 0, rows
    mark = marks[-1]
    return (
        _ckpt_dir(arm) / f"S24-1-{arm}_{mark}.zip",
        mark,
        [r for r in rows if r["mark"] <= mark],
    )


def train(arm: str, device: str = "cpu") -> dict:
    from sb3_contrib import MaskablePPO

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
        callback=probes.callback,  # the callback is what stops at TOTAL env steps
        reset_num_timesteps=resume_ckpt is None,
    )
    env_steps = offset + env.env_steps
    record = {
        "arm": arm,
        **ARMS[arm],
        "round": "S24.1 Step 1.1 — learned pursuer tactic selector (masked PPO)",
        "prereg": PREREG,
        "ratio": RATIO,
        "gamma": GAMMA,
        "total_env_steps_budget": TOTAL,
        "probe_every": EVERY,
        "obs_len": int(env.observation_space.shape[0]),
        "action_space": list(ACTIONS),
        "env_seed0": ARMS[arm]["seed"] * 1000,
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
        "provenance": provenance(),
    }
    final = _ckpt_dir(arm) / f"S24-1-{arm}_{TOTAL}.zip"
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
        except KeyboardInterrupt:
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
            f"on {PROBE_SEED0}+ (N = {PROBE_N}) vs {OPPONENT}; obs {OBS_LEN}\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
