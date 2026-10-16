"""S24-FLAT Step F.2 — PPO training of the STRUCT structured-search pursuer.

Pre-registration: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md`
section 3 (arm `STRUCT-s`: "the project's structured learner (`SearchMDP`:
fixed intercept law when visible, PPO chooses search commands otherwise;
S17 BASE config, `shield_obs`/`escape_obs`) through the same recovery shield
against the same evader -- the strongest non-hierarchical learner the
project has", 3 seeds 80/81/82, "Probes every 250 k on 2 600 000+ ... N = 200
vs `TACTIC-V0`"), section 2 (what is frozen: the S24.1 game assembly, ratio
1.2, 3 M env steps, S17 BASE hyper-parameters, gamma = 0.99) and section 5
(CG-F1, STRUCT half: asserted on the built env before the first step).

THIS SCRIPT PRINTS NO CAPTURE RATE, mirroring Step F.1
(`scripts/derisk_s24_flat_train.py`) exactly: every measured quantity goes to
the per-arm JSON and nothing else.

Readings, where the brief and the prereg leave a choice:

1. **The evader factory is `scripts/s24_tactic_game.py::build_game`'s evader
   construction, reproduced rather than called.** `build_game` builds BOTH
   roles -- `rules["evader"] = EvaderRulesV0(cfg)` then
   `evader = TacticOpponent(rules["evader"].tactics, rules["evader"])`, and
   `rules["pursuer"] = PursuerRulesV0(cfg)` plus `TacticLaw(...)`. This arm's
   pursuer never consults `TacticLaw` -- `SearchMDP` plays its own
   `InterceptLaw` (fixed chase / last-seen law) and the learner's held
   command, per the prereg's "S1-S3/S17 structured learner" -- so building
   the pursuer side only to discard it is both wasted work and a needless
   coupling to `PursuerRulesV0`'s internals. `evader_factory_for` below is
   `build_game`'s own two evader lines, copied verbatim, with nothing of the
   pursuer side instantiated.
2. **S17 BASE hyper-parameters = SB3 defaults apart from gamma**, exactly
   F.1's reading 1 (`scripts/derisk_s24_flat_train.py::make_learner`), with
   `seed` = the arm's seed and `gamma` = `GAMMA` = 0.99. `MlpPolicy` is the
   DEFAULT continuous head -- `SearchMDP`'s action space is the pursuer's
   `(v, omega)` Box regardless of which branch (law or learner) is driving a
   given env step, so this is the same PPO construction F.1 uses, not a new
   one.
3. **The env construction is the deliverable's own list, literally**:
   `SearchMDP(cfg, evader_factory_for(cfg), hold=10, shield=<the game's
   RecoveryShieldV2>, shield_obs=True, escape_obs=True, r_acq=0.0,
   reward_wrapper=None)` at ratio 1.2. `stall_penalty` and
   `intervention_penalty` are left at `SearchMDP`'s own default (0.0): the
   brief's list does not mention them, and the frozen game's own
   `build_game` kwargs (`scripts/s24_tactic_game.py`) pass neither, so 0.0
   is what "the tactic game's ... recovery shield" already runs under. `law`
   is left `None` so `SearchMDP.__init__` builds its own `InterceptLaw(cfg)`
   (reading 7 of the brief).
4. **The shield is a single `RecoveryShieldV2`, held for the whole training
   run** (S17's own `make_env`, `scripts/derisk_s17_fixes_b.py`, builds its
   `CBFShield` once, not per episode) -- plain `SearchMDP._begin` does not
   rebuild `self.shield`, unlike the frozen game's `TacticGameMDP._begin`
   (`scripts/s24_tactic_game.py::_TacticGameMemory._begin`), which rebuilds a
   fresh shield every episode. The TRAINING env is therefore built with one
   shield instance, exactly as every prior structured-learner round trained.
   PROBING is different: `make_gate_env` builds the env with `shield=None`
   and the probe loop assigns `env.shield = build_shield(cfg)` before every
   episode -- S17's per-episode discipline
   (`scripts/derisk_s17_fixes_b.py::evaluate_arm_policy`, the spec requirement
   that a `RecoveryShieldV2`'s immobile run and recovery counters have no
   episode hook, so one shield across a paired-seed block would carry an open
   run across the boundary) -- because the probe and the Step F.3 gate are
   seeded, paired-episode measurements, not a single training rollout.
5. **The training env's seed stream is `seed * 1000`** (S17's `make_env`
   convention, S24.1's and F.1's reading), disjoint from the probe stream
   2 600 000+ and the gate stream 2 400 000+.
6. **Resumable at ARM granularity**, exactly as F.1 and S24.1: an arm whose
   record exists with a verdict is skipped; an arm with checkpoints but no
   record restarts from the newest checkpoint with the remaining budget and
   `env_steps` offset by that mark.
7. **The checkpoint label is `S24-STRUCT-{seed}`.** `_label` is F.1's own
   function (`f"S24-{arm}"`); the arm NAMES are the literal `STRUCT-80/81/82`
   the brief fixes, so `_label("STRUCT-80") == "S24-STRUCT-80"` already --
   no doubling to read around, unlike F.1's own note about `FLAT`.
8. **The v0b record's sha256 is provenance, not a dependency** (F.1's
   reading 6, S24.1's reading 4): the hash pins which baseline file the
   numbers came from.
9. **`make_gate_env` is exposed so Step F.3 imports it rather than re-typing
   the construction** (the brief's explicit ask): it is also what `probe`
   below calls, so the probe this script runs during training and the gate
   that re-plays checkpoints afterwards cannot drift apart.

Usage (nothing launches on import):
  ./venv/bin/python scripts/derisk_s24_struct_train.py --arms STRUCT-80
  ./venv/bin/python scripts/derisk_s24_struct_train.py --arms STRUCT-80 STRUCT-81
  ./venv/bin/python scripts/derisk_s24_struct_train.py --list
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # BEFORE numpy builds its pools

import argparse
import dataclasses
import hashlib
import importlib.util
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
from pe2.config import EnvConfig
from pe2.eval import win_count
from pe2.obs import obs_dim
from pe2.selectors import EvaderRulesV0
from pe2.shield import RECOVERY_AFTER
from pe2.structured import HOLD, SearchMDP
from pe2.tactics import TacticOpponent

PREREG = "docs/pathway/tactics_prereg_s24_flat_2026-10-14.md"
V0B_RECORD = pathlib.Path("results/derisk/s24_tactic_game_v0b.json")
OUT = pathlib.Path("results/derisk")
PREFIX = "s24_struct"
ROUND = "S24-STRUCT"

# --- pre-registered (prereg sections 2 and 3). Do not tune. ---
SEEDS = (80, 81, 82)
RATIO = 1.2
TOTAL = 3_000_000  # env steps
EVERY = 250_000  # env steps between probes
PROBE_SEED0 = 2_600_000  # "probes every 250 k on 2 600 000+" (STRUCT)
PROBE_N = 200
GAMMA = 0.99  # S17's GAMMA and the prereg's "gamma = 0.99 per decision"
OPPONENT = "TACTIC-V0"  # the scripted tactical evader; the only probe opponent

ARMS: dict[str, dict] = {f"STRUCT-{seed}": {"family": "STRUCT", "seed": seed} for seed in SEEDS}

# shield_obs (3 entries) + escape_obs (2 entries), both ON for this arm.
OBS_LEN = obs_dim(EnvConfig()) + 3 + 2

if TOTAL % EVERY:
    raise SystemExit(f"TOTAL {TOTAL} must be a multiple of EVERY {EVERY}")


# ----------------------------------------------------------------- io
def _label(arm: str) -> str:
    """F.1's own function: `STRUCT-80` -> `S24-STRUCT-80`."""
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


def _game_module():
    """`scripts/s24_tactic_game.py`, loaded by path (it is a script, not a
    package module) -- F.1's `_label_branch` loader, generalised: this arm
    also needs `build_shield` from the same module (reading 4)."""
    module = sys.modules.get("s24_tactic_game")
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "s24_tactic_game", _HERE / "s24_tactic_game.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["s24_tactic_game"] = module
        spec.loader.exec_module(module)
    return module


def _label_branch() -> str:
    """The v0b game's own guard, kept (F.1's `_label_branch`): the EVADER's
    rule table is `EvaderRulesV0`, written on the ruling that both
    recogniser gates FAILED (D147/D148). It refuses to run if that record
    reads PASS."""
    return _game_module().label_branch()


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
        "learner": "STRUCT (SearchMDP + PPO, S17 BASE)",
    }


# -------------------------------------------------------------- the arm
def evader_factory_for(cfg: EnvConfig):
    """Reading 1: `build_game`'s evader, reproduced rather than called --
    `rules["evader"] = EvaderRulesV0(cfg)` then
    `evader = TacticOpponent(rules["evader"].tactics, rules["evader"])`
    (`scripts/s24_tactic_game.py::build_game`), with none of the pursuer side
    (`PursuerRulesV0`, `TacticLaw`) instantiated. Returns a factory closing
    over ONE shared `TacticOpponent`, which is what `build_game` does too
    (`lambda: evader`) -- `SingleAgentWrapper.reset` calls the factory and
    then `.reset()` on what it returns every episode, so one shared,
    resettable instance is the correct shape, not a bug to fix."""
    rules = EvaderRulesV0(cfg)
    evader = TacticOpponent(rules.tactics, rules)
    return lambda: evader


def make_gate_env() -> SearchMDP:
    """Reading 9: the exact construction `play_episode` runs under for both
    this script's `probe` and Step F.3's gate (prereg section 2: ratio 1.2,
    hold 10, `shield_obs`/`escape_obs` on, `r_acq` 0.0, no reward wrapper).
    `shield=None` here: a paired-seed evaluation needs a FRESH
    `RecoveryShieldV2` per episode (reading 4), so the caller assigns
    `env.shield` before every `play_episode` call rather than this function
    building one shield for every episode to share."""
    cfg = EnvConfig(evader_speed_ratio=RATIO)
    return SearchMDP(
        cfg,
        evader_factory_for(cfg),
        hold=HOLD,
        r_acq=0.0,
        shield=None,
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=None,
    )


def make_env(arm: str) -> SearchMDP:
    """The TRAINING env (reading 3, reading 4): one `RecoveryShieldV2` held
    for the whole run, built the same way the frozen game builds its own
    (`s24_tactic_game.build_shield`)."""
    cfg = EnvConfig(evader_speed_ratio=RATIO)
    shield = _game_module().build_shield(cfg)
    return SearchMDP(
        cfg,
        evader_factory_for(cfg),
        seed0=ARMS[arm]["seed"] * 1000,
        hold=HOLD,
        r_acq=0.0,
        shield=shield,
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=None,
    )


def make_learner(arm: str, env, device: str = "cpu"):
    """Reading 2: S17's `make_learner` -- SB3 defaults apart from gamma --
    with the DEFAULT continuous head."""
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        seed=ARMS[arm]["seed"],
        gamma=GAMMA,
        verbose=0,
        device=device,
    )


def assert_cg_f1(env: SearchMDP) -> dict:
    """CG-F1 (STRUCT half, prereg section 5), on the BUILT env, before the
    first step: STRUCT's env differs from the game's only in the law/learner
    split -- same `EnvConfig`, the same `RecoveryShieldV2` parameters, the
    same evader (`TacticOpponent` over `EvaderRulesV0`). A failure is a
    refusal about the CONFIGURATION, so it aborts the launch (`SystemExit`)
    the way the v0b assembly's own guards and F.1's CG-F1 do, rather than
    becoming three UNDETERMINED records."""
    game_cfg = dataclasses.asdict(EnvConfig(evader_speed_ratio=RATIO))
    env_cfg = dataclasses.asdict(env.cfg)
    config_diff = sorted(f for f, v in env_cfg.items() if v != game_cfg.get(f))

    shield = env.shield
    recovery_after = getattr(shield, "recovery_after", None)
    shield_params = {
        "class": type(shield).__name__,
        "margin": float(shield.margin),
        "gamma": float(shield.gamma),
        "recovery_after": None if recovery_after is None else int(recovery_after),
    }
    shield_ok = (
        shield_params["class"] == "RecoveryShieldV2"
        and shield_params["margin"] == 0.0
        and shield_params["gamma"] == 0.2
        and shield_params["recovery_after"] == RECOVERY_AFTER
    )

    opponent = env.inner.opponent_factory()
    opponent_params = {
        "class": type(opponent).__name__,
        "rules_class": type(opponent.select).__name__,
    }
    opponent_ok = (
        opponent_params["class"] == "TacticOpponent"
        and opponent_params["rules_class"] == "EvaderRulesV0"
    )
    cg = {
        "gate": "CG-F1",
        "prereg": PREREG,
        "config_diff": config_diff,
        "shield": shield_params,
        "opponent": opponent_params,
        "ok": bool(not config_diff and shield_ok and opponent_ok),
    }
    if not cg["ok"]:
        raise SystemExit(f"CG-F1 FAILED: {cg}")
    return cg


def probe(model, arm: str, n: int | None = None, seed0: int | None = None) -> dict:
    """`n` paired-seed episodes against `TACTIC-V0` only (prereg section 3),
    through `make_gate_env` (reading 9) with a FRESH `RecoveryShieldV2`
    assigned before every episode (reading 4).
    """
    # The module constants are read HERE, not bound as defaults, so a test
    # can shrink the probe without the pre-registered value drifting.
    n = PROBE_N if n is None else int(n)
    seed0 = PROBE_SEED0 if seed0 is None else int(seed0)
    env = make_gate_env()
    build_shield = _game_module().build_shield
    logs = []
    for i in range(n):
        env.shield = build_shield(env.cfg)
        logs.append(env.play_episode(model.predict, seed0 + i))
    k, total = win_count(logs, "pursuer")
    return {
        "opponent": OPPONENT,
        "seed0": int(seed0),
        "n": int(total),
        "k": int(k),
        "capture_rate": k / total,
        "mean_steps": sum(log["steps"] for log in logs) / total,
        "violations": int(sum(log["violations"] for log in logs)),
        "episodes_with_violations": int(sum(bool(log["violations"]) for log in logs)),
    }


def train_evidence(vec_env) -> dict:
    """H-SAFE's counter read off the TRAINING env, written into every probe
    row so a crashed arm still carries it (the arm record may never be
    written, but the probe file is flushed at every mark)."""
    return {"train_violations": int(vec_env.get_attr("violations_total")[0])}


class _Probe:
    """Checkpoint + probe at every `EVERY` ENV steps, rows flushed atomically
    after each one, and `learn` STOPPED once `TOTAL` env steps are reached --
    F.1's `_Probe`, unchanged apart from the arm's env.

    `offset` is the env-step mark a resumed arm starts from (reading 6).
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
    """The newest checkpoint of an unfinished arm, its mark, and the probe
    rows recorded up to it (reading 6)."""
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
    cg_f1 = assert_cg_f1(env)  # prereg section 5, before the first step
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
        "round": "S24-STRUCT Step F.2 — structured search pursuer (SearchMDP + PPO, S17 BASE)",
        "prereg": PREREG,
        "ratio": RATIO,
        "gamma": GAMMA,
        "hold": int(env.hold),
        "total_env_steps_budget": TOTAL,
        "probe_every": EVERY,
        "probe_seed0": PROBE_SEED0,
        "probe_n": PROBE_N,
        "obs_len": int(env.observation_space.shape[0]),
        "action_space": str(env.action_space),
        "env_seed0": ARMS[arm]["seed"] * 1000,
        "cg_f1": cg_f1,
        # F.1's A2.4 note, recorded and not changed (S17's own behaviour, the
        # same `SingleAgentWrapper` mechanics):
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
            print(f"[{arm}] record exists; skipping (reading 6)", flush=True)
            continue
        started = time.time()
        print(f"[{arm}] training {TOTAL} env steps, probes every {EVERY}", flush=True)
        try:
            record = train(arm, device)
        except (KeyboardInterrupt, SystemExit):
            # The v0b assembly's guards (`build_shield`, `label_branch`) and
            # CG-F1 raise `SystemExit`. That is a launch-blocking refusal
            # about the configuration, not a crash of this arm, and must
            # abort instead of becoming three UNDETERMINED records.
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
            f"continuous (v, omega) held {HOLD} env steps through InterceptLaw "
            f"at ratio {RATIO}\n"
            "nothing runs without --arms"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    run(args.arms, args.device)


if __name__ == "__main__":
    main()
