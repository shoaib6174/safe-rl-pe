"""S16 — Phase 2: five remedies for the loop, head to head.
Spec: docs/pathway/loop_fixes_prereg_2026-09-20.md. Nothing here may change a
threshold, a seed, a stream, a rule or a budget after a number is seen.

The failure under test (D125-D128): a shielded pursuer spends hundreds of steps
asking for a forward command the filter refuses while a pure turn is admissible
the whole time. Five remedies, one configuration, one gate:

  BASE-{60,61,62}  the S5 RC control (gamma 0.99, stall penalty 1e-3)
  GAM-{60,61,62}   gamma 0.999          — D128's discount half
  OBS-{60,61,62}   +2 escape features   — D128's frozen-input half
  MEM-{60,61,62}   RecurrentPPO (LSTM)  — can memory alone break the loop?
  PEN-{60,61,62}   stall penalty 1e-2   — D113's negative, 10x
  REC-{60,61,62}   NO TRAINING: BASE's three policies with `RecoveryShield`
                   at evaluation — the scripted baseline every learned arm is
                   measured against (H-FIX-2, the round's headline).

Gate: stream 1 720 000+, N = 500, every arm on the same episodes, against
ATK4-45, ObstacleHuggingEvader and GreedyEvader. Primary metric the pooled
immobile-step rate.

Two evaluation-path corrections this round needs, both pinned by tests:
  - `pe2.structured.evaluate_structured` builds ONE shield for a whole call and
    `RecoveryShield`'s immobile run is shield state with no episode hook, so
    `REC` would carry an open run across the paired-seed boundary.
    `evaluate_arm_policy` rebuilds the shield per episode and is otherwise
    identical to the library call (pinned bit-for-bit for a stateless shield).
  - `RecurrentPPO.predict` takes and returns the LSTM state, which
    `SearchMDP.play_episode` neither supplies nor keeps; `RecurrentPredictor`
    threads it through an episode and resets it at each episode boundary, so
    MEM is evaluated as the recurrent policy it was trained as.

Two readings the spec leaves open, fixed here BEFORE any number exists:
  - H-FIX-1's capture guard ("its GreedyEvader capture is not significantly
    below BASE's") is applied SEED-MATCHED and by the round's own >= 2/3 rule:
    an arm is downgraded to `SOLVES (weaker)` iff BASE is significantly better
    (paired exact McNemar, p < 0.05) on at least PASS_MIN of the three seeds.
    The pooled seed-matched test is recorded beside it, report-only.
  - H-FIX-2's "within 0.05 of REC's" is read ONE-SIDED: an arm matches iff its
    mean is at most REC's + 0.05. An arm far BELOW REC beats the scripted
    recovery and cannot sensibly be called a non-match; the two-sided flag and
    a `beats` flag are recorded so both readings are visible.

Usage:
  ./venv/bin/python scripts/derisk_s16_fixes.py --measure-start
  ./venv/bin/python scripts/derisk_s16_fixes.py --arms BASE-60      (one per process)
  ./venv/bin/python scripts/derisk_s16_fixes.py --rec               (after the BASE arms)
  ./venv/bin/python scripts/derisk_s16_fixes.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.eval import wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.shield import RECOVERY_AFTER
from pe2.structured import HOLD

SPEC = "docs/pathway/loop_fixes_prereg_2026-09-20.md"
ARTEFACT_PREFIX = "s16"
OUT = pathlib.Path("results/derisk")
START_PATH = OUT / f"{ARTEFACT_PREFIX}_start.json"
ATK_CKPT = OUT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"

# --- pre-registered (spec sections 2-4). Do not tune. ---
SEEDS = (60, 61, 62)
TOTAL = 5_000_000
EVERY = 500_000
PROBE_SEED0 = 1_730_000
GATE_SEED0 = 1_720_000
GATE_N = 500
EVAL_N = 200
GAMMA = 0.99
GAMMA_LONG = 0.999
STALL_W = 1e-3
STALL_W_HIGH = 1e-2
R_ACQ = 0.0
SHIELD_OBS = True
SOLVE_BAR = 0.10  # H-FIX-1: immobile-step rate below this, on >= PASS_MIN seeds
MATCH_BAR = 0.05  # H-FIX-2: mean immobile-step rate within this of REC's
GAM_BAR = 0.05  # H-FIX-3: GAM below BASE by at least this, on >= PASS_MIN seeds
PASS_MIN = 2
P_GATE = 0.05
GATE_OPPONENTS = ("ATK4-45", "HUG7", "GREEDY")
PRIMARY_OPPONENT = "ATK4-45"  # the immobile-step rate every verdict reads
CAPTURE_OPPONENT = "GREEDY"  # the capture guard's opponent
EVADER_HOLD = 10  # ATK4-45's own hold, as in S6/S9/S15

LEARNED = ("BASE", "GAM", "OBS", "MEM", "PEN")
REMEDIES = ("GAM", "OBS", "MEM", "PEN")  # LEARNED minus the control
FAMILY_ORDER = (*LEARNED, "REC")
FAMILIES = {
    "BASE": {
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "algo": "PPO",
    },
    "GAM": {
        "gamma": GAMMA_LONG,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "algo": "PPO",
    },
    "OBS": {
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": True,
        "algo": "PPO",
    },
    "MEM": {
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "algo": "RecurrentPPO",
    },
    "PEN": {
        "gamma": GAMMA,
        "stall_penalty": STALL_W_HIGH,
        "escape_obs": False,
        "algo": "PPO",
    },
}
ARMS: dict = {
    f"{fam}-{s}": {"family": fam, "seed": s, **cfg}
    for fam, cfg in FAMILIES.items()
    for s in SEEDS
}
REC_ARMS: dict = {
    f"REC-{s}": {"family": "REC", "seed": s, "source": f"BASE-{s}"} for s in SEEDS
}
ALL_ARMS: dict = ARMS | REC_ARMS
POLICY_STRING = {"PPO": "MlpPolicy", "RecurrentPPO": "MlpLstmPolicy"}

_MODELS: dict = {}


# ----------------------------------------------------------------------------
# artefact paths, io, provenance
# ----------------------------------------------------------------------------
def _arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_arm_{arm}.json"


def _probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_probe_{arm}.json"


def _round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def _ckpt_dir(arm: str) -> pathlib.Path:
    return OUT / f"ckpts_{arm}"


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def provenance() -> dict:
    import subprocess

    import sb3_contrib
    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "sb3_version": stable_baselines3.__version__,
        "sb3_contrib_version": sb3_contrib.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


# ----------------------------------------------------------------------------
# arms: algorithm, env, learner, opponents
# ----------------------------------------------------------------------------
def is_recurrent(arm: str) -> bool:
    """REC replays BASE's PPO policy, so only the MEM arms are recurrent."""
    return ARMS.get(arm, {}).get("algo") == "RecurrentPPO"


def algo_class(algo: str):
    if algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    from stable_baselines3 import PPO

    return PPO


def make_env(arm: str):
    """Spec section 2: the S5 RC configuration from scratch, plus the arm's one
    change. Every arm trains against GreedyEvader with the CBF shield in the
    loop and `shield_obs`; only gamma, the stall penalty, the escape features
    and the algorithm differ."""
    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP

    spec = ARMS[arm]
    cfg = EnvConfig()
    return SearchMDP(
        cfg,
        GreedyEvader,
        seed0=spec["seed"] * 1000,
        hold=HOLD,
        r_acq=R_ACQ,
        zero_reward=False,
        shield=CBFShield(cfg),
        shield_obs=SHIELD_OBS,
        stall_penalty=spec["stall_penalty"],
        escape_obs=spec["escape_obs"],
    )


def make_learner(arm: str, env, device: str = "cpu"):
    """SB3 defaults apart from the arm's gamma; MEM is the LSTM policy."""
    spec = ARMS[arm]
    cls = algo_class(spec["algo"])
    return cls(
        POLICY_STRING[spec["algo"]],
        env,
        seed=spec["seed"],
        gamma=spec["gamma"],
        verbose=0,
        device=device,
    )


def _model(path, algo: str, device: str):
    key = (str(path), algo)
    if key not in _MODELS:
        _MODELS[key] = algo_class(algo).load(path, device=device)
    return _MODELS[key]


def gate_opponent_factory(name: str, device: str):
    if name == "GREEDY":
        return GreedyEvader
    if name == "HUG7":
        return ObstacleHuggingEvader
    from pe2.evade import D_POS, StructuredEvaderPolicy

    cfg = EnvConfig()
    model = _model(ATK_CKPT, "PPO", device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=EVADER_HOLD, d_pos=D_POS)


# ----------------------------------------------------------------------------
# evaluation: a fresh shield per episode, and the LSTM state threaded
# ----------------------------------------------------------------------------
class RecurrentPredictor:
    """`model.predict` with the LSTM state carried through ONE episode.

    `SearchMDP.play_episode` calls `predict(obs, deterministic=True)` and
    discards the second return value, so a `RecurrentPPO` driven through it
    directly would start every decision from a zeroed hidden state — a
    memoryless policy wearing an LSTM, and arm MEM would be silently broken.
    `reset()` is called at each episode boundary, which is also what
    `episode_start` tells the model on the first decision of an episode."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._state = None
        self._episode_start = True

    def predict(self, obs, deterministic: bool = True):
        action, self._state = self.model.predict(
            obs,
            state=self._state,
            episode_start=np.array([self._episode_start]),
            deterministic=deterministic,
        )
        self._episode_start = False
        return action, self._state


def predictor_for(model, recurrent: bool):
    return RecurrentPredictor(model) if recurrent else model


def evaluate_arm_policy(
    model,
    cfg: EnvConfig,
    opponent_factory,
    n_episodes: int,
    seed0: int,
    *,
    shield_factory=None,
    shield_obs: bool = SHIELD_OBS,
    escape_obs: bool = False,
    recurrent: bool = False,
    hold: int = HOLD,
) -> list[dict]:
    """`pe2.structured.evaluate_structured` with two corrections.

    1. A FRESH shield per episode. The library builds one shield for a whole
       call on the documented assumption that a shield carries no cross-episode
       state; `RecoveryShield` breaks that assumption (its immobile run has no
       episode hook), so arm REC's first steps of episode i+1 would inherit
       episode i's open run.
    2. The LSTM state threaded through an episode and reset at its boundary,
       for arm MEM.

    Everything else is the library call unchanged: one `SearchMDP`, the same
    paired seeds `seed0 + i`, `play_episode` per seed — pinned identical to it
    for a stateless shield and a memoryless policy by
    `test_identical_to_evaluate_structured_for_a_stateless_shield`.
    """
    from pe2.structured import SearchMDP

    # Built with no shield and given one per episode below: `play_episode`
    # rebuilds `shield_metrics` from `self.shield` at every reset, so the only
    # difference from the library's construction is WHICH shield object each
    # episode runs under -- which is the whole point.
    mdp = SearchMDP(
        cfg,
        opponent_factory,
        hold=hold,
        shield=None,
        shield_obs=shield_obs,
        escape_obs=escape_obs,
    )
    predictor = predictor_for(model, recurrent)
    logs = []
    for i in range(n_episodes):
        if shield_factory is not None:
            mdp.shield = shield_factory()
        if recurrent:
            predictor.reset()
        logs.append(mdp.play_episode(predictor.predict, seed0 + i))
    return logs


def shield_factory_for(cfg: EnvConfig, recovery: bool):
    from pe2.shield import CBFShield, RecoveryShield

    if recovery:
        return lambda: RecoveryShield(cfg, recovery_after=RECOVERY_AFTER)
    return lambda: CBFShield(cfg)


def gate_arm(
    model,
    *,
    escape_obs: bool = False,
    recurrent: bool = False,
    recovery: bool = False,
    device: str = "cpu",
    holder: dict | None = None,
) -> dict:
    """Spec section 3: the gate stream, N = GATE_N, all three opponents, one
    entry each. Each entry is stashed into `holder["gate"]` as it completes, so
    a crash part-way through still hands `run_arm` the violations seen."""
    cfg = EnvConfig()
    factory = shield_factory_for(cfg, recovery)
    out: dict = {}
    for name in GATE_OPPONENTS:
        logs = evaluate_arm_policy(
            model,
            cfg,
            gate_opponent_factory(name, device),
            GATE_N,
            GATE_SEED0,
            shield_factory=factory,
            shield_obs=SHIELD_OBS,
            escape_obs=escape_obs,
            recurrent=recurrent,
        )
        k, n = win_count(logs, "pursuer")
        sh = summarize_shield(logs)
        out[name] = {
            "capture_ind": [lg["outcome"] == "capture" for lg in logs],
            "deadlock_ind": [bool(lg["shield"]["deadlock"]) for lg in logs],
            "capture_rate": k / n,
            "capture": _ci(k, n),
            "pursuer": {
                key: sh[key]
                for key in (
                    "immobile_step_rate",
                    "deadlock_episode_rate",
                    "stall_step_rate",
                    "intervention_rate",
                    "infeasibility_rate",
                    "backup_rate",
                    "min_clearance_floor",
                )
            },
            "n_steps": int(sh["n_steps"]),
            "n_immobile": int(sh["n_immobile"]),
            "pursuer_violations": int(sh["violation_steps"]),
        }
        if holder is not None:
            holder.setdefault("gate", {})[name] = out[name]
        print(
            f"    vs {name:8s}: immobile {sh['immobile_step_rate']:.3f}, "
            f"deadlock eps {sh['deadlock_episode_rate']:.3f}, capture {k / n:.3f}, "
            f"violations {sh['violation_steps']}",
            flush=True,
        )
    return out


# ----------------------------------------------------------------------------
# guards (spec section 6's stop rule)
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule, spec section 6).")
    if any(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json")) or any(
        _ckpt_dir(a).exists() for a in ALL_ARMS
    ):
        raise SystemExit(
            "S16 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if _arm_file(arm).exists() or _ckpt_dir(arm).exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running it."
        )


def load_start() -> dict:
    """An arm refuses to run before the round has a recorded starting point,
    and refuses to run against a gate opponent that has changed since."""
    if not START_PATH.exists():
        raise SystemExit(f"round not started: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    if "attacker" not in s or "sha256" not in (s.get("attacker") or {}):
        raise SystemExit(
            f"{START_PATH} has no attacker record; re-run --measure-start."
        )
    ckpt = pathlib.Path(s["attacker"]["ckpt"])
    if ckpt.exists() and _sha256(ckpt) != s["attacker"]["sha256"]:
        raise SystemExit(
            f"{ckpt} has changed since the round started; the gate opponent is not the "
            "one this round was pre-registered against."
        )
    return s


def measure_start() -> dict:
    """The round's starting point. Spec section 3 pre-registers no reference to
    measure — every comparison is within-round — so this records what must be
    fixed before any arm runs rather than a number: the code version, the
    constants, and the identity of the learned gate opponent."""
    require_not_measured()
    if not ATK_CKPT.exists():
        raise SystemExit(f"gate opponent missing: {ATK_CKPT}")
    out = {
        "spec": SPEC,
        "seeds": list(SEEDS),
        "arms": sorted(ALL_ARMS),
        "gate_seed0": GATE_SEED0,
        "gate_n": GATE_N,
        "probe_seed0": PROBE_SEED0,
        "constants": {
            "total": TOTAL,
            "every": EVERY,
            "eval_n": EVAL_N,
            "hold": HOLD,
            "r_acq": R_ACQ,
            "shield_obs": SHIELD_OBS,
            "gamma": GAMMA,
            "gamma_long": GAMMA_LONG,
            "stall_penalty": STALL_W,
            "stall_penalty_high": STALL_W_HIGH,
            "recovery_after": RECOVERY_AFTER,
            "solve_bar": SOLVE_BAR,
            "match_bar": MATCH_BAR,
            "gam_bar": GAM_BAR,
            "pass_min": PASS_MIN,
            "p_gate": P_GATE,
        },
        "gate_opponents": list(GATE_OPPONENTS),
        "attacker": {
            "ckpt": str(ATK_CKPT),
            "bytes": ATK_CKPT.stat().st_size,
            "sha256": _sha256(ATK_CKPT),
        },
        "provenance": provenance(),
    }
    _write_json(START_PATH, out)
    print(
        f"S16 started: {len(ARMS)} training arms + {len(REC_ARMS)} REC evaluations, "
        f"gate stream {GATE_SEED0}+ x {GATE_N}, attacker {ATK_CKPT.name} "
        f"sha256 {out['attacker']['sha256'][:12]}",
        flush=True,
    )
    return out


# ----------------------------------------------------------------------------
# training arms
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """S5's shielded env-step-grid probe, carrying this round's primary metric
    (the immobile-step rate) and the arm's own evaluation path — the escape
    features when it has them, the LSTM state when it is recurrent."""

    def __init__(
        self,
        cfg,
        opponent_factory,
        every,
        total,
        eval_n,
        seed0,
        ckpt_dir,
        report_path,
        label,
        escape_obs,
        recurrent,
        hold=HOLD,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []
        factory = shield_factory_for(cfg, recovery=False)

        def evaluate_at(
            model, env_steps: int, mark: int, train_violations: int
        ) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_arm_policy(
                model,
                cfg,
                opponent_factory,
                eval_n,
                seed0,
                shield_factory=factory,
                shield_obs=SHIELD_OBS,
                escape_obs=escape_obs,
                recurrent=recurrent,
                hold=hold,
            )
            k, n = win_count(logs, "pursuer")
            sh = summarize_shield(logs)
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "win_rate": k / n,
                "k": k,
                "n": n,
                "violations": int(sh["violation_steps"]),
                "train_violations": int(train_violations),
                "immobile_step_rate": sh["immobile_step_rate"],
                "intervention_rate": sh["intervention_rate"],
                "stall_step_rate": sh["stall_step_rate"],
                "deadlock_episode_rate": sh["deadlock_episode_rate"],
            }
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): capture {k}/{n} = {k / n:.3f}, "
                f"immobile {row['immobile_step_rate']:.3f}, deadlock eps "
                f"{row['deadlock_episode_rate']:.3f}, violations {row['violations']} "
                f"(train {row['train_violations']})",
                flush=True,
            )
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1
                env_steps = int(counts[0])
                viol = int(inner_self.training_env.get_attr("violations_total")[0])
                while (
                    inner_self.next_mark <= total and env_steps >= inner_self.next_mark
                ):
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark, viol)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()


def _arm_verdict(gate: dict) -> tuple[str, str]:
    """The per-seed clause of H-FIX-1, and the row the summary reads."""
    e = gate[PRIMARY_OPPONENT]
    rate = e["pursuer"]["immobile_step_rate"]
    verdict = "BELOW BAR" if rate < SOLVE_BAR else "AT OR ABOVE BAR"
    reason = (
        f"immobile vs {PRIMARY_OPPONENT} {rate:.3f} (bar {SOLVE_BAR:.2f}); "
        + "; ".join(
            f"{o} immobile {gate[o]['pursuer']['immobile_step_rate']:.3f} "
            f"capture {gate[o]['capture_rate']:.3f}"
            for o in GATE_OPPONENTS
        )
    )
    return verdict, reason


def run_train_arm(arm: str, device: str, holder: dict | None = None) -> dict:
    spec = ARMS[arm]
    cfg = EnvConfig()
    d = _ckpt_dir(arm)
    d.mkdir(parents=True, exist_ok=True)
    recurrent = is_recurrent(arm)
    probe = ShieldedStructuredProbe(
        cfg,
        GreedyEvader,
        every=EVERY,
        total=TOTAL,
        eval_n=EVAL_N,
        seed0=PROBE_SEED0,
        ckpt_dir=d,
        report_path=_probe_file(arm),
        label=arm,
        escape_obs=spec["escape_obs"],
        recurrent=recurrent,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = make_env(arm)
    if holder is not None:
        holder["env"] = env
    model = make_learner(arm, env, device)
    model.learn(total_timesteps=TOTAL, callback=probe.callback)
    result = {
        "arm": arm,
        **spec,
        "obs_width": int(env.observation_space.shape[0]),
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "train_violations": int(env.violations_total),
        "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
        "checkpoints": probe.rows,
        "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not any(r["mark"] == TOTAL for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = (
            "UNDETERMINED",
            f"final probe at mark {TOTAL} never fired (env steps {env.env_steps})",
        )
        return result
    print(f"[{arm}] gate:", flush=True)
    gate = gate_arm(
        algo_class(spec["algo"]).load(ckpt, device=device),
        escape_obs=spec["escape_obs"],
        recurrent=recurrent,
        recovery=False,
        device=device,
        holder=holder,
    )
    result["final"] = {
        "ckpt": str(ckpt),
        "gate": gate,
        "pursuer_violations": int(sum(e["pursuer_violations"] for e in gate.values())),
    }
    verdict, reason = _arm_verdict(gate)
    result["verdict"] = verdict
    result["reason"] = (
        f"{reason}; violations train {result['train_violations']} probe "
        f"{result['probe_violations']} gate {result['final']['pursuer_violations']}"
    )
    return result


def run_arm(arm: str, device: str) -> dict:
    require_arm_not_run(arm)  # refuse BEFORE reading or writing anything
    load_start()
    holder: dict = {}
    try:
        result = run_train_arm(arm, device, holder)
    except BaseException as e:
        _write_json(
            _arm_file(arm),
            {
                "arm": arm,
                **ARMS[arm],
                "verdict": "UNDETERMINED",
                "reason": f"crashed: {type(e).__name__}: {e}",
                "checkpoints": holder.get("rows", []),
                "train_violations": int(
                    getattr(holder.get("env"), "violations_total", 0)
                ),
                "probe_violations": int(
                    sum(r.get("violations", 0) for r in holder.get("rows", []))
                ),
                "gate_violations": int(
                    sum(
                        e["pursuer_violations"] for e in holder.get("gate", {}).values()
                    )
                ),
                "gate_partial": holder.get("gate", {}),
                "provenance": provenance(),
            },
        )
        raise
    _write_json(_arm_file(arm), result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# the REC arm: no training, BASE's policies with the rotation recovery
# ----------------------------------------------------------------------------
def rec_seeds() -> list[int]:
    """The seeds `--rec` still has to evaluate: a BASE final checkpoint exists
    and no REC record does. Refuses outright when no BASE arm has produced a
    final checkpoint — REC is BASE's policies, and there are none yet."""
    have_base = [
        s for s in SEEDS if (_ckpt_dir(f"BASE-{s}") / f"BASE-{s}_{TOTAL}.zip").exists()
    ]
    if not have_base:
        raise SystemExit(
            "--rec has nothing to evaluate: REC replays the BASE policies and no BASE "
            f"arm has a final checkpoint yet (looked for ckpts_BASE-<seed>/BASE-<seed>_{TOTAL}.zip). "
            "Run the BASE arms first."
        )
    return [s for s in have_base if not _arm_file(f"REC-{s}").exists()]


def run_rec_arm(seed: int, device: str) -> dict:
    arm = f"REC-{seed}"
    require_arm_not_run(arm)
    load_start()
    source = REC_ARMS[arm]["source"]
    ckpt = _ckpt_dir(source) / f"{source}_{TOTAL}.zip"
    if not ckpt.exists():
        raise SystemExit(f"{arm} needs {source}'s final checkpoint: {ckpt} is missing.")
    holder: dict = {}
    print(
        f"[{arm}] gate ({source} + RecoveryShield after {RECOVERY_AFTER} immobile steps):",
        flush=True,
    )
    try:
        gate = gate_arm(
            algo_class(ARMS[source]["algo"]).load(ckpt, device=device),
            escape_obs=ARMS[source]["escape_obs"],
            recurrent=is_recurrent(source),
            recovery=True,
            device=device,
            holder=holder,
        )
    except BaseException as e:
        _write_json(
            _arm_file(arm),
            {
                "arm": arm,
                **REC_ARMS[arm],
                "verdict": "UNDETERMINED",
                "reason": f"crashed: {type(e).__name__}: {e}",
                "gate_violations": int(
                    sum(
                        x["pursuer_violations"] for x in holder.get("gate", {}).values()
                    )
                ),
                "gate_partial": holder.get("gate", {}),
                "provenance": provenance(),
            },
        )
        raise
    result = {
        "arm": arm,
        **REC_ARMS[arm],
        "recovery_after": RECOVERY_AFTER,
        "source_ckpt": str(ckpt),
        # REC trains nothing: its training and probe contact steps ARE the
        # source arm's, counted there, and double-counting them here would
        # inflate the H-SAFE census.
        "train_violations": 0,
        "probe_violations": 0,
        "checkpoints": [],
        "final": {
            "ckpt": str(ckpt),
            "gate": gate,
            "pursuer_violations": int(
                sum(e["pursuer_violations"] for e in gate.values())
            ),
        },
        "provenance": provenance(),
    }
    verdict, reason = _arm_verdict(gate)
    result["verdict"] = verdict
    result["reason"] = (
        f"{reason}; gate violations {result['final']['pursuer_violations']}"
    )
    _write_json(_arm_file(arm), result)
    print(f"[{arm}] {verdict}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 4)
# ----------------------------------------------------------------------------
def _entry(results: dict, arm: str, opponent: str) -> dict | None:
    gate = ((results.get(arm) or {}).get("final") or {}).get("gate") or {}
    return gate.get(opponent)


def immobile_rate(results: dict, arm: str, opponent: str = PRIMARY_OPPONENT):
    e = _entry(results, arm, opponent)
    return None if e is None else float(e["pursuer"]["immobile_step_rate"])


def _capture_ind(results: dict, arm: str) -> list[bool] | None:
    e = _entry(results, arm, CAPTURE_OPPONENT)
    return None if e is None else list(e["capture_ind"])


def _mean(values) -> float:
    return sum(values) / len(values)


def _capture_guard(results: dict, arm: str, seed: int) -> dict | None:
    """H-FIX-1's second clause, seed-matched: is BASE significantly better at
    capture than this arm on the same episodes? `pass` = yes."""
    base = _capture_ind(results, f"BASE-{seed}")
    arm_ind = _capture_ind(results, arm)
    if base is None or arm_ind is None:
        return None
    return paired_indicator_test(base, arm_ind, P_GATE)


def fix1_verdicts(results: dict) -> dict:
    """H-FIX-1: an arm SOLVES iff its immobile-step rate against the primary
    opponent is < SOLVE_BAR on >= PASS_MIN seeds AND BASE is not significantly
    better at capture (see the module docstring for how the guard is applied)."""
    out: dict = {}
    for fam in FAMILY_ORDER:
        rates = {s: immobile_rate(results, f"{fam}-{s}") for s in SEEDS}
        guards = {s: _capture_guard(results, f"{fam}-{s}", s) for s in SEEDS}
        pooled = None
        base_ind: list[bool] = []
        arm_ind: list[bool] = []
        for s in SEEDS:
            b, a = (
                _capture_ind(results, f"BASE-{s}"),
                _capture_ind(results, f"{fam}-{s}"),
            )
            if b is not None and a is not None:
                base_ind += b
                arm_ind += a
        if base_ind:
            pooled = paired_indicator_test(base_ind, arm_ind, P_GATE)
        if any(v is None for v in rates.values()) or any(
            g is None for g in guards.values()
        ):
            out[fam] = {
                "verdict": "INCOMPLETE",
                "immobile": rates,
                "n_below_bar": sum(
                    v is not None and v < SOLVE_BAR for v in rates.values()
                ),
                "capture_guard": {s: g for s, g in guards.items() if g is not None},
                "pooled_capture_guard": pooled,
            }
            continue
        n_below = sum(v < SOLVE_BAR for v in rates.values())
        weaker = sum(bool(g["pass"]) for g in guards.values()) >= PASS_MIN
        if n_below >= PASS_MIN:
            verdict = "SOLVES (weaker)" if weaker else "SOLVES"
        else:
            verdict = "FAILS"
        out[fam] = {
            "verdict": verdict,
            "immobile": rates,
            "mean_immobile": _mean(list(rates.values())),
            "n_below_bar": n_below,
            "capture": {
                s: _entry(results, f"{fam}-{s}", CAPTURE_OPPONENT)["capture_rate"]
                for s in SEEDS
            },
            "capture_guard": guards,
            "pooled_capture_guard": pooled,
        }
    return out


def solving(fix1: dict) -> list[str]:
    """The families that SOLVE the loop, in the arm table's order."""
    return [f for f in FAMILY_ORDER if str(fix1[f]["verdict"]).startswith("SOLVES")]


def fix2_verdicts(results: dict) -> dict:
    """H-FIX-2, the headline: does a learned arm MATCH the scripted recovery?
    Read one-sided (see the module docstring); the two-sided flag is recorded."""
    rec = [immobile_rate(results, f"REC-{s}") for s in SEEDS]
    rec_mean = None if any(v is None for v in rec) else _mean(rec)
    per: dict = {}
    for fam in LEARNED:
        rates = [immobile_rate(results, f"{fam}-{s}") for s in SEEDS]
        if any(v is None for v in rates) or rec_mean is None:
            per[fam] = {
                "verdict": "INCOMPLETE",
                "mean": None if any(v is None for v in rates) else _mean(rates),
                "delta": None,
                "beats": None,
                "two_sided_within": None,
            }
            continue
        mean = _mean(rates)
        delta = mean - rec_mean
        per[fam] = {
            "verdict": "MATCHES" if delta <= MATCH_BAR else "DOES NOT MATCH",
            "mean": mean,
            "delta": delta,
            "beats": bool(delta < -MATCH_BAR),
            "two_sided_within": bool(abs(delta) <= MATCH_BAR),
        }
    matching = [f for f in REMEDIES if per[f]["verdict"] == "MATCHES"]
    if rec_mean is None or any(v["verdict"] == "INCOMPLETE" for v in per.values()):
        headline = "INCOMPLETE"
    elif matching:
        headline = (
            "REFUTED: the pre-registered expectation was that no learned arm matches "
            f"REC; {', '.join(matching)} does"
        )
    else:
        headline = "CONFIRMED: no learned arm matches REC"
    return {
        "rec_mean": rec_mean,
        "rec": {s: v for s, v in zip(SEEDS, rec, strict=True)},
        "per_family": per,
        "matching": matching,
        "base_matches": per["BASE"]["verdict"] == "MATCHES",
        "H_FIX_2": headline,
    }


def fix3_verdicts(results: dict) -> dict:
    """H-FIX-3, the discount test: GAM below BASE by >= GAM_BAR on >= PASS_MIN
    seeds. Reported whether or not GAM solves the loop."""
    per: dict = {}
    for s in SEEDS:
        base = immobile_rate(results, f"BASE-{s}")
        gam = immobile_rate(results, f"GAM-{s}")
        per[s] = {
            "base": base,
            "gam": gam,
            "delta": None if base is None or gam is None else base - gam,
        }
    if any(v["delta"] is None for v in per.values()):
        return {"verdict": "INCOMPLETE", "per_seed": per, "n_pass": 0}
    n_pass = sum(v["delta"] >= GAM_BAR for v in per.values())
    return {
        "verdict": "PASS" if n_pass >= PASS_MIN else "FAIL",
        "per_seed": per,
        "n_pass": n_pass,
    }


def hsafe(results: dict) -> dict:
    """H-SAFE: 0 pursuer contact steps in training, probes and gates, every arm
    including REC (whose training and probe steps are BASE's and counted there)."""
    steps, offenders = 0, []
    missing = [a for a in ALL_ARMS if a not in results or "final" not in results[a]]
    for a in sorted(results):
        r = results[a]
        v = (
            int(r.get("train_violations", 0))
            + int(r.get("probe_violations", 0))
            # a crash record has no "final"; gate_violations is what the gate
            # had already seen when the process died
            + int(r.get("gate_violations", 0))
            + int((r.get("final") or {}).get("pursuer_violations", 0))
        )
        if "probe_violations" not in r:
            v += int(sum(x.get("violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    return {
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "pursuer_violation_steps": steps,
        "offenders": offenders,
        "missing": missing,
    }


# ----------------------------------------------------------------------------
# what each outcome fixes (spec section 5)
# ----------------------------------------------------------------------------
MATCH_DIRECTION = {
    "OBS": (
        "OBS matches REC: the frozen-input half is the binding constraint and the fix is an "
        "observation-design rule — expose the filter's LINEARISED admissible set to the policy "
        "(amendment S16-A1: the feature is the DCBF rows at v = 0, not the filter's exact "
        "one-step test, and a failure of OBS does not rule out that the exact set would work)."
    ),
    "GAM": (
        "GAM matches REC: the discount half is binding; the fix is a horizon rule and "
        "generalises to any long-delay filtered task."
    ),
    "MEM": (
        "MEM matches REC: memory alone suffices; the claim becomes one about memoryless "
        "architectures specifically."
    ),
    "PEN": (
        "PEN matches REC: D113 was a budget artefact and the simplest fix works after all — "
        "report and re-scope D113."
    ),
}


def direction(h: dict) -> str:
    """Spec section 5, row by row."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no "
            "further round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["incomplete"]:
        return (
            f"incomplete: all {len(ARMS)} training arms and the {len(REC_ARMS)} REC "
            "evaluations must run"
        )
    warning = (
        "BASE matches REC: the control is already within the match bar of the scripted "
        "recovery, so the loop is not present at this budget and the round cannot "
        "discriminate between remedies. "
        if h.get("base_matches")
        else ""
    )
    solving_now = list(h["solving"])
    matching = list(h["matching"])
    if not solving_now:
        return warning + (
            "No arm solves the loop, REC included: the recovery's effect in D125 was specific "
            "to the policies it was applied to; the round is reported as a failure and Phase 2 "
            "is redesigned."
        )
    if "REC" not in solving_now:
        learned = [f for f in solving_now if f != "REC"]
        return warning + (
            f"REC does NOT solve the loop but {', '.join(learned)} does: the scripted recovery "
            "is not the baseline this round assumed. Report the learned arms' result and "
            "re-examine D125's recovery on these policies"
            + (f" (matching REC: {', '.join(matching)})" if matching else "")
            + "."
        )
    if matching:
        text = MATCH_DIRECTION[matching[0]]
        if len(matching) > 1:
            text += f" Also matching REC: {', '.join(matching[1:])}."
        return warning + text
    return warning + (
        "REC solves and no learned arm matches it — the expected and most useful outcome: the "
        "remedy is an engineering one, not a learning one, and the paper says so plainly."
    )


# ----------------------------------------------------------------------------
# summary and figure
# ----------------------------------------------------------------------------
def load_results() -> dict:
    return {
        p.stem.removeprefix(f"{ARTEFACT_PREFIX}_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json"))
    }


def summarize(results: dict, start: dict | None = None) -> dict:
    f1 = fix1_verdicts(results)
    f2 = fix2_verdicts(results)
    f3 = fix3_verdicts(results)
    hs = hsafe(results)
    complete = all(a in results and "final" in results[a] for a in ALL_ARMS)
    table = {}
    for a in ALL_ARMS:
        r = results.get(a)
        gate = ((r or {}).get("final") or {}).get("gate") or {}
        table[a] = {
            "verdict": (r or {}).get("verdict"),
            "immobile": immobile_rate(results, a),
            "capture": (gate.get(CAPTURE_OPPONENT) or {}).get("capture_rate"),
            "deadlock": ((gate.get(PRIMARY_OPPONENT) or {}).get("pursuer") or {}).get(
                "deadlock_episode_rate"
            ),
            "gate": {
                o: {
                    "immobile": gate[o]["pursuer"]["immobile_step_rate"],
                    "deadlock": gate[o]["pursuer"]["deadlock_episode_rate"],
                    "capture": gate[o]["capture_rate"],
                }
                for o in gate
            },
        }
    return {
        "spec": SPEC,
        "hsafe": hs,
        "hypotheses": {"H_FIX_1": f1, "H_FIX_2": f2, "H_FIX_3": f3},
        "solving": solving(f1),
        "arms": table,
        "gate": {
            "seed0": GATE_SEED0,
            "n": GATE_N,
            "opponents": list(GATE_OPPONENTS),
            "primary": PRIMARY_OPPONENT,
        },
        "start": (start or {}).get("provenance"),
        "complete": complete,
        "direction": direction(
            {
                "hsafe": hs["verdict"],
                "incomplete": not complete,
                "solving": solving(f1),
                "matching": f2["matching"],
                "base_matches": f2["base_matches"],
            }
        ),
    }


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {
        "BASE": "k",
        "GAM": "tab:blue",
        "OBS": "tab:green",
        "MEM": "tab:orange",
        "PEN": "tab:purple",
        "REC": "tab:red",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    for arm, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x.get("immobile_step_rate") for x in rows],
                marker=".",
                color=colour.get(r.get("family"), "0.5"),
                alpha=0.8,
                label=arm,
            )
    ax.axhline(SOLVE_BAR, ls=":", color="k", lw=1.2, label=f"solve bar = {SOLVE_BAR}")
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"probe immobile-step rate (GREEDY, seeds {PROBE_SEED0}+)")
    ax.set_title("S16 — the loop during training")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    names = [a for a in ALL_ARMS if summary["arms"][a]["immobile"] is not None]
    if names:
        xs = range(len(names))
        ax.bar(
            list(xs),
            [summary["arms"][a]["immobile"] for a in names],
            color=[colour.get(a.split("-")[0], "0.5") for a in names],
        )
        ax.axhline(
            SOLVE_BAR, ls=":", color="k", lw=1.2, label=f"solve bar = {SOLVE_BAR}"
        )
        rec_mean = summary["hypotheses"]["H_FIX_2"]["rec_mean"]
        if rec_mean is not None:
            ax.axhline(
                rec_mean + MATCH_BAR,
                ls="--",
                color="tab:red",
                lw=1.0,
                label=f"REC mean + {MATCH_BAR} = {rec_mean + MATCH_BAR:.3f}",
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=7, rotation=60)
        ax.set_ylabel(f"gate immobile-step rate vs {PRIMARY_OPPONENT}")
        ax.set_title(f"H-FIX-2: {summary['hypotheses']['H_FIX_2']['H_FIX_2']}")
        ax.legend(fontsize=7)
    fig.suptitle(
        "S16 — five remedies for the loop"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--rec", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start()
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.rec:
        todo = rec_seeds()
        if not todo:
            print(
                "--rec: every REC evaluation already has a record (stop rule).",
                flush=True,
            )
        for seed in todo:
            run_rec_arm(seed, args.device)
    if args.summarize:
        results = load_results()
        start = json.loads(START_PATH.read_text()) if START_PATH.exists() else None
        summary = summarize(results, start)
        _write_json(_round_file("json"), {"summary": summary, "arms": results})
        write_figure(results, summary, _round_file("png"))
        print(
            json.dumps(
                {
                    "hsafe": summary["hsafe"],
                    "H_FIX_1": {
                        f: summary["hypotheses"]["H_FIX_1"][f]["verdict"]
                        for f in FAMILY_ORDER
                    },
                    "H_FIX_2": summary["hypotheses"]["H_FIX_2"]["H_FIX_2"],
                    "H_FIX_3": summary["hypotheses"]["H_FIX_3"]["verdict"],
                    "solving": summary["solving"],
                },
                indent=2,
                default=str,
            )
        )
        print(json.dumps(summary["arms"], indent=2, default=str))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
