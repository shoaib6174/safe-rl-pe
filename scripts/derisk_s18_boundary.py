"""S18 — the boundary condition: when does penalising filter intervention help?

Spec: docs/pathway/boundary_condition_prereg_2026-09-23.md INCLUDING Amendment A1.
Nothing here may change a threshold, a seed, a stream, a rule or a budget after a
number is seen (spec section 7).

The disagreement this round settles. Pizarro Bejarano et al. (arXiv:2410.11671)
report that penalising safety-filter interventions during training IMPROVES the
learned policy; D129 found the opposite on three seeds — the penalty backfired,
teaching the agent to stop asking to move. Hypothesis B says the two results
differ because of WHERE the task reward has gradient, not because of the penalty.

So the independent variable is gradient-on-`I` (the set of actions available
where the filter intervenes), not reward density, and the design is 2 x 3:

  column   reward variant                    dense?  varies across actions on I?
  SPARSE   terminal +/-1 and r_acq only      no      no
  DFLAT    + w_vis (DenseRewardWrapper)      YES     no  (the evader is invisible
                                                          inside a stalled pocket)
  DVARY    + w_search (staleness)            YES     YES

  row      BASE  intervention_penalty = 0
           PEN   intervention_penalty = lambda

DFLAT is the discriminating cell: it is dense and flat on `I`, so if the variable
were density it would pattern with DVARY and if it is gradient-on-`I` it patterns
with SPARSE (H-MECH).

Three things this round gets right on purpose, each of them a past error:

  - THE PENALTY IS THEIRS, NOT D129's. `SearchMDP.intervention_penalty` subtracts
    lambda * ||a_nominal - a_safe||_2 per shielded env step in normalised action
    units — the magnitude of the filter's correction, defined everywhere.
    `stall_penalty` (D129's stalled-step penalty) is a DIFFERENT quantity, is not
    an arm of this round, and is pinned to 0.0 in every cell here.

  - THE MECHANISM METRIC MEASURES THE REQUEST, NOT THE EXECUTION. `H-ZERO`'s
    zero-motion fraction counts the LEARNER's nominal — what the policy asked
    for, before the filter touched it — because the whole claim is that the agent
    stops ASKING to move. S16 measured the execution and thereby counted a
    remedy's own mechanism as the disease (D129); `RecordingSearchMDP` records
    the action handed to `step()`, which is exactly the request.

  - EVALUATION IS UNWRAPPED (amendment A1.3). `play_episode` discards rewards and
    reports outcomes, and `TestRewardWrapperHook::test_reward_shaping_cannot_
    change_the_trajectory` proves a wrapper cannot change an episode — but the
    gate still builds its `SearchMDP` with `reward_wrapper=None` explicitly, so
    the property does not rest on that argument alone.

lambda, w_vis and w_search are CALIBRATED, never chosen: they are read from
`results/derisk/s18_lambda.json`, written by `scripts/derisk_s18_lambda.py`
(spec section 3 + A1.2). An arm that cannot read that file REFUSES TO RUN.

Usage:
  ./venv/bin/python scripts/derisk_s18_boundary.py --arms SPARSE-PEN-70  (one/process)
  ./venv/bin/python scripts/derisk_s18_boundary.py --arms                (all not yet run)
  ./venv/bin/python scripts/derisk_s18_boundary.py --arms DVARY-PEN-71 --gate-only
  ./venv/bin/python scripts/derisk_s18_boundary.py --summarize
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
from pe2.obs import obs_dim
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.shield import STALL_V_FRAC
from pe2.structured import HOLD, SearchMDP

SPEC = "docs/pathway/boundary_condition_prereg_2026-09-23.md"
ARTEFACT_PREFIX = "s18"
OUT = pathlib.Path("results/derisk")

# --- pre-registered (spec sections 3-4 and amendment A1). Do not tune. ---
SEEDS = (70, 71, 72)
TOTAL = 5_000_000
EVERY = 500_000
GATE_SEED0 = 1_900_000  # A1: the gate
PROBE_SEED0 = 1_950_000  # A1: training probes
GATE_N = 500
EVAL_N = 200  # probe episodes; the spec fixes the probe STREAM, not its N
GAMMA = 0.99
R_ACQ = 0.0
SHIELD_OBS = True
ESCAPE_OBS = True
# D129's quantity, explicitly off: the spec (section 3) keeps `stall_penalty` in
# the code untouched and makes it no arm of this round.
STALL_PENALTY = 0.0
# RecoveryShieldV2 is OFF (section 4): the recovery is D131's remedy and would
# mask the effect under test. The shield is the plain hard filter.
RECOVERY = False
P_GATE = 0.05
PASS_MIN = 2  # ">= 2/3 seeds", every hypothesis
# H-ZERO's threshold IS the filter's own immobility threshold, so "the policy
# asked for no motion" and "the filter calls this stopped" cannot disagree.
ZERO_MOTION_V_FRAC = STALL_V_FRAC

GATE_OPPONENTS = ("GREEDY", "HUG")
CAPTURE_OPPONENT = "GREEDY"  # the primary metric's opponent (section 4)
# AMENDMENT A2. Section 4 originally named `StructuredEvaderPolicy` as the
# robustness opponent while also saying "the learned attacker family is
# deliberately absent" -- but that class takes a LEARNED MODEL as its first
# argument, so the two clauses could not both hold. Resolved in the spec before
# any arm ran: the robustness opponent is the SCRIPTED full-state
# `ObstacleHuggingEvader` (S17's HUG7). No learned attacker is trained,
# selected or loaded anywhere in S18.

# Spec section 2's table, column by column. `weight` names the
# `DenseRewardWrapper` keyword the column's CALIBRATED weight is passed as;
# `None` is the SPARSE column, which takes no wrapper at all.
COLUMNS: dict[str, dict] = {
    "SPARSE": {"weight": None, "dense": False, "varies_on_I": False},
    "DFLAT": {"weight": "w_vis", "dense": True, "varies_on_I": False},
    "DVARY": {"weight": "w_search", "dense": True, "varies_on_I": True},
}
ROWS: dict[str, bool] = {"BASE": False, "PEN": True}  # penalty off / on
CELLS: dict[str, dict] = {
    f"{col}-{row}": {"column": col, "row": row, "penalty": pen, **spec}
    for col, spec in COLUMNS.items()
    for row, pen in ROWS.items()
}
CELL_ORDER: tuple[str, ...] = tuple(CELLS)
# `DFLAT-BASE-70`.split("-") happens to recover the parts, but nothing may rely
# on that: the cell and the seed are STORED and every reader goes through the
# table (S17's convention, after `RECF-BASE-60` broke the parsing route).
ARMS: dict[str, dict] = {
    f"{cell}-{s}": {"cell": cell, "seed": s, **spec}
    for cell, spec in CELLS.items()
    for s in SEEDS
}

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


def _lambda_file() -> pathlib.Path:
    """The calibration artefact (spec section 3 + A1.2), written by
    `scripts/derisk_s18_lambda.py`. A function rather than a constant so it
    follows `OUT`."""
    return OUT / f"{ARTEFACT_PREFIX}_lambda.json"


def _ckpt_dir(arm: str) -> pathlib.Path:
    # Spec section 8: `results/derisk/ckpts_S18-{CELL}-{SEED}/`.
    return OUT / f"ckpts_S18-{arm}"


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
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


# ----------------------------------------------------------------------------
# the calibration (spec section 3 + A1.2): read, never chosen
# ----------------------------------------------------------------------------
CALIBRATION_KEYS = ("lambda", "w_vis", "w_search")


def load_calibration() -> dict:
    """The three calibrated weights. **An arm that cannot read the calibration
    file refuses to run** (spec section 3, restated in A1.2).

    The refusal is not defensive coding: one `lambda`, shared by all three
    penalty cells, is what keeps the column comparison unconfounded by penalty
    strength, and both dense weights are calibrated so the columns differ in
    WHERE the reward has gradient and in nothing else. A hardcoded or defaulted
    weight would silently destroy the round's only claim."""
    path = _lambda_file()
    if not path.exists():
        raise SystemExit(
            f"missing calibration {path}: spec section 3 requires lambda, w_vis "
            "and w_search to be measured before any arm trains. Run "
            "scripts/derisk_s18_lambda.py first. No arm may run without it and "
            "no weight here may be hardcoded."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"calibration {path} is not readable JSON: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit(f"calibration {path} is not an object: {type(data).__name__}")
    missing = [k for k in CALIBRATION_KEYS if k not in data]
    if missing:
        raise SystemExit(
            f"calibration {path} is missing {missing}; the schema is "
            f"{{{', '.join(CALIBRATION_KEYS)}, means, n_episodes, seed0, provenance}}."
        )
    for key in CALIBRATION_KEYS:
        value = data[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise SystemExit(
                f"calibration {path}: {key} = {value!r} is not a positive finite "
                "number. A non-positive weight means the calibrated term never "
                "fired, and the column it defines would be mislabelled."
            )
    return data


# ----------------------------------------------------------------------------
# arms: reward variant, env, learner, opponents
# ----------------------------------------------------------------------------
def reward_wrapper_for(column: str, calibration: dict):
    """The column's TRAINING-ONLY reward variant (spec section 2), as the
    callable `SearchMDP(reward_wrapper=...)` applies to the inner env.

    `None` for SPARSE — no wrapper at all, so the SPARSE column really is the
    S16/S17 setting and not a dense wrapper at weight zero."""
    weight = COLUMNS[column]["weight"]
    if weight is None:
        return None
    from pe2.rewards import DenseRewardWrapper

    w = float(calibration[weight])
    return lambda env: DenseRewardWrapper(env, **{weight: w})


def intervention_penalty_for(arm: str, calibration: dict) -> float:
    """`lambda` on a PEN arm, 0.0 on a BASE arm. The one place the row becomes
    a number."""
    return float(calibration["lambda"]) if ARMS[arm]["penalty"] else 0.0


def obs_width() -> int:
    """Every arm of this round has the same observation: the env observation,
    plus `shield_obs`'s three features, plus `escape_obs`'s two."""
    return obs_dim(EnvConfig()) + 3 * int(SHIELD_OBS) + 2 * int(ESCAPE_OBS)


def make_env(arm: str, calibration: dict):
    """The S17 `BASE` recipe with this round's two changes: the arm's reward
    variant (column) and its intervention penalty (row). Everything else —
    hold, shield, shield_obs, escape_obs, r_acq — is section 4's protocol."""
    from pe2.shield import CBFShield

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
        stall_penalty=STALL_PENALTY,
        escape_obs=ESCAPE_OBS,
        intervention_penalty=intervention_penalty_for(arm, calibration),
        reward_wrapper=reward_wrapper_for(spec["column"], calibration),
    )


def make_learner(arm: str, env, device: str = "cpu"):
    """PPO exactly as S17's BASE family configured it: SB3 defaults apart from
    the seed and gamma."""
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        seed=ARMS[arm]["seed"],
        gamma=GAMMA,
        verbose=0,
        device=device,
    )


def _model(path, device: str):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


def gate_opponent_factory(name: str, device: str):
    if name == "GREEDY":
        return GreedyEvader
    if name == "HUG":
        return ObstacleHuggingEvader
    raise ValueError(f"unknown gate opponent {name!r}; expected {GATE_OPPONENTS}")


# ----------------------------------------------------------------------------
# H-ZERO's mechanism metric: the REQUEST, not the execution
# ----------------------------------------------------------------------------
def is_zero_motion(action, cfg: EnvConfig) -> bool:
    """H-ZERO's per-decision test on the POLICY'S NOMINAL: the commanded speed
    is `<= ZERO_MOTION_V_FRAC * v_max`.

    `action` is the normalised Box action the learner emitted, clipped exactly
    as `SearchMDP._step_inner` clips it; the commanded speed is
    `(a0 + 1) / 2 * v_max`, the same inverse `SingleAgentWrapper` applies.

    This measures what the agent ASKED FOR, before the filter touched it. That
    is the whole of the claim: under the penalty the agent stops asking to move,
    which satisfies the penalty perfectly and destroys the task. Measuring the
    EXECUTED speed instead would count the filter's own corrections as the
    agent's choice — the D129 error, which made S16 unreadable."""
    a0 = float(np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)[0])
    return (a0 + 1.0) / 2.0 * cfg.v_max <= ZERO_MOTION_V_FRAC * cfg.v_max


class RecordingSearchMDP(SearchMDP):
    """`SearchMDP` that records, per episode, the learner's REQUESTS and the
    authoritative intervention magnitude.

    Both quantities pass through `step()` and nowhere else: its `action`
    argument IS the policy's nominal (H-ZERO), and its info carries
    `intervention_magnitude`, the env's own
    `sum ||a_nominal - a_safe||_2` over the decision's span — the same quantity
    the penalty is computed from, so the metric cannot drift from the treatment.
    `play_episode` discards both, hence this subclass rather than a second copy
    of the norm here.

    EVALUATION ONLY. `play_episode` calls `_begin` exactly once per episode, so
    `episodes` has one entry per episode; `reset()` retries `_begin` in a loop
    when the scripted law finishes an episode before any decision, which would
    append entries with no episode of their own. Training uses plain
    `SearchMDP` (`make_env`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes: list[dict] = []
        self._episode: dict = {"decisions": 0, "zero_motion": 0, "intervention": 0.0}

    def _begin(self, seed: int | None) -> np.ndarray:
        obs = super()._begin(seed)
        self._episode = {"decisions": 0, "zero_motion": 0, "intervention": 0.0}
        self.episodes.append(self._episode)
        return obs

    def step(self, action):
        out = super().step(action)
        ep = self._episode
        ep["decisions"] += 1
        ep["zero_motion"] += int(is_zero_motion(action, self.cfg))
        ep["intervention"] += float(out[4]["intervention_magnitude"])
        return out


def request_summary(episodes: list[dict]) -> dict:
    """H-ZERO's fraction and the mean per-episode intervention magnitude.

    The fraction is DECISION-weighted over the pooled episodes, not a mean of
    per-episode fractions: an episode that ends after two decisions must not
    weigh as much as one that took two hundred."""
    decisions = sum(e["decisions"] for e in episodes)
    zero = sum(e["zero_motion"] for e in episodes)
    total = sum(e["intervention"] for e in episodes)
    n = len(episodes)
    return {
        "zero_motion": {
            "n_zero_motion": int(zero),
            "n_decisions": int(decisions),
            "fraction": zero / decisions if decisions else 0.0,
        },
        "intervention": {
            "total": float(total),
            "n_episodes": n,
            "per_episode_mean": float(total / n) if n else 0.0,
        },
    }


# ----------------------------------------------------------------------------
# evaluation: unwrapped, a fresh shield per episode
# ----------------------------------------------------------------------------
def evaluate_arm_policy(
    model,
    cfg: EnvConfig,
    opponent_factory,
    n_episodes: int,
    seed0: int,
    *,
    shield_factory=None,
) -> tuple[list[dict], list[dict]]:
    """Paired-seed evaluation in a `RecordingSearchMDP`, returning the logs and
    the per-episode request records.

    Amendment A1.3: the env is built with `reward_wrapper=None` EXPLICITLY. The
    reward variant is training-only and `play_episode` discards rewards, so a
    wrapper provably cannot change an episode — but the gate does not rest the
    property on that argument.

    `intervention_penalty` is likewise left at 0.0: the penalty is a training
    treatment, and a gate that shaped the reward would be scoring the treatment
    rather than the policy it produced.

    A fresh shield per episode. `CBFShield` is memoryless so this is equivalent
    to the library's one-shield-per-call, but the equivalence is a property of
    the shield rather than of the protocol, and rebuilding costs nothing."""
    from pe2.shield import CBFShield

    factory = (lambda: CBFShield(cfg)) if shield_factory is None else shield_factory
    mdp = RecordingSearchMDP(
        cfg,
        opponent_factory,
        hold=HOLD,
        shield=None,
        shield_obs=SHIELD_OBS,
        escape_obs=ESCAPE_OBS,
        reward_wrapper=None,
    )
    logs = []
    for i in range(n_episodes):
        mdp.shield = factory()
        logs.append(mdp.play_episode(model.predict, seed0 + i))
    return logs, mdp.episodes


def gate_model(model, *, device: str = "cpu", holder: dict | None = None) -> dict:
    """Spec section 4: the gate stream, N = GATE_N, both opponents, every arm on
    the same episodes. Each entry is stashed into `holder["gate"]` as it
    completes, so a crash part-way through still hands the caller the
    violations seen."""
    cfg = EnvConfig()
    out: dict = {}
    for name in GATE_OPPONENTS:
        logs, episodes = evaluate_arm_policy(
            model, cfg, gate_opponent_factory(name, device), GATE_N, GATE_SEED0
        )
        k, n = win_count(logs, "pursuer")
        sh = summarize_shield(logs)
        req = request_summary(episodes)
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
            "n_episodes": int(sh["n_episodes"]),
            "pursuer_violations": int(sh["violation_steps"]),
            **req,
        }
        if holder is not None:
            holder.setdefault("gate", {})[name] = out[name]
        print(
            f"    vs {name:7s}: capture {k / n:.3f}, deadlock eps "
            f"{sh['deadlock_episode_rate']:.3f}, zero-motion requests "
            f"{req['zero_motion']['fraction']:.3f} "
            f"({req['zero_motion']['n_zero_motion']}/"
            f"{req['zero_motion']['n_decisions']}), intervention/ep "
            f"{req['intervention']['per_episode_mean']:.3f}, violations "
            f"{sh['violation_steps']}",
            flush=True,
        )
    return out


# ----------------------------------------------------------------------------
# guards (spec section 7's stop rule)
# ----------------------------------------------------------------------------
def require_arm_not_recorded(arm: str) -> None:
    """The stop rule: nothing is re-run after a number is seen."""
    if _arm_file(arm).exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 7) forbids "
            "re-running it."
        )


def require_arm_not_run(arm: str) -> None:
    """The stop rule for a TRAINING run. Every arm of this round trains, and its
    checkpoint directory belongs to this round alone, so the directory's
    existence is independent evidence that the arm already started."""
    require_arm_not_recorded(arm)
    if _ckpt_dir(arm).exists():
        raise SystemExit(
            f"arm {arm} has already trained ({_ckpt_dir(arm)} exists); the stop "
            "rule (spec section 7) forbids re-running it. To gate a checkpoint "
            "whose gate never completed, use --gate-only."
        )


def require_final_ckpt(arm: str) -> pathlib.Path:
    ckpt = _ckpt_dir(arm) / f"{arm}_{TOTAL}.zip"
    if not ckpt.exists():
        raise SystemExit(f"{arm} has no final checkpoint to gate: {ckpt} is missing.")
    return ckpt


def _check_width(arm: str, model) -> None:
    """Refuse a policy whose input size disagrees with the round's observation.

    Every S18 arm is 16-dim (`shield_obs` and `escape_obs` both on). Feeding a
    saved policy the wrong width either crashes deep inside `predict` or, with a
    width that matches for the wrong reason, silently mis-shapes its input."""
    want, got = obs_width(), int(model.observation_space.shape[0])
    if got != want:
        raise SystemExit(
            f"{arm}: the checkpoint expects an observation of width {got} but the "
            f"protocol declares {want} (shield_obs={SHIELD_OBS}, "
            f"escape_obs={ESCAPE_OBS}). One of the two is wrong; do NOT gate "
            "this arm until they agree."
        )


def todo(names: list[str] | None) -> list[str]:
    """The arms still to run, in the arm table's order. An arm with a record on
    disk is SKIPPED rather than refused — the round is driven one process at a
    time and a bare `--arms` must be safe to re-issue. `require_arm_not_run` is
    still the guard at the point of running."""
    wanted = list(ARMS) if not names else [n for n in ARMS if n in set(names)]
    return [n for n in wanted if not _arm_file(n).exists()]


def load_results() -> dict:
    return {
        p.stem.removeprefix(f"{ARTEFACT_PREFIX}_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json"))
    }


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """The env-step-grid probe, carrying this round's own metrics — capture and
    the zero-motion REQUEST fraction — along the training curve.

    The probe evaluates on the unwrapped gate path (`evaluate_arm_policy`), so a
    dense column's curve is comparable with SPARSE's: what is plotted is the
    task, never the shaped return."""

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
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(
            model, env_steps: int, mark: int, train_violations: int
        ) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs, episodes = evaluate_arm_policy(
                model, cfg, opponent_factory, eval_n, seed0
            )
            k, n = win_count(logs, "pursuer")
            sh = summarize_shield(logs)
            req = request_summary(episodes)
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "win_rate": k / n,
                "k": k,
                "n": n,
                "violations": int(sh["violation_steps"]),
                "train_violations": int(train_violations),
                "zero_motion_fraction": req["zero_motion"]["fraction"],
                "mean_intervention_magnitude": req["intervention"]["per_episode_mean"],
                "immobile_step_rate": sh["immobile_step_rate"],
                "intervention_rate": sh["intervention_rate"],
                "stall_step_rate": sh["stall_step_rate"],
                "deadlock_episode_rate": sh["deadlock_episode_rate"],
            }
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): capture {k}/{n} = "
                f"{k / n:.3f}, zero-motion {row['zero_motion_fraction']:.3f}, "
                f"intervention/ep {row['mean_intervention_magnitude']:.3f}, "
                f"deadlock eps {row['deadlock_episode_rate']:.3f}, violations "
                f"{row['violations']} (train {row['train_violations']})",
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


def _train_and_gate(
    arm: str, calibration: dict, device: str, holder: dict | None = None
) -> dict:
    cfg = EnvConfig()
    d = _ckpt_dir(arm)
    d.mkdir(parents=True, exist_ok=True)
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
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = make_env(arm, calibration)
    if holder is not None:
        holder["env"] = env
    model = make_learner(arm, env, device)
    model.learn(total_timesteps=TOTAL, callback=probe.callback)
    result = {
        "arm": arm,
        **ARMS[arm],
        "intervention_penalty": intervention_penalty_for(arm, calibration),
        "calibration": {k: calibration[k] for k in CALIBRATION_KEYS},
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
        result["verdict"] = "UNDETERMINED"
        result["reason"] = (
            f"final probe at mark {TOTAL} never fired (env steps {env.env_steps})"
        )
        return result
    result |= _gate_into(arm, ckpt, device, holder)
    return result


def _gate_into(arm: str, ckpt: pathlib.Path, device: str, holder: dict | None) -> dict:
    """Gate the checkpoint and build the record's `final` block and headline."""
    from stable_baselines3 import PPO

    spec = ARMS[arm]
    print(
        f"[{arm}] gate (column {spec['column']}, penalty "
        f"{'ON' if spec['penalty'] else 'OFF'}):",
        flush=True,
    )
    model = PPO.load(ckpt, device=device)
    _check_width(arm, model)
    gate = gate_model(model, device=device, holder=holder)
    zero = sum(e["zero_motion"]["n_zero_motion"] for e in gate.values())
    decisions = sum(e["zero_motion"]["n_decisions"] for e in gate.values())
    episodes = sum(e["intervention"]["n_episodes"] for e in gate.values())
    final = {
        "ckpt": str(ckpt),
        "ckpt_sha256": _sha256(ckpt),
        "gate": gate,
        "pursuer_violations": int(sum(e["pursuer_violations"] for e in gate.values())),
        "zero_motion": {
            "n_zero_motion": zero,
            "n_decisions": decisions,
            "fraction": zero / decisions if decisions else 0.0,
        },
        "mean_intervention_magnitude": (
            sum(e["intervention"]["total"] for e in gate.values()) / episodes
            if episodes
            else 0.0
        ),
    }
    reason = (
        f"capture vs {CAPTURE_OPPONENT} {gate[CAPTURE_OPPONENT]['capture_rate']:.3f}; "
        f"zero-motion requests {final['zero_motion']['fraction']:.3f}; "
        f"intervention/ep {final['mean_intervention_magnitude']:.3f}; "
        f"gate violations {final['pursuer_violations']}"
    )
    return {"final": final, "verdict": "MEASURED", "reason": reason}


def _crash_record(arm: str, exc: BaseException, holder: dict) -> dict:
    """Spec section 7: a crashed arm is UNDETERMINED with its probe history
    preserved."""
    return {
        "arm": arm,
        **ARMS[arm],
        "verdict": "UNDETERMINED",
        "reason": f"crashed: {type(exc).__name__}: {exc}",
        "checkpoints": holder.get("rows", []),
        "train_violations": int(getattr(holder.get("env"), "violations_total", 0)),
        "gate_violations": int(
            sum(e["pursuer_violations"] for e in holder.get("gate", {}).values())
        ),
        "gate_partial": holder.get("gate", {}),
        "provenance": provenance(),
    }


def run_arm(arm: str, device: str = "cpu") -> dict:
    """Train one arm and gate it."""
    calibration = load_calibration()  # refuse BEFORE touching anything else
    require_arm_not_run(arm)
    holder: dict = {}
    try:
        result = _train_and_gate(arm, calibration, device, holder)
    except BaseException as e:
        _write_json(_arm_file(arm), _crash_record(arm, e, holder))
        raise
    _write_json(_arm_file(arm), result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


def gate_arm(arm: str, device: str = "cpu") -> dict:
    """Gate an arm whose training finished but whose gate did not. No training.

    The checkpoint directory is NOT evidence of a completed arm here — that is
    the precondition — so only the arm record guards this path."""
    calibration = load_calibration()
    require_arm_not_recorded(arm)
    ckpt = require_final_ckpt(arm)
    holder: dict = {}
    try:
        result = {
            "arm": arm,
            **ARMS[arm],
            "intervention_penalty": intervention_penalty_for(arm, calibration),
            "calibration": {k: calibration[k] for k in CALIBRATION_KEYS},
            "obs_width": obs_width(),
            # The training and probe contact steps of this arm were incurred by
            # the training process and recorded in its probe file; this path
            # re-gates a checkpoint and must not invent a second census.
            "train_violations": 0,
            "probe_violations": int(
                sum(
                    r.get("violations", 0)
                    for r in json.loads(_probe_file(arm).read_text())["checkpoints"]
                )
                if _probe_file(arm).exists()
                else 0
            ),
            "checkpoints": (
                json.loads(_probe_file(arm).read_text())["checkpoints"]
                if _probe_file(arm).exists()
                else []
            ),
            "gate_only": True,
            "provenance": provenance(),
        }
        result |= _gate_into(arm, ckpt, device, holder)
    except BaseException as e:
        _write_json(_arm_file(arm), _crash_record(arm, e, holder))
        raise
    _write_json(_arm_file(arm), result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# readers over the records
# ----------------------------------------------------------------------------
def _entry(results: dict, arm: str, opponent: str) -> dict | None:
    gate = ((results.get(arm) or {}).get("final") or {}).get("gate") or {}
    return gate.get(opponent)


def capture_ind(
    results: dict, arm: str, opponent: str = CAPTURE_OPPONENT
) -> list[bool] | None:
    e = _entry(results, arm, opponent)
    return None if e is None else list(e["capture_ind"])


def capture_rate(
    results: dict, arm: str, opponent: str = CAPTURE_OPPONENT
) -> float | None:
    e = _entry(results, arm, opponent)
    return None if e is None else float(e["capture_rate"])


def _pooled(results: dict, arm: str, num, den) -> float | None:
    """A pooled rate over BOTH gate opponents. Every mechanism metric is
    reported as one number per arm, and the pool is weighted by the denominator
    rather than averaging the two per-opponent rates."""
    gate = ((results.get(arm) or {}).get("final") or {}).get("gate") or {}
    if set(gate) != set(GATE_OPPONENTS):
        return None
    n = sum(den(gate[o]) for o in GATE_OPPONENTS)
    return sum(num(gate[o]) for o in GATE_OPPONENTS) / n if n else 0.0


def deadlock_rate(results: dict, arm: str) -> float | None:
    """Episode-weighted deadlock-episode rate pooled over both gate opponents."""
    return _pooled(
        results,
        arm,
        lambda e: e["pursuer"]["deadlock_episode_rate"] * e["n_episodes"],
        lambda e: e["n_episodes"],
    )


def zero_motion_fraction(results: dict, arm: str) -> float | None:
    """H-ZERO's metric: decision-weighted fraction of the learner's REQUESTS
    whose commanded speed is at or below the filter's immobility threshold,
    pooled over both gate opponents."""
    return _pooled(
        results,
        arm,
        lambda e: e["zero_motion"]["n_zero_motion"],
        lambda e: e["zero_motion"]["n_decisions"],
    )


def mean_intervention_magnitude(results: dict, arm: str) -> float | None:
    """Episode-weighted mean of the per-episode sum of ||a_nominal - a_safe||."""
    return _pooled(
        results,
        arm,
        lambda e: e["intervention"]["total"],
        lambda e: e["intervention"]["n_episodes"],
    )


# ----------------------------------------------------------------------------
# the hypotheses (spec section 5)
# ----------------------------------------------------------------------------
def penalty_effect(results: dict, column: str) -> dict:
    """What the penalty did to capture in one column — the raw material every
    hypothesis below reads, and the reason a FAILED hypothesis is still legible.

    Reports, per seed, the PEN and BASE capture rates against the primary
    opponent and their signed difference; and, pooled over the three seeds, the
    paired test in BOTH directions (`drop` = BASE captures more, `gain` = PEN
    captures more). Every arm gates the same GATE_N episodes, so the seeds pool
    into 3 * GATE_N matched pairs."""
    per_seed: dict = {}
    for s in SEEDS:
        pen = capture_rate(results, f"{column}-PEN-{s}")
        base = capture_rate(results, f"{column}-BASE-{s}")
        per_seed[s] = (
            None
            if pen is None or base is None
            else {
                "base": base,
                "pen": pen,
                "delta": pen - base,
                "lowers": pen < base,
                "raises": pen > base,
            }
        )
    complete = all(v is not None for v in per_seed.values())
    n_lower = sum(1 for v in per_seed.values() if v and v["lowers"])
    n_raise = sum(1 for v in per_seed.values() if v and v["raises"])
    pooled = None
    if complete:
        pen_ind: list[bool] = []
        base_ind: list[bool] = []
        for s in SEEDS:
            pen_ind.extend(capture_ind(results, f"{column}-PEN-{s}"))
            base_ind.extend(capture_ind(results, f"{column}-BASE-{s}"))
        pooled = {
            "drop": paired_indicator_test(base_ind, pen_ind, P_GATE),
            "gain": paired_indicator_test(pen_ind, base_ind, P_GATE),
        }
    direction = "INCOMPLETE"
    if complete:
        if n_lower >= PASS_MIN:
            direction = "LOWERS"
        elif n_raise >= PASS_MIN:
            direction = "RAISES"
        else:
            direction = "MIXED"
    return {
        "column": column,
        "complete": complete,
        "per_seed": per_seed,
        "n_lower": n_lower,
        "n_raise": n_raise,
        "direction": direction,
        "pooled": pooled,
    }


def _effect_sentence(eff: dict) -> str:
    if not eff["complete"]:
        return f"{eff['column']}: INCOMPLETE"
    deltas = ", ".join(
        f"{s}: {eff['per_seed'][s]['base']:.3f}->{eff['per_seed'][s]['pen']:.3f}"
        for s in SEEDS
    )
    p_drop = eff["pooled"]["drop"]["p"]
    p_gain = eff["pooled"]["gain"]["p"]
    return (
        f"{eff['column']}: penalty {eff['direction']} capture "
        f"({eff['n_lower']}/3 seeds down, {eff['n_raise']}/3 up; {deltas}; "
        f"pooled p(drop)={p_drop:.4g}, p(gain)={p_gain:.4g})"
    )


def h_cross(results: dict) -> dict:
    """H-CROSS (primary). A crossover interaction: the penalty LOWERS capture in
    SPARSE and RAISES it in DENSE-VARY. The sign must hold on >= 2/3 seeds in
    EACH direction, and the SPARSE drop and the DVARY gain must each be
    significant by `paired_indicator_test` at p < 0.05 pooled.

    A failed H-CROSS is this round's refutation, so the direction actually
    observed is reported either way and never collapsed into "FAIL"."""
    sparse = penalty_effect(results, "SPARSE")
    dvary = penalty_effect(results, "DVARY")
    complete = sparse["complete"] and dvary["complete"]
    sign = complete and sparse["n_lower"] >= PASS_MIN and dvary["n_raise"] >= PASS_MIN
    significant = complete and bool(
        sparse["pooled"]["drop"]["pass"] and dvary["pooled"]["gain"]["pass"]
    )
    verdict = (
        "INCOMPLETE" if not complete else ("PASS" if sign and significant else "FAIL")
    )
    return {
        "verdict": verdict,
        "sign_holds": bool(sign),
        "significant": bool(significant),
        "SPARSE": sparse,
        "DVARY": dvary,
        "observed": f"{_effect_sentence(sparse)}. {_effect_sentence(dvary)}.",
    }


def h_mech(results: dict) -> dict:
    """H-MECH (the discriminating test). DENSE-FLAT patterns with SPARSE, not
    with DENSE-VARY: the penalty lowers capture there too, on >= 2/3 seeds.

    This is what separates gradient-on-`I` from density: DFLAT is dense
    everywhere and flat on the intervened set, so density predicts it behaves
    like DVARY and Hypothesis B predicts it behaves like SPARSE."""
    eff = penalty_effect(results, "DFLAT")
    patterns_with = {
        "LOWERS": "SPARSE",
        "RAISES": "DVARY",
        "MIXED": "NEITHER",
        "INCOMPLETE": None,
    }[eff["direction"]]
    verdict = (
        "INCOMPLETE"
        if not eff["complete"]
        else ("PASS" if eff["n_lower"] >= PASS_MIN else "FAIL")
    )
    return {
        "verdict": verdict,
        "patterns_with": patterns_with,
        "DFLAT": eff,
        "observed": _effect_sentence(eff),
    }


def h_zero(results: dict) -> dict:
    """H-ZERO (mechanism). In every CELL where the penalty lowers capture, the
    zero-motion REQUEST fraction rises relative to its matched no-penalty
    control. Without it H-CROSS is a correlation.

    READING (stated, not silently chosen): the spec's "cell" is a grid entry of
    section 2's 2 x 3 (`Six cells, 3 seeds each`), so "every cell where the
    penalty lowers capture" is read as every COLUMN whose PEN cell lowers
    capture, on the same >= 2/3-seed bar the other hypotheses use; the rise is
    then required on >= 2/3 seeds of that column, each PEN arm against the BASE
    arm of its OWN seed. Per-seed detail is reported so the stricter per-seed
    reading can be read straight off the record."""
    per_column: dict = {}
    for column in COLUMNS:
        eff = penalty_effect(results, column)
        per_seed: dict = {}
        for s in SEEDS:
            pen = zero_motion_fraction(results, f"{column}-PEN-{s}")
            base = zero_motion_fraction(results, f"{column}-BASE-{s}")
            per_seed[s] = (
                None
                if pen is None or base is None
                else {
                    "base": base,
                    "pen": pen,
                    "delta": pen - base,
                    "rises": pen > base,
                }
            )
        complete = eff["complete"] and all(v is not None for v in per_seed.values())
        n_rise = sum(1 for v in per_seed.values() if v and v["rises"])
        lowered = complete and eff["n_lower"] >= PASS_MIN
        if not complete:
            verdict = "INCOMPLETE"
        elif not lowered:
            verdict = "N/A"  # the penalty did not lower capture here
        else:
            verdict = "PASS" if n_rise >= PASS_MIN else "FAIL"
        per_column[column] = {
            "verdict": verdict,
            "capture_lowered": bool(lowered),
            "n_rise": n_rise,
            "per_seed": per_seed,
            "complete": complete,
        }
    incomplete = [c for c, v in per_column.items() if not v["complete"]]
    affected = [c for c, v in per_column.items() if v["capture_lowered"]]
    if incomplete:
        verdict = "INCOMPLETE"
    elif not affected:
        verdict = "VACUOUS"  # no cell lowered capture; there is nothing to explain
    else:
        verdict = (
            "PASS"
            if all(per_column[c]["verdict"] == "PASS" for c in affected)
            else "FAIL"
        )
    return {
        "verdict": verdict,
        "affected_columns": affected,
        "per_column": per_column,
        "observed": "; ".join(
            f"{c}: zero-motion "
            + ", ".join(
                (
                    f"{s}: n/a"
                    if per_column[c]["per_seed"][s] is None
                    else f"{s}: {per_column[c]['per_seed'][s]['base']:.3f}->"
                    f"{per_column[c]['per_seed'][s]['pen']:.3f}"
                )
                for s in SEEDS
            )
            for c in COLUMNS
        ),
    }


def hsafe(results: dict) -> dict:
    """H-SAFE (standing): zero pursuer contact steps across all 18 arms —
    training, probes and gate alike."""
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a in sorted(results):
        r = results[a]
        v = (
            int(r.get("train_violations", 0))
            + int(r.get("probe_violations", 0))
            # a crash record has no "final"; gate_violations is what the gate had
            # already seen when the process died
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
# what each outcome means (spec section 5's refutation paragraph)
# ----------------------------------------------------------------------------
def direction(h: dict) -> str:
    """Spec section 5, stated plainly there and quoted here.

    No interaction refutes Hypothesis B and the paper does not exist in this
    form; H-CROSS passing while H-MECH fails means the variable is density after
    all and the S20 proposition is rewritten to match."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; "
            "no further round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["incomplete"]:
        return (
            f"incomplete: all {len(ARMS)} arms must be gated; no verdict is read "
            "until they are. Observed so far: " + h["cross_observed"]
        )
    if h["cross"] == "FAIL":
        return (
            "H-CROSS FAILS: no crossover interaction. Hypothesis B is REFUTED and the "
            "paper does not exist in this form. Spec section 7: stop and reconsider "
            "rather than re-cut the data. Observed: " + h["cross_observed"]
        )
    if h["mech"] == "FAIL":
        return (
            "H-CROSS holds but H-MECH FAILS: DENSE-FLAT behaves like DENSE-VARY, so "
            "the variable is reward DENSITY after all, not gradient on the intervened "
            "set. The interaction is real but the S20 proposition is wrong as drafted "
            "and gets rewritten to match. Observed: " + h["mech_observed"]
        )
    if h["zero"] == "FAIL":
        return (
            "H-CROSS and H-MECH hold but H-ZERO FAILS: the penalty lowers capture "
            "without the zero-motion request fraction rising, so the mechanism "
            "claimed for the interaction is not the one measured and H-CROSS stands "
            "as a correlation only. Observed: " + h["zero_observed"]
        )
    return (
        "The boundary condition HOLDS: penalising filter intervention helps where the "
        "task reward varies across the actions available on the intervened set and "
        "hurts where it does not — DENSE-FLAT patterns with SPARSE, which separates "
        "gradient-on-I from density"
        + (
            ", and the zero-motion request fraction rises exactly where capture falls."
            if h["zero"] == "PASS"
            else " (H-ZERO is vacuous: no cell lowered capture)."
        )
    )


# ----------------------------------------------------------------------------
# summary and figure
# ----------------------------------------------------------------------------
def summarize(results: dict, start: dict | None = None) -> dict:
    cross, mech, zero = h_cross(results), h_mech(results), h_zero(results)
    hs = hsafe(results)
    complete = all(a in results and "final" in results[a] for a in ARMS)
    table = {}
    for a in ARMS:
        r = results.get(a)
        gate = ((r or {}).get("final") or {}).get("gate") or {}
        table[a] = {
            "verdict": (r or {}).get("verdict"),
            "cell": ARMS[a]["cell"],
            "seed": ARMS[a]["seed"],
            "penalty": ARMS[a]["penalty"],
            "capture": capture_rate(results, a),
            "deadlock": deadlock_rate(results, a),
            "zero_motion": zero_motion_fraction(results, a),
            "intervention": mean_intervention_magnitude(results, a),
            "gate": {
                o: {
                    "capture": gate[o]["capture_rate"],
                    "deadlock": gate[o]["pursuer"]["deadlock_episode_rate"],
                    "immobile": gate[o]["pursuer"]["immobile_step_rate"],
                    "zero_motion": gate[o]["zero_motion"]["fraction"],
                    "intervention": gate[o]["intervention"]["per_episode_mean"],
                }
                for o in gate
            },
        }
    return {
        "spec": SPEC,
        "hsafe": hs,
        "hypotheses": {"H-CROSS": cross, "H-MECH": mech, "H-ZERO": zero},
        "effects": {c: penalty_effect(results, c) for c in COLUMNS},
        "arms": table,
        "cells": {c: dict(CELLS[c]) for c in CELL_ORDER},
        "gate": {
            "seed0": GATE_SEED0,
            "n": GATE_N,
            "opponents": list(GATE_OPPONENTS),
            "capture_opponent": CAPTURE_OPPONENT,
            "reward_wrapper": None,  # amendment A1.3
        },
        "bars": {
            "p": P_GATE,
            "pass_min": PASS_MIN,
            "zero_motion_v_frac": ZERO_MOTION_V_FRAC,
            "gamma": GAMMA,
            "total": TOTAL,
        },
        "complete": complete,
        "direction": direction(
            {
                "hsafe": hs["verdict"],
                "incomplete": not complete,
                "cross": cross["verdict"],
                "mech": mech["verdict"],
                "zero": zero["verdict"],
                "cross_observed": cross["observed"],
                "mech_observed": mech["observed"],
                "zero_observed": zero["observed"],
            }
        ),
    }


COLUMN_COLOUR = {"SPARSE": "tab:blue", "DFLAT": "tab:orange", "DVARY": "tab:green"}


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    ax = axes[0]
    for arm, r in results.items():
        rows = r.get("checkpoints") or []
        spec = ARMS.get(arm)
        if rows and spec:
            ax.plot(
                [x["steps"] for x in rows],
                [x.get("win_rate") for x in rows],
                marker=".",
                ls="--" if spec["penalty"] else "-",
                color=COLUMN_COLOUR.get(spec["column"], "0.5"),
                alpha=0.8,
                label=arm,
            )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"probe capture (GREEDY, seeds {PROBE_SEED0}+)")
    ax.set_title("training curves (dashed = penalty ON)")
    ax.legend(fontsize=5, ncol=2)

    # The interaction, cell by cell: BASE beside PEN in each column.
    ax = axes[1]
    width = 0.35
    for i, column in enumerate(COLUMNS):
        for j, row in enumerate(ROWS):
            vals = [
                summary["arms"][f"{column}-{row}-{s}"]["capture"]
                for s in SEEDS
                if summary["arms"][f"{column}-{row}-{s}"]["capture"] is not None
            ]
            if not vals:
                continue
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                sum(vals) / len(vals),
                width=width,
                color=COLUMN_COLOUR[column],
                alpha=1.0 if row == "BASE" else 0.45,
                hatch="" if row == "BASE" else "//",
                label=row if i == 0 else None,
            )
            ax.plot([x] * len(vals), vals, "k.", ms=5)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(list(COLUMNS))
    ax.set_ylabel(f"gate capture vs {CAPTURE_OPPONENT}")
    ax.set_title("H-CROSS / H-MECH — capture by cell (dots = seeds)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, column in enumerate(COLUMNS):
        for j, row in enumerate(ROWS):
            vals = [
                summary["arms"][f"{column}-{row}-{s}"]["zero_motion"]
                for s in SEEDS
                if summary["arms"][f"{column}-{row}-{s}"]["zero_motion"] is not None
            ]
            if not vals:
                continue
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                sum(vals) / len(vals),
                width=width,
                color=COLUMN_COLOUR[column],
                alpha=1.0 if row == "BASE" else 0.45,
                hatch="" if row == "BASE" else "//",
                label=row if i == 0 else None,
            )
            ax.plot([x] * len(vals), vals, "k.", ms=5)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(list(COLUMNS))
    ax.set_ylabel("zero-motion REQUEST fraction")
    ax.set_title("H-ZERO — what the policy asked for (not what ran)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "S18 — penalising filter intervention, by where the reward has gradient"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arms",
        nargs="*",
        choices=list(ARMS),
        default=None,
        help="arms to run; bare --arms means every arm not yet recorded",
    )
    p.add_argument(
        "--gate-only",
        action="store_true",
        help="gate the named arms' existing final checkpoints; train nothing",
    )
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.arms is not None or args.gate_only:
        for arm in todo(args.arms):
            if args.gate_only:
                gate_arm(arm, args.device)
            else:
                run_arm(arm, args.device)
    if args.summarize:
        results = load_results()
        summary = summarize(results)
        _write_json(_round_file("json"), {"summary": summary, "arms": results})
        write_figure(results, summary, _round_file("png"))
        print(
            json.dumps(
                {
                    "hsafe": summary["hsafe"],
                    "H-CROSS": {
                        k: summary["hypotheses"]["H-CROSS"][k]
                        for k in ("verdict", "sign_holds", "significant", "observed")
                    },
                    "H-MECH": {
                        k: summary["hypotheses"]["H-MECH"][k]
                        for k in ("verdict", "patterns_with", "observed")
                    },
                    "H-ZERO": {
                        k: summary["hypotheses"]["H-ZERO"][k]
                        for k in ("verdict", "affected_columns", "observed")
                    },
                },
                indent=2,
                default=str,
            )
        )
        print(json.dumps(summary["arms"], indent=2, default=str))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
