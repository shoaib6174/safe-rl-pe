"""S17 — Phase 2 redesigned: a forward-offering recovery, judged on the task.

Spec: docs/pathway/loop_fixes_b_prereg_2026-09-21.md plus amendment S17-A1.
Nothing here may change a threshold, a seed, a stream, a rule or a budget
after a number is seen.

What this round corrects (D129, D130). S16's primary metric was the
immobile-step rate, which counts a remedy's own MECHANISM as the disease: the
scripted recovery drove deadlock to zero by rotating on the spot and the metric
called that a fix. This round judges on the TASK — capture against
`GreedyEvader` — on deadlock episodes, and on a new SPIN FRACTION that exists
specifically so no arm can win by turning in place. D130 also showed a straight
command moves the body in 100 % of states sampled deep inside those spins,
which is why the recovery now offers forward candidates first
(`pe2.shield.RecoveryShieldV2`, Task 1).

Seven families x three seeds; only the combination arm is new:

  BASE-{60,61,62}         S16's checkpoints, re-gated. The reference.
  OBS-{60,61,62}          S16's checkpoints, re-gated. +2 escape features.
  GAM-{60,61,62}          S16's checkpoints, re-gated. gamma = 0.999.
  OBSGAM-{60,61,62}       NEW TRAINING: both halves of D128 at once.
  RECF-BASE-{60,61,62}    BASE's policies + RecoveryShieldV2 at evaluation.
  RECF-OBS-{60,61,62}     OBS's policies + RecoveryShieldV2.
  RECF-OBSGAM-{60,61,62}  OBSGAM's policies + RecoveryShieldV2.

Gate: stream 1 740 000+, N = 500, every arm on the same episodes, against
ATK4-45, ObstacleHuggingEvader and GreedyEvader.

Three evaluation-path requirements, each pinned by tests:

  - A FRESH `RecoveryShieldV2` per episode. `pe2.structured.evaluate_structured`
    builds ONE shield for a whole call and the recovery's immobile run and
    counters are shield state with no episode hook, so a RECF arm would carry
    an open run across the paired-seed boundary. `evaluate_arm_policy` rebuilds
    the shield per episode and is otherwise the library call, pinned
    bit-for-bit for a stateless shield.

  - PER-STEP LOGGING (spec section 2, forced by D130): the policy's nominal and
    the executed command are recorded as SEPARATE fields, plus the recovery
    category, so attribution is measured rather than inferred. `StepRecorder`
    wraps the shield the episode runs under; it is also the only place the
    executed omega exists, since `CBFShield.filter`'s info reports `v_exec` and
    no omega at all and H3's spin test needs both. Aggregates are kept for all
    GATE_N episodes, full traces for the first TRACE_EPISODES only.

  - `escape_obs` ON for the 16-dim families (OBS, OBSGAM and their RECF arms)
    and OFF for the 14-dim ones (BASE, GAM). Four families replay S16
    checkpoints and handing one the wrong observation width is silent, not
    loud, so `obs_width` is pinned against the real checkpoints by a test.

Amendment S17-A1, implemented here: at `RecoveryShieldV2`'s default
`move_eps` the `policy` recovery category is UNREACHABLE (the recovery fires
because the nominal came back immobile, and that candidate merely re-tries it).
A zero `policy` count in this round's records is expected, not a bug; the
category is reported anyway so the tally is the spec's four, not three.

Usage:
  ./venv/bin/python scripts/derisk_s17_fixes_b.py --train OBSGAM-60   (one per process)
  ./venv/bin/python scripts/derisk_s17_fixes_b.py --gate BASE-60 RECF-BASE-60
  ./venv/bin/python scripts/derisk_s17_fixes_b.py --gate             (every arm not yet run)
  ./venv/bin/python scripts/derisk_s17_fixes_b.py --summarize
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
from pe2.shield import RECOVERY_AFTER, STALL_V_FRAC
from pe2.structured import HOLD

SPEC = "docs/pathway/loop_fixes_b_prereg_2026-09-21.md"
ARTEFACT_PREFIX = "s17"
OUT = pathlib.Path("results/derisk")
ATK_CKPT = OUT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"

# --- pre-registered (spec sections 3-4). Do not tune. ---
SEEDS = (60, 61, 62)
TOTAL = 5_000_000
EVERY = 500_000
GATE_SEED0 = 1_740_000  # spec section 3: "stream 1 740 000+"
GATE_N = 500
EVAL_N = 200
# The spec pre-registers the GATE stream only; the training probe needs one
# too, and it must not overlap the gate. Disjoint by construction:
# gate 1 740 000 .. 1 740 499, probe 1 750 000 .. 1 750 199.
PROBE_SEED0 = 1_750_000
GAMMA = 0.99
GAMMA_LONG = 0.999
STALL_W = 1e-3
R_ACQ = 0.0
SHIELD_OBS = True
DEADLOCK_BAR = 0.05  # H2: deadlock-episode rate vs the deadlock opponent
SPIN_BAR = 0.05  # H3: spin fraction over all gate steps
PASS_MIN = 2  # ">= 2 of 3 seeds", every hypothesis
P_GATE = 0.05  # H1: McNemar
# H3's definition, spec section 4. The speed clause is the filter's OWN
# immobility threshold, so "spinning" and "immobile" cannot disagree.
SPIN_V_FRAC = STALL_V_FRAC
SPIN_OMEGA_FRAC = 0.5
GATE_OPPONENTS = ("ATK4-45", "HUG7", "GREEDY")
DEADLOCK_OPPONENT = "ATK4-45"  # H2's opponent
CAPTURE_OPPONENT = "GREEDY"  # H1's opponent
EVADER_HOLD = 10  # ATK4-45's own hold, as in S6/S9/S15/S16
TRACE_EPISODES = 5  # full per-step traces kept for the first N episodes
REFERENCE = "BASE"  # the reference family; not eligible to solve

BARE = ("BASE", "OBS", "GAM", "OBSGAM")
RECF = ("RECF-BASE", "RECF-OBS", "RECF-OBSGAM")
FAMILY_ORDER = (*BARE, *RECF)

# Spec section 3's table, column by column. `source` is the family whose
# CHECKPOINTS the arm gates (BASE/OBS/GAM are S16's, unretrained); `trains` is
# the "new training?" column; `recovery` is the RecoveryShieldV2 column;
# `escape_obs` is the observation width the source policy was trained at and
# must never disagree with it.
FAMILIES: dict[str, dict] = {
    "BASE": {
        "source": "BASE",
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "recovery": False,
        "trains": False,
    },
    "OBS": {
        "source": "OBS",
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": True,
        "recovery": False,
        "trains": False,
    },
    "GAM": {
        "source": "GAM",
        "gamma": GAMMA_LONG,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "recovery": False,
        "trains": False,
    },
    "OBSGAM": {
        "source": "OBSGAM",
        "gamma": GAMMA_LONG,
        "stall_penalty": STALL_W,
        "escape_obs": True,
        "recovery": False,
        "trains": True,
    },
    "RECF-BASE": {
        "source": "BASE",
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": False,
        "recovery": True,
        "trains": False,
    },
    "RECF-OBS": {
        "source": "OBS",
        "gamma": GAMMA,
        "stall_penalty": STALL_W,
        "escape_obs": True,
        "recovery": True,
        "trains": False,
    },
    "RECF-OBSGAM": {
        "source": "OBSGAM",
        "gamma": GAMMA_LONG,
        "stall_penalty": STALL_W,
        "escape_obs": True,
        "recovery": True,
        "trains": False,
    },
}
# `RECF-BASE-60`.split("-") does not recover the family, so the family and the
# seed are STORED here and every reader goes through the table.
ARMS: dict[str, dict] = {
    f"{fam}-{s}": {"family": fam, "seed": s, **cfg}
    for fam, cfg in FAMILIES.items()
    for s in SEEDS
}
TRAIN_ARMS: tuple[str, ...] = tuple(a for a, s in ARMS.items() if s["trains"])

RECOVERY_KINDS = ("forward", "policy", "rotation")
A1_NOTE = (
    "Amendment S17-A1: at RecoveryShieldV2's default move_eps the `policy` "
    "recovery category is unreachable (the recovery fires because the policy's "
    "nominal came back immobile, and that candidate merely re-tries it), so a "
    "zero `policy` count is expected and is not a bug."
)

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
# arms: sources, widths, env, learner, opponents
# ----------------------------------------------------------------------------
def source_arm(arm: str) -> str:
    """The arm whose checkpoint `arm` gates. `RECF-OBS-61` -> `OBS-61`."""
    spec = ARMS[arm]
    return f"{spec['source']}-{spec['seed']}"


def source_ckpt(arm: str) -> pathlib.Path:
    src = source_arm(arm)
    return _ckpt_dir(src) / f"{src}_{TOTAL}.zip"


def obs_width(family: str) -> int:
    """The observation width the family's policy expects: the env observation,
    plus `shield_obs`'s three features, plus `escape_obs`'s two.

    Pinned against the real S16 checkpoints by a test. This is the number that
    must never drift from the source policy's input size — feeding a saved
    policy the wrong width either crashes or silently mis-shapes its input."""
    return (
        obs_dim(EnvConfig())
        + 3 * int(SHIELD_OBS)
        + 2 * int(FAMILIES[family]["escape_obs"])
    )


def make_env(arm: str):
    """The S16 `BASE` recipe with the arm's changes (spec section 3). Only the
    OBSGAM arms train, and they change gamma at the learner and `escape_obs`
    here; everything else is S16's configuration unchanged."""
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
    """SB3 defaults apart from the arm's gamma."""
    from stable_baselines3 import PPO

    spec = ARMS[arm]
    return PPO(
        "MlpPolicy",
        env,
        seed=spec["seed"],
        gamma=spec["gamma"],
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
    if name == "HUG7":
        return ObstacleHuggingEvader
    from pe2.evade import D_POS, StructuredEvaderPolicy

    cfg = EnvConfig()
    model = _model(ATK_CKPT, device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=EVADER_HOLD, d_pos=D_POS)


# ----------------------------------------------------------------------------
# per-step logging (spec section 2) and the spin fraction (H3)
# ----------------------------------------------------------------------------
def is_spin(v_exec: float, w_exec: float, v_max: float, omega_max: float) -> bool:
    """H3's per-step test: `v_exec <= 0.05.v_max` AND `|w_exec| >= 0.5.w_max`.

    Both bounds inclusive, exactly as spec section 4 writes them. The speed
    clause is the filter's own immobility threshold; the turn clause is what
    makes a spinning step distinguishable from a merely stopped one, which is
    the distinction S16's immobile-step rate could not draw (D129)."""
    return float(v_exec) <= SPIN_V_FRAC * float(v_max) and abs(
        float(w_exec)
    ) >= SPIN_OMEGA_FRAC * float(omega_max)


class StepRecorder:
    """A shield wrapper that logs every filtered step (spec section 2).

    D130 could not attribute the long spins because nothing recorded what the
    POLICY asked for beside what the filter EXECUTED. This records both as
    separate fields, plus the recovery category, so the attribution is a
    measurement here rather than an inference.

    It is also the only place the executed omega exists: `CBFShield.filter`
    reports `v_exec` in its info and no omega at all, and H3's spin test needs
    both components of the executed command.

    The wrapper is observation-free — it returns the filter's `(u, info)`
    untouched and delegates every other attribute to the shield it wraps, so
    `pe2.shield.escape_features` and the recovery counters reach through it.
    `keep_trace` decides whether the per-step rows are STORED; the aggregates
    are kept either way, which is how a round keeps totals for all GATE_N
    episodes and full traces for the first few.
    """

    def __init__(self, shield, keep_trace: bool = False):
        self.shield = shield
        self.keep_trace = bool(keep_trace)
        self.n_steps = 0
        self.n_spin = 0
        self.kinds = dict.fromkeys((*RECOVERY_KINDS, "none"), 0)
        self.trace: list[dict] = []

    def __getattr__(self, name: str):
        # Only reached for attributes this object does not define; the guard
        # breaks the recursion when `shield` itself is not set yet.
        if name == "shield":
            raise AttributeError(name)
        return getattr(self.shield, name)

    def filter(self, action, state, obstacles, v_max=None):
        u, info = self.shield.filter(action, state, obstacles, v_max)
        cap = self.shield.cfg.v_max if v_max is None else float(v_max)
        omega_max = self.shield.cfg.omega_max
        v_exec, w_exec = float(u[0]), float(u[1])
        # `info["v_nom"]` is the POLICY's nominal even on a recovery step (the
        # recovery reports what was executed in the barrier fields and leaves
        # the nominal describing what the policy asked for), so the two fields
        # below really are the two different things.
        v_nom = float(info["v_nom"])
        w_nom = float(np.clip(float(action[1]), -omega_max, omega_max))
        kind = str(info.get("recovery_kind", "none"))
        spin = is_spin(v_exec, w_exec, cap, omega_max)
        self.n_steps += 1
        self.n_spin += int(spin)
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        if self.keep_trace:
            self.trace.append(
                {
                    "v_nom": v_nom,
                    "w_nom": w_nom,
                    "v_exec": v_exec,
                    "w_exec": w_exec,
                    "recovery_kind": kind,
                    "spin": bool(spin),
                    "immobile": bool(info["immobile"]),
                    "intervention": bool(info["intervention"]),
                }
            )
        return u, info

    @property
    def spin_fraction(self) -> float:
        return self.n_spin / self.n_steps if self.n_steps else 0.0


class _StepTally:
    """Builds a fresh recording shield per episode and totals the logs.

    The shield is per episode (spec section 3) and so, therefore, is the
    recorder, which means a gate cell's numbers only exist as a sum over the
    episodes. Only the first `trace_episodes` recorders keep their per-step
    rows; the rest aggregate and discard, so a 500-episode cell stays small.
    """

    def __init__(self, factory, trace_episodes: int = TRACE_EPISODES):
        self._factory = factory
        self.trace_episodes = int(trace_episodes)
        self.recorders: list[StepRecorder] = []

    def __call__(self) -> StepRecorder:
        rec = StepRecorder(
            self._factory(), keep_trace=len(self.recorders) < self.trace_episodes
        )
        self.recorders.append(rec)
        return rec

    @property
    def n_steps(self) -> int:
        return sum(r.n_steps for r in self.recorders)

    @property
    def n_spin(self) -> int:
        return sum(r.n_spin for r in self.recorders)

    @property
    def spin_fraction(self) -> float:
        return self.n_spin / self.n_steps if self.n_steps else 0.0

    @property
    def kinds(self) -> dict:
        out = dict.fromkeys((*RECOVERY_KINDS, "none"), 0)
        for r in self.recorders:
            for k, v in r.kinds.items():
                out[k] = out.get(k, 0) + v
        return out

    @property
    def traces(self) -> list[list[dict]]:
        return [r.trace for r in self.recorders if r.keep_trace]

    @property
    def recoveries(self) -> int:
        return sum(getattr(r.shield, "n_recoveries", 0) for r in self.recorders)


# ----------------------------------------------------------------------------
# evaluation: a fresh shield per episode
# ----------------------------------------------------------------------------
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
    hold: int = HOLD,
) -> list[dict]:
    """`pe2.structured.evaluate_structured` with a FRESH shield per episode.

    The library builds one shield for a whole call on the documented assumption
    that a shield carries no cross-episode state; `RecoveryShieldV2` breaks that
    assumption (its immobile run and its three counters have no episode hook),
    so a RECF arm's first steps of episode i+1 would inherit episode i's open
    run. Everything else is the library call unchanged: one `SearchMDP`, the
    same paired seeds `seed0 + i`, `play_episode` per seed — pinned identical to
    it for a stateless shield by
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
    logs = []
    for i in range(n_episodes):
        if shield_factory is not None:
            mdp.shield = shield_factory()
        logs.append(mdp.play_episode(model.predict, seed0 + i))
    return logs


def shield_factory_for(cfg: EnvConfig, recovery: bool):
    from pe2.shield import CBFShield, RecoveryShieldV2

    if recovery:
        return lambda: RecoveryShieldV2(cfg, recovery_after=RECOVERY_AFTER)
    return lambda: CBFShield(cfg)


def gate_model(
    model,
    *,
    escape_obs: bool = False,
    recovery: bool = False,
    device: str = "cpu",
    holder: dict | None = None,
) -> dict:
    """Spec section 3: the gate stream, N = GATE_N, all three opponents, one
    entry each. Each entry is stashed into `holder["gate"]` as it completes, so
    a crash part-way through still hands the caller the violations seen."""
    cfg = EnvConfig()
    base_factory = shield_factory_for(cfg, recovery)
    out: dict = {}
    for name in GATE_OPPONENTS:
        tally = _StepTally(base_factory, TRACE_EPISODES)
        logs = evaluate_arm_policy(
            model,
            cfg,
            gate_opponent_factory(name, device),
            GATE_N,
            GATE_SEED0,
            shield_factory=tally,
            shield_obs=SHIELD_OBS,
            escape_obs=escape_obs,
        )
        k, n = win_count(logs, "pursuer")
        sh = summarize_shield(logs)
        kinds = tally.kinds
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
            # H3's raw material, pooled over this cell's episodes.
            "spin": {
                "n_spin": tally.n_spin,
                "n_steps": tally.n_steps,
                "fraction": tally.spin_fraction,
            },
            # S17-A1: `policy` is expected to be identically zero.
            "recovery_kinds": {k2: kinds[k2] for k2 in RECOVERY_KINDS},
            "recoveries": tally.recoveries,
            # Full per-step traces for the first TRACE_EPISODES episodes only.
            "traces": tally.traces,
        }
        if holder is not None:
            holder.setdefault("gate", {})[name] = out[name]
        print(
            f"    vs {name:8s}: capture {k / n:.3f}, deadlock eps "
            f"{sh['deadlock_episode_rate']:.3f}, spin {tally.spin_fraction:.3f} "
            f"({tally.n_spin}/{tally.n_steps}), violations {sh['violation_steps']}"
            + (
                f", recoveries {tally.recoveries} "
                f"(forward {kinds['forward']}, rotation {kinds['rotation']}, "
                f"policy {kinds['policy']})"
                if recovery
                else ""
            ),
            flush=True,
        )
    return out


# ----------------------------------------------------------------------------
# guards (spec section 6's stop rule)
# ----------------------------------------------------------------------------
def require_arm_not_run(arm: str) -> None:
    """The stop rule: nothing is re-run after a number is seen.

    A checkpoint directory is evidence only for an arm that TRAINS. Six of the
    seven families gate a checkpoint directory that ALREADY EXISTS before this
    round starts -- `ckpts_BASE-60` is S16's, and `RECF-OBSGAM-60` gates
    `ckpts_OBSGAM-60`. For those the arm record is the only marker."""
    if _arm_file(arm).exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 6) forbids "
            "re-running it."
        )
    if ARMS[arm]["trains"] and _ckpt_dir(arm).exists():
        raise SystemExit(
            f"arm {arm} has already trained ({_ckpt_dir(arm)} exists); the stop rule "
            "(spec section 6) forbids re-running it."
        )


def require_trainable(arm: str) -> None:
    if not ARMS[arm]["trains"]:
        raise SystemExit(
            f"{arm} does not train: spec section 3 re-gates the S16 checkpoints of "
            f"family {ARMS[arm]['family']} and trains OBSGAM only. Use --gate."
        )


def require_source_ckpt(arm: str) -> pathlib.Path:
    ckpt = source_ckpt(arm)
    if not ckpt.exists():
        raise SystemExit(
            f"{arm} needs {source_arm(arm)}'s final checkpoint: {ckpt} is missing."
        )
    return ckpt


def todo(names: list[str] | None) -> list[str]:
    """The arms still to run: skip-existing, in the arm table's order.

    An arm with a record already on disk is SKIPPED rather than refused — the
    round is driven one process at a time and a bare `--gate` must be safe to
    re-issue. `require_arm_not_run` is still the guard at the point of running."""
    wanted = list(ARMS) if names is None else [n for n in ARMS if n in set(names)]
    return [n for n in wanted if not _arm_file(n).exists()]


def load_results() -> dict:
    return {
        p.stem.removeprefix(f"{ARTEFACT_PREFIX}_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json"))
    }


# ----------------------------------------------------------------------------
# the training arm (OBSGAM only)
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """S5's shielded env-step-grid probe, carrying this round's own metrics —
    capture and the spin fraction — along the training curve."""

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
            tally = _StepTally(factory, trace_episodes=0)
            logs = evaluate_arm_policy(
                model,
                cfg,
                opponent_factory,
                eval_n,
                seed0,
                shield_factory=tally,
                shield_obs=SHIELD_OBS,
                escape_obs=escape_obs,
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
                "spin_fraction": tally.spin_fraction,
                "immobile_step_rate": sh["immobile_step_rate"],
                "intervention_rate": sh["intervention_rate"],
                "stall_step_rate": sh["stall_step_rate"],
                "deadlock_episode_rate": sh["deadlock_episode_rate"],
            }
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): capture {k}/{n} = "
                f"{k / n:.3f}, spin {row['spin_fraction']:.3f}, deadlock eps "
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


def _train_and_gate(arm: str, device: str, holder: dict | None = None) -> dict:
    spec = ARMS[arm]
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
        escape_obs=spec["escape_obs"],
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
        f"[{arm}] gate ({source_arm(arm)}"
        + (
            f" + RecoveryShieldV2 after {RECOVERY_AFTER} immobile steps"
            if spec["recovery"]
            else ""
        )
        + "):",
        flush=True,
    )
    model = PPO.load(ckpt, device=device)
    _check_width(arm, model)
    gate = gate_model(
        model,
        escape_obs=spec["escape_obs"],
        recovery=spec["recovery"],
        device=device,
        holder=holder,
    )
    final = {
        "ckpt": str(ckpt),
        "gate": gate,
        "pursuer_violations": int(sum(e["pursuer_violations"] for e in gate.values())),
        "recoveries": int(sum(e["recoveries"] for e in gate.values())),
        "recovery_kinds": {
            k: int(sum(e["recovery_kinds"][k] for e in gate.values()))
            for k in RECOVERY_KINDS
        },
    }
    spin_n = sum(e["spin"]["n_spin"] for e in gate.values())
    spin_steps = sum(e["spin"]["n_steps"] for e in gate.values())
    final["spin"] = {
        "n_spin": spin_n,
        "n_steps": spin_steps,
        "fraction": spin_n / spin_steps if spin_steps else 0.0,
    }
    reason = (
        f"capture vs {CAPTURE_OPPONENT} {gate[CAPTURE_OPPONENT]['capture_rate']:.3f}; "
        f"deadlock eps vs {DEADLOCK_OPPONENT} "
        f"{gate[DEADLOCK_OPPONENT]['pursuer']['deadlock_episode_rate']:.3f} "
        f"(bar {DEADLOCK_BAR}); spin fraction {final['spin']['fraction']:.3f} "
        f"(bar {SPIN_BAR}); gate violations {final['pursuer_violations']}"
    )
    return {"final": final, "verdict": "MEASURED", "reason": reason}


def _check_width(arm: str, model) -> None:
    """Refuse a policy whose input size disagrees with the arm's `escape_obs`.

    Four families replay S16 checkpoints and the two widths in this round (14
    and 16) differ only by `escape_obs`; SB3 would either raise deep inside
    `predict` or, with a matching width for the wrong reason, feed the policy
    features it was never trained on. Check it once, at load, by name."""
    want = obs_width(ARMS[arm]["family"])
    got = int(model.observation_space.shape[0])
    if got != want:
        raise SystemExit(
            f"{arm}: {source_ckpt(arm)} expects an observation of width {got} but the "
            f"arm table declares {want} (escape_obs={ARMS[arm]['escape_obs']}). "
            "One of the two is wrong; do NOT gate this arm until they agree."
        )


def _crash_record(arm: str, exc: BaseException, holder: dict) -> dict:
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
    """Train an OBSGAM arm and gate it. The only family that trains."""
    require_trainable(arm)
    require_arm_not_run(arm)  # refuse BEFORE reading or writing anything
    holder: dict = {}
    try:
        result = _train_and_gate(arm, device, holder)
    except BaseException as e:
        _write_json(_arm_file(arm), _crash_record(arm, e, holder))
        raise
    _write_json(_arm_file(arm), result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


def gate_arm(arm: str, device: str = "cpu") -> dict:
    """Gate one arm's source checkpoint. No training of any kind."""
    require_arm_not_run(arm)
    ckpt = require_source_ckpt(arm)
    holder: dict = {}
    try:
        result = {
            "arm": arm,
            **ARMS[arm],
            "source_arm": source_arm(arm),
            "source_ckpt": str(ckpt),
            "source_sha256": _sha256(ckpt),
            "obs_width": obs_width(ARMS[arm]["family"]),
            "recovery_after": RECOVERY_AFTER if ARMS[arm]["recovery"] else None,
            # A re-gated arm trains nothing: its training and probe contact
            # steps ARE the source arm's, counted there (for OBSGAM) or in S16
            # (for BASE/OBS/GAM), and double-counting them here would inflate
            # the H-SAFE census.
            "train_violations": 0,
            "probe_violations": 0,
            "checkpoints": [],
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
# verdicts (spec section 4)
# ----------------------------------------------------------------------------
def _entry(results: dict, arm: str, opponent: str) -> dict | None:
    gate = ((results.get(arm) or {}).get("final") or {}).get("gate") or {}
    return gate.get(opponent)


def capture_ind(results: dict, arm: str) -> list[bool] | None:
    e = _entry(results, arm, CAPTURE_OPPONENT)
    return None if e is None else list(e["capture_ind"])


def deadlock_rate(results: dict, arm: str) -> float | None:
    e = _entry(results, arm, DEADLOCK_OPPONENT)
    return None if e is None else float(e["pursuer"]["deadlock_episode_rate"])


def spin_fraction(results: dict, arm: str) -> float | None:
    """H3's "over all gate steps": a STEP-WEIGHTED pool across every gate
    opponent, not a mean of the three per-opponent rates."""
    gate = ((results.get(arm) or {}).get("final") or {}).get("gate") or {}
    if set(gate) != set(GATE_OPPONENTS):
        return None
    n_spin = sum(gate[o]["spin"]["n_spin"] for o in GATE_OPPONENTS)
    n_steps = sum(gate[o]["spin"]["n_steps"] for o in GATE_OPPONENTS)
    return n_spin / n_steps if n_steps else 0.0


def _family_verdict(per_seed: dict, passes: dict) -> tuple[str, int]:
    if any(v is None for v in per_seed.values()):
        return "INCOMPLETE", sum(bool(p) for p in passes.values() if p is not None)
    n = sum(bool(p) for p in passes.values())
    return ("PASS" if n >= PASS_MIN else "FAIL"), n


def h1_verdicts(results: dict) -> dict:
    """H1 (task): the arm's `GreedyEvader` capture is significantly ABOVE
    `BASE`'s on the same seed (paired exact McNemar, p < P_GATE) on at least
    PASS_MIN of the three seeds.

    `BASE` is the reference: testing it against itself is meaningless (every
    pair is concordant), so it is labelled REFERENCE rather than given a
    verdict it could never pass."""
    out: dict = {}
    for fam in FAMILY_ORDER:
        per_seed: dict = {}
        for s in SEEDS:
            arm_ind = capture_ind(results, f"{fam}-{s}")
            base_ind = capture_ind(results, f"{REFERENCE}-{s}")
            if arm_ind is None or base_ind is None:
                per_seed[s] = None
                continue
            # a = the arm, b = BASE: `pass` means the ARM captures more often
            # than BASE on the same episodes.
            per_seed[s] = paired_indicator_test(arm_ind, base_ind, P_GATE)
        passes = {
            s: (None if t is None else bool(t["pass"])) for s, t in per_seed.items()
        }
        verdict, n_pass = _family_verdict(per_seed, passes)
        if fam == REFERENCE and verdict != "INCOMPLETE":
            verdict = "REFERENCE"
        out[fam] = {
            "verdict": verdict,
            "n_pass": n_pass,
            "per_seed": per_seed,
            "capture": {s: _cap_rate(results, f"{fam}-{s}") for s in SEEDS},
        }
    return out


def _cap_rate(results: dict, arm: str) -> float | None:
    e = _entry(results, arm, CAPTURE_OPPONENT)
    return None if e is None else float(e["capture_rate"])


def h2_verdicts(results: dict) -> dict:
    """H2 (deadlock): the deadlock-episode rate against `DEADLOCK_OPPONENT` is
    < DEADLOCK_BAR on at least PASS_MIN of the three seeds."""
    out: dict = {}
    for fam in FAMILY_ORDER:
        rates = {s: deadlock_rate(results, f"{fam}-{s}") for s in SEEDS}
        passes = {
            s: (None if v is None else v < DEADLOCK_BAR) for s, v in rates.items()
        }
        verdict, n_pass = _family_verdict(rates, passes)
        out[fam] = {"verdict": verdict, "n_pass": n_pass, "deadlock": rates}
    return out


def h3_verdicts(results: dict) -> dict:
    """H3 (no spinning): the spin fraction over ALL gate steps is < SPIN_BAR on
    at least PASS_MIN of the three seeds.

    The bar this round exists to impose. S16's `REC` drove deadlock to 0.000 by
    rotating on the spot and the immobile-step rate called that a fix (D129);
    a step that turns at half omega_max while going nowhere now counts against
    the remedy rather than for it."""
    out: dict = {}
    for fam in FAMILY_ORDER:
        fracs = {s: spin_fraction(results, f"{fam}-{s}") for s in SEEDS}
        passes = {s: (None if v is None else v < SPIN_BAR) for s, v in fracs.items()}
        verdict, n_pass = _family_verdict(fracs, passes)
        out[fam] = {"verdict": verdict, "n_pass": n_pass, "spin": fracs}
    return out


def solves(h1: dict, h2: dict, h3: dict) -> list[str]:
    """Spec section 4: an arm SOLVES iff H1, H2 and H3 all hold. `BASE` is the
    reference and is not eligible."""
    return [
        fam
        for fam in FAMILY_ORDER
        if fam != REFERENCE
        and h1[fam]["verdict"] == "PASS"
        and h2[fam]["verdict"] == "PASS"
        and h3[fam]["verdict"] == "PASS"
    ]


def hsafe(results: dict) -> dict:
    """H-SAFE: 0 pursuer contact steps in every cell. The re-gated families
    train nothing here, so their training and probe steps are counted where
    they were incurred (S16, or this round's OBSGAM arms)."""
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
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
_SPIN_ROW = (
    "RECF-BASE FAILS H3: the forward-offering recovery still spins; D130's 100 % "
    "result did not transfer to closed loop, and the mechanism needs re-diagnosing "
    "before any remedy is proposed. "
)
_ENGINEERING_ROW = (
    "The RECF arms solve and the bare learned arms do not: the remedy is an "
    "engineering one — offer the filtered agent a forward escape. That is the "
    "paper's practical recommendation, now with a metric that cannot be won by "
    "spinning."
)
_LEARNING_ROW = (
    "OBSGAM solves alone: the fix is a learning one after all — expose the "
    "admissible set and lengthen the horizon — and the scripted recovery is "
    "unnecessary."
)
_FAILURE_ROW = (
    "No arm solves: report as a failure. The three metrics are then known to be "
    "jointly unsatisfiable by anything tried, which is itself a result about "
    "filtered agents."
)


def direction(h: dict) -> str:
    """Spec section 5, row by row."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no "
            "further round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["incomplete"]:
        return f"incomplete: all {len(ARMS)} arms must be gated"
    # The RECF-BASE row is a diagnosis, not an outcome: it is reported whenever
    # it holds, ahead of whichever outcome row applies.
    lead = _SPIN_ROW if h.get("recf_base_h3") == "FAIL" else ""
    solving = list(h["solving"])
    if not solving:
        return lead + _FAILURE_ROW
    bare = [f for f in BARE if f in solving]
    recf = [f for f in RECF if f in solving]
    if recf and not bare:
        return lead + _ENGINEERING_ROW + f" Solving: {', '.join(recf)}."
    if solving == ["OBSGAM"]:
        return lead + _LEARNING_ROW
    return lead + (
        f"A mixed outcome: {', '.join(solving)} solve. Learned and scripted remedies "
        "both clear all three bars, so the paper reports both and the practical "
        "recommendation is whichever is cheaper to deploy."
    )


# ----------------------------------------------------------------------------
# summary and figure
# ----------------------------------------------------------------------------
def summarize(results: dict, start: dict | None = None) -> dict:
    h1, h2, h3 = h1_verdicts(results), h2_verdicts(results), h3_verdicts(results)
    solving = solves(h1, h2, h3)
    hs = hsafe(results)
    complete = all(a in results and "final" in results[a] for a in ARMS)
    table = {}
    for a in ARMS:
        r = results.get(a)
        final = (r or {}).get("final") or {}
        gate = final.get("gate") or {}
        table[a] = {
            "verdict": (r or {}).get("verdict"),
            "capture": _cap_rate(results, a),
            "deadlock": deadlock_rate(results, a),
            "spin": spin_fraction(results, a),
            "recoveries": final.get("recoveries"),
            "recovery_kinds": final.get("recovery_kinds"),
            "gate": {
                o: {
                    "capture": gate[o]["capture_rate"],
                    "deadlock": gate[o]["pursuer"]["deadlock_episode_rate"],
                    "immobile": gate[o]["pursuer"]["immobile_step_rate"],
                    "spin": gate[o]["spin"]["fraction"],
                }
                for o in gate
            },
        }
    return {
        "spec": SPEC,
        "hsafe": hs,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3},
        "solving": solving,
        "arms": table,
        "gate": {
            "seed0": GATE_SEED0,
            "n": GATE_N,
            "opponents": list(GATE_OPPONENTS),
            "capture_opponent": CAPTURE_OPPONENT,
            "deadlock_opponent": DEADLOCK_OPPONENT,
            "trace_episodes": TRACE_EPISODES,
        },
        "bars": {
            "deadlock": DEADLOCK_BAR,
            "spin": SPIN_BAR,
            "p": P_GATE,
            "pass_min": PASS_MIN,
            "spin_v_frac": SPIN_V_FRAC,
            "spin_omega_frac": SPIN_OMEGA_FRAC,
        },
        "recovery_kinds_note": A1_NOTE,
        "start": (start or {}).get("provenance"),
        "complete": complete,
        "direction": direction(
            {
                "hsafe": hs["verdict"],
                "incomplete": not complete,
                "solving": solving,
                "recf_base_h3": h3["RECF-BASE"]["verdict"],
            }
        ),
    }


def write_figure(results: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {
        "BASE": "k",
        "OBS": "tab:green",
        "GAM": "tab:blue",
        "OBSGAM": "tab:purple",
        "RECF-BASE": "tab:red",
        "RECF-OBS": "tab:orange",
        "RECF-OBSGAM": "tab:brown",
    }
    names = [a for a in ARMS if summary["arms"][a]["capture"] is not None]
    bars = [colour.get(ARMS[a]["family"], "0.5") for a in names]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    ax = axes[0]
    for arm, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x.get("spin_fraction") for x in rows],
                marker=".",
                color=colour.get(ARMS.get(arm, {}).get("family"), "0.5"),
                alpha=0.8,
                label=arm,
            )
    ax.axhline(SPIN_BAR, ls=":", color="k", lw=1.2, label=f"H3 bar = {SPIN_BAR}")
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"probe spin fraction (GREEDY, seeds {PROBE_SEED0}+)")
    ax.set_title("S17 — spinning during OBSGAM's training")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    if names:
        xs = list(range(len(names)))
        ax.bar(xs, [summary["arms"][a]["capture"] for a in names], color=bars)
        base = [summary["arms"][f"BASE-{s}"]["capture"] for s in SEEDS]
        base = [v for v in base if v is not None]
        if base:
            ax.axhline(
                sum(base) / len(base),
                ls="--",
                color="k",
                lw=1.0,
                label=f"BASE mean = {sum(base) / len(base):.3f}",
            )
            ax.legend(fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=7, rotation=70)
        ax.set_ylabel(f"gate capture vs {CAPTURE_OPPONENT}")
        ax.set_title("H1 — the task")

    ax = axes[2]
    if names:
        xs = np.arange(len(names), dtype=float)
        ax.bar(
            xs - 0.2,
            [summary["arms"][a]["spin"] or 0.0 for a in names],
            width=0.4,
            color=bars,
            label="spin fraction",
        )
        ax.bar(
            xs + 0.2,
            [summary["arms"][a]["deadlock"] or 0.0 for a in names],
            width=0.4,
            color=bars,
            alpha=0.45,
            hatch="//",
            label="deadlock episodes",
        )
        ax.axhline(SPIN_BAR, ls=":", color="k", lw=1.2, label=f"bar = {SPIN_BAR}")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=7, rotation=70)
        ax.set_ylabel("rate")
        ax.set_title("H3 (solid) and H2 (hatched)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "S17 — a forward-offering recovery, judged on the task"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="*", choices=list(TRAIN_ARMS), default=None)
    p.add_argument("--gate", nargs="*", choices=list(ARMS), default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.train is not None:
        for arm in todo(args.train or list(TRAIN_ARMS)):
            run_arm(arm, args.device)
    if args.gate is not None:
        for arm in todo(args.gate or None):
            gate_arm(arm, args.device)
    if args.summarize:
        results = load_results()
        summary = summarize(results)
        _write_json(_round_file("json"), {"summary": summary, "arms": results})
        write_figure(results, summary, _round_file("png"))
        print(
            json.dumps(
                {
                    "hsafe": summary["hsafe"],
                    "H1": {
                        f: summary["hypotheses"]["H1"][f]["verdict"]
                        for f in FAMILY_ORDER
                    },
                    "H2": {
                        f: summary["hypotheses"]["H2"][f]["verdict"]
                        for f in FAMILY_ORDER
                    },
                    "H3": {
                        f: summary["hypotheses"]["H3"][f]["verdict"]
                        for f in FAMILY_ORDER
                    },
                    "solving": summary["solving"],
                },
                indent=2,
                default=str,
            )
        )
        print(json.dumps(summary["arms"], indent=2, default=str))
        print(f"\n{A1_NOTE}")
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
