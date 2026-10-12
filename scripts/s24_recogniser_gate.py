"""S24.0 Step 3 — the recogniser construct gate: are the opponent-tactic labels
right enough for a selector to consume?

Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1 (the recogniser "is a
construct and gets its own gate (>= 0.8 accuracy on held-out episodes, confusion
matrix reported) before any selector consumes it") and section 2 (S24.0's
construct gates).

**The gate, fixed before this ran and printed above the table:**

    The recogniser is usable by a selector only if held-out accuracy >= 0.80 for
    the role at ratio 1.2 with UNKNOWN <= 0.30 of steps.

Protocol. Self-play on the round's construct stream (2 190 000+): the evader is
`TacticOpponent` over its five tactic FAMILIES with a uniform-random `select`,
the pursuer is `TacticLaw` over its five, each held until DONE or
`SPAN_LIMIT` = 80 steps, and the pursuer's command goes through `CBFShield` at
the S18/S19 defaults. Both roles are labelled by their OWN adapter's
`current.name`, which is the tactic that produced that step's action — so the
ground truth is the generating tactic and not a reconstruction of it.

Three readings of the brief, taken here and recorded:
1. The library has five FAMILIES, two of which are parameterised by an obstacle
   (`ORBIT(k)`, `HOP(j)`; `COUNTER-ORBIT(k)`, `AMBUSH(k)`). `select` draws a
   family uniformly from those that are APPLICABLE, then an instance uniformly
   from that family's applicable instances — the adapters never test
   `applicable` themselves (Step 1 ruling), so without this filter most draws
   would be no-ops and the labels would be polluted by tactics that never ran.
   The fallback when nothing is applicable is `FLEE` / `INTERCEPT`.
2. The 300 episodes at ratio 1.2 are seeds 2 190 000 .. 2 190 299, train = the
   first 200, held-out = the last 100. The 100 held-out episodes at ratios 0.8
   and 1.0 use the SAME 100 seeds, so the three ratios are PAIRED on the
   initial state and the tactic draws and only the ratio differs.
3. Both classifiers are scored on one pass: the episode loop records the
   recogniser's feature vector per step, and `rule_label` and the fitted
   `SoftmaxModel` are applied to those same vectors afterwards. UNKNOWN is a
   property of the window (fewer than 10 visible steps), so it is identical for
   the two and is reported once per cell.

Reported beside the gate, gating nothing: the per-ratio accuracies at 0.8 and
1.0 (the speed-agnostic check), the rule classifier's accuracy next to the
fitted one, and both 6x6 confusion matrices. The true-label UNKNOWN row is
empty by construction (the generating tactic is always known) and is printed
anyway so the matrix is the 6x6 the brief asks for.

Usage:
  ./venv/bin/python scripts/s24_recogniser_gate.py
  ./venv/bin/python scripts/s24_recogniser_gate.py --summarize
"""

from __future__ import annotations

import os

OMP_THREADS = "4"  # plan section 4; BEFORE numpy builds its pools
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pe2
from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.recogniser import (
    FEATURES,
    UNKNOWN,
    SoftmaxModel,
    TacticRecogniser,
    library_of,
    rule_label,
    tactic_family,
)
from pe2.shield import CBFShield
from pe2.tactics import (
    Ambush,
    CounterOrbit,
    Flee,
    HoldHidden,
    Hop,
    Intercept,
    Lead,
    Orbit,
    Sweep,
    TacticLaw,
    TacticOpponent,
    WallSlide,
)

PLAN = "docs/pathway/tactics_plan_2026-10-13.md"
OUT = pathlib.Path("results/derisk") / "s24_recogniser_gate.json"

# --- fixed here before the run. Do not tune. -------------------------------
CONSTRUCT_SEED0 = 2_190_000  # the round's construct stream (plan section 5)
N_TRAIN = 200
N_HELDOUT = 100
TRAIN_RATIO = 1.2  # the ratio the recogniser is FITTED at
OTHER_RATIOS = (0.8, 1.0)  # reported, not gated
WINDOW = 40  # ~2 s at dt = 0.05 (plan section 1)
SPAN_LIMIT = 80  # a tactic is held until DONE or this many steps
GATE = (
    "The recogniser is usable by a selector only if held-out accuracy >= 0.80 "
    "for the role at ratio 1.2 with UNKNOWN <= 0.30 of steps."
)
GATE_ACCURACY = 0.80
GATE_UNKNOWN = 0.30
ROLES = ("evader", "pursuer")  # the role of the OPPONENT being recognised


# ----------------------------------------------------------------------------
# the libraries and the uniform-random selector
# ----------------------------------------------------------------------------
class _Capped:
    """`inner`, held for at most `limit` steps of one span, then DONE.

    The adapters switch on DONE only; the brief's "held until DONE or 80 steps"
    is therefore a property of the tactic as offered to `select`, which is what
    this wrapper is. Everything else is forwarded, so the controller, the
    precondition and the name are Step 1's unchanged.
    """

    def __init__(self, inner, limit: int = SPAN_LIMIT) -> None:
        self.inner = inner
        self.limit = int(limit)
        self.steps = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def reset(self, env=None) -> None:
        self.inner.reset(env)
        self.steps = 0

    def enter(self) -> None:
        self.inner.enter()
        self.steps = 0

    def applicable(self, obs, env) -> bool:
        return self.inner.applicable(obs, env)

    def act(self, *args):
        if self.steps >= self.limit:
            return None
        action = self.inner.act(*args)
        if action is None:
            return None
        self.steps += 1
        return action


def families(role_of_opponent: str, cfg: EnvConfig) -> dict[str, list]:
    """The role's five families, each a list of `_Capped` instances (one per
    obstacle for the parameterised ones)."""
    ks = range(cfg.n_obstacles)
    if role_of_opponent == "evader":
        raw = {
            "FLEE": [Flee(cfg)],
            "ORBIT": [Orbit(cfg, k) for k in ks],
            "HOP": [Hop(cfg, k) for k in ks],
            "WALL-SLIDE": [WallSlide(cfg)],
            "HOLD-HIDDEN": [HoldHidden(cfg)],
        }
    else:
        raw = {
            "INTERCEPT": [Intercept(cfg)],
            "LEAD": [Lead(cfg)],
            "COUNTER-ORBIT": [CounterOrbit(cfg, k) for k in ks],
            "AMBUSH": [Ambush(cfg, k) for k in ks],
            "SWEEP": [Sweep(cfg)],
        }
    if tuple(raw) != library_of(role_of_opponent):
        raise SystemExit(f"library order diverged from pe2.recogniser: {tuple(raw)}")
    return {f: [_Capped(t) for t in ts] for f, ts in raw.items()}


def uniform_select(pool: dict[str, list], fallback: str, rng: np.random.Generator):
    """Uniform over APPLICABLE families, then uniform over that family's
    applicable instances (reading 1 in the module docstring)."""
    names = tuple(pool)

    def select(obs, env, history):
        options = [
            (f, [t for t in pool[f] if t.applicable(obs, env)]) for f in names
        ]
        options = [(f, ts) for f, ts in options if ts]
        if not options:
            return pool[fallback][0]
        _, ts = options[int(rng.integers(len(options)))]
        return ts[int(rng.integers(len(ts)))]

    return select


def flat(pool: dict[str, list]) -> list:
    return [t for ts in pool.values() for t in ts]


# ----------------------------------------------------------------------------
# one episode
# ----------------------------------------------------------------------------
def _unscale_pursuer(action, cfg: EnvConfig) -> tuple[float, float]:
    """A law's normalised action back to (v, omega) — exactly
    `SingleAgentWrapper._scale_action` for the pursuer role."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return (a[0] + 1.0) / 2.0 * cfg.v_max, a[1] * cfg.omega_max


def play_episode(seed: int, cfg: EnvConfig) -> dict:
    """One labelled self-play episode. Returns, per observed role, the stacked
    feature vectors, the true tactic family per step, and the step count."""
    env = PursuitEvasionEnv(cfg)
    obs, _ = env.reset(seed=seed)
    shield = CBFShield(cfg)
    if (shield.margin, shield.gamma) != (0.0, 0.2):
        raise SystemExit("CBFShield defaults moved; this round runs at 0.0 / 0.2")

    evader_pool = families("evader", cfg)
    pursuer_pool = families("pursuer", cfg)
    evader = TacticOpponent(
        flat(evader_pool),
        uniform_select(evader_pool, "FLEE", np.random.default_rng([seed, 101])),
    )
    law = TacticLaw(
        flat(pursuer_pool),
        uniform_select(pursuer_pool, "INTERCEPT", np.random.default_rng([seed, 202])),
    )
    evader.reset(env)
    law.reset(env)
    # The recogniser of the evader is carried by the PURSUER and vice versa.
    rec = {
        "evader": TacticRecogniser(cfg, "evader", WINDOW),
        "pursuer": TacticRecogniser(cfg, "pursuer", WINDOW),
    }
    rows: dict[str, list] = {"evader": [], "pursuer": []}

    while True:
        action = law.act(obs["pursuer"])
        if action is None:  # nothing to act on: `FovLimitedPursuer`'s fallback
            v_p, w_p = 0.0, 0.5 * cfg.omega_max
        else:
            v_p, w_p = _unscale_pursuer(action, cfg)
        v_e, w_e = evader.act(obs["evader"], env)
        true = {
            "evader": tactic_family(evader.current.name),
            "pursuer": tactic_family(law.current.name),
        }
        (v_p, w_p), _ = shield.filter(
            (v_p, w_p), env.pursuer_pose, env.obstacles, v_max=cfg.v_max
        )
        obs, _, terminated, truncated, _ = env.step([v_p, w_p, v_e, w_e])

        for role, (observer, target, seen) in {
            "evader": (env.pursuer_pose, env.evader_pose, obs["pursuer"][6] > 0.5),
            "pursuer": (env.evader_pose, env.pursuer_pose, obs["evader"][6] > 0.5),
        }.items():
            rec[role].observe(
                observer, target[:2] if seen else None, env.obstacles
            )
            rows[role].append((rec[role].features(), true[role]))
        if terminated or truncated:
            break

    out = {}
    for role, recorded in rows.items():
        known = [(f, t) for f, t in recorded if f is not None]
        out[role] = {
            "X": np.array([f for f, _ in known], dtype=float).reshape(-1, len(FEATURES)),
            "y": [t for _, t in known],
            "unknown_true": [t for f, t in recorded if f is None],
            "n_steps": len(recorded),
        }
    return out


def play_block(seeds, ratio: float) -> dict:
    """`play_episode` over a block of seeds, pooled per observed role."""
    cfg = EnvConfig(evader_speed_ratio=ratio)
    pooled = {
        role: {"X": [], "y": [], "unknown_true": [], "n_steps": 0} for role in ROLES
    }
    for seed in seeds:
        episode = play_episode(seed, cfg)
        for role in ROLES:
            pooled[role]["X"].append(episode[role]["X"])
            pooled[role]["y"] += episode[role]["y"]
            pooled[role]["unknown_true"] += episode[role]["unknown_true"]
            pooled[role]["n_steps"] += episode[role]["n_steps"]
    for role in ROLES:
        pooled[role]["X"] = np.concatenate(pooled[role]["X"], axis=0)
    return pooled


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score(pooled_role: dict, role: str, model: SoftmaxModel | None) -> dict:
    """Accuracy and the 6x6 confusion matrix for one classifier on one cell.

    UNKNOWN steps are excluded from the accuracy (the brief) and counted in the
    UNKNOWN column of the matrix; the UNKNOWN ROW stays empty because the
    generating tactic is always known.
    """
    labels = library_of(role)
    axis = list(labels) + [UNKNOWN]
    X, y = pooled_role["X"], pooled_role["y"]
    if model is None:
        predicted = [rule_label(x, role) for x in X]
    else:
        predicted = model.predict(X) if len(X) else []
    matrix = np.zeros((len(axis), len(axis)), dtype=int)
    index = {name: i for i, name in enumerate(axis)}
    for true, pred in zip(y, predicted, strict=True):
        matrix[index[true], index[pred]] += 1
    for true in pooled_role["unknown_true"]:
        matrix[index[true], index[UNKNOWN]] += 1
    correct = int(np.trace(matrix[: len(labels), : len(labels)]))
    n_known = len(y)
    n_steps = pooled_role["n_steps"]
    return {
        "accuracy": correct / n_known if n_known else None,
        "n_known": n_known,
        "n_steps": n_steps,
        "unknown_fraction": (n_steps - n_known) / n_steps if n_steps else None,
        "axis": axis,
        "confusion": matrix.tolist(),
        "per_true_support": {
            name: int(matrix[i].sum()) for i, name in enumerate(axis)
        },
    }


def run() -> dict:
    t0 = time.time()
    train_seeds = range(CONSTRUCT_SEED0, CONSTRUCT_SEED0 + N_TRAIN)
    heldout_seeds = range(
        CONSTRUCT_SEED0 + N_TRAIN, CONSTRUCT_SEED0 + N_TRAIN + N_HELDOUT
    )
    train = play_block(train_seeds, TRAIN_RATIO)
    blocks = {TRAIN_RATIO: play_block(heldout_seeds, TRAIN_RATIO)}
    for ratio in OTHER_RATIOS:
        blocks[ratio] = play_block(heldout_seeds, ratio)

    models = {
        role: SoftmaxModel.fit(train[role]["X"], train[role]["y"], library_of(role))
        for role in ROLES
    }
    cells = {}
    for role in ROLES:
        cells[role] = {
            "train": {
                "fitted": score(train[role], role, models[role]),
                "rule": score(train[role], role, None),
            },
            "heldout": {
                str(ratio): {
                    "fitted": score(blocks[ratio][role], role, models[role]),
                    "rule": score(blocks[ratio][role], role, None),
                }
                for ratio in blocks
            },
        }
    # The classifier PROPOSED FOR USE: the better held-out accuracy at the
    # training ratio. The gate's verdict is read on that one.
    proposed = {}
    verdict = {}
    for role in ROLES:
        gated = cells[role]["heldout"][str(TRAIN_RATIO)]
        use = max(("fitted", "rule"), key=lambda c: gated[c]["accuracy"] or 0.0)
        proposed[role] = use
        verdict[role] = {
            "classifier": use,
            "accuracy": gated[use]["accuracy"],
            "unknown_fraction": gated[use]["unknown_fraction"],
            "passed": bool(
                (gated[use]["accuracy"] or 0.0) >= GATE_ACCURACY
                and (gated[use]["unknown_fraction"] or 1.0) <= GATE_UNKNOWN
            ),
        }

    record = {
        "round": "S24.0 Step 3 — opponent-tactic recogniser construct gate",
        "gate": GATE,
        "gate_thresholds": {
            "accuracy_min": GATE_ACCURACY,
            "unknown_max": GATE_UNKNOWN,
            "ratio": TRAIN_RATIO,
        },
        "protocol": {
            "construct_seed0": CONSTRUCT_SEED0,
            "n_train": N_TRAIN,
            "n_heldout": N_HELDOUT,
            "train_ratio": TRAIN_RATIO,
            "other_ratios": list(OTHER_RATIOS),
            "window": WINDOW,
            "span_limit": SPAN_LIMIT,
            "heldout_seeds_paired_across_ratios": True,
            "pursuer_shielded": True,
            "sklearn_available": False,
        },
        "features": list(FEATURES),
        "cells": cells,
        "proposed_classifier": proposed,
        "verdict": verdict,
        "fitted_weights": {role: models[role].to_dict() for role in ROLES},
        "wall_clock_s": round(time.time() - t0, 1),
        "provenance": provenance(),
    }
    _write_json(OUT, record)
    record["path"] = str(OUT)
    return record


# ----------------------------------------------------------------------------
# provenance, output, table
# ----------------------------------------------------------------------------
def provenance() -> dict:
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    plan = _REPO / PLAN
    return {
        "git_commit": sha,
        "pe2_file": pe2.__file__,
        "spec": PLAN,
        "spec_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def confusion_text(cell: dict) -> list[str]:
    axis = cell["axis"]
    short = [a[:6] for a in axis]
    corner = "true \\ pred"
    head = f"{corner:<13}" + "".join(f"{s:>7}" for s in short) + f"{'n':>8}"
    lines = [head, "-" * len(head)]
    for i, name in enumerate(axis):
        row = cell["confusion"][i]
        lines.append(
            f"{name:<13}" + "".join(f"{v:>7}" for v in row) + f"{sum(row):>8}"
        )
    return lines


def table(record: dict) -> str:
    lines = [record["round"], "", GATE, ""]
    for role in ROLES:
        v = record["verdict"][role]
        lines.append(
            f"GATE [{role:<7}] {'PASS' if v['passed'] else 'FAIL'}: "
            f"{v['classifier']} accuracy {_f(v['accuracy'])} at ratio "
            f"{TRAIN_RATIO} (bar {GATE_ACCURACY}), UNKNOWN "
            f"{_f(v['unknown_fraction'])} (bar {GATE_UNKNOWN})"
        )
    lines.append("")
    head = (
        f"{'opponent':<8} {'clf':<7} {'acc 1.2':>8} {'acc 0.8':>8} {'acc 1.0':>8} "
        f"{'UNK 1.2':>8} {'UNK 0.8':>8} {'UNK 1.0':>8} {'train acc':>10} "
        f"{'steps 1.2':>10}"
    )
    lines += [head, "-" * len(head)]
    for role in ROLES:
        cell = record["cells"][role]
        for clf in ("fitted", "rule"):
            held = {r: cell["heldout"][str(r)][clf] for r in (1.2, 0.8, 1.0)}
            lines.append(
                f"{role:<8} {clf:<7} "
                + "".join(f"{_f(held[r]['accuracy']):>9}" for r in (1.2, 0.8, 1.0))
                + "".join(
                    f"{_f(held[r]['unknown_fraction']):>9}" for r in (1.2, 0.8, 1.0)
                )
                + f"{_f(cell['train'][clf]['accuracy']):>10}"
                + f"{held[1.2]['n_steps']:>11}"
            )
    for role in ROLES:
        use = record["proposed_classifier"][role]
        cell = record["cells"][role]["heldout"][str(TRAIN_RATIO)][use]
        lines += [
            "",
            (
                f"confusion, opponent = {role}, classifier = {use}, held-out "
                f"at ratio {TRAIN_RATIO} (UNKNOWN row empty by construction):"
            ),
        ]
        lines += confusion_text(cell)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summarize", action="store_true", help="re-print the existing record"
    )
    args = ap.parse_args(argv)
    if args.summarize:
        record = json.loads(OUT.read_text())
    else:
        record = run()
        print(f"wrote {record['path']}")
    print(table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
