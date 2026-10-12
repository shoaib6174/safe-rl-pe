"""S24.0 Step 3 — the opponent-tactic recogniser.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1 ("**opponent tactic
history** via a **recogniser** that labels the opponent's last ~2 s of motion as
one of the library's tactics ... it is a construct and gets its own gate (>= 0.8
accuracy on held-out episodes, confusion matrix reported) before any selector
consumes it") and section 3.

A map from the LAST `window` STEPS OF OBSERVED OPPONENT MOTION to one of the
five tactic families of that opponent's library, or `UNKNOWN` when the window
holds fewer than `MIN_VISIBLE` visible steps. It sees what any agent in this
game sees — the opponent's position on the steps it was visible, its own pose
and the static map (`pe2/obs.py`: obstacle features are never masked).

Two classifiers over the SAME features, both measured by the gate; its verdict
applies to the one proposed for use.
- `rule_label` — one readable rule per tactic, thresholds written down here.
- `SoftmaxModel` — multinomial logistic regression fitted on labelled self-play
  (`scripts/s24_recogniser_gate.py`). `sklearn` is NOT in this venv (checked)
  and `pe2` carries no scipy dependency either (`pe2/shield.py` names the same
  constraint: its CI installs only numpy / gymnasium / matplotlib / imageio), so
  the fit is the numpy one the brief allows instead — Nesterov accelerated
  gradient on the convex objective from a zero start: one answer, no seed, no
  learning rate to choose.

The binding rule, as for the tactics themselves: **nothing here reads the speed
ratio or any speed bound** (`evader_speed_ratio`, `evader_v_max`). Every feature
is dimensionless — displacements over `v_ref`, the largest per-step opponent
displacement OBSERVED so far this episode, or over the window's own path length;
distances over the half arena. `tests/pe2/test_recogniser.py` asserts that on
the source and at runtime at ratios 0.8 / 1.0 / 1.2.

Labels are tactic FAMILIES: `ORBIT(0)` and `ORBIT(1)` are both `ORBIT`
(`tactic_family`). The parameter is an obstacle index, which a recogniser of
MOTION cannot be asked to recover and no selector input in section 1 needs.
"""

from __future__ import annotations

import numpy as np

from pe2.config import EnvConfig

UNKNOWN = "UNKNOWN"
EVADER_TACTICS = ("FLEE", "ORBIT", "HOP", "WALL-SLIDE", "HOLD-HIDDEN")
PURSUER_TACTICS = ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
MIN_VISIBLE = 10  # fewer visible steps in the window -> UNKNOWN (brief)
TREND_CLIP = 3.0  # trends are per-step rates in units of a step; clip outliers

FEATURES = (
    "speed_frac",  # mean per-step displacement / v_ref            (mean speed)
    "tortuosity",  # net displacement / path length
    "sweep_ratio",  # tangential arc length about obstacle k / path length
    "obstacle_trend",  # d(surface distance to obstacle k)/dstep / step length
    "observer_trend",  # d(distance to the observer)/dstep / step length
    "wall_trend",  # d(distance to the nearest wall)/dstep / step length
    "visible_frac",  # visible steps in the window / window
    "obstacle_level",  # mean surface distance to obstacle k / half arena
    "observer_level",  # mean distance to the observer / half arena
    "wall_level",  # mean distance to the nearest wall / half arena
)
# The first seven are the brief's list. The three `_level` features are added
# because that list is TRENDS only while two tactics are defined by a LEVEL:
# `WALL-SLIDE` is "pressed against the wall" and `ORBIT` / `COUNTER-ORBIT` are
# "on obstacle k's hug circle" — without them a wall-sliding evader and one
# fleeing INTO a wall are identical. Normalised by the half arena, never a speed.


def tactic_family(name: str) -> str:
    """`ORBIT(1)` -> `ORBIT`; the label space is the family (see module doc)."""
    return str(name).split("(")[0]


def library_of(role_of_opponent: str) -> tuple[str, ...]:
    if role_of_opponent == "evader":
        return EVADER_TACTICS
    if role_of_opponent == "pursuer":
        return PURSUER_TACTICS
    raise ValueError(f"role_of_opponent must be evader|pursuer, got {role_of_opponent!r}")


# ------------------------------------------------------------------ features
def _slope(steps: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of `values` per unit `steps` (0 for a single point)."""
    centred = steps - steps.mean()
    denom = float(centred @ centred)
    return 0.0 if denom < 1e-12 else float(centred @ (values - values.mean()) / denom)


def _nearest_obstacle(point: np.ndarray, obstacles: list[dict]) -> int | None:
    if not obstacles:
        return None
    surface = [
        float(np.hypot(point[0] - o["x"], point[1] - o["y"])) - float(o["radius"])
        for o in obstacles
    ]
    return int(np.argmin(surface))


def window_features(
    samples: list[tuple[int, np.ndarray, np.ndarray]],
    obstacles: list[dict],
    cfg: EnvConfig,
    v_ref: float,
    window: int,
) -> np.ndarray:
    """The `FEATURES` vector for the visible samples `(step, own_xy, opp_xy)`.

    Obstacle k is the one nearest the LATEST visible opponent position and is
    held fixed across the window, so the sweep and both obstacle features
    describe one obstacle rather than a changing nearest one.
    """
    idx = np.array([s[0] for s in samples], dtype=float)
    own = np.array([s[1] for s in samples], dtype=float)
    opp = np.array([s[2] for s in samples], dtype=float)

    step_xy = np.diff(opp, axis=0)
    path = float(np.hypot(step_xy[:, 0], step_xy[:, 1]).sum())
    span = float(idx[-1] - idx[0])
    mean_step = path / span if span > 0.0 else 0.0
    net = float(np.hypot(*(opp[-1] - opp[0])))
    # The normaliser for a per-step rate: the window's own step length, floored
    # at a tenth of the observed top step so a near-stationary window does not
    # amplify position noise into a huge trend.
    norm = max(mean_step, 0.1 * float(v_ref), 1e-6)

    half = cfg.arena_size / 2.0
    d_own = np.hypot(opp[:, 0] - own[:, 0], opp[:, 1] - own[:, 1])
    d_wall = half - np.maximum(np.abs(opp[:, 0]), np.abs(opp[:, 1]))

    k = _nearest_obstacle(opp[-1], obstacles)
    if k is None:  # an obstacle-free arena: no obstacle to be near or to circle
        sweep_ratio, obstacle_trend, obstacle_level = 0.0, 0.0, 1.0
    else:
        centre = np.array([obstacles[k]["x"], obstacles[k]["y"]], dtype=float)
        rel = opp - centre
        radius = np.hypot(rel[:, 0], rel[:, 1])
        phi = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        # The TANGENTIAL part of the path: r * |dphi| summed over the window,
        # as a fraction of the path length. 1 for a circular arc about k, 0 for
        # a radial run at k, and — unlike (total swept angle) * r / path —
        # small for a straight chord past k, whose far segments are radial.
        arc = float(
            (0.5 * (radius[:-1] + radius[1:]) * np.abs(np.diff(phi))).sum()
        )
        sweep_ratio = arc / path if path > 1e-9 else 0.0
        d_obstacle = radius - float(obstacles[k]["radius"])
        obstacle_trend = _slope(idx, d_obstacle) / norm
        obstacle_level = float(d_obstacle.mean()) / half

    return np.array(
        [
            mean_step / v_ref if v_ref > 1e-9 else 0.0,
            net / path if path > 1e-9 else 0.0,
            sweep_ratio,
            np.clip(obstacle_trend, -TREND_CLIP, TREND_CLIP),
            np.clip(_slope(idx, d_own) / norm, -TREND_CLIP, TREND_CLIP),
            np.clip(_slope(idx, d_wall) / norm, -TREND_CLIP, TREND_CLIP),
            len(samples) / float(window),
            obstacle_level,
            float(d_own.mean()) / half,
            float(d_wall.mean()) / half,
        ],
        dtype=float,
    )


# ---------------------------------------------------------- rule classifier
HOLD_SPEED = 0.15  # below this fraction of the observed top speed: standing
SWEEP_RATIO = 0.55  # at/above this: the path is an arc about obstacle k
NEAR_OBSTACLE = 0.10  # x half arena (0.50 m): on the hug circle
NEAR_WALL = 0.06  # x half arena (0.30 m): pressed against the wall
STRAIGHT = 0.75  # tortuosity at/above this: a straight run
OPENING = 0.35  # observer trend at/above this: actively opening the gap
CLOSING_HARD = -0.60  # observer trend at/below this: closing straight at you


def rule_label(f: np.ndarray, role_of_opponent: str) -> str:
    """One readable rule per tactic, in order; the last is the role's default.

    The weakest rule, stated as such: `INTERCEPT` vs `LEAD` differ only in
    aiming 1 s of velocity ahead, which changes the closing RATE and nothing
    else an observer of motion can see, so the split is a closing-rate
    threshold and the confusion matrix is expected to show it.
    """
    g = dict(zip(FEATURES, f, strict=True))
    if role_of_opponent == "evader":
        if g["speed_frac"] < HOLD_SPEED:
            return "HOLD-HIDDEN"  # the only evader tactic that commands v = 0
        if g["sweep_ratio"] >= SWEEP_RATIO and g["obstacle_level"] <= NEAR_OBSTACLE:
            return "ORBIT"  # arcing around an obstacle it is hugging
        if g["wall_level"] <= NEAR_WALL and g["obstacle_level"] > NEAR_OBSTACLE:
            return "WALL-SLIDE"  # pressed against a wall, away from obstacles
        if g["tortuosity"] >= STRAIGHT and g["observer_trend"] < OPENING:
            return "HOP"  # a straight run that does NOT open the gap: a transit
        return "FLEE"  # radial flight: straight and opening the gap
    if g["speed_frac"] < HOLD_SPEED:
        return "AMBUSH"  # the only pursuer tactic that holds position
    if g["sweep_ratio"] >= SWEEP_RATIO and g["obstacle_level"] <= NEAR_OBSTACLE:
        return "COUNTER-ORBIT"  # arcing around the obstacle between the two
    if g["observer_trend"] >= -OPENING:
        return "SWEEP"  # moving, but not closing: it is not following you
    if g["observer_trend"] <= CLOSING_HARD:
        return "INTERCEPT"  # closing at full step rate: straight at you
    return "LEAD"  # closing, but off the direct bearing (aiming ahead)


# --------------------------------------------------------- fitted classifier
class SoftmaxModel:
    """Multinomial logistic regression on `FEATURES`, L2-regularised. The
    training mean/scale are part of the model, so a loaded model standardises
    its input exactly as the fit did."""

    def __init__(self, classes, mean, scale, W, b) -> None:
        self.classes = tuple(classes)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.W = np.asarray(W, dtype=float)
        self.b = np.asarray(b, dtype=float)

    @classmethod
    def fit(cls, X, y, classes, l2: float = 1e-3, iterations: int = 4000) -> SoftmaxModel:
        """Minimise the L2-regularised multinomial cross-entropy. The objective
        is convex and smooth, so Nesterov accelerated gradient at the step 1/L —
        L from the design's own Gram matrix (the standard 0.5 lambda_max / n
        bound for the softmax Hessian) plus `l2` — needs no tuning; it stops on
        the gradient norm, so `iterations` is a guard, not a schedule."""
        X = np.asarray(X, dtype=float)
        classes = tuple(classes)
        index = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(y), len(classes)))
        Y[np.arange(len(y)), [index[label] for label in y]] = 1.0
        mean = X.mean(axis=0)
        spread = X.std(axis=0)
        scale = np.where(spread < 1e-9, 1.0, spread)
        Z = np.hstack([(X - mean) / scale, np.ones((len(X), 1))])
        n, d = Z.shape
        gram = Z.T @ Z
        step = 1.0 / (0.5 * float(np.linalg.eigvalsh(gram)[-1]) / n + l2)

        theta = np.zeros((d, len(classes)))
        momentum = theta.copy()
        for t in range(int(iterations)):
            logits = Z @ momentum
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            grad = Z.T @ (p - Y) / n
            grad[:-1] += l2 * momentum[:-1]  # the bias row is not penalised
            nxt = momentum - step * grad
            momentum = nxt + (t / (t + 3.0)) * (nxt - theta)
            theta = nxt
            if float(np.abs(grad).max()) < 1e-8:
                break
        return cls(classes, mean, scale, theta[:-1], theta[-1])

    def predict(self, X) -> list[str]:
        Z = (np.atleast_2d(np.asarray(X, dtype=float)) - self.mean) / self.scale
        return [self.classes[i] for i in np.argmax(Z @ self.W + self.b, axis=1)]

    def to_dict(self) -> dict:
        return {
            "classes": list(self.classes),
            "features": list(FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SoftmaxModel:
        return cls(d["classes"], d["mean"], d["scale"], d["W"], d["b"])


# ---------------------------------------------------------------- recogniser
class TacticRecogniser:
    """Per-step labels and the run history of the opponent's tactic.

    `observe` is called once per env step with what the observer has — its own
    pose, the opponent's position when visible (`None` when not), the static map
    — and returns `label(t)`. `history` is the run log
    `(label, start_step, end_step)`, `end_step` exclusive, so adjacent steps
    with the same label are ONE run. `model` chooses the classifier (the fitted
    one when given, else the rule one); `features()` is the window's vector, or
    `None` on an UNKNOWN step, so a caller can score both on one pass.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        role_of_opponent: str,
        window: int = 40,
        model: SoftmaxModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.role_of_opponent = role_of_opponent
        self.labels = library_of(role_of_opponent)
        self.window = int(window)
        self.model = model
        self.classifier = "rule" if model is None else "fitted"
        self.reset()

    def reset(self) -> None:
        self._samples: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._step = -1
        self._last: tuple[int, np.ndarray] | None = None
        self.v_ref = 0.0  # largest per-step opponent displacement seen so far
        self._features: np.ndarray | None = None
        self.label = UNKNOWN
        self.history: list[tuple[str, int, int]] = []

    def observe(self, own_pose, opponent_xy, obstacles: list[dict]) -> str:
        self._step += 1
        if opponent_xy is not None:
            # COPIES, not views: callers hand in the env's own live pose arrays
            # (`env.evader_pose`), which the env mutates in place, so a stored
            # view would make every sample in the window read as the LATEST
            # position and every motion feature would be exactly zero.
            opp = np.array(opponent_xy, dtype=float)[:2]
            if self._last is not None:
                gap = self._step - self._last[0]
                self.v_ref = max(
                    self.v_ref, float(np.hypot(*(opp - self._last[1]))) / gap
                )
            self._last = (self._step, opp)
            self._samples.append(
                (self._step, np.array(own_pose, dtype=float)[:2], opp)
            )
        oldest = self._step - self.window + 1
        self._samples = [s for s in self._samples if s[0] >= oldest]

        self._features = None
        if len(self._samples) >= MIN_VISIBLE:
            self._features = window_features(
                self._samples, list(obstacles), self.cfg, self.v_ref, self.window
            )
        if self._features is None:
            self.label = UNKNOWN
        elif self.model is None:
            self.label = rule_label(self._features, self.role_of_opponent)
        else:
            self.label = self.model.predict(self._features)[0]
        self._record(self.label)
        return self.label

    def features(self) -> np.ndarray | None:
        return self._features

    def _record(self, label: str) -> None:
        if self.history and self.history[-1][0] == label:
            name, start, _ = self.history[-1]
            self.history[-1] = (name, start, self._step + 1)
        else:
            self.history.append((label, self._step, self._step + 1))
