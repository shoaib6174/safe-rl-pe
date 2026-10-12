"""S24.0 Step 3 — unit checks for the opponent-tactic recogniser.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1. The construct GATE
(>= 0.8 held-out accuracy, confusion matrices) is measured on 300 self-play
episodes by `scripts/s24_recogniser_gate.py` and is NOT re-run here; this file
checks the pieces the gate rests on, on trajectories built by hand so that what
the features should say is known independently of any episode:

- a pure arc about an obstacle, a straight run, a wall slide, a stop and a
  transit that loses line of sight — what each feature must read, and the label
  the rule classifier must give;
- `UNKNOWN` below `MIN_VISIBLE` visible steps in the window;
- the fitted model's round trip (fit, predict, to_dict / from_dict);
- the run boundaries of `history`;
- the speed-agnostic rule, twice: on the source (no `evader_speed_ratio`, no
  `evader_v_max`) and at runtime, by feeding ONE world-frame trajectory to
  recognisers built at ratios 0.8 / 1.0 / 1.2 and requiring identical features.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import numpy as np
import pytest

from pe2 import recogniser as R
from pe2.config import EnvConfig
from pe2.recogniser import (
    FEATURES,
    MIN_VISIBLE,
    UNKNOWN,
    SoftmaxModel,
    TacticRecogniser,
    library_of,
    rule_label,
    tactic_family,
    window_features,
)

CFG = EnvConfig()
RATIOS = (0.8, 1.0, 1.2)
OBSTACLE = {"x": 0.0, "y": 0.0, "radius": 1.0}
FAR_OBSTACLE = {"x": 0.0, "y": 3.5, "radius": 1.0}
STEP = 0.06  # one step at the locked dt and v_max, so the paths are plausible


# ------------------------------------------------------------------- harness
def feed(
    positions,
    observer=lambda i: (0.0, -4.0),
    visible=lambda i: True,
    role: str = "evader",
    obstacles=(OBSTACLE,),
    cfg: EnvConfig = CFG,
    window: int = 40,
) -> TacticRecogniser:
    """Drive a recogniser over a hand-built trajectory, one step per position."""
    rec = TacticRecogniser(cfg, role, window=window)
    for i, point in enumerate(positions):
        rec.observe(
            np.array([*observer(i), 0.0]),
            np.array(point, dtype=float) if visible(i) else None,
            list(obstacles),
        )
    return rec


def named(rec: TacticRecogniser) -> dict:
    return dict(zip(FEATURES, rec.feature_vector(), strict=True))


def arc(radius: float, dphi: float, n: int, centre=(0.0, 0.0)):
    return [
        (centre[0] + radius * np.cos(dphi * i), centre[1] + radius * np.sin(dphi * i))
        for i in range(n)
    ]


def line(start, direction, n: int, step: float = STEP):
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.hypot(*unit)
    return [tuple(np.asarray(start, dtype=float) + unit * step * i) for i in range(n)]


# ------------------------------------------------------------------ features
class TestFeatures:
    def test_a_pure_orbit_is_all_sweep_and_little_net_displacement(self):
        # Just over half a turn on the hug circle of the obstacle at the origin.
        g = named(feed(arc(1.15, 0.08, 40)))
        assert g["sweep_ratio"] > 0.95  # the whole path is tangential
        assert g["tortuosity"] < 0.70  # and it comes back on itself
        assert abs(g["obstacle_trend"]) < 0.05  # at a held radius
        assert g["obstacle_level"] == pytest.approx(0.03, abs=0.01)
        assert g["visible_fraction"] == 1.0

    def test_a_straight_run_is_tortuosity_one_and_no_sweep(self):
        g = named(feed(line((1.0, 1.0), (1.0, 1.0), 40), observer=lambda i: (-4.0, -4.0)))
        assert g["tortuosity"] == pytest.approx(1.0)
        assert g["sweep_ratio"] < 0.05  # radially outward from the obstacle
        assert g["observer_trend"] == pytest.approx(1.0, abs=0.01)  # opening
        assert g["obstacle_trend"] == pytest.approx(1.0, abs=0.01)

    def test_a_hop_is_tortuosity_one_then_the_line_of_sight_breaks(self):
        # A transit across the far side of the obstacle; seen for 28 steps.
        hop = line((-2.0, 1.6), (1.0, 0.0), 40)
        g = named(feed(hop, visible=lambda i: i < 28))
        assert g["tortuosity"] == pytest.approx(1.0)
        assert g["visible_fraction"] == pytest.approx(28 / 40)
        assert g["observer_trend"] < 0.35  # a transit, not a flight
        # Keep going unseen and the window empties out below MIN_VISIBLE.
        rec = feed(hop + line((2.0, 1.6), (1.0, 0.0), 30), visible=lambda i: i < 28)
        assert rec.feature_vector() is None
        assert rec.label == UNKNOWN

    def test_a_hold_has_zero_speed_and_zero_trends(self):
        g = named(feed([(2.0, 2.0)] * 40))
        assert g["speed_frac"] == 0.0
        assert g["tortuosity"] == 0.0
        assert g["sweep_ratio"] == 0.0
        assert (g["obstacle_trend"], g["observer_trend"], g["wall_trend"]) == (
            0.0,
            0.0,
            0.0,
        )

    def test_speed_frac_is_the_fraction_of_the_observed_top_speed(self):
        # Full speed for 40 steps, then half speed for 40: v_ref is the top
        # speed SEEN, so the second window reads about one half.
        fast = line((-3.0, 2.0), (1.0, 0.0), 40, step=STEP)
        slow = line(fast[-1], (1.0, 0.0), 41, step=STEP / 2)[1:]
        rec = TacticRecogniser(CFG, "evader")
        for point in fast:
            rec.observe(np.array([0.0, -4.0, 0.0]), np.array(point), [OBSTACLE])
        assert named(rec)["speed_frac"] == pytest.approx(1.0)
        for point in slow:
            rec.observe(np.array([0.0, -4.0, 0.0]), np.array(point), [OBSTACLE])
        assert named(rec)["speed_frac"] == pytest.approx(0.5, abs=0.02)

    def test_samples_are_copied_not_aliased_to_the_callers_arrays(self):
        """The env hands out its OWN pose arrays and mutates them in place; a
        stored view makes every sample read as the latest position and every
        motion feature collapses to exactly zero."""
        pose = np.array([0.0, -4.0, 0.0])
        live = np.array([-1.0, 1.6])
        rec = TacticRecogniser(CFG, "evader")
        for _ in range(40):
            live += np.array([STEP, 0.0])  # in place, as `env.step` does
            rec.observe(pose, live, [OBSTACLE])
        g = named(rec)
        assert g["speed_frac"] > 0.9
        assert g["tortuosity"] == pytest.approx(1.0)

    def test_unknown_below_min_visible_visible_steps(self):
        seen = [i % 4 == 0 for i in range(40)]  # 10 visible steps in the window
        assert sum(seen) == MIN_VISIBLE
        rec = feed(line((-1.0, 1.6), (1.0, 0.0), 40), visible=lambda i: seen[i])
        assert rec.feature_vector() is not None
        rec = feed(line((-1.0, 1.6), (1.0, 0.0), 40), visible=lambda i: seen[i] and i > 0)
        assert rec.feature_vector() is None  # one fewer: nine visible steps
        assert rec.label == UNKNOWN

    def test_features_do_not_depend_on_the_speed_ratio_at_runtime(self):
        traj = arc(1.15, 0.08, 40)
        vectors = [
            feed(traj, cfg=EnvConfig(evader_speed_ratio=r)).feature_vector() for r in RATIOS
        ]
        for other in vectors[1:]:
            assert np.array_equal(vectors[0], other)

    def test_the_module_source_never_reads_a_speed_bound(self):
        """A NAME-token scan (as `test_tactics.py` does), so the module's own
        prose about the rule neither satisfies nor breaks it."""
        names = {
            token.string
            for token in tokenize.generate_tokens(
                io.StringIO(inspect.getsource(R)).readline
            )
            if token.type == tokenize.NAME
        }
        assert "evader_speed_ratio" not in names
        assert "evader_v_max" not in names

    def test_an_obstacle_free_arena_has_no_obstacle_to_circle(self):
        g = dict(
            zip(
                FEATURES,
                window_features(
                    [(i, np.array([0.0, -4.0]), np.array(p)) for i, p in
                     enumerate(line((1.0, 1.0), (1.0, 1.0), 40))],
                    [],
                    CFG,
                    STEP,
                    40,
                ),
                strict=True,
            )
        )
        assert (g["sweep_ratio"], g["obstacle_trend"]) == (0.0, 0.0)


# ------------------------------------------------------------ public product
class TestFeaturesDict:
    """`features()` is what a selector consumes (D147), so its contract is
    checked directly: fixed keys, no None, no NaN, and `valid` flipping at
    exactly `MIN_VISIBLE` visible steps in the window."""

    def test_the_keys_are_fixed_and_the_values_are_floats(self):
        rec = feed(arc(1.15, 0.08, 40))
        product = rec.features()
        assert set(product) == set(FEATURES) | {"n_visible_in_window", "valid"}
        assert product["valid"] is True
        assert product["n_visible_in_window"] == 40
        numbers = [product[name] for name in FEATURES]
        assert all(isinstance(v, float) for v in numbers)
        assert not any(np.isnan(v) or np.isinf(v) for v in numbers)
        assert numbers == [float(v) for v in rec.feature_vector()]

    def test_an_invalid_window_is_zeros_and_not_none(self):
        rec = feed(line((-1.0, 1.6), (1.0, 0.0), 40), visible=lambda i: i < 5)
        product = rec.features()
        assert product["valid"] is False
        assert product["n_visible_in_window"] == 5
        assert [product[name] for name in FEATURES] == [0.0] * len(FEATURES)
        assert rec.feature_vector() is None

    def test_valid_flips_at_exactly_min_visible_visible_steps(self):
        """One visible step per window position, so the count IS the step index
        plus one and the flip can be read off step by step."""
        rec = TacticRecogniser(CFG, "evader", window=40)
        flips = []
        for i, point in enumerate(line((-1.2, 1.6), (1.0, 0.0), 2 * MIN_VISIBLE)):
            rec.observe(np.array([0.0, -4.0, 0.0]), np.array(point), [OBSTACLE])
            flips.append((i + 1, rec.features()["valid"]))
        assert next(n for n, valid in flips if valid) == MIN_VISIBLE
        assert all(not valid for n, valid in flips if n < MIN_VISIBLE)
        assert all(valid for n, valid in flips if n >= MIN_VISIBLE)

    def test_the_same_ground_traversed_at_three_speeds_reads_the_same(self):
        """The speed-agnostic claim in its strongest form: ONE geometric path,
        walked at the step of each ratio, with the window scaled so all three
        windows cover the SAME ground. Every feature is dimensionless (a ratio
        of lengths, or a rate per unit path), so all three must agree.

        The sibling check in `TestFeatures` is the exact one (same trajectory,
        three `EnvConfig` ratios, bit-identical); this one varies the thing the
        ratio actually changes — how far the opponent moves per step.
        """
        total, radius = 1.6, 1.15  # rad of arc on the obstacle's hug circle
        products = []
        for ratio in RATIOS:
            n = round(radius * total / (0.05 * ratio))
            traj = arc(radius, total / (n - 1), n)
            products.append(
                feed(traj, cfg=EnvConfig(evader_speed_ratio=ratio), window=n).features()
            )
        assert len({p["n_visible_in_window"] for p in products}) == 3  # 3 speeds
        for other in products[1:]:
            for name in FEATURES:
                assert other[name] == pytest.approx(products[0][name], abs=0.02)


# ------------------------------------------------------------ rule classifier
class TestRuleClassifier:
    """One hand-built case per tactic, per role. These are the cases the
    thresholds in `pe2.recogniser` were written for, before any episode was
    generated."""

    def test_the_evader_cases(self):
        assert feed(arc(1.15, 0.08, 40)).label == "ORBIT"
        assert feed([(2.0, 2.0)] * 40).label == "HOLD-HIDDEN"
        assert (
            feed(line((1.0, 1.0), (1.0, 1.0), 40), observer=lambda i: (-4.0, -4.0)).label
            == "FLEE"
        )
        assert feed(line((-2.0, 1.6), (1.0, 0.0), 40)).label == "HOP"
        # Pressed against the +x wall, sliding along it, obstacle far away.
        assert (
            feed(
                line((4.75, -2.0), (0.0, 1.0), 40), observer=lambda i: (1.0, -3.0)
            ).label
            == "WALL-SLIDE"
        )

    def test_the_pursuer_cases(self):
        # The observer is the evader at the origin; the obstacle is well away
        # from every path so no case can read as an orbit by accident.
        here = (0.0, 0.0)
        closing = line((4.0, 0.0), (-1.0, 0.0), 40)
        assert feed(closing, observer=lambda i: here, role="pursuer",
                    obstacles=(FAR_OBSTACLE,)).label == "INTERCEPT"
        # Closing at about 45 % of its step: aiming ahead of, not at, the target.
        lead = line((4.0, -1.5), (-0.45, np.sqrt(1 - 0.45**2)), 40)
        rec = feed(lead, observer=lambda i: here, role="pursuer",
                   obstacles=(FAR_OBSTACLE,))
        assert -0.60 < named(rec)["observer_trend"] < -0.35
        assert rec.label == "LEAD"
        # Crossing at constant range: moving, but not following.
        assert feed(arc(4.0, 0.015, 40), observer=lambda i: here, role="pursuer",
                    obstacles=(FAR_OBSTACLE,)).label == "SWEEP"
        assert feed([(3.0, 0.0)] * 40, observer=lambda i: here, role="pursuer",
                    obstacles=(FAR_OBSTACLE,)).label == "AMBUSH"
        # Round the obstacle that is between the two, at a held radius.
        assert feed(arc(1.15, 0.08, 40), observer=lambda i: (0.0, -4.0),
                    role="pursuer").label == "COUNTER-ORBIT"

    def test_every_rule_label_is_in_the_role_library(self):
        rng = np.random.default_rng(0)
        for role in ("evader", "pursuer"):
            for _ in range(200):
                f = rng.normal(0.0, 1.0, size=len(FEATURES))
                assert rule_label(f, role) in library_of(role)


# ----------------------------------------------------------- fitted classifier
class TestSoftmaxModel:
    def test_it_round_trips_on_a_separable_set(self):
        rng = np.random.default_rng(24)
        classes = library_of("evader")
        centres = np.eye(len(classes), len(FEATURES)) * 8.0
        X = np.concatenate(
            [centres[i] + 0.2 * rng.normal(size=(80, len(FEATURES)))
             for i in range(len(classes))]
        )
        y = [c for c in classes for _ in range(80)]
        model = SoftmaxModel.fit(X, y, classes)
        assert np.mean([p == t for p, t in zip(model.predict(X), y)]) == 1.0
        reloaded = SoftmaxModel.from_dict(model.to_dict())
        assert reloaded.predict(X) == model.predict(X)
        assert model.to_dict()["features"] == list(FEATURES)

    def test_a_fitted_model_drives_the_recogniser(self):
        classes = library_of("evader")
        # A model that always answers ORBIT: zero weights, one hot bias.
        bias = np.array([1.0 if c == "ORBIT" else 0.0 for c in classes])
        model = SoftmaxModel(
            classes, np.zeros(len(FEATURES)), np.ones(len(FEATURES)),
            np.zeros((len(FEATURES), len(classes))), bias,
        )
        rec = TacticRecogniser(CFG, "evader", model=model)
        assert rec.classifier == "fitted"
        for point in line((1.0, 1.0), (1.0, 1.0), 40):
            rec.observe(np.array([-4.0, -4.0, 0.0]), np.array(point), [OBSTACLE])
        assert rec.label == "ORBIT"  # the rule classifier would say FLEE


# ------------------------------------------------------------------- history
class TestHistory:
    def test_runs_start_at_the_first_step_and_merge_equal_labels(self):
        # 9 steps blind (UNKNOWN), then a visible orbit.
        rec = feed(
            [(1.15, 0.0)] * 9 + arc(1.15, 0.08, 40), visible=lambda i: i >= 9
        )
        labels = [run[0] for run in rec.history]
        assert labels[0] == UNKNOWN
        assert labels.count(UNKNOWN) == 1  # one merged run, not nine
        assert rec.history[0][1] == 0
        # Runs tile the steps exactly, end exclusive, with no gap or overlap.
        assert rec.history[-1][2] == 49
        for (_, _, end), (_, start, _) in zip(rec.history, rec.history[1:]):
            assert end == start
        assert all(end > start for _, start, end in rec.history)

    def test_reset_clears_the_history_and_the_speed_reference(self):
        rec = feed(arc(1.15, 0.08, 40))
        assert rec.v_ref > 0.0
        rec.reset()
        assert (rec.history, rec.label, rec.v_ref) == ([], UNKNOWN, 0.0)
        assert rec.feature_vector() is None


# -------------------------------------------------------------------- labels
def test_tactic_family_strips_the_obstacle_index():
    assert tactic_family("ORBIT(1)") == "ORBIT"
    assert tactic_family("COUNTER-ORBIT(0)") == "COUNTER-ORBIT"
    assert tactic_family("HOLD-HIDDEN") == "HOLD-HIDDEN"


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        TacticRecogniser(CFG, "referee")
