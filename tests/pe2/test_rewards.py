"""Tests for pe2.rewards — Tier 0 dense training rewards (T0.2)."""

import numpy as np
import pytest

from pe2.config import EnvConfig, SensingConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.rewards import (
    DenseRewardWrapper,
    SearchStalenessTracker,
    nondegeneracy_report,
    signal_nondegeneracy,
    visibility_monotone_report,
    visibility_term,
)
from pe2.scripted import GreedyEvader, GreedyPursuer

CFG = EnvConfig()


class TestSearchStalenessTracker:
    def test_first_observation_is_fully_stale(self):
        # After reset every cell has last_observed = -t_stale, so at step 0
        # steps_since = t_stale -> staleness clipped to 1.0 for every
        # observed cell -> mean 1.0.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        r = tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        assert r == pytest.approx(1.0)

    def test_immediate_reobservation_is_fresh(self):
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        # Same pose, next step: every observed cell was seen 1 step ago ->
        # staleness 1/50 = 0.02 for all -> mean 0.02.
        r = tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=1)
        assert r == pytest.approx(1 / 50)

    def test_cells_behind_the_robot_are_not_observed(self):
        # Heading +x at the origin with a 90 deg cone: the cell centred at
        # (-4.5, 0) is behind the robot and must stay stale; the cell at
        # (+4.5, 0) is ahead (within 8 m range) and must be marked observed.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), [], current_step=0)
        idx_behind = tr.cell_index(-4.5, 0.0)
        idx_ahead = tr.cell_index(4.5, 0.0)
        assert tr.last_observed[idx_behind] == -50
        assert tr.last_observed[idx_ahead] == 0

    def test_occluded_cell_is_not_observed(self):
        # Obstacle of radius 1.0 at (2, 0) blocks LOS from the origin to the
        # cell centred at (4.5, 0) (segment passes through the disk).
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        obstacles = [{"x": 2.0, "y": 0.0, "radius": 1.0}]
        tr.observe_and_reward(np.array([0.0, 0.0, 0.0]), obstacles, current_step=0)
        assert tr.last_observed[tr.cell_index(4.5, 0.0)] == -50

    def test_nothing_visible_returns_zero(self):
        # Robot facing +x from the east wall: the cone leaves the arena, no
        # cell centre is in range -> 0.0 and no cell marked.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        r = tr.observe_and_reward(np.array([4.9, 0.0, 0.0]), [], current_step=0)
        assert r == 0.0
        assert (tr.last_observed == -50).all()

    def test_reward_varies_along_a_trajectory(self):
        # Non-degeneracy (derisk v2 T0.2 plateau check): driving straight
        # across the arena must produce a staleness signal with nonzero
        # spread, i.e. something to learn from.
        tr = SearchStalenessTracker(CFG, grid_size=10, t_stale=50)
        tr.reset()
        rs = [
            tr.observe_and_reward(
                np.array([-4.0 + 0.1 * k, 0.0, 0.0]), [], current_step=k
            )
            for k in range(60)
        ]
        assert np.std(rs) > 0.0
        assert 0.0 < np.mean(rs) < 1.0


def _place(w, pursuer, evader):
    w.env.pursuer_pose = np.array(pursuer, dtype=float)
    w.env.evader_pose = np.array(evader, dtype=float)
    w._last_joint_obs = w.env._get_obs()


class TestDenseRewardWrapper:
    def test_visibility_term_hand_computed(self):
        # Pursuer at origin heading +x, evader 2 m ahead: visible -> +0.1.
        # Evader behind: not visible -> +0.0. Terminal env reward is 0 mid-episode.
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([-1.0, 0.0]))  # v=0: nothing moves but the evader
        assert r == pytest.approx(0.1, abs=1e-9)
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (-3.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([-1.0, 0.0]))
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_search_term_uses_tracker(self):
        # First step after reset: every visible cell is fully stale -> mean
        # staleness 1.0 -> reward w_search * 1.0.
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_search=1e-4
        )
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (-4.0, -4.0, 0.0))  # evader behind: no vis term
        _, r, *_ = w.step(np.array([-1.0, 0.0]))
        assert r == pytest.approx(1e-4, abs=1e-12)

    def test_evader_visibility_requires_pbrs(self):
        cfg = EnvConfig()
        with pytest.raises(ValueError, match="PBRS"):
            DenseRewardWrapper(
                SingleAgentWrapper(cfg, "evader", GreedyPursuer), w_evis=0.1
            )

    def test_evader_visibility_sign_and_pbrs_hand_computed(self):
        # One obstacle r=1.0 at (3,0); evader at origin heading +x, pursuer
        # far behind it (not visible): hidden -> +w_evis. PBRS with gamma=1:
        # Phi(s) = -(3 - 1) = -2 before; the evader moves v_max*dt = 1.2*0.05
        # = 0.06 toward the obstacle -> Phi(s') = -(2.94 - 1) = -1.94;
        # F = -1.94 - (-2) = +0.06. Total = 0.1 + 0.06 = 0.16.
        cfg = EnvConfig(n_obstacles=1)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "evader", GreedyPursuer),
            w_evis=0.1,
            w_pbrs=1.0,
            gamma=1.0,
        )
        w.reset(seed=0)
        w.env.env.obstacles = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
        _place(w.env, (-4.5, -4.5, 0.0), (0.0, 0.0, 0.0))
        _, r, *_ = w.step(np.array([1.0, 0.0]))  # full speed forward
        assert r == pytest.approx(0.16, abs=1e-6)

    def test_terminal_outcome_unchanged(self):
        # Adjacent agents: capture on the first step; the env's terminal +1
        # must dominate and be present unchanged (dense terms are O(0.1)).
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (0.2, 0.0, 0.0))
        _, r, terminated, *_ = w.step(np.array([1.0, 0.0]))
        assert terminated
        assert r >= 1.0

    def test_dense_reward_key_isolates_the_dense_term(self):
        # info["dense_reward"] must carry the dense term ALONE, distinct
        # from the combined reward once a terminal outcome mixes in — this
        # is what signal_nondegeneracy must read instead of the raw reward,
        # or a rollout with a nonzero capture rate would contaminate it.
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        _, r, _, _, info = w.step(np.array([-1.0, 0.0]))  # visible, no terminal: agree
        assert r == pytest.approx(0.1, abs=1e-9)
        assert info["dense_reward"] == pytest.approx(r, abs=1e-9)

        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (0.2, 0.0, 0.0))  # adjacent: captures this step
        _, r, terminated, _, info = w.step(np.array([1.0, 0.0]))
        assert terminated
        assert r == pytest.approx(1.0 + info["dense_reward"], abs=1e-9)
        assert info["dense_reward"] != pytest.approx(r, abs=1e-9)

    def test_nondegeneracy_on_a_real_rollout(self):
        # T0.2 plateau check, run for real: the FOV-limited pursuer rolled
        # against the greedy evader must see the opponent on SOME but not
        # ALL steps, so the visibility term is a live signal. Reads the
        # isolated dense term from info, not the combined reward, so a rare
        # terminal +/-1 cannot contaminate the check.
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig()
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        dense_rewards = []
        for ep in range(3):
            _obs, _ = w.reset(seed=790100 + ep)
            teacher = FovLimitedPursuer()
            teacher.reset()
            while True:
                v, om = teacher.act(w.env._last_joint_obs["pursuer"], w.env.env)
                a = np.array(
                    [2.0 * v / cfg.v_max - 1.0, om / cfg.omega_max], dtype=np.float32
                )
                _obs, _r, term, trunc, info = w.step(a)
                dense_rewards.append(info["dense_reward"])
                if term or trunc:
                    break
        nd = signal_nondegeneracy(dense_rewards)
        assert 0.05 < nd["frac_positive"] < 0.95
        assert nd["std"] > 0.0


class TestPerTermNondegeneracy:
    """The T0.2 plateau check is a PER-TERM question. Asked of the sum it is
    not interpretable — terms live on different scales (w_vis=0.1 against
    w_search=1e-4), and the sum of the shipped P-A configuration reads
    frac_positive = 0.97 purely because the staleness term is non-negative."""

    def test_dense_terms_are_reported_separately_and_sum_to_the_dense_reward(self):
        cfg = EnvConfig(n_obstacles=0)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1, w_search=1e-4
        )
        w.reset(seed=0)
        _place(w.env, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))  # evader visible ahead
        _, _r, _, _, info = w.step(np.array([-1.0, 0.0]))
        assert set(info["dense_terms"]) == {"vis", "search"}
        # visible -> 0.1; first step after reset -> every visible cell fully
        # stale -> 1e-4 * 1.0. Sum is the reported dense term.
        assert info["dense_terms"]["vis"] == pytest.approx(0.1, abs=1e-9)
        assert info["dense_terms"]["search"] == pytest.approx(1e-4, abs=1e-12)
        assert info["dense_reward"] == pytest.approx(
            sum(info["dense_terms"].values()), abs=1e-12
        )

    def test_an_always_on_indicator_is_degenerate(self):
        # 100 steps, the term on at every one: frac_positive 1.0, std 0.
        # An indicator that never turns off carries no information. Asked of
        # `evis`, which is the term that still carries the band (A7 moved
        # `vis` to the analytic monotone criterion; the band is unchanged).
        rep = nondegeneracy_report({"evis": [0.1] * 100})
        assert rep["terms"]["evis"]["frac_positive"] == 1.0
        assert rep["terms"]["evis"]["passes"] is False
        assert rep["all_pass"] is False

    def test_a_live_indicator_passes(self):
        # on for 7 of 100 steps -> frac_positive 0.07, inside (0.05, 0.95).
        rep = nondegeneracy_report({"evis": [0.1] * 7 + [0.0] * 93})
        assert rep["terms"]["evis"]["frac_positive"] == pytest.approx(0.07)
        assert rep["terms"]["evis"]["criterion"] == "positive_fraction_band"
        assert rep["terms"]["evis"]["stats_role"] == "criterion"
        assert rep["terms"]["evis"]["passes"] is True
        assert rep["all_pass"] is True

    def test_a_varying_non_negative_continuous_term_passes(self):
        # k/100 for k = 1..100: frac_positive is 1.0 BY CONSTRUCTION (the
        # staleness term is non-negative), which fails the indicator band —
        # yet every step carries a different value, so there is a gradient
        # to learn from. This is the case the band would wrongly reject.
        vals = [k / 100 for k in range(1, 101)]
        rep = nondegeneracy_report({"search": vals})
        assert rep["terms"]["search"]["frac_positive"] == 1.0
        assert rep["terms"]["search"]["frac_modal"] == pytest.approx(0.01)
        assert rep["terms"]["search"]["passes"] is True

    def test_a_frozen_continuous_term_is_degenerate(self):
        # one value on every step: std 0, frac_modal 1.0 — the flat region
        # the check exists to reject.
        rep = nondegeneracy_report({"search": [5e-5] * 100})
        assert rep["terms"]["search"]["std"] == 0.0
        assert rep["terms"]["search"]["frac_modal"] == 1.0
        assert rep["terms"]["search"]["passes"] is False

    def test_a_near_frozen_continuous_term_is_degenerate(self):
        # 96 steps on one value, 4 elsewhere: std > 0, so std alone would
        # pass it; frac_modal 0.96 >= 0.95 rejects it.
        vals = [5e-5] * 96 + [1e-5, 2e-5, 3e-5, 4e-5]
        rep = nondegeneracy_report({"search": vals})
        assert rep["terms"]["search"]["std"] > 0.0
        assert rep["terms"]["search"]["frac_modal"] == pytest.approx(0.96)
        assert rep["terms"]["search"]["passes"] is False

    def test_an_unknown_term_name_is_refused(self):
        # A new term must declare which check applies to it; silently
        # defaulting would apply the wrong criterion to it.
        with pytest.raises(ValueError, match="kind"):
            nondegeneracy_report({"mystery": [1.0, 2.0]})

    def test_the_shipped_p_a_configuration_on_a_real_rollout(self):
        # The gate applied to what actually trains (w_vis=0.1, w_search=1e-4).
        # Measured on the gated clone over 6211 steps: the SUM reads
        # frac_positive 0.971, outside the band, entirely because staleness
        # is non-negative; per term, vis is on 7% of steps and staleness
        # takes ~700 distinct values. Rolled here with the scripted teacher
        # so the test needs no saved model. `vis` is now judged by the
        # analytic monotone criterion (A7), so the rollout supplies only its
        # recorded observations; `search` is still judged by this rollout.
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig()
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1, w_search=1e-4
        )
        terms: dict[str, list[float]] = {}
        for ep in range(2):
            _obs, _ = w.reset(seed=790500 + ep)
            teacher = FovLimitedPursuer()
            teacher.reset()
            while True:
                v, om = teacher.act(w.env._last_joint_obs["pursuer"], w.env.env)
                a = np.array(
                    [2.0 * v / cfg.v_max - 1.0, om / cfg.omega_max], dtype=np.float32
                )
                _obs, _r, term, trunc, info = w.step(a)
                for k, val in info["dense_terms"].items():
                    terms.setdefault(k, []).append(val)
                if term or trunc:
                    break
        rep = nondegeneracy_report(terms, cfg=cfg, w_vis=0.1)
        assert set(rep["terms"]) == {"vis", "search"}
        assert rep["terms"]["vis"]["stats_role"] == "observation"
        assert rep["terms"]["search"]["stats_role"] == "criterion"
        assert rep["all_pass"] is True, rep
        # not vacuous: the sum would fail the indicator band it is not owed.
        summed = [sum(v) for v in zip(*terms.values(), strict=True)]
        assert signal_nondegeneracy(summed)["frac_positive"] > 0.95


# ---------------------------------------------------------------------------
# T0.2 visibility criterion (amendment A7): the spec's own check for the
# pursuer visibility term — "reward-vs-bearing curve must be monotone into
# the cone" — replaces the 0.05-0.95 positive-fraction band.
#
# Hand-computed geometry, EnvConfig defaults throughout:
#   fov_angle  = pi/2 = 1.5707963268 rad, so the cone HALF-angle is
#                pi/4 = 0.7853981634 rad = 45 deg;
#   fov_range  = 8.0 m, so the default swept range is 8.0 / 2 = 4.0 m,
#                strictly inside sensing range;
#   n_bearings = 361 over [-pi, pi] -> grid step 2*pi/360 = pi/180
#                = 0.0174532925 rad = 1 deg, and 45 deg = 45 * (pi/180) is
#                exactly a grid point (index 225 counting from -pi).
# The sweep is run with NO obstacles, so LOS is clear at every bearing and
# the only thing that moves the reward is the cone.
# ---------------------------------------------------------------------------

HALF_ANGLE = np.pi / 4  # 0.7853981633974483 rad = 45 deg
GRID_STEP = np.pi / 180  # 0.017453292519943295 rad = 1 deg


def _blind_dead_ahead(w_vis, obs):
    """NEGATIVE CONTROL: pays inside the cone EXCEPT in a blind spot of
    |bearing| < pi/8 = 22.5 deg around the heading. Turning toward the
    opponent from 30 deg to 10 deg then LOSES the reward — exactly the
    degeneracy the monotone criterion exists to catch."""
    if obs[6] <= 0.5:
        return 0.0
    bearing = float(np.arctan2(obs[4], obs[3]))  # body frame, from obs itself
    return 0.0 if abs(bearing) < np.pi / 8 else w_vis


def _always_on(w_vis, obs):
    """NEGATIVE CONTROL: the same value at every bearing. Non-decreasing by
    construction, so monotonicity alone would pass it; it carries no
    information at all."""
    return w_vis


def _narrow_by_22_5_degrees(w_vis, obs):
    """NEGATIVE CONTROL: monotone and varying, but the step sits at
    |bearing| = pi/8 = 22.5 deg instead of the configured half-angle of
    45 deg — the reward and the sensing configuration disagree."""
    if obs[6] <= 0.5:
        return 0.0
    bearing = float(np.arctan2(obs[4], obs[3]))
    return w_vis if abs(bearing) <= np.pi / 8 else 0.0


def _reward_at(rep, degrees):
    b = np.asarray(rep["bearings"])
    r = np.asarray(rep["rewards"])
    return float(r[int(np.argmin(np.abs(b - np.deg2rad(degrees))))])


class TestVisibilityTerm:
    def test_the_term_is_the_indicator_the_wrapper_pays(self):
        obs = np.zeros(11)
        obs[6] = 1.0
        assert visibility_term(0.1, obs) == pytest.approx(0.1)
        obs[6] = 0.0
        assert visibility_term(0.1, obs) == pytest.approx(0.0)


class TestVisibilityMonotoneCheck:
    def test_the_shipped_visibility_term_passes(self):
        rep = visibility_monotone_report(CFG, w_vis=0.1)
        assert rep["monotone_into_cone"] is True
        assert rep["varies"] is True
        assert rep["transition_bearing"] == pytest.approx(HALF_ANGLE, abs=GRID_STEP)
        assert rep["expected_transition_bearing"] == pytest.approx(HALF_ANGLE)
        assert rep["transition_matches"] is True
        assert rep["passes"] is True

    def test_the_swept_curve_is_the_step_the_configuration_specifies(self):
        # 0 deg (dead ahead) and +-44 deg are inside the 45 deg half-angle
        # -> +w_vis = 0.1; 46 deg, 90 deg and 180 deg are outside -> 0.0.
        rep = visibility_monotone_report(CFG, w_vis=0.1)
        assert _reward_at(rep, 0) == pytest.approx(0.1)
        assert _reward_at(rep, 44) == pytest.approx(0.1)
        assert _reward_at(rep, -44) == pytest.approx(0.1)
        assert _reward_at(rep, 46) == pytest.approx(0.0)
        assert _reward_at(rep, -46) == pytest.approx(0.0)
        assert _reward_at(rep, 90) == pytest.approx(0.0)
        assert _reward_at(rep, 180) == pytest.approx(0.0)

    def test_the_evidence_is_a_full_sweep_at_a_range_inside_sensing_range(self):
        rep = visibility_monotone_report(CFG, w_vis=0.1)
        assert len(rep["bearings"]) == len(rep["rewards"]) == 361
        assert rep["bearings"][0] == pytest.approx(-np.pi)
        assert rep["bearings"][-1] == pytest.approx(np.pi)
        assert rep["grid_step"] == pytest.approx(GRID_STEP)
        assert rep["range_m"] == pytest.approx(4.0)  # 8.0 / 2, inside range
        assert rep["obstacles"] == 0

    def test_the_check_is_deterministic(self):
        # No sampling: two calls must agree exactly, which is the whole
        # reason the runner does not bootstrap this verdict.
        a = visibility_monotone_report(CFG, w_vis=0.1)
        b = visibility_monotone_report(CFG, w_vis=0.1)
        assert a == b

    def test_a_narrower_cone_moves_the_transition(self):
        # fov_angle = pi/3 (60 deg) -> half-angle 30 deg = 0.5235987756 rad.
        # The check reads the transition off the reward and compares it with
        # THIS configuration, so it tracks the config rather than a constant.
        cfg = EnvConfig(sensing=SensingConfig(fov_angle=np.pi / 3))
        rep = visibility_monotone_report(cfg, w_vis=0.1)
        assert rep["expected_transition_bearing"] == pytest.approx(np.pi / 6)
        assert rep["transition_bearing"] == pytest.approx(np.pi / 6, abs=GRID_STEP)
        assert rep["passes"] is True

    # ---- NEGATIVE CONTROLS: the criterion must be able to fail ----

    def test_a_reward_that_dips_dead_ahead_is_rejected(self):
        # THE CASE THE CRITERION EXISTS FOR. Blind spot inside |b| < 22.5 deg:
        # at 30 deg the reward is 0.1 and at 10 deg it is 0.0, so turning
        # toward the opponent LOSES reward. Varies, and its outermost
        # positive bearing is still 45 deg, so only monotonicity catches it.
        rep = visibility_monotone_report(CFG, w_vis=0.1, term_fn=_blind_dead_ahead)
        assert _reward_at(rep, 30) == pytest.approx(0.1)
        assert _reward_at(rep, 10) == pytest.approx(0.0)
        assert rep["varies"] is True
        assert rep["transition_matches"] is True
        assert rep["monotone_into_cone"] is False
        assert rep["passes"] is False

    def test_a_constant_reward_is_rejected(self):
        # Non-decreasing by construction (every difference is 0), so the
        # monotonicity test alone would pass it; `varies` rejects it.
        rep = visibility_monotone_report(CFG, w_vis=0.1, term_fn=_always_on)
        assert rep["monotone_into_cone"] is True
        assert rep["varies"] is False
        assert rep["passes"] is False

    def test_a_transition_at_the_wrong_angle_is_rejected(self):
        # Monotone and varying, but the step is at 22.5 deg while the config
        # says 45 deg — a 22.5 deg disagreement against a 1 deg tolerance.
        rep = visibility_monotone_report(
            CFG, w_vis=0.1, term_fn=_narrow_by_22_5_degrees
        )
        assert rep["monotone_into_cone"] is True
        assert rep["varies"] is True
        assert rep["transition_bearing"] == pytest.approx(np.pi / 8, abs=GRID_STEP)
        assert rep["transition_matches"] is False
        assert rep["passes"] is False

    def test_the_check_scores_the_reward_the_wrapper_actually_pays(self):
        # The criterion must not be able to pass while the trained reward
        # does something else. Same two bearings, two routes: the analytic
        # sweep, and a real DenseRewardWrapper step with the evader placed
        # at 4 m and 30 deg (inside the 45 deg half-angle -> 0.1) and at
        # 4 m and 75 deg (outside -> 0.0). The evader moves at most
        # 1.2 * 0.05 = 0.06 m during the step, i.e. <= 0.86 deg of bearing
        # at 4 m, so neither placement can cross the 45 deg edge.
        cfg = EnvConfig(n_obstacles=0)
        rep = visibility_monotone_report(cfg, w_vis=0.1)
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        for degrees in (30.0, 75.0):
            rad = np.deg2rad(degrees)
            w.reset(seed=0)
            _place(
                w.env,
                (0.0, 0.0, 0.0),
                (4.0 * np.cos(rad), 4.0 * np.sin(rad), 0.0),
            )
            _, _r, _, _, info = w.step(np.array([-1.0, 0.0]))  # v = 0
            assert info["dense_terms"]["vis"] == pytest.approx(
                _reward_at(rep, degrees), abs=1e-12
            )


class TestVisTermDispatch:
    """A7: `vis` is dispatched to the analytic criterion; every other term's
    criterion is untouched."""

    def test_the_vis_term_is_judged_by_geometry_not_by_frequency(self):
        # 3 positives in 100 steps -> frac_positive 0.03, BELOW the old 0.05
        # band edge, and the term still passes: the criterion is now the
        # reward's shape, and the frequency is only recorded.
        rep = nondegeneracy_report({"vis": [0.1] * 3 + [0.0] * 97}, cfg=CFG, w_vis=0.1)
        vis = rep["terms"]["vis"]
        assert vis["kind"] == "monotone"
        assert vis["criterion"] == "monotone_into_cone"
        assert vis["stats_role"] == "observation"
        assert vis["frac_positive"] == pytest.approx(0.03)
        assert vis["monotone"]["passes"] is True
        assert vis["passes"] is True
        assert rep["all_pass"] is True

    def test_a_degenerate_vis_reward_fails_the_report(self, monkeypatch):
        # And the report says False when the geometry is wrong: the same
        # rollout values, with the analytic check fed the blind-spot reward.
        # The production signature takes no injection point, so the check is
        # substituted here rather than made configurable for a test.
        import pe2.rewards as rewards_mod

        def _degenerate(cfg, *, w_vis):
            return visibility_monotone_report(
                cfg, w_vis=w_vis, term_fn=_blind_dead_ahead
            )

        monkeypatch.setattr(rewards_mod, "visibility_monotone_report", _degenerate)
        rep = nondegeneracy_report({"vis": [0.1] * 3 + [0.0] * 97}, cfg=CFG, w_vis=0.1)
        assert rep["terms"]["vis"]["passes"] is False
        assert rep["all_pass"] is False

    def test_the_monotone_criterion_refuses_to_run_without_the_configuration(self):
        # A frequency check needs only numbers; a geometric one needs the
        # configuration. Silently falling back to the band is the failure
        # this guards.
        with pytest.raises(ValueError, match="cfg"):
            nondegeneracy_report({"vis": [0.1, 0.0]})

    def test_the_search_term_keeps_its_non_constancy_check(self):
        vals = [k / 100 for k in range(1, 101)]
        rep = nondegeneracy_report({"search": vals})
        assert rep["terms"]["search"]["criterion"] == "non_constancy"
        assert rep["terms"]["search"]["stats_role"] == "criterion"
        assert rep["terms"]["search"]["passes"] is True


class TestNegatedRewardWrapper:
    """T1.0 spec §3's negative control: the pursuer is paid for NOT capturing.

    Horizon-independent by construction, which is why it is the control for a
    round whose hypothesis is about the horizon — the usual naive-fine-tune
    control might legitimately SUCCEED here if the hypothesis is true, and a
    control the hypothesis predicts may pass is no control at all.
    """

    def _wrapped(self, max_steps=40):
        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=max_steps)
        return NegatedRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader))

    def test_reward_sign_is_flipped_and_the_true_value_is_kept(self):
        env = self._wrapped()
        env.reset(seed=970_000)
        seen = []
        while True:
            _obs, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            seen.append((reward, info["true_reward"]))
            if terminated or truncated:
                break
        assert all(r == -t for r, t in seen)
        # A terminal-only env pays exactly once, at the end.
        assert seen[-1][1] != 0.0
        assert all(t == 0.0 for _, t in seen[:-1])

    def test_everything_but_the_reward_passes_through(self):
        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        cfg = EnvConfig(max_steps=40)
        plain = SingleAgentWrapper(cfg, "pursuer", GreedyEvader)
        wrapped = NegatedRewardWrapper(SingleAgentWrapper(cfg, "pursuer", GreedyEvader))
        a = np.array([0.6, -0.2], dtype=np.float32)
        o1, _ = plain.reset(seed=970_001)
        o2, _ = wrapped.reset(seed=970_001)
        assert np.array_equal(o1, o2)
        for _ in range(20):
            s1 = plain.step(a)
            s2 = wrapped.step(a)
            assert np.array_equal(s1[0], s2[0])  # obs
            assert s1[2] == s2[2] and s1[3] == s2[3]  # terminated / truncated
            assert s1[1] == -s2[1]  # reward, and only the reward
            if s1[2] or s1[3]:
                break

    def test_it_is_a_training_wrapper_only(self):
        """The eval path builds its own unshaped wrapper; a negated reward must
        never reach a probe or the final evaluation."""
        import inspect

        from pe2 import exploitability

        assert "Negated" not in inspect.getsource(exploitability.evaluate_sb3_policy)
        assert "Negated" not in inspect.getsource(exploitability.CheckpointEvalProbe)


class TestZeroRewardWrapper:
    def test_every_reward_is_zero_and_the_true_reward_is_kept(self):
        from pe2.rewards import ZeroRewardWrapper

        w = ZeroRewardWrapper(SingleAgentWrapper(CFG, "pursuer", GreedyEvader))
        w.reset(seed=900_800)
        seen_nonzero_true = False
        while True:
            _, r, term, trunc, info = w.step(np.array([1.0, 0.0], dtype=np.float32))
            assert r == 0.0
            assert "true_reward" in info
            seen_nonzero_true |= info["true_reward"] != 0.0
            if term or trunc:
                break
        assert (
            seen_nonzero_true
        )  # the terminal +/-1 was paid by the env, hidden by the wrapper
