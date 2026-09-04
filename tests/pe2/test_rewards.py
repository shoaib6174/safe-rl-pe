"""Tests for pe2.rewards — Tier 0 dense training rewards (T0.2)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.rewards import DenseRewardWrapper, SearchStalenessTracker, signal_nondegeneracy
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

    def test_nondegeneracy_on_a_real_rollout(self):
        # T0.2 plateau check, run for real: the FOV-limited pursuer rolled
        # against the greedy evader must see the opponent on SOME but not
        # ALL steps, so the visibility term is a live signal.
        from pe2.scripted import FovLimitedPursuer

        cfg = EnvConfig()
        w = DenseRewardWrapper(
            SingleAgentWrapper(cfg, "pursuer", GreedyEvader), w_vis=0.1
        )
        rewards = []
        for ep in range(3):
            _obs, _ = w.reset(seed=790100 + ep)
            teacher = FovLimitedPursuer()
            teacher.reset()
            while True:
                v, om = teacher.act(w.env._last_joint_obs["pursuer"], w.env.env)
                a = np.array(
                    [2.0 * v / cfg.v_max - 1.0, om / cfg.omega_max], dtype=np.float32
                )
                _obs, r, term, trunc, _ = w.step(a)
                rewards.append(r)
                if term or trunc:
                    break
        nd = signal_nondegeneracy(rewards)
        assert 0.05 < nd["frac_positive"] < 0.95
        assert nd["std"] > 0.0
