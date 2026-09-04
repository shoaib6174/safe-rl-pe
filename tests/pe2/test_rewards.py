"""Tests for pe2.rewards — Tier 0 dense training rewards (T0.2)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.rewards import SearchStalenessTracker

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
