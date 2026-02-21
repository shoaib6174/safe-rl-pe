"""Unit tests for the reward computation."""

import numpy as np
import pytest

from envs.rewards import RewardComputer


class TestRewardComputer:
    """Tests for the reward function."""

    def make_reward(self, **kwargs):
        defaults = dict(
            distance_scale=1.0,
            capture_bonus=100.0,
            timeout_penalty=-50.0,
            d_max=28.28,
        )
        defaults.update(kwargs)
        return RewardComputer(**defaults)

    def test_zero_sum(self):
        """r_P + r_E should always be 0."""
        rc = self.make_reward()

        # Normal step
        r_p, r_e = rc.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

        # Capture
        r_p, r_e = rc.compute(d_curr=0.3, d_prev=0.6, captured=True, timed_out=False)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

        # Timeout
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=10.0, captured=False, timed_out=True)
        assert r_p + r_e == pytest.approx(0.0, abs=1e-10)

    def test_closing_positive(self):
        """Pursuer gets positive reward when closing distance."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p > 0
        assert r_e < 0

    def test_retreating_negative(self):
        """Pursuer gets negative reward when distance increases."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=9.0, captured=False, timed_out=False)
        assert r_p < 0
        assert r_e > 0

    def test_no_change_zero(self):
        """No distance change, no terminal -> zero reward."""
        rc = self.make_reward()
        r_p, r_e = rc.compute(d_curr=5.0, d_prev=5.0, captured=False, timed_out=False)
        assert r_p == pytest.approx(0.0, abs=1e-10)
        assert r_e == pytest.approx(0.0, abs=1e-10)

    def test_capture_bonus(self):
        """Capture should add large positive reward for pursuer."""
        rc = self.make_reward(capture_bonus=100.0)
        r_p, r_e = rc.compute(d_curr=0.3, d_prev=0.5, captured=True, timed_out=False)
        assert r_p > 90  # capture bonus dominates
        assert r_e < -90

    def test_timeout_penalty(self):
        """Timeout should penalize pursuer."""
        rc = self.make_reward(timeout_penalty=-50.0)
        r_p, r_e = rc.compute(d_curr=10.0, d_prev=10.0, captured=False, timed_out=True)
        assert r_p < 0
        assert r_e > 0

    def test_distance_scale(self):
        """Distance scale should proportionally affect shaping reward."""
        rc1 = self.make_reward(distance_scale=1.0)
        rc2 = self.make_reward(distance_scale=2.0)

        r1, _ = rc1.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)
        r2, _ = rc2.compute(d_curr=8.0, d_prev=9.0, captured=False, timed_out=False)

        assert r2 == pytest.approx(2 * r1, abs=1e-10)
