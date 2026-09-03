"""Tests for pe2.belief — Step 4 last-seen dead-reckoning baseline."""

from itertools import pairwise

import numpy as np
import pytest

from pe2.belief import SIGMA_GROWTH, SIGMA_OBS, VELOCITY_DECAY, LastSeenBelief
from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv

CFG = EnvConfig()


def make_obs(own=(0.0, 0.0, 0.0), opp_world=None, visible=True):
    """Hand-encode an 11-D observation (no obstacles)."""
    obs = np.zeros(7 + 2 * CFG.n_obstacles, dtype=np.float32)
    half = CFG.arena_size / 2.0
    obs[0], obs[1], obs[2] = own[0] / half, own[1] / half, own[2] / np.pi
    if visible and opp_world is not None:
        dx = opp_world[0] - own[0]
        dy = opp_world[1] - own[1]
        c, s = np.cos(own[2]), np.sin(own[2])
        obs[3] = (c * dx + s * dy) / CFG.arena_size
        obs[4] = (-s * dx + c * dy) / CFG.arena_size
        obs[6] = 1.0
    return obs


class TestVisibleTracking:
    def test_mean_equals_observed_position(self):
        b = LastSeenBelief(CFG)
        b.update(make_obs(opp_world=(3.0, 1.0)))
        assert b.seen
        assert b.mean == pytest.approx([3.0, 1.0])
        assert b.sigma == SIGMA_OBS
        assert b.t_since_seen == 0

    def test_velocity_estimate_converges_to_true(self):
        """Constant-velocity target: low-passed finite difference -> true v."""
        b = LastSeenBelief(CFG)
        v_true = np.array([1.2, 0.0])  # evader_v_max straight +x
        for k in range(30):
            pos = np.array([0.06, 0.0]) * k  # 1.2 m/s at dt=0.05
            b.update(make_obs(opp_world=pos))
        assert b.velocity == pytest.approx(v_true, abs=1e-6)

    def test_in_fov_error_is_zero(self):
        """Noiseless sim: in-FOV belief error must be ~0 (RMSE < 2m trivially)."""
        b = LastSeenBelief(CFG)
        for k in range(10):
            pos = (1.0 + 0.06 * k, 2.0)
            b.update(make_obs(opp_world=pos))
            err = np.hypot(*(b.mean - np.array(pos)))
            assert err < 1e-6


class TestOccludedDeadReckoning:
    def _seen_tracker(self):
        b = LastSeenBelief(CFG)
        # target moving +x at 1.0 m/s, seen for several steps
        for k in range(20):
            b.update(make_obs(opp_world=(0.05 * k, 0.0)))
        return b

    def test_propagates_along_last_velocity(self):
        b = self._seen_tracker()
        last = b.mean.copy()
        b.update(make_obs(visible=False))
        assert b.t_since_seen == 1
        assert b.just_lost
        # moved ~velocity * dt along +x
        assert b.mean[0] > last[0]
        assert b.mean[1] == pytest.approx(0.0, abs=1e-9)

    def test_sigma_grows_linearly(self):
        b = self._seen_tracker()
        sigmas = []
        for _ in range(10):
            b.update(make_obs(visible=False))
            sigmas.append(b.sigma)
        assert all(s2 > s1 for s1, s2 in pairwise(sigmas))
        assert sigmas[-1] == pytest.approx(SIGMA_OBS + SIGMA_GROWTH * 10 * CFG.dt)

    def test_velocity_decays(self):
        b = self._seen_tracker()
        v0 = np.hypot(*b.velocity)
        for _ in range(20):
            b.update(make_obs(visible=False))
        v1 = np.hypot(*b.velocity)
        assert v1 == pytest.approx(v0 * VELOCITY_DECAY**20, rel=1e-6)

    def test_no_snap_to_truth_for_occluded_moving_target(self):
        """Plan Step 4 verify: belief error > 0 for an occluded MOVING target.
        The target turns while occluded; the belief must NOT follow it."""
        b = self._seen_tracker()  # believes target moves +x at ~1 m/s
        true_pos = np.array([0.05 * 19, 0.0])
        for _ in range(40):  # 2s occluded; truth turns north instead
            true_pos = true_pos + np.array([0.0, 1.2 * CFG.dt])
            b.update(make_obs(visible=False))
        err = np.hypot(*(b.mean - true_pos))
        assert err > 1.0  # truth is ~1m north of the dead-reckoned ghost

    def test_belief_is_pure_function_of_obs(self):
        """No-leak by construction: identical obs streams -> identical beliefs,
        regardless of what the (unobserved) truth was doing."""
        obs_seq = [make_obs(opp_world=(0.05 * k, 0.0)) for k in range(5)]
        obs_seq += [make_obs(visible=False)] * 10
        b1, b2 = LastSeenBelief(CFG), LastSeenBelief(CFG)
        for obs in obs_seq:
            b1.update(obs)
            b2.update(obs.copy())
        assert b1.mean == pytest.approx(b2.mean)
        assert b1.velocity == pytest.approx(b2.velocity)
        assert b1.sigma == b2.sigma


class TestNeverSeen:
    def test_stays_unseen_and_excludable(self):
        b = LastSeenBelief(CFG)
        for _ in range(5):
            b.update(make_obs(visible=False))
        assert not b.seen
        assert not b.just_lost
        assert b.sigma == float("inf")
        assert b.t_since_seen is None


class TestReacquisition:
    def test_snaps_to_observation_within_one_step(self):
        """Plan gate: on re-acquisition belief converges within ~1s (here:
        1 step — observations are noiseless)."""
        b = LastSeenBelief(CFG)
        for k in range(10):
            b.update(make_obs(opp_world=(0.05 * k, 0.0)))
        for _ in range(30):
            b.update(make_obs(visible=False))
        drifted = b.mean.copy()
        # target reappears somewhere far from the drifted ghost
        b.update(make_obs(opp_world=(2.0, 1.0)))
        assert b.mean == pytest.approx([2.0, 1.0])
        assert np.hypot(*(b.mean - drifted)) > 0.5  # it really had drifted
        assert b.sigma == SIGMA_OBS
        assert b.t_since_seen == 0
        assert not b.just_lost

    def test_velocity_kept_not_recomputed_over_gap(self):
        """Re-acquire after gap: velocity must NOT be the gap finite
        difference (position jump / long gap -> garbage)."""
        b = LastSeenBelief(CFG)
        for k in range(20):
            b.update(make_obs(opp_world=(0.05 * k, 0.0)))
        v_prior = np.hypot(*b.velocity)
        for _ in range(10):
            b.update(make_obs(visible=False))
        b.update(make_obs(opp_world=(5.0, 5.0)))  # teleported reappearance
        assert np.hypot(*b.velocity) < v_prior  # decayed, not gap-derived


class TestEnvIntegration:
    def test_tracker_matches_env_obs_stream(self):
        """Tracker fed by env obs: visible error ~0 whenever flag is set."""
        env = PursuitEvasionEnv(CFG)
        obs, _ = env.reset(seed=3)
        b = LastSeenBelief(CFG)
        rng = np.random.default_rng(0)
        for _ in range(200):
            b.update(obs["pursuer"])
            if obs["pursuer"][6] > 0.5:
                err = np.hypot(*(b.mean - env.evader_pose[:2]))
                assert err < 1e-6
            obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 4))
            if terminated or truncated:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
