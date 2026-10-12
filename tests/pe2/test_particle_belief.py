"""Tests for pe2.belief.ParticleBelief — S24.0 Step 2, the occlusion-aware
particle filter — and for the error computation of `scripts/s24_belief_gate.py`.

No episode runs here: the gate's own 200-episode stream is the gate's, and a
test suite that replays it would make the single run meaningless.
"""

from __future__ import annotations

import inspect
import pathlib
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from pe2.belief import SIGMA_OBS, ParticleBelief
from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.sensing import is_visible

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.s24_belief_gate import BeliefProbe

CFG = EnvConfig()
V_OPP = 1.2  # the opponent's speed bound, supplied by the caller as it must be


def stub_env(obstacles=(), evader=(0.0, 0.0, 0.0)):
    """The two attributes a belief / the probe reads off an env."""
    return SimpleNamespace(
        obstacles=[dict(o) for o in obstacles],
        evader_pose=np.array(evader, dtype=float),
    )


def make_obs(own=(0.0, 0.0, 0.0), opp_world=None, visible=True):
    """Hand-encode an observation (pe2/obs.py layout), as test_belief.py does."""
    obs = np.zeros(7 + 2 * CFG.n_obstacles, dtype=np.float32)
    half = CFG.arena_size / 2.0
    obs[0], obs[1], obs[2] = own[0] / half, own[1] / half, own[2] / np.pi
    if visible and opp_world is not None:
        dx, dy = opp_world[0] - own[0], opp_world[1] - own[1]
        c, s = np.cos(own[2]), np.sin(own[2])
        obs[3] = (c * dx + s * dy) / CFG.arena_size
        obs[4] = (-s * dx + c * dy) / CFG.arena_size
        obs[6] = 1.0
    return obs


class TestOcclusionAwareKill:
    """ "I can see there and it is not there" — the negative information."""

    def setup_method(self):
        # One obstacle 2 m ahead of a pursuer at the origin facing +x.
        self.obstacle = {"x": 2.0, "y": 0.0, "radius": 0.5}
        self.pose = np.array([0.0, 0.0, 0.0])
        self.pb = ParticleBelief(CFG, V_OPP, n=200, seed=3)
        self.pb.reset(stub_env([self.obstacle]))

    def test_visible_region_particles_die_and_shadow_particles_survive(self):
        pb = self.pb
        # Half the cloud in open view at (1, 0); half in the obstacle's shadow
        # at (3.5, 0). Both are > 0.06 m (one predict step) from the shadow
        # boundary, so the jitter cannot change either verdict.
        pb.particles = np.repeat([[1.0, 0.0], [3.5, 0.0]], 100, axis=0)
        pb.headings = np.zeros(200)
        assert self.pb.visible_mask(self.pose, pb.particles)[:100].all()
        assert not self.pb.visible_mask(self.pose, pb.particles)[100:].any()

        pb.update(self.pose, False, None)

        assert pb.summary()["visible_mass"] == 0.0
        alive = pb.particles[pb.weights > 0]
        assert len(alive) > 0
        assert not pb.visible_mask(self.pose, alive).any()
        # the surviving mass is the shadow half, not the seen half
        assert pb.mean[0] == pytest.approx(3.5, abs=0.2)

    def test_all_particles_in_view_resamples_into_the_hidden_free_space(self):
        pb = self.pb
        pb.particles = np.repeat([[1.0, 0.0]], 200, axis=0)
        pb.headings = np.zeros(200)
        pb.update(self.pose, False, None)
        assert not pb.visible_mask(self.pose, pb.particles).any()
        assert pb._free(pb.particles).all()
        assert pb.weights == pytest.approx(np.full(200, 1 / 200))

    def test_sighting_collapses_the_cloud_onto_the_observation(self):
        pb = self.pb
        pb.update(self.pose, False, None)  # spread out first
        assert pb.summary()["spread"] > 1.0
        pb.update(self.pose, True, (1.5, -1.0))
        s = pb.summary()
        assert s["mean"] == pytest.approx([1.5, -1.0], abs=5 * SIGMA_OBS)
        assert s["spread"] < 5 * SIGMA_OBS
        assert s["steps_since_seen"] == 0
        assert s["ever_seen"]


class TestPredictStaysInTheFreeSpace:
    def test_never_inside_an_inflated_obstacle_or_outside_the_inset_arena(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=11)
        pb = ParticleBelief(CFG, V_OPP, n=300, seed=5)
        pb.reset(env)
        lim = CFG.arena_size / 2.0 - CFG.agent_radius
        for _ in range(200):
            pb.update(env.pursuer_pose, False, None, env.obstacles)
            assert np.abs(pb.particles).max() <= lim
            for o in env.obstacles:
                d = np.linalg.norm(pb.particles - np.array([o["x"], o["y"]]), axis=1)
                assert d.min() >= o["radius"] + CFG.agent_radius

    def test_predict_never_moves_a_particle_farther_than_the_speed_bound(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=12)
        pb = ParticleBelief(CFG, V_OPP, n=300, seed=6)
        pb.reset(env)
        bound = V_OPP * CFG.dt
        for _ in range(200):
            before = pb.particles.copy()
            pb._predict()
            assert np.linalg.norm(pb.particles - before, axis=1).max() <= bound

    def test_the_whole_update_respects_the_bound_up_to_resampling(self):
        """The correction may RESAMPLE (copy) survivors, which relocates a
        particle. The invariant that still holds through the whole update is
        the one that matters: every particle afterwards sits within one bounded
        step of SOME particle before it — the filter never invents motion
        faster than the bound it was given."""
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=12)
        pb = ParticleBelief(CFG, V_OPP, n=200, seed=6)
        pb.reset(env)
        bound = V_OPP * CFG.dt
        for _ in range(100):
            before = pb.particles.copy()
            pb.update(env.pursuer_pose, False, None, env.obstacles)
            d = np.linalg.norm(pb.particles[:, None, :] - before[None, :, :], axis=2)
            # The uniform-resample fallback (every particle in view) would
            # break this, and cannot fire here: a 90-degree, 8 m cone never
            # contains an arena-wide cloud.
            assert d.min(axis=1).max() <= bound + 1e-12

    def test_zero_speed_bound_freezes_the_cloud(self):
        """The bound is the constructor's argument and nothing else."""
        pb = ParticleBelief(CFG, 0.0, n=50, seed=7)
        pb.reset(stub_env())
        before = pb.particles.copy()
        pb._predict()
        assert pb.particles == pytest.approx(before)


class TestVisibilityIsTheSensingModule:
    def test_vectorised_mask_equals_is_visible_exactly(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=13)
        pb = ParticleBelief(CFG, V_OPP, n=10, seed=8)
        pb.reset(env)
        rng = np.random.default_rng(0)
        for _ in range(30):
            pose = np.array(
                [rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-np.pi, np.pi)]
            )
            pts = rng.uniform(-5, 5, size=(200, 2))
            got = pb.visible_mask(pose, pts)
            want = np.array(
                [
                    is_visible(pose, (p[0], p[1]), env.obstacles, CFG.sensing)
                    for p in pts
                ]
            )
            assert (got == want).all()

    def test_tangency_convention_matches(self):
        """A sight line exactly tangent to a disc does NOT block (pe2/sensing)."""
        obstacle = {"x": 0.0, "y": 0.5, "radius": 0.5}
        pb = ParticleBelief(CFG, V_OPP, n=1, seed=9)
        pb.reset(stub_env([obstacle]))
        pose = np.array([0.0, 0.0, 0.0])
        pts = np.array([[3.0, 0.0]])  # the x axis is tangent to the disc
        assert pb.visible_mask(pose, pts)[0]
        assert is_visible(pose, (3.0, 0.0), [obstacle], CFG.sensing)


class TestContract:
    def test_deterministic_given_the_seed(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=14)
        runs = []
        for _ in range(2):
            pb = ParticleBelief(CFG, V_OPP, n=100, seed=99)
            pb.reset(env)
            for t in range(40):
                visible = t % 13 == 0
                pb.update(
                    env.pursuer_pose,
                    visible,
                    env.evader_pose[:2] if visible else None,
                    env.obstacles,
                )
            runs.append((pb.particles.copy(), pb.weights.copy()))
        assert runs[0][0] == pytest.approx(runs[1][0])
        assert runs[0][1] == pytest.approx(runs[1][1])

    def test_different_seeds_differ(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=14)
        out = []
        for seed in (1, 2):
            pb = ParticleBelief(CFG, V_OPP, n=100, seed=seed)
            pb.reset(env)
            pb.update(env.pursuer_pose, False, None, env.obstacles)
            out.append(pb.particles.copy())
        assert not np.allclose(out[0], out[1])

    def test_summary_fields(self):
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=15)
        pb = ParticleBelief(CFG, V_OPP, n=100, seed=10)
        pb.reset(env)
        pb.update(env.pursuer_pose, False, None, env.obstacles)
        pb.update(env.pursuer_pose, False, None, env.obstacles)
        s = pb.summary()
        assert set(s) == {
            "mean",
            "spread",
            "mode",
            "visible_mass",
            "steps_since_seen",
            "ever_seen",
        }
        assert s["mean"].shape == (2,)
        assert s["mode"].shape == (2,)
        assert s["spread"] > 0.0
        assert 0.0 <= s["visible_mass"] <= 1.0
        assert s["steps_since_seen"] == 2
        assert s["ever_seen"] is False
        # the strongest mode is a point of the cloud's support, not the mean
        assert pb._free(s["mode"][None, :]).all()

    def test_speed_agnostic_rule_on_the_source(self):
        """Plan section 1: the class takes the bound as an argument and never
        reads the config's speed ratio."""
        src = inspect.getsource(ParticleBelief)
        assert "evader_speed_ratio" not in src
        assert "evader_v_max" not in src
        assert "v_max_opp" in inspect.signature(ParticleBelief.__init__).parameters
        # the module names the field only in the comment that refuses to read
        # it; what the test forbids is a READ of the config's speed fields.
        src_body = inspect.getsource(ParticleBelief)
        assert "cfg.evader" not in src_body
        assert "self.cfg.evader" not in src_body


class TestGateErrorComputation:
    """The gate script's error record, exact on a known path.

    The pursuer sits at the origin facing +x; the evader walks along -x, i.e.
    behind the pursuer and outside its 90-degree cone, so every step is
    occluded and dead reckoning never sees it — its mean stays at the origin
    (`LastSeenBelief` reset state), so the expected error is exactly the
    evader's distance from the origin.
    """

    PATH: ClassVar[list] = [(-1.0, 0.0), (-2.0, 0.5), (-3.0, 1.5)]

    def probe_over_path(self):
        env = stub_env()
        probe = BeliefProbe(CFG, V_OPP, n=50, seed=4)
        probe.reset(env)
        for xy in self.PATH:
            env.evader_pose = np.array([xy[0], xy[1], 0.0])
            probe.observe(make_obs(own=(0.0, 0.0, 0.0), visible=False), env)
        return probe, env

    def test_dead_reckoning_errors_are_exact(self):
        probe, _ = self.probe_over_path()
        assert not probe.dr.seen  # never in the cone
        assert probe.dr_err == pytest.approx([np.hypot(*xy) for xy in self.PATH])
        assert probe.after_sighting == [False, False, False]

    def test_particle_error_is_the_mean_to_the_truth(self):
        probe, env = self.probe_over_path()
        truth = env.evader_pose[:2]
        assert probe.pf_err[-1] == pytest.approx(
            float(np.linalg.norm(probe.pf.mean - truth))
        )
        assert probe.pf_spread[-1] == pytest.approx(probe.pf.summary()["spread"])

    def test_occlusion_bookkeeping(self):
        probe, env = self.probe_over_path()
        # one visible step: it must NOT be recorded as an occluded error
        env.evader_pose = np.array([2.0, 0.0, 0.0])
        probe.observe(make_obs(opp_world=(2.0, 0.0), visible=True), env)
        st = probe.stats()
        assert st["n_steps"] == 4
        assert st["n_occluded"] == 3
        assert st["occluded_fraction"] == pytest.approx(3 / 4)
        assert st["dr_median"] == pytest.approx(np.hypot(-2.0, 0.5))
        assert st["n_episodes_with_occlusion"] == 1
        assert st["episode_median"][0]["n_occluded"] == 3
        assert st["dr_median_after_sighting"] is None  # never seen on this path

    def test_each_episode_records_its_own_median(self):
        probe, env = self.probe_over_path()
        probe.reset(env)  # closes episode 1
        env.evader_pose = np.array([-4.0, 0.0, 0.0])
        probe.observe(make_obs(visible=False), env)
        st = probe.stats()
        assert len(st["episode_median"]) == 2
        assert st["episode_median"][1]["dr_median"] == pytest.approx(4.0)
        assert st["n_occluded"] == 4
