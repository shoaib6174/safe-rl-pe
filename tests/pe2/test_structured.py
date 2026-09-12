"""pe2.structured — the S1 structured pursuer (spec:
docs/pathway/structured_search_prereg_2026-09-12.md)."""

import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.scripted import GreedyEvader, SearchingPursuer
from pe2.structured import (
    HOLD,
    R_ACQ,
    InterceptLaw,
    OrbitSearch,
    RandomSearch,
    SearchMDP,
    evaluate_scripted,
    evaluate_structured,
)

CFG = EnvConfig()
LOG_KEYS = {"seed", "role", "outcome", "steps", "final_distance", "los_breaks"}

# Poses for the tests that STEP the env are given in the env's own frame:
# the arena is centred on the origin, so a pose is only in bounds for
# |x|, |y| <= arena_size / 2 - agent_radius = 4.85 (pe2.env lines 68-72).
# Out-of-bounds poses are clipped on the first step, which would put both
# agents in the same corner (an instant capture at distance 0).
#
# Spin-to-sight pair: the evader sits 2.5 m from the pursuer at bearing
# +60 deg, just outside the 45 deg half-cone, so spinning in place at
# omega_max sweeps it into view within a couple of env steps (2.5 m is
# inside the 8 m range, and 15 deg / (2.84 * 0.05) = 1.8 steps).
SPIN_POSES = ((0.0, 0.0, 0.0), (1.25, 2.165, 0.0))
SPIN_SEED = 4
SPIN_ACTION = np.array([-1.0, 1.0], dtype=np.float32)  # stop and spin


def _teacher_action(obs, wrapper):
    """SearchingPursuer's (v, w) on this obs, normalized to the Box."""
    t = SearchingPursuer()
    t.reset()
    v, w = t.act(obs, wrapper.env)
    return np.array([2.0 * v / CFG.v_max - 1.0, w / CFG.omega_max], dtype=np.float32), t


def _reset_with_evader(wrapper, pursuer_pose, evader_pose, seed=1):
    obs, _ = wrapper.reset(seed=seed, init_poses=(pursuer_pose, evader_pose))
    return obs


class TestInterceptLaw:
    def test_constants_are_the_spec_values(self):
        assert HOLD == 10
        assert R_ACQ == 0.2

    def test_visible_branch_matches_the_teacher(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        # evader 2 m dead ahead: visible
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (7.0, 5.0, 0.0))
        assert obs[6] > 0.5
        law = InterceptLaw(CFG)
        expected, teacher = _teacher_action(obs, w)
        got = law.act(obs)
        assert got is not None
        np.testing.assert_allclose(got, expected, atol=1e-6)
        np.testing.assert_allclose(law.last_seen, teacher.last_seen)

    def test_nothing_to_act_on_returns_none(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        # evader 3 m directly behind: invisible, no last_seen
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (2.0, 5.0, 0.0))
        assert obs[6] < 0.5
        law = InterceptLaw(CFG)
        assert law.act(obs) is None
        assert law.last_seen is None

    def test_last_seen_branch_matches_the_teacher_then_clears(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        obs_vis = _reset_with_evader(w, (5.0, 5.0, 0.0), (7.0, 5.0, 0.0))
        law = InterceptLaw(CFG)
        law.act(obs_vis)  # sets last_seen
        teacher = SearchingPursuer()
        teacher.reset()
        teacher.act(obs_vis, w.env)
        # now an invisible observation from a pose far from last_seen
        obs_inv = _reset_with_evader(w, (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), seed=2)
        assert obs_inv[6] < 0.5
        v, om = teacher.act(obs_inv, w.env)
        expected = np.array(
            [2.0 * v / CFG.v_max - 1.0, om / CFG.omega_max], dtype=np.float32
        )
        got = law.act(obs_inv)
        assert got is not None
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # standing on the last-seen point: the law clears it and has nothing to act on
        law.last_seen = np.array([1.0, 1.0])
        assert law.act(obs_inv) is None
        assert law.last_seen is None


class TestSearchShims:
    def test_orbit_search_is_the_teacher_fallback(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        obs = _reset_with_evader(w, (5.0, 5.0, 0.0), (2.0, 5.0, 0.0))
        expected, _ = _teacher_action(obs, w)  # teacher falls through to the orbit
        got, _ = OrbitSearch(CFG).predict(obs, deterministic=True)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_random_search_is_seeded_and_in_the_box(self):
        a, _ = RandomSearch(seed=0).predict(np.zeros(11))
        b, _ = RandomSearch(seed=0).predict(np.zeros(11))
        np.testing.assert_array_equal(a, b)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)


class TestSearchMDP:
    def test_spaces_match_the_inner_wrapper(self):
        m = SearchMDP(CFG, GreedyEvader)
        assert m.observation_space == m.inner.observation_space
        assert m.action_space == m.inner.action_space

    def test_v1_orbit_at_hold_1_reproduces_the_teacher_episode_for_episode(self):
        # Spec §2 validity check V1 on a small seed set.
        seeds0, n = 900_500, 12
        teacher_logs = evaluate_scripted(CFG, SearchingPursuer, GreedyEvader, n, seeds0)
        mdp_logs = evaluate_structured(
            OrbitSearch(CFG), CFG, GreedyEvader, n, seeds0, hold=1
        )
        assert [(l["outcome"], l["steps"]) for l in teacher_logs] == [
            (l["outcome"], l["steps"]) for l in mdp_logs
        ]
        assert [l["los_breaks"] for l in teacher_logs] == [
            l["los_breaks"] for l in mdp_logs
        ]

    def test_a_decision_is_held_for_hold_env_steps_while_invisible(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10)
        # evader 3 m behind and driving away; the learner drives straight
        # ahead, so the evader stays invisible for the whole hold
        m.inner.reset(seed=3, init_poses=((0.0, 0.0, 0.0), (-3.0, 0.0, np.pi)))
        m.law.reset()
        m._visible = False
        before = m.env_steps
        _obs, r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert m.env_steps - before == 10
        assert info["env_steps"] == m.env_steps
        assert info["onsets"] == 0
        assert r == 0.0 and not term and not trunc

    def test_hold_ends_early_on_sight_and_the_law_then_plays(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10)
        # evader 2.5 m away at bearing +60 deg; spinning in place brings it
        # into the 90 deg cone in ~2 steps, then the chase law runs until
        # sight is lost or the episode ends — so more than 10 env steps elapse.
        m.inner.reset(seed=SPIN_SEED, init_poses=SPIN_POSES)
        m.law.reset()
        m._visible = False
        _obs, r, _term, _trunc, info = m.step(SPIN_ACTION)
        assert info["onsets"] >= 1
        assert m.env_steps > 10
        assert r == pytest.approx(info["raw_reward"] + R_ACQ * info["onsets"])

    def test_zero_reward_arm_returns_zero_but_reports_the_raw_reward(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10, zero_reward=True)
        m.inner.reset(seed=SPIN_SEED, init_poses=SPIN_POSES)
        m.law.reset()
        m._visible = False
        _, r, _, _, info = m.step(SPIN_ACTION)
        assert r == 0.0
        assert info["onsets"] >= 1

    def test_terminal_only_arm_has_no_bonus(self):
        m = SearchMDP(CFG, GreedyEvader, hold=10, r_acq=0.0)
        m.inner.reset(seed=SPIN_SEED, init_poses=SPIN_POSES)
        m.law.reset()
        m._visible = False
        _, r, _, _, info = m.step(SPIN_ACTION)
        assert r == pytest.approx(info["raw_reward"])

    def test_gym_reset_and_step_run_a_whole_episode(self):
        m = SearchMDP(CFG, GreedyEvader, seed0=7000)
        obs, info = m.reset()
        assert obs.shape == (11,)
        assert obs[6] < 0.5  # a decision point is always one the law cannot act on
        done = False
        decisions = 0
        while not done:
            obs, _r, term, trunc, info = m.step(m.action_space.sample())
            done = term or trunc
            decisions += 1
        assert decisions <= 1200
        assert info["outcome"] in ("capture", "timeout")

    def test_evaluators_share_the_log_schema(self):
        s = evaluate_scripted(CFG, SearchingPursuer, GreedyEvader, 2, 900_600)
        e = evaluate_structured(RandomSearch(0), CFG, GreedyEvader, 2, 900_600)
        assert len(s) == len(e) == 2
        assert set(s[0]) == set(e[0]) == LOG_KEYS
        assert s[0]["seed"] == e[0]["seed"] == 900_600


class TestStructuredEvalProbe:
    def test_probes_on_the_env_step_grid_and_stops_at_total(self, tmp_path):
        from pe2.structured import StructuredEvalProbe

        class _VecEnv:
            def __init__(self):
                self.env_steps = 0

            def get_attr(self, name):
                return [getattr(self, name)]

        class _Model:
            def __init__(self, venv):
                self.venv = venv

            def get_env(self):  # what BaseCallback.training_env reads
                return self.venv

            def save(self, path):
                pathlib.Path(path).write_bytes(b"ckpt")

            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        probe = StructuredEvalProbe(
            CFG,
            GreedyEvader,
            every=100,
            total=300,
            eval_n=2,
            seed0=900_700,
            ckpt_dir=tmp_path,
            report_path=tmp_path / "r.json",
            label="T",
        )
        cb, venv = probe.callback, _VecEnv()
        cb.model = _Model(venv)  # what SB3's init_callback sets
        venv.env_steps = 57
        assert cb._on_step() is True and probe.rows == []
        venv.env_steps = 130  # first decision boundary past the 100 mark
        assert cb._on_step() is True
        assert [(r["mark"], r["steps"]) for r in probe.rows] == [(100, 130)]
        venv.env_steps = 310  # one stretch jumped over two marks
        assert cb._on_step() is False  # total reached: learn() must stop
        assert [(r["mark"], r["steps"]) for r in probe.rows] == [
            (100, 130),
            (200, 310),
            (300, 310),
        ]
        assert (tmp_path / "T_300.zip").exists()
        report = json.loads((tmp_path / "r.json").read_text())
        assert [r["mark"] for r in report["checkpoints"]] == [100, 200, 300]
        assert {"steps", "mark", "ckpt", "win_rate", "k", "n"} <= set(probe.rows[0])

    def test_total_must_be_a_multiple_of_every(self, tmp_path):
        from pe2.structured import StructuredEvalProbe

        with pytest.raises(ValueError, match="multiple"):
            StructuredEvalProbe(
                CFG,
                GreedyEvader,
                every=100,
                total=250,
                eval_n=2,
                seed0=900_700,
                ckpt_dir=tmp_path,
                report_path=tmp_path / "r.json",
                label="T",
            )

    def test_rejects_a_vec_env_with_more_than_one_env(self, tmp_path):
        from pe2.structured import StructuredEvalProbe

        class _TwoEnvs:
            def get_attr(self, name):
                return [10, 10]

        class _Model:
            def get_env(self):
                return _TwoEnvs()

        probe = StructuredEvalProbe(
            CFG,
            GreedyEvader,
            every=100,
            total=300,
            eval_n=2,
            seed0=900_700,
            ckpt_dir=tmp_path,
            report_path=tmp_path / "r.json",
            label="T",
        )
        probe.callback.model = _Model()
        with pytest.raises(AssertionError, match="single env"):
            probe.callback._on_step()
