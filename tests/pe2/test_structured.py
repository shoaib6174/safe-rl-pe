"""pe2.structured — the S1 structured pursuer (spec:
docs/pathway/structured_search_prereg_2026-09-12.md)."""

import json
import pathlib
from itertools import pairwise

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


class TestTraceAndCharacterise:
    def test_trace_records_every_env_step_with_its_branch(self):
        from pe2.structured import SearchMDP

        seed = 900_900
        m = SearchMDP(CFG, GreedyEvader, hold=10)
        m.trace = []
        log = m.play_episode(OrbitSearch(CFG).predict, seed)
        assert len(m.trace) == log["steps"]
        branches = {b for _, b, _, _ in m.trace}
        assert branches <= {"law", "search"} and "search" in branches
        vis = [v for v, _, _, _ in m.trace]
        # The trace records visibility only AFTER each step, but _los_breaks uses
        # the reset visibility as the first `prev` — so seed the sequence with it
        # (the wrapper's reset is deterministic per seed) or a visible-at-reset ->
        # invisible-after-step-1 episode would be counted by one side only.
        obs0, _ = SingleAgentWrapper(CFG, "pursuer", GreedyEvader).reset(seed=seed)
        reset_vis = bool(obs0[6] > 0.5)
        # every recorded "law" step follows a visible or last-seen state; the
        # count of visible->invisible transitions must equal the log's los_breaks
        breaks = sum(1 for a, b in pairwise([reset_vis, *vis]) if a and not b)
        assert breaks == log["los_breaks"]
        for _, _, v, w in m.trace:
            assert -1.0 <= v <= 1.0 and -1.0 <= w <= 1.0

    def test_trace_is_off_by_default(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader)
        assert m.trace is None
        m.play_episode(OrbitSearch(CFG).predict, 900_901)
        assert m.trace is None

    def test_characterise_orbit_matches_the_design_numbers_in_shape(self):
        from pe2.structured import characterise

        c = characterise(
            OrbitSearch(CFG).predict, CFG, GreedyEvader, 12, 900_910, hold=1
        )
        assert c["n_episodes"] == 12
        assert 0.0 < c["visible_frac"] < 1.0
        assert 0.0 < c["search_frac"] < 1.0
        assert c["first_sight_median"] <= c["first_sight_p90"]
        assert c["stretch_median"] <= c["stretch_p90"]
        assert c["search_speed_mean"] == pytest.approx(
            CFG.v_max, abs=1e-6
        )  # the orbit drives full speed
        assert 0.0 <= c["search_abs_omega_mean"] <= CFG.omega_max
        assert 0.0 <= c["capture_rate"] <= 1.0 and 0.0 <= c["found_frac"] <= 1.0

    def test_characterise_stand_still_has_zero_search_speed(self):
        from pe2.structured import characterise

        class Still:
            def predict(self, obs, deterministic=True):
                return np.array([-1.0, 0.5], dtype=np.float32), None

        c = characterise(Still().predict, CFG, GreedyEvader, 4, 900_920, hold=10)
        assert c["search_speed_mean"] == pytest.approx(0.0, abs=1e-6)
        assert c["search_abs_omega_mean"] == pytest.approx(
            0.5 * CFG.omega_max, abs=1e-6
        )


class TestShieldInSearchMDP:
    def test_no_shield_leaves_logs_unchanged(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader)
        log = m.play_episode(OrbitSearch(CFG).predict, 900_930)
        assert "shield" not in log and "shielded_role" not in log
        assert m.shield is None and m.shield_metrics is None
        assert m.violations_total >= 0

    def test_noop_shield_round_trip_is_exact(self):
        # nominal -> NoOpShield -> renormalise must be the identity on the
        # clipped Box action, or the NONE arm is not the unshielded episode
        from pe2.shield import NoOpShield

        shield = NoOpShield(CFG)
        pose = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1.0, 1.0, size=(20_000, 2)).astype(np.float32)
        for action in actions:
            a = np.clip(action, -1.0, 1.0)
            v_nom = (float(a[0]) + 1.0) / 2.0 * CFG.v_max
            w_nom = float(a[1]) * CFG.omega_max
            (v, w), _ = shield.filter((v_nom, w_nom), pose, [], v_max=CFG.v_max)
            back = np.array(
                [2.0 * v / CFG.v_max - 1.0, w / CFG.omega_max], dtype=np.float32
            )
            assert np.array_equal(back, a)

    def test_noop_shield_reproduces_the_unshielded_episode_and_measures(self):
        from pe2.metrics import summarize_shield
        from pe2.shield import NoOpShield
        from pe2.structured import SearchMDP

        plain = SearchMDP(CFG, GreedyEvader, hold=1).play_episode(
            OrbitSearch(CFG).predict, 900_931
        )
        m = SearchMDP(CFG, GreedyEvader, hold=1, shield=NoOpShield(CFG))
        log = m.play_episode(OrbitSearch(CFG).predict, 900_931)
        assert (log["outcome"], log["steps"], log["los_breaks"]) == (
            plain["outcome"],
            plain["steps"],
            plain["los_breaks"],
        )
        # exact, not approx: the NONE control must be the unshielded episode
        assert log["final_distance"] == plain["final_distance"]
        assert log["shielded_role"] == "pursuer"
        assert log["shield"]["n_steps"] == log["steps"]
        assert log["shield"]["tier_histogram"]["none"] == log["steps"]
        assert log["violations"] == m.episode_violations
        s = summarize_shield([log])
        assert s["n_episodes"] == 1 and s["violation_steps"] == log["violations"]

    def test_cbf_shield_filters_every_step_and_counts_no_violation(self):
        from pe2.metrics import TIERS
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG))
        for i in range(3):
            log = m.play_episode(RandomSearch(i).predict, 900_940 + i)
            assert log["shield"]["n_steps"] == log["steps"]
            assert log["shield"]["tier_histogram"]["none"] == 0
            assert log["violations"] == 0  # D97: zero contact steps at delta = 0
            assert m._last_sinfo["method"] in TIERS  # R1: Task 2's overlay source
        assert m.violations_total == 0

    def test_unshielded_random_search_does_hit_walls(self):
        # the contact counter must be live without a shield, and a full-speed
        # random searcher collides somewhere in a few episodes (G4's logic)
        from pe2.structured import SearchMDP

        class Straight:
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        m = SearchMDP(CFG, GreedyEvader, hold=10)
        for i in range(5):
            m.play_episode(Straight().predict, 900_950 + i)
        assert m.violations_total > 0

    def test_shield_is_applied_to_the_law_branch_too(self):
        # every executed command goes through the filter: n_steps equals env steps
        # even on an episode where the law (chase) plays most of the steps
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=1, shield=CBFShield(CFG))
        m.trace = []
        log = m.play_episode(OrbitSearch(CFG).predict, 900_960)
        assert any(b == "law" for _, b, _, _ in m.trace)
        assert log["shield"]["n_steps"] == len(m.trace) == log["steps"]

    def test_evaluate_structured_shield_factory(self):
        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        logs = evaluate_structured(
            RandomSearch(0),
            CFG,
            GreedyEvader,
            2,
            900_970,
            hold=10,
            shield_factory=lambda: CBFShield(CFG),
        )
        assert all("shield" in lg and lg["shielded_role"] == "pursuer" for lg in logs)
        plain = evaluate_structured(
            RandomSearch(0), CFG, GreedyEvader, 2, 900_970, hold=10
        )
        assert all("shield" not in lg for lg in plain)


class TestShieldAwareIngredients:
    def test_features_default_without_a_shield(self):
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, shield_obs=True)
        np.testing.assert_allclose(m.shield_features(), [0.0, 2.0, 0.0])
        assert m.observation_space.shape == (14,)
        obs, _ = m.reset(seed=900_980)
        assert obs.shape == (14,) and obs.dtype == np.float32
        np.testing.assert_allclose(obs[-3:], [0.0, 2.0, 0.0])

    def test_features_come_from_the_last_executed_step(self):
        from pe2.metrics import DEADLOCK_STEPS
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        # 900_989, not 900_981: at 900_981 the law has nothing to act on at
        # reset, so no step is executed and `_last_sinfo` is still None --
        # there is no "last executed step" to compare the features against.
        m = SearchMDP(
            CFG, GreedyEvader, hold=10, shield=CBFShield(CFG), shield_obs=True
        )
        obs, _ = m.reset(seed=900_989)
        assert obs.shape == (14,)
        f = m.shield_features()
        s = m._last_sinfo
        assert f[0] == float(bool(s["intervention"]))
        assert f[1] == pytest.approx(float(np.clip(s["min_clearance"], -0.5, 2.0)))
        assert f[2] == pytest.approx(
            float(
                np.clip(m.shield_metrics._current_stall_run / DEADLOCK_STEPS, 0.0, 2.0)
            )
        )
        # the law still sees the raw 11-dim observation: play a full episode without error
        log = m.play_episode(RandomSearch(1).predict, 900_989)
        assert log["violations"] == 0

    def test_shield_obs_off_leaves_the_observation_unchanged(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(CFG, GreedyEvader, hold=10, shield=CBFShield(CFG))
        obs, _ = m.reset(seed=900_982)
        assert obs.shape == (11,) and m.observation_space.shape == (11,)

    def test_stall_penalty_subtracts_per_stalled_step_in_the_span(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        class Wall:  # drive straight into the nearest wall: the shield must stall it
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        m = SearchMDP(
            CFG, GreedyEvader, hold=10, shield=CBFShield(CFG), stall_penalty=1e-3
        )
        obs, _ = m.reset(seed=900_983)
        total_stalled = 0
        total_penalty = 0.0
        done = False
        while not done:
            obs, r, term, trunc, info = m.step(Wall().predict(obs)[0])
            done = term or trunc
            assert info["stalled_steps"] == m.span_stalled
            total_stalled += info["stalled_steps"]
            total_penalty += r - info["raw_reward"]
        assert total_stalled > 0
        assert total_penalty == pytest.approx(-1e-3 * total_stalled)

    def test_stall_penalty_is_zero_under_zero_reward(self):
        from pe2.shield import CBFShield
        from pe2.structured import SearchMDP

        m = SearchMDP(
            CFG,
            GreedyEvader,
            hold=10,
            shield=CBFShield(CFG),
            stall_penalty=1e-3,
            zero_reward=True,
        )
        m.reset(seed=900_984)
        _, r, _, _, _ = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert r == 0.0

    def test_evaluate_structured_with_shield_obs(self):
        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        class Echo:
            def predict(self, obs, deterministic=True):
                assert obs.shape == (14,)
                return np.array([0.5, 0.0], dtype=np.float32), None

        logs = evaluate_structured(
            Echo(),
            CFG,
            GreedyEvader,
            2,
            900_990,
            hold=10,
            shield_factory=lambda: CBFShield(CFG),
            shield_obs=True,
        )
        assert len(logs) == 2 and all("shield" in lg for lg in logs)
