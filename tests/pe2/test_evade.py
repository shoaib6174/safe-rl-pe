"""pe2.evade — the structured full-state attacker (S6, Step 9)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.shield import CBFShield
from pe2.structured import HOLD, StructuredPursuerPolicy

CFG = EnvConfig()
LOG_KEYS = {
    "seed",
    "role",
    "outcome",
    "steps",
    "escaped",
    "pursuer_violations",
    "pursuer_deadlock",
    "pursuer_shield",
}


def _str14():
    from stable_baselines3 import PPO

    return PPO.load("results/derisk/ckpts_STR14/STR14_10000000.zip", device="cpu")


def _d1_factory(model):
    return lambda: StructuredPursuerPolicy(model, CFG, CBFShield(CFG), HOLD, False)


class TestEvadeMDP:
    def test_spaces_and_full_state_obs(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()))
        assert m.observation_space.shape == (11,)
        obs, _ = m.reset(seed=900_1300)
        assert (
            obs.shape == (11,) and obs[6] == 1.0
        )  # full state: pursuer always 'visible'

    def test_law_acts_within_d_pos_and_learner_beyond(self):
        from pe2.evade import D_POS, EvadeMDP

        assert D_POS == 4.0
        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        m.trace = []
        log = m.play_episode(
            lambda o, deterministic=True: (
                np.array([1.0, 0.0], dtype=np.float32),
                None,
            ),
            900_1301,
        )
        assert {b for b, _ in m.trace} <= {"law", "search"}
        for branch, dist in m.trace:
            if branch == "law":
                assert dist <= D_POS + 1e-9
            else:
                assert dist > D_POS
        assert log["steps"] == len(m.trace)

    def test_stall_bonus_counts_pursuer_stalls_in_the_span(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10, beta=1e-3)
        m.reset(seed=900_1302)
        total_stalled = 0
        total_bonus = 0.0
        done = False
        while not done:
            _obs, r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
            done = term or trunc
            total_stalled += info["pursuer_stalled_steps"]
            total_bonus += r - info["raw_reward"]
        assert total_bonus == pytest.approx(1e-3 * total_stalled)

    def test_zero_reward(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10, beta=1e-3, zero_reward=True)
        m.reset(seed=900_1303)
        _, r, _, _, _ = m.step(np.array([1.0, 0.0], dtype=np.float32))
        assert r == 0.0

    def test_play_episode_log_and_pursuer_metrics(self):
        from pe2.evade import EvadeMDP, SplitHugEvader

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        log = m.play_episode(SplitHugEvader(CFG).predict, 900_1304)
        assert set(log) >= LOG_KEYS and log["role"] == "evader"
        assert log["escaped"] == (log["outcome"] == "timeout")
        assert log["pursuer_shield"]["n_steps"] == log["steps"]
        assert log["pursuer_violations"] == 0  # D97: shielded defender never contacts
        assert isinstance(log["pursuer_deadlock"], bool)

    def test_terminal_reward_is_the_evaders(self):
        from pe2.evade import EvadeMDP

        m = EvadeMDP(CFG, _d1_factory(_str14()), hold=10)
        m.reset(seed=900_1305)
        total = 0.0
        done = False
        while not done:
            _obs, _r, term, trunc, info = m.step(np.array([1.0, 0.0], dtype=np.float32))
            done = term or trunc
            total += info["raw_reward"]
        assert total == (1.0 if info["outcome"] == "timeout" else -1.0)


class TestSplitHugEvader:
    def test_flees_directly_away_in_body_frame(self):
        from pe2.evade import SplitHugEvader
        from pe2.obs import full_state_observation

        own = np.array([0.0, 0.0, 0.0])
        opp = np.array([2.0, 0.0, 0.0])  # pursuer dead ahead
        obs = full_state_observation(own, opp, [], CFG)
        a, _ = SplitHugEvader(CFG).predict(obs)
        assert a[0] == pytest.approx(1.0)  # full speed
        assert abs(a[1]) == pytest.approx(1.0)  # turn hard: flee direction is behind


class TestEvaluators:
    def test_evaluate_evader_and_scripted_share_the_schema(self):
        from pe2.evade import SplitHugEvader, evaluate_evader, play_scripted_evader

        model = _str14()
        a = evaluate_evader(SplitHugEvader(CFG), CFG, _d1_factory(model), 2, 900_1310)
        b = play_scripted_evader(
            CFG, ObstacleHuggingEvader, _d1_factory(model), 2, 900_1310
        )
        c = play_scripted_evader(CFG, GreedyEvader, _d1_factory(model), 2, 900_1310)
        for logs in (a, b, c):
            assert len(logs) == 2 and all(set(lg) >= LOG_KEYS for lg in logs)
            assert [lg["seed"] for lg in logs] == [900_1310, 900_1311]
            assert all(lg["pursuer_violations"] == 0 for lg in logs)

    def test_probe_grid_and_stop(self, tmp_path):
        from pe2.evade import EvaderEvalProbe

        class _VecEnv:
            env_steps = 0

            def get_attr(self, name):
                return [getattr(self, name)]

        class _Model:
            def __init__(self, venv):
                self.venv = venv

            def get_env(self):  # what BaseCallback.training_env reads
                return self.venv

            def save(self, path):
                import pathlib

                pathlib.Path(path).write_bytes(b"x")

            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0], dtype=np.float32), None

        p = EvaderEvalProbe(
            CFG,
            _d1_factory(_str14()),
            every=100,
            total=200,
            eval_n=2,
            seed0=900_1320,
            ckpt_dir=tmp_path,
            report_path=tmp_path / "r.json",
            label="T",
            hold=10,
            d_pos=4.0,
        )
        cb, venv = p.callback, _VecEnv()
        cb.model = _Model(venv)  # what SB3's init_callback sets
        venv.env_steps = 150
        assert cb._on_step() is True and [r["mark"] for r in p.rows] == [100]
        venv.env_steps = 200
        assert cb._on_step() is False and [r["mark"] for r in p.rows] == [100, 200]
        assert {
            "steps",
            "mark",
            "ckpt",
            "escape_rate",
            "deadlock_rate",
            "k",
            "n",
            "pursuer_violations",
        } <= set(p.rows[0])

    def test_pursuer_summary_pools_the_defender_block(self):
        from pe2.evade import SplitHugEvader, evaluate_evader, pursuer_summary

        logs = evaluate_evader(
            SplitHugEvader(CFG), CFG, _d1_factory(_str14()), 2, 900_1310
        )
        s = pursuer_summary(logs)
        assert s["n_episodes"] == 2 and s["shielded_roles"] == ["pursuer"]
        assert s["n_steps"] == sum(lg["steps"] for lg in logs)
        assert s["violation_steps"] == 0
        assert s["escape_rate"] == sum(lg["escaped"] for lg in logs) / 2
        assert s["timeout_rate"] == s["escape_rate"]  # outcome must reach the pooler


class TestStructuredEvaderPolicy:
    def test_v1e_reproduces_evademdp_against_a_frozen_defender(self):
        from stable_baselines3 import PPO

        from pe2.evade import EvadeMDP, StructuredEvaderPolicy
        from pe2.exploitability import SingleAgentWrapper

        d1 = _str14()
        ev_model = PPO.load(
            "results/derisk/ckpts_CTL-D125/CTL-D125_10000000.zip", device="cpu"
        )
        for i in range(20):
            seed = 900_1400 + i
            ref = EvadeMDP(CFG, _d1_factory(d1), hold=10).play_episode(
                ev_model.predict, seed
            )
            w = SingleAgentWrapper(
                CFG, "pursuer", lambda: StructuredEvaderPolicy(ev_model, CFG, hold=10)
            )
            w.reset(seed=seed)
            pol = StructuredPursuerPolicy(d1, CFG, CBFShield(CFG), HOLD, False)
            pol.reset()
            while True:
                v, om = pol.act(w._last_joint_obs["pursuer"], w.env)
                _, _, term, trunc, info = w.step(
                    np.array(
                        [2 * v / CFG.v_max - 1, om / CFG.omega_max], dtype=np.float32
                    )
                )
                pol.record(
                    bool(
                        info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]
                    )
                )
                if term or trunc:
                    break
            got = (
                info["outcome"],
                w.env.current_step,
                pol.metrics.n_stalled,
                pol.metrics.longest_stall_run,
            )
            exp = (
                ref["outcome"],
                ref["steps"],
                ref["pursuer_shield"]["n_stalled"],
                ref["pursuer_shield"]["longest_stall_run"],
            )
            assert got == exp, f"seed {seed}: {got} != {exp}"

    def test_decisions_only_beyond_d_pos(self):
        from stable_baselines3 import PPO

        from pe2.evade import D_POS, StructuredEvaderPolicy
        from pe2.exploitability import SingleAgentWrapper

        ev_model = PPO.load(
            "results/derisk/ckpts_CTL-D125/CTL-D125_10000000.zip", device="cpu"
        )
        w = SingleAgentWrapper(
            CFG, "pursuer", lambda: StructuredEvaderPolicy(ev_model, CFG, hold=10)
        )
        w.reset(seed=900_1450)
        pol = w._opponent
        n_before = pol.decisions
        # a stationary pursuer far away: the law never applies, decisions tick every `hold` steps
        far = float(np.hypot(*(w.env.pursuer_pose[:2] - w.env.evader_pose[:2]))) > D_POS
        for _ in range(30):
            _, _, term, trunc, _ = w.step(np.array([-1.0, 0.0], dtype=np.float32))
            if term or trunc:
                break
        if far:
            assert pol.decisions - n_before >= 1


class TestMixtureOpponentFactory:
    def test_draws_are_seeded_and_uniform_ish(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import FovLimitedEvader, GreedyEvader, ObstacleHuggingEvader

        members = [GreedyEvader, ObstacleHuggingEvader, FovLimitedEvader]
        f1 = MixtureOpponentFactory(members, seed=5)
        f2 = MixtureOpponentFactory(members, seed=5)
        kinds1 = [type(f1()).__name__ for _ in range(300)]
        kinds2 = [type(f2()).__name__ for _ in range(300)]
        assert kinds1 == kinds2
        counts = {k: kinds1.count(k) for k in set(kinds1)}
        assert set(counts) == {
            "GreedyEvader",
            "ObstacleHuggingEvader",
            "FovLimitedEvader",
        }
        assert all(60 <= c <= 140 for c in counts.values())
        assert f1.draws[:3] == f2.draws[:3]

    def test_works_as_a_training_opponent(self):
        from pe2.evade import MixtureOpponentFactory
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
        from pe2.structured import SearchMDP

        m = SearchMDP(
            CFG,
            MixtureOpponentFactory([GreedyEvader, ObstacleHuggingEvader], seed=1),
            hold=10,
        )
        m.reset()
        for _ in range(3):
            _obs, _, term, trunc, _ = m.step(np.array([1.0, 0.0], dtype=np.float32))
            if term or trunc:
                m.reset()
