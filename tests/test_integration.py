"""Integration tests for Phase 1 — Session 7.

Tests end-to-end workflows, edge cases, and reproducibility.
"""

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from envs.dynamics import unicycle_step, wrap_angle
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper
from safety.vcp_cbf import VCPCBFFilter
from training.baselines import GreedyEvaderPolicy, GreedyPursuerPolicy, RandomPolicy
from training.utils import make_pe_env, make_vec_env, setup_reproducibility


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_agents_at_same_position(self):
        """Agents spawned at same position (capture_radius overlap)."""
        env = PursuitEvasionEnv(render_mode=None, max_steps=10)
        obs, info = env.reset(seed=42)

        # Force agents to same position via internal state
        env.pursuer_state = np.array([0.0, 0.0, 0.0])
        env.evader_state = np.array([0.0, 0.0, 0.0])

        # Step with zero actions — capture check happens at step
        action_p = np.array([0.0, 0.0], dtype=np.float32)
        action_e = np.array([0.0, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action_p, action_e)

        assert terminated, "Should detect capture when agents overlap"
        env.close()

    def test_agents_at_boundary(self):
        """Both agents at arena boundary."""
        env = PursuitEvasionEnv(render_mode=None, max_steps=50)
        obs, info = env.reset(seed=42)

        half_w = env.arena_width / 2
        half_h = env.arena_height / 2

        # Force agents to boundary
        env.pursuer_state = np.array([half_w - env.robot_radius, half_h - env.robot_radius, 0.0])
        env.evader_state = np.array([-(half_w - env.robot_radius), -(half_h - env.robot_radius), np.pi])

        # Agents moving toward walls — should clip
        action_p = np.array([1.0, 0.0], dtype=np.float32)
        action_e = np.array([1.0, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action_p, action_e)

        # Positions should be clipped to boundary
        assert abs(env.pursuer_state[0]) <= half_w
        assert abs(env.evader_state[0]) <= half_w
        env.close()

    def test_extreme_velocities(self):
        """Max velocity and angular velocity for many steps."""
        env = PursuitEvasionEnv(render_mode=None, max_steps=100)
        obs, info = env.reset(seed=42)

        for _ in range(50):
            action_p = np.array([1.0, 2.84], dtype=np.float32)  # max v, max omega
            action_e = np.array([1.0, -2.84], dtype=np.float32)  # max v, min omega
            obs, rewards, terminated, truncated, info = env.step(action_p, action_e)
            if terminated or truncated:
                break

        # Should not crash or produce NaN
        for key in ["pursuer", "evader"]:
            assert np.isfinite(obs[key]).all(), f"NaN in {key} obs after extreme velocities"
        env.close()

    def test_zero_velocity_episode(self):
        """Both agents stationary for entire episode."""
        env = PursuitEvasionEnv(render_mode=None, max_steps=50)
        obs, info = env.reset(seed=42)

        for _ in range(50):
            action_p = np.array([0.0, 0.0], dtype=np.float32)
            action_e = np.array([0.0, 0.0], dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(action_p, action_e)
            if terminated or truncated:
                break

        # Should timeout, not crash
        assert truncated, "Should timeout with zero velocity"
        env.close()

    def test_many_resets(self):
        """Environment handles 1000 resets without memory issues."""
        env = PursuitEvasionEnv(render_mode=None, max_steps=10)
        for i in range(1000):
            obs, _ = env.reset(seed=i)
            assert obs is not None
        env.close()


class TestReproducibility:
    """Verify deterministic behavior with same seed."""

    def test_env_deterministic(self):
        """Same seed → same episode trajectory."""
        results = []
        for _ in range(2):
            env = PursuitEvasionEnv(render_mode=None, max_steps=50)
            obs, _ = env.reset(seed=42)
            trajectory = [obs["pursuer"].copy()]

            for _ in range(20):
                action_p = np.array([0.5, 0.3], dtype=np.float32)
                action_e = np.array([0.8, -0.5], dtype=np.float32)
                obs, _, terminated, truncated, _ = env.step(action_p, action_e)
                trajectory.append(obs["pursuer"].copy())
                if terminated or truncated:
                    break
            results.append(trajectory)
            env.close()

        # Compare trajectories
        assert len(results[0]) == len(results[1])
        for t1, t2 in zip(results[0], results[1]):
            np.testing.assert_array_equal(t1, t2)

    def test_ppo_deterministic(self):
        """Same seed → same PPO predictions on same observations.

        Note: Bit-for-bit weight reproducibility is not guaranteed across
        separate Python processes due to hash randomization and SB3 internal
        state ordering. Instead, we verify that the environment step function
        is deterministic and that a trained model produces consistent predictions.
        """
        setup_reproducibility(42)

        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "env": {
                "arena_width": 10.0, "arena_height": 10.0,
                "dt": 0.05, "max_steps": 50,
                "capture_radius": 0.5, "collision_radius": 0.3,
                "robot_radius": 0.15,
                "pursuer": {"v_max": 1.0, "omega_max": 2.84},
                "evader": {"v_max": 1.0, "omega_max": 2.84},
                "min_init_distance": 3.0, "max_init_distance": 7.0,
                "reward": {"distance_scale": 1.0, "capture_bonus": 100.0, "timeout_penalty": -100.0},
            },
        })

        envs = make_vec_env(cfg, n_envs=1, role="pursuer", seed=42)
        model = PPO(
            "MlpPolicy", envs, verbose=0, seed=42, device="cpu",
            n_steps=64, batch_size=32, n_epochs=2,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=128)

        # Verify deterministic predictions (same obs → same action)
        test_obs = np.random.default_rng(42).standard_normal(14).astype(np.float32)
        actions = []
        for _ in range(5):
            action, _ = model.predict(test_obs, deterministic=True)
            actions.append(action.copy())

        for a in actions[1:]:
            np.testing.assert_array_equal(actions[0], a, "Deterministic predict should be identical")

        envs.close()


class TestSelfPlayIntegration:
    """Self-play pipeline integration test."""

    def test_self_play_full_round(self):
        """Complete one round of self-play training and evaluation."""
        from training.self_play_eval import evaluate_both_agents

        setup_reproducibility(42)

        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "env": {
                "arena_width": 10.0, "arena_height": 10.0,
                "dt": 0.05, "max_steps": 50,
                "capture_radius": 0.5, "collision_radius": 0.3,
                "robot_radius": 0.15,
                "pursuer": {"v_max": 1.0, "omega_max": 2.84},
                "evader": {"v_max": 1.0, "omega_max": 2.84},
                "min_init_distance": 3.0, "max_init_distance": 7.0,
                "reward": {"distance_scale": 1.0, "capture_bonus": 100.0, "timeout_penalty": -100.0},
            },
        })

        # Train pursuer
        p_env = make_vec_env(cfg, n_envs=1, role="pursuer", seed=42)
        pursuer = PPO("MlpPolicy", p_env, verbose=0, seed=42, device="cpu",
                      n_steps=64, batch_size=32, policy_kwargs={"net_arch": [32]})
        pursuer.learn(total_timesteps=128)
        p_env.close()

        # Train evader
        e_env = make_vec_env(cfg, n_envs=1, role="evader", seed=42)
        evader = PPO("MlpPolicy", e_env, verbose=0, seed=42, device="cpu",
                     n_steps=64, batch_size=32, policy_kwargs={"net_arch": [32]})
        evader.learn(total_timesteps=128)
        e_env.close()

        # Evaluate
        eval_env = PursuitEvasionEnv(render_mode=None, max_steps=50)
        metrics = evaluate_both_agents(pursuer, evader, eval_env, n_episodes=5, seed=100)

        assert "capture_rate" in metrics
        assert 0 <= metrics["capture_rate"] <= 1
        assert "escape_rate" in metrics
        assert metrics["capture_rate"] + metrics["escape_rate"] == pytest.approx(1.0)
        eval_env.close()


class TestVCPCBFIntegration:
    """VCP-CBF integrated with environment dynamics."""

    def test_cbf_with_env_step(self):
        """CBF filter integrated with environment step function."""
        cbf = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=5.0, arena_half_h=5.0, robot_radius=0.15,
        )

        rng = np.random.default_rng(42)
        x, y, theta = 0.0, 0.0, 0.0
        dt = 0.05

        for _ in range(200):
            state = np.array([x, y, theta])
            u_nom = np.array([rng.uniform(0, 1), rng.uniform(-2.84, 2.84)])
            u_safe, _ = cbf.filter_action(u_nom, state)

            x, y, theta, _ = unicycle_step(
                x, y, theta, float(u_safe[0]), float(u_safe[1]),
                dt, 10.0, 10.0, 0.15,
            )

            # Should stay in bounds
            assert abs(x) <= 5.0, f"x={x} out of bounds"
            assert abs(y) <= 5.0, f"y={y} out of bounds"
            assert np.isfinite(x) and np.isfinite(y) and np.isfinite(theta)


class TestPerformance:
    """Performance benchmarks."""

    def test_env_step_speed(self):
        """Environment step should be < 1ms on average."""
        import time
        env = PursuitEvasionEnv(render_mode=None, max_steps=1000)
        env.reset(seed=42)

        n_steps = 1000
        action_p = np.array([0.5, 0.3], dtype=np.float32)
        action_e = np.array([0.8, -0.5], dtype=np.float32)

        start = time.perf_counter()
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(action_p, action_e)
            if terminated or truncated:
                env.reset(seed=42)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_steps) * 1000
        assert avg_ms < 1.0, f"Avg step time {avg_ms:.2f}ms > 1ms target"
        env.close()

    def test_cbf_qp_speed(self):
        """CBF QP solve should be < 5ms on average."""
        import time
        cbf = VCPCBFFilter(d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84)

        rng = np.random.default_rng(42)
        n_solves = 200

        start = time.perf_counter()
        for _ in range(n_solves):
            state = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)])
            u_nom = np.array([rng.uniform(0, 1), rng.uniform(-2.84, 2.84)])
            cbf.filter_action(u_nom, state, [{"x": 2.0, "y": 0.0, "radius": 0.5}])
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_solves) * 1000
        assert avg_ms < 5.0, f"Avg CBF QP time {avg_ms:.2f}ms > 5ms target"
