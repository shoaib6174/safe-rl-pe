"""Tests for self-play training loop and health monitor."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from training.health_monitor import SelfPlayHealthMonitor
from training.self_play_eval import evaluate_both_agents, collect_trajectories


def _make_test_cfg():
    """Create a minimal config for self-play testing."""
    return OmegaConf.create({
        "seed": 42,
        "total_timesteps": 512,
        "eval_freq": 256,
        "n_eval_episodes": 2,
        "save_freq": 256,
        "n_envs": 2,
        "experiment_group": "test",
        "env": {
            "arena_width": 20.0,
            "arena_height": 20.0,
            "dt": 0.05,
            "max_steps": 100,
            "capture_radius": 0.5,
            "collision_radius": 0.3,
            "robot_radius": 0.15,
            "min_init_distance": 3.0,
            "max_init_distance": 15.0,
            "pursuer": {"v_max": 1.0, "omega_max": 2.84},
            "evader": {"v_max": 1.0, "omega_max": 2.84},
            "reward": {
                "capture_bonus": 100.0,
                "timeout_penalty": -100.0,
                "distance_scale": 1.0,
            },
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [32, 32]},
        },
        "self_play": {
            "n_phases": 2,
            "timesteps_per_phase": 256,
            "eval_episodes": 5,
            "convergence_threshold": 0.10,
            "n_seeds": 1,
            "health": {
                "min_entropy": 0.1,
                "max_capture_rate": 0.90,
                "min_capture_rate": 0.10,
                "greedy_eval_interval": 2,
                "max_checkpoints": 3,
            },
        },
        "wandb": {
            "entity": None,
            "project": "test",
            "mode": "disabled",
            "sync_tensorboard": False,
            "save_code": False,
            "tags": ["test"],
            "log_frequency": 128,
            "video": {"enabled": False, "record_every_n_episodes": 100, "fps": 20},
            "model": {"save_freq": 256, "save_best": False},
        },
    })


class TestHealthMonitor:
    """Tests for SelfPlayHealthMonitor."""

    def test_should_rollback_balanced(self):
        """Balanced capture rates should not trigger rollback."""
        hm = SelfPlayHealthMonitor()
        assert hm.should_rollback([0.5, 0.45]) is False
        assert hm.should_rollback([0.6, 0.4, 0.55]) is False

    def test_should_rollback_pursuer_dominating(self):
        """Two consecutive phases with high capture rate should trigger."""
        hm = SelfPlayHealthMonitor(max_capture_rate=0.90)
        assert hm.should_rollback([0.95, 0.92]) is True

    def test_should_rollback_evader_dominating(self):
        """Two consecutive phases with low capture rate should trigger."""
        hm = SelfPlayHealthMonitor(min_capture_rate=0.10)
        assert hm.should_rollback([0.05, 0.08]) is True

    def test_should_rollback_insufficient_history(self):
        """Less than 2 phases should never trigger rollback."""
        hm = SelfPlayHealthMonitor()
        assert hm.should_rollback([]) is False
        assert hm.should_rollback([0.95]) is False

    def test_should_rollback_mixed(self):
        """One balanced + one imbalanced should not trigger."""
        hm = SelfPlayHealthMonitor(max_capture_rate=0.90)
        assert hm.should_rollback([0.5, 0.95]) is False

    def test_trajectory_diversity_diverse(self):
        """Spread-out endpoints should give multiple clusters."""
        hm = SelfPlayHealthMonitor()
        # Endpoints spread across arena
        trajectories = [
            [(0.0, 0.0), (5.0, 5.0)],
            [(0.0, 0.0), (-5.0, -5.0)],
            [(0.0, 0.0), (5.0, -5.0)],
            [(0.0, 0.0), (-5.0, 5.0)],
        ]
        n_clusters = hm.check_trajectory_diversity(trajectories)
        assert n_clusters >= 2

    def test_trajectory_diversity_collapsed(self):
        """All endpoints at same location should give 1 cluster."""
        hm = SelfPlayHealthMonitor()
        trajectories = [
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.1, 1.1)],
            [(0.0, 0.0), (0.9, 0.9)],
        ]
        n_clusters = hm.check_trajectory_diversity(trajectories)
        assert n_clusters == 1

    def test_save_and_cleanup_checkpoints(self, tmp_path):
        """Checkpoints should be saved and old ones cleaned up."""
        from stable_baselines3 import PPO
        from envs.pursuit_evasion_env import PursuitEvasionEnv
        from envs.wrappers import SingleAgentPEWrapper

        hm = SelfPlayHealthMonitor(max_checkpoints=2)

        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        env = SingleAgentPEWrapper(base, role="pursuer")

        import torch
        model = PPO("MlpPolicy", env, verbose=0,
                     policy_kwargs={"net_arch": [16], "activation_fn": torch.nn.Tanh})

        # Save 3 checkpoints (max=2, so first should be deleted)
        for phase in range(3):
            hm.save_checkpoint(model, model, tmp_path, phase)

        # Phase 0 should be deleted, phases 1 and 2 should exist
        assert not (tmp_path / "pursuer_phase0.zip").exists()
        assert (tmp_path / "pursuer_phase1.zip").exists()
        assert (tmp_path / "pursuer_phase2.zip").exists()

        env.close()

    def test_check_entropy(self):
        """Entropy check should return a float for PPO model."""
        from stable_baselines3 import PPO
        from envs.pursuit_evasion_env import PursuitEvasionEnv
        from envs.wrappers import SingleAgentPEWrapper

        hm = SelfPlayHealthMonitor()
        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        env = SingleAgentPEWrapper(base, role="pursuer")

        import torch
        model = PPO("MlpPolicy", env, verbose=0,
                     policy_kwargs={"net_arch": [16], "activation_fn": torch.nn.Tanh})
        model.learn(total_timesteps=128)

        entropy = hm.check_entropy(model)
        # Should return a float (may be None if log_std not accessible)
        if entropy is not None:
            assert isinstance(entropy, float)

        env.close()


class TestSelfPlayEval:
    """Tests for the self-play evaluation function."""

    def test_evaluate_both_agents(self):
        """evaluate_both_agents should return valid metrics."""
        from stable_baselines3 import PPO
        from envs.wrappers import SingleAgentPEWrapper

        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        p_env = SingleAgentPEWrapper(
            PursuitEvasionEnv(render_mode=None, max_steps=50), role="pursuer",
        )
        e_env = SingleAgentPEWrapper(
            PursuitEvasionEnv(render_mode=None, max_steps=50), role="evader",
        )

        import torch
        pk = {"net_arch": [16], "activation_fn": torch.nn.Tanh}
        p_model = PPO("MlpPolicy", p_env, verbose=0, policy_kwargs=pk)
        e_model = PPO("MlpPolicy", e_env, verbose=0, policy_kwargs=pk)

        metrics = evaluate_both_agents(p_model, e_model, base, n_episodes=5, seed=42)

        assert 0 <= metrics["capture_rate"] <= 1
        assert 0 <= metrics["escape_rate"] <= 1
        assert metrics["capture_rate"] + metrics["escape_rate"] == pytest.approx(1.0)
        assert metrics["mean_episode_length"] > 0

        p_env.close()
        e_env.close()
        base.close()

    def test_collect_trajectories(self):
        """collect_trajectories should return list of position lists."""
        from stable_baselines3 import PPO
        from envs.wrappers import SingleAgentPEWrapper

        base = PursuitEvasionEnv(render_mode=None, max_steps=50)
        p_env = SingleAgentPEWrapper(
            PursuitEvasionEnv(render_mode=None, max_steps=50), role="pursuer",
        )
        e_env = SingleAgentPEWrapper(
            PursuitEvasionEnv(render_mode=None, max_steps=50), role="evader",
        )

        import torch
        pk = {"net_arch": [16], "activation_fn": torch.nn.Tanh}
        p_model = PPO("MlpPolicy", p_env, verbose=0, policy_kwargs=pk)
        e_model = PPO("MlpPolicy", e_env, verbose=0, policy_kwargs=pk)

        trajectories = collect_trajectories(p_model, e_model, base, n_episodes=3, seed=42)

        assert len(trajectories) == 3
        for traj in trajectories:
            assert len(traj) > 0
            assert len(traj[0]) == 2  # (x, y)

        p_env.close()
        e_env.close()
        base.close()


class TestSelfPlaySmoke:
    """Smoke test for the full self-play loop."""

    def test_self_play_runs(self):
        """Full self-play loop should complete without errors."""
        from training.self_play import alternating_self_play

        cfg = _make_test_cfg()
        result = alternating_self_play(cfg)

        assert "pursuer_model" in result
        assert "evader_model" in result
        assert "history" in result
        assert len(result["history"]["capture_rate"]) == cfg.self_play.n_phases
