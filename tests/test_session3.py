"""Tests for Phase 3 Session 3: AMS-DRL Self-Play.

Tests cover:
  - PartialObsOpponentAdapter
  - NavigationEnv
  - CheckpointManager
  - SB3 Callbacks (entropy, health monitor, baseline eval)
  - AMSDRLSelfPlay (smoke)
  - NE tools
  - Scripted baseline policies
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


# ─── PartialObsOpponentAdapter Tests ───


class TestPartialObsOpponentAdapter:
    """Tests for the partial-obs opponent adapter."""

    def _make_adapter(self):
        from envs.opponent_adapter import PartialObsOpponentAdapter

        base_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        base_env.reset(seed=42)

        # Create a mock model with predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.array([0.5, 0.1], dtype=np.float32),
            None,
        )

        adapter = PartialObsOpponentAdapter(
            model=mock_model,
            role="evader",
            base_env=base_env,
            history_length=10,
        )
        return adapter, base_env, mock_model

    def test_predict_returns_action(self):
        """Adapter predict returns (action, None) like SB3."""
        adapter, base_env, _ = self._make_adapter()
        action, state = adapter.predict(np.zeros(14), deterministic=True)
        assert action.shape == (2,)
        assert state is None

    def test_predict_calls_model_with_dict_obs(self):
        """Adapter processes obs to Dict format before calling model."""
        adapter, base_env, mock_model = self._make_adapter()
        adapter.predict(np.zeros(14))

        # Check that model.predict was called with a dict
        call_args = mock_model.predict.call_args
        obs_dict = call_args[0][0]
        assert isinstance(obs_dict, dict)
        assert "obs_history" in obs_dict
        assert "lidar" in obs_dict
        assert "state" in obs_dict

    def test_obs_dict_shapes(self):
        """Adapter produces correct observation shapes."""
        adapter, base_env, mock_model = self._make_adapter()

        # Capture the obs dict passed to model.predict
        captured_obs = {}
        def capture_predict(obs, **kwargs):
            captured_obs.update(obs)
            return np.array([0.5, 0.1], dtype=np.float32), None
        mock_model.predict.side_effect = capture_predict

        adapter.predict(np.zeros(14))

        assert captured_obs["obs_history"].shape == (10, 43)
        assert captured_obs["lidar"].shape == (1, 36)
        assert captured_obs["state"].shape == (7,)

    def test_reset_clears_buffer(self):
        """Adapter reset clears observation history."""
        adapter, base_env, _ = self._make_adapter()

        # Make some predictions to fill buffer
        for _ in range(5):
            adapter.predict(np.zeros(14))
        assert adapter.steps_since_seen >= 0

        # Reset should clear everything
        adapter.reset()
        assert adapter.steps_since_seen == 0
        assert adapter.last_known_opp_pos is None

    def test_works_with_single_agent_wrapper(self):
        """Adapter integrates with SingleAgentPEWrapper.set_opponent()."""
        from envs.opponent_adapter import PartialObsOpponentAdapter

        base_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        single_env = SingleAgentPEWrapper(base_env, role="pursuer")

        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.array([0.5, 0.0], dtype=np.float32),
            None,
        )

        adapter = PartialObsOpponentAdapter(
            model=mock_model,
            role="evader",
            base_env=base_env,
            deterministic=True,
        )

        single_env.set_opponent(adapter)
        obs, info = single_env.reset(seed=42)
        # Should not crash — adapter handles obs conversion
        obs2, reward, term, trunc, info = single_env.step(np.array([0.5, 0.0]))
        assert obs2.shape[0] == 14  # Full-state obs dim

    def test_reset_called_on_episode_boundary(self):
        """SingleAgentPEWrapper calls adapter.reset() on env reset."""
        from envs.opponent_adapter import PartialObsOpponentAdapter

        base_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        single_env = SingleAgentPEWrapper(base_env, role="pursuer")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.5, 0.0]), None)

        adapter = PartialObsOpponentAdapter(
            model=mock_model, role="evader", base_env=base_env,
        )
        single_env.set_opponent(adapter)

        single_env.reset(seed=42)
        # Make some steps to advance the buffer
        for _ in range(3):
            single_env.step(np.array([0.5, 0.0]))

        # Reset should clear the adapter's buffer
        single_env.reset(seed=43)
        assert adapter.steps_since_seen == 0


# ─── NavigationEnv Tests ───


class TestNavigationEnv:
    """Tests for the cold-start navigation environment."""

    def _make_nav_env(self, include_flee=True):
        from envs.navigation_env import NavigationEnv

        base_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0,
                                     max_steps=1200)
        nav_env = NavigationEnv(
            base_env,
            role="evader",
            include_flee_phase=include_flee,
            max_steps=50,
            goal_radius=0.5,
        )
        return nav_env

    def test_obs_space_matches_partial_obs(self):
        """NavigationEnv observation space matches PartialObsWrapper format."""
        nav_env = self._make_nav_env()
        obs_space = nav_env.observation_space

        assert "obs_history" in obs_space.spaces
        assert "lidar" in obs_space.spaces
        assert "state" in obs_space.spaces
        assert obs_space["obs_history"].shape == (10, 43)
        assert obs_space["lidar"].shape == (1, 36)
        assert obs_space["state"].shape == (7,)

    def test_reset_returns_dict_obs(self):
        """Reset returns Dict observation with correct keys."""
        nav_env = self._make_nav_env(include_flee=False)
        obs, info = nav_env.reset(seed=42)

        assert isinstance(obs, dict)
        assert obs["obs_history"].shape == (10, 43)
        assert obs["lidar"].shape == (1, 36)
        assert obs["state"].shape == (7,)

    def test_step_returns_correct_format(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        nav_env = self._make_nav_env(include_flee=False)
        obs, _ = nav_env.reset(seed=42)

        action = nav_env.action_space.sample()
        obs2, reward, terminated, truncated, info = nav_env.step(action)

        assert isinstance(obs2, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "goals_reached" in info

    def test_goal_reaching_mode(self):
        """Goal-reaching mode provides distance-based reward."""
        nav_env = self._make_nav_env(include_flee=False)
        obs, info = nav_env.reset(seed=42)

        assert info["mode"] == "goal_reaching"
        assert nav_env.goal is not None

        # Goal should be within arena bounds
        assert 1.0 <= nav_env.goal[0] <= 19.0
        assert 1.0 <= nav_env.goal[1] <= 19.0

    def test_truncation_at_max_steps(self):
        """Episode truncates at max_steps."""
        nav_env = self._make_nav_env(include_flee=False)
        obs, _ = nav_env.reset(seed=42)

        for _ in range(50):  # max_steps = 50
            action = nav_env.action_space.sample()
            obs, reward, terminated, truncated, info = nav_env.step(action)
            if terminated or truncated:
                break

        assert truncated  # Should truncate at step 50

    def test_flee_mode_has_pursuer(self):
        """Flee mode generates pursuer actions."""
        from envs.navigation_env import NavigationEnv

        base_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        # Force flee mode by disabling random mode selection
        nav_env = NavigationEnv(base_env, role="evader", include_flee_phase=True,
                                max_steps=10)

        # Run several resets to get a flee mode episode
        for _ in range(20):
            obs, info = nav_env.reset(seed=None)
            if info["mode"] == "flee":
                break
        # Just verify it runs without error in flee mode
        if info["mode"] == "flee":
            action = nav_env.action_space.sample()
            obs2, reward, _, _, _ = nav_env.step(action)
            assert isinstance(reward, float)

    def test_goal_sampled_away_from_obstacles(self):
        """Goal sampling avoids obstacles."""
        from envs.navigation_env import NavigationEnv

        base_env = PursuitEvasionEnv(
            arena_width=20.0, arena_height=20.0, n_obstacles=3,
        )
        nav_env = NavigationEnv(base_env, role="evader", include_flee_phase=False)
        nav_env.reset(seed=42)

        goal = nav_env.goal
        for obs in base_env.obstacles:
            dist = np.hypot(goal[0] - obs["x"], goal[1] - obs["y"])
            assert dist >= obs["radius"]  # Goal not inside obstacle


# ─── CheckpointManager Tests ───


class TestCheckpointManager:
    """Tests for checkpoint management."""

    def _make_manager(self):
        from training.checkpoint_manager import CheckpointManager

        tmpdir = tempfile.mkdtemp()
        mgr = CheckpointManager(checkpoint_dir=tmpdir, max_rolling=3)
        return mgr, tmpdir

    def _make_mock_model(self):
        """Create a minimal mock SB3 model."""
        model = MagicMock()
        model.save = MagicMock()
        return model

    def _make_mock_encoder(self):
        """Create a minimal encoder-like object."""
        encoder = torch.nn.Linear(10, 5)
        return encoder

    def test_save_rolling_creates_directory(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        mgr.save_rolling(model=model, step=1000)

        ckpt_dir = Path(tmpdir) / "rolling_1000"
        assert ckpt_dir.exists()
        model.save.assert_called_once()
        assert (ckpt_dir / "meta.json").exists()

        shutil.rmtree(tmpdir)

    def test_rolling_cleanup(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        # Save 4 checkpoints (max_rolling=3)
        for step in [1000, 2000, 3000, 4000]:
            mgr.save_rolling(model=model, step=step)

        # Only 3 should remain
        assert len(mgr.rolling) == 3
        # Oldest (1000) should be removed
        assert not (Path(tmpdir) / "rolling_1000").exists()
        assert (Path(tmpdir) / "rolling_2000").exists()

        shutil.rmtree(tmpdir)

    def test_save_with_encoder(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()
        encoder = self._make_mock_encoder()

        mgr.save_rolling(model=model, encoder=encoder, step=1000)

        assert (Path(tmpdir) / "rolling_1000" / "encoder.pt").exists()

        shutil.rmtree(tmpdir)

    def test_save_milestone(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        mgr.save_milestone(model=model, phase=3, role="pursuer")

        milestone_dir = Path(tmpdir) / "milestone_phase3_pursuer"
        assert milestone_dir.exists()
        assert (milestone_dir / "meta.json").exists()

        with open(milestone_dir / "meta.json") as f:
            meta = json.load(f)
        assert meta["phase"] == 3
        assert meta["role"] == "pursuer"

        shutil.rmtree(tmpdir)

    def test_save_best_only_improves(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        assert mgr.save_best(model=model, metric_value=0.5) is True
        assert mgr.save_best(model=model, metric_value=0.3) is False  # Not better
        assert mgr.save_best(model=model, metric_value=0.7) is True   # Better

        shutil.rmtree(tmpdir)

    def test_rollback_raises_if_insufficient(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        mgr.save_rolling(model=model, step=1000)

        with pytest.raises(ValueError, match="Cannot rollback"):
            mgr.perform_rollback(type(model), rollback_steps=3)

        shutil.rmtree(tmpdir)

    def test_meta_json_content(self):
        mgr, tmpdir = self._make_manager()
        model = self._make_mock_model()

        mgr.save_rolling(
            model=model, step=5000,
            meta={"phase": 2, "capture_rate": 0.45},
        )

        with open(Path(tmpdir) / "rolling_5000" / "meta.json") as f:
            meta = json.load(f)

        assert meta["step"] == 5000
        assert meta["phase"] == 2
        assert meta["capture_rate"] == 0.45

        shutil.rmtree(tmpdir)


# ─── Callback Tests ───


class TestEntropyMonitorCallback:
    """Tests for the entropy monitor callback."""

    def test_callback_instantiates(self):
        from training.selfplay_callbacks import EntropyMonitorCallback

        cb = EntropyMonitorCallback(check_freq=100, log_std_floor=-2.0)
        assert cb.check_freq == 100
        assert cb.log_std_floor == -2.0
        assert cb.enable_clamp is True


class TestSelfPlayHealthMonitorCallback:
    """Tests for the self-play health monitor callback."""

    def test_instantiates_with_defaults(self):
        from training.selfplay_callbacks import SelfPlayHealthMonitorCallback
        from training.checkpoint_manager import CheckpointManager

        tmpdir = tempfile.mkdtemp()
        ckpt_mgr = CheckpointManager(tmpdir)

        cb = SelfPlayHealthMonitorCallback(checkpoint_manager=ckpt_mgr)
        assert cb.entropy_collapse == -2.0
        assert cb.capture_collapse == 0.02
        assert cb.capture_domination == 0.98
        assert cb.rollback_count == 0

        shutil.rmtree(tmpdir)

    def test_reset_phase(self):
        from training.selfplay_callbacks import SelfPlayHealthMonitorCallback
        from training.checkpoint_manager import CheckpointManager

        tmpdir = tempfile.mkdtemp()
        ckpt_mgr = CheckpointManager(tmpdir)

        cb = SelfPlayHealthMonitorCallback(checkpoint_manager=ckpt_mgr)
        cb.capture_history.extend([1.0, 0.0, 1.0])
        assert len(cb.capture_history) == 3

        cb.reset_phase()
        assert len(cb.capture_history) == 0

        shutil.rmtree(tmpdir)


class TestFixedBaselineEvalCallback:
    """Tests for the baseline evaluation callback."""

    def test_instantiates(self):
        from training.selfplay_callbacks import FixedBaselineEvalCallback

        eval_env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        cb = FixedBaselineEvalCallback(
            eval_env=eval_env,
            baselines={"random": None},
            role="pursuer",
        )
        assert cb.elo["training_agent"] == 1200.0
        assert cb.elo["random"] == 1200.0
        eval_env.close()


# ─── Scripted Baseline Tests ───


class TestScriptedBaselines:
    """Tests for scripted baseline policies."""

    def test_pure_pursuit_moves_toward_evader(self):
        from training.selfplay_callbacks import pure_pursuit_policy

        env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        env.reset(seed=42)

        action = pure_pursuit_policy(env)
        assert action.shape == (2,)
        assert action[0] == env.pursuer_v_max  # Full speed
        assert -env.pursuer_omega_max <= action[1] <= env.pursuer_omega_max
        env.close()

    def test_flee_away_moves_from_pursuer(self):
        from training.selfplay_callbacks import flee_away_policy

        env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        env.reset(seed=42)

        action = flee_away_policy(env)
        assert action.shape == (2,)
        assert action[0] == env.evader_v_max  # Full speed
        env.close()

    def test_flee_to_corner_valid_action(self):
        from training.selfplay_callbacks import flee_to_corner_policy

        env = PursuitEvasionEnv(arena_width=20.0, arena_height=20.0)
        env.reset(seed=42)

        action = flee_to_corner_policy(env)
        assert action.shape == (2,)
        assert action[0] == env.evader_v_max
        env.close()


# ─── NE Tools Tests ───


class TestNETools:
    """Tests for Nash Equilibrium analysis tools."""

    def test_compute_ne_gap(self):
        from training.ne_tools import compute_ne_gap

        history = [
            {"capture_rate": 0.8, "escape_rate": 0.2},
            {"capture_rate": 0.6, "escape_rate": 0.4},
            {"capture_rate": 0.5, "escape_rate": 0.5},
        ]
        gaps = compute_ne_gap(history)
        assert len(gaps) == 3
        assert abs(gaps[0] - 0.6) < 1e-6
        assert abs(gaps[1] - 0.2) < 1e-6
        assert abs(gaps[2] - 0.0) < 1e-6

    def test_analyze_convergence_converged(self):
        from training.ne_tools import analyze_convergence

        history = [
            {"capture_rate": 0.8, "escape_rate": 0.2},
            {"capture_rate": 0.6, "escape_rate": 0.4},
            {"capture_rate": 0.52, "escape_rate": 0.48},
        ]
        result = analyze_convergence(history, eta=0.10)
        assert result["converged"] is True
        assert result["final_ne_gap"] < 0.10

    def test_analyze_convergence_not_converged(self):
        from training.ne_tools import analyze_convergence

        history = [
            {"capture_rate": 0.8, "escape_rate": 0.2},
            {"capture_rate": 0.7, "escape_rate": 0.3},
        ]
        result = analyze_convergence(history, eta=0.10)
        assert result["converged"] is False

    def test_analyze_empty_history(self):
        from training.ne_tools import analyze_convergence

        result = analyze_convergence([], eta=0.10)
        assert result["converged"] is False

    def test_plot_ne_convergence_creates_file(self):
        from training.ne_tools import plot_ne_convergence

        history = [
            {"capture_rate": 0.8, "escape_rate": 0.2, "phase": "S1", "role": "pursuer"},
            {"capture_rate": 0.6, "escape_rate": 0.4, "phase": "S2", "role": "evader"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_ne_convergence(history, save_path=str(save_path))
            assert save_path.exists()
            assert save_path.stat().st_size > 0


# ─── AMSDRLSelfPlay Smoke Tests ───


class TestAMSDRLSelfPlaySmoke:
    """Smoke tests for AMS-DRL (very short runs, just checks it doesn't crash)."""

    def test_cold_start_creates_models(self):
        """Cold start creates both pursuer and evader models."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmpdir:
            amsdrl = AMSDRLSelfPlay(
                output_dir=tmpdir,
                max_phases=0,  # Skip alternating phases
                cold_start_timesteps=256,  # Minimal
                n_envs=1,
                use_dcbf=False,
                encoder_type="mlp",  # Fastest encoder
                seed=42,
                device="cpu",
            )

            amsdrl._cold_start()
            assert amsdrl.evader_model is not None
            assert amsdrl.pursuer_model is not None

    def test_evaluate_head_to_head(self):
        """Head-to-head evaluation produces valid metrics."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmpdir:
            amsdrl = AMSDRLSelfPlay(
                output_dir=tmpdir,
                cold_start_timesteps=256,
                n_envs=1,
                use_dcbf=False,
                encoder_type="mlp",
                eval_episodes=5,
                seed=42,
                device="cpu",
            )

            amsdrl._cold_start()
            metrics = amsdrl._evaluate()

            assert "capture_rate" in metrics
            assert "escape_rate" in metrics
            assert 0.0 <= metrics["capture_rate"] <= 1.0
            assert 0.0 <= metrics["escape_rate"] <= 1.0
            assert abs(metrics["capture_rate"] + metrics["escape_rate"] - 1.0) < 1e-6

    def test_make_partial_obs_env(self):
        """Environment factory creates correct wrapper stack."""
        from training.amsdrl import _make_partial_obs_env
        from envs.partial_obs_wrapper import PartialObsWrapper

        env, base_env = _make_partial_obs_env(
            role="pursuer", use_dcbf=False, n_obstacles=0,
        )

        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "obs_history" in obs
        assert "lidar" in obs
        assert "state" in obs

        env.close()

    def test_make_partial_obs_env_with_dcbf(self):
        """Environment factory applies DCBF for pursuer."""
        from training.amsdrl import _make_partial_obs_env
        from envs.dcbf_action_wrapper import DCBFActionWrapper

        env, base_env = _make_partial_obs_env(
            role="pursuer", use_dcbf=True, gamma=0.2,
        )
        assert isinstance(env, DCBFActionWrapper)
        env.close()

    def test_make_vec_env(self):
        """Vectorized env factory creates correct number of envs."""
        from training.amsdrl import _make_vec_env

        vec_env, base_envs = _make_vec_env(
            role="pursuer", n_envs=2, use_dcbf=False,
        )
        assert len(base_envs) == 2
        assert vec_env.num_envs == 2

        obs = vec_env.reset()
        # Dict obs with batch dim
        assert "obs_history" in obs
        assert obs["obs_history"].shape[0] == 2  # batch of 2

        vec_env.close()


# ─── Integration: SingleAgentPEWrapper + PartialObsOpponentAdapter ───


class TestSingleAgentWithPartialObsOpponent:
    """Integration test: full step loop with partial-obs opponent."""

    def test_full_episode_runs(self):
        """Run a full episode with partial-obs opponent adapter."""
        from envs.opponent_adapter import PartialObsOpponentAdapter
        from envs.partial_obs_wrapper import PartialObsWrapper

        base_env = PursuitEvasionEnv(arena_width=10.0, arena_height=10.0, max_steps=50)
        single_env = SingleAgentPEWrapper(base_env, role="pursuer")
        partial_env = PartialObsWrapper(single_env, role="pursuer")

        # Mock evader model
        mock_evader = MagicMock()
        mock_evader.predict.return_value = (
            np.array([0.5, 0.0], dtype=np.float32), None,
        )

        adapter = PartialObsOpponentAdapter(
            model=mock_evader, role="evader", base_env=base_env,
        )
        single_env.set_opponent(adapter)

        obs, info = partial_env.reset(seed=42)
        done = False
        steps = 0

        while not done and steps < 50:
            action = partial_env.action_space.sample()
            obs, reward, terminated, truncated, info = partial_env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0  # Should have taken at least one step
        partial_env.close()
