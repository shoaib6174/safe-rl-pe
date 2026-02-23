"""Tests for Session 7: Opponent Pool for Self-Play Diversity.

Tests cover:
  - OpponentPool: add, sample, eviction, caching, repr
  - Pool with random policy included/excluded
  - Integration: AMSDRLSelfPlay accepts opponent_pool_size param
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─── OpponentPool Unit Tests ───


class TestOpponentPool:
    """Tests for the OpponentPool class."""

    def test_empty_pool_sample_returns_none(self):
        """Empty pool with include_random=True samples None (random)."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=True)
        samples = pool.sample(4)
        assert len(samples) == 4
        assert all(s is None for s in samples)

    def test_empty_pool_no_random_returns_none(self):
        """Empty pool with include_random=False also returns None."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=False)
        samples = pool.sample(4)
        assert len(samples) == 4
        assert all(s is None for s in samples)

    def test_add_checkpoint(self):
        """Adding checkpoints increases pool size."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)
        assert len(pool) == 0

        pool.add_checkpoint("/path/to/ckpt1")
        assert len(pool) == 1

        pool.add_checkpoint("/path/to/ckpt2")
        assert len(pool) == 2

    def test_add_duplicate_ignored(self):
        """Adding the same checkpoint path twice doesn't create duplicates."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)
        pool.add_checkpoint("/path/to/ckpt1")
        pool.add_checkpoint("/path/to/ckpt1")
        assert len(pool) == 1

    def test_fifo_eviction(self):
        """Oldest checkpoint is evicted when pool exceeds max_size."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=3, include_random=False)
        pool.add_checkpoint("/ckpt/1")
        pool.add_checkpoint("/ckpt/2")
        pool.add_checkpoint("/ckpt/3")
        assert len(pool) == 3
        assert pool.checkpoints == ["/ckpt/1", "/ckpt/2", "/ckpt/3"]

        # Adding 4th should evict /ckpt/1
        pool.add_checkpoint("/ckpt/4")
        assert len(pool) == 3
        assert pool.checkpoints == ["/ckpt/2", "/ckpt/3", "/ckpt/4"]
        assert "/ckpt/1" not in pool.checkpoints

    def test_eviction_clears_cache(self):
        """Evicted checkpoints are removed from cache."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=2)
        pool.add_checkpoint("/ckpt/1")
        pool.add_checkpoint("/ckpt/2")

        # Manually populate cache
        pool._cache["/ckpt/1"] = MagicMock()
        pool._cache["/ckpt/2"] = MagicMock()

        # Adding 3rd evicts /ckpt/1
        pool.add_checkpoint("/ckpt/3")
        assert "/ckpt/1" not in pool._cache
        assert "/ckpt/2" in pool._cache

    def test_sample_with_random(self):
        """Sampling with include_random=True can return None."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=True)
        pool.add_checkpoint("/ckpt/1")

        # With 1 checkpoint + random, sample many times to check both appear
        np.random.seed(42)
        samples = pool.sample(100)
        assert any(s is None for s in samples), "Random (None) should appear in samples"
        assert any(s == "/ckpt/1" for s in samples), "Checkpoint should appear in samples"

    def test_sample_without_random(self):
        """Sampling with include_random=False never returns None."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=False)
        pool.add_checkpoint("/ckpt/1")
        pool.add_checkpoint("/ckpt/2")

        samples = pool.sample(50)
        assert all(s is not None for s in samples)
        assert all(s in ["/ckpt/1", "/ckpt/2"] for s in samples)

    def test_sample_correct_count(self):
        """sample(n) returns exactly n items."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)
        pool.add_checkpoint("/ckpt/1")

        assert len(pool.sample(1)) == 1
        assert len(pool.sample(4)) == 4
        assert len(pool.sample(16)) == 16

    def test_get_model_loads_and_caches(self):
        """get_model loads PPO from checkpoint and caches it."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt1"
            ckpt_dir.mkdir()

            # Create a minimal PPO model and save it
            from stable_baselines3 import PPO

            model = PPO("MlpPolicy", "CartPole-v1", n_steps=32)
            model.save(str(ckpt_dir / "ppo.zip"))

            pool.add_checkpoint(str(ckpt_dir))

            # Load via pool
            loaded = pool.get_model(str(ckpt_dir))
            assert loaded is not None

            # Second call should return cached instance
            loaded2 = pool.get_model(str(ckpt_dir))
            assert loaded2 is loaded  # Same object (cached)

    def test_clear_cache(self):
        """clear_cache empties the model cache."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)
        pool._cache["a"] = MagicMock()
        pool._cache["b"] = MagicMock()
        assert len(pool._cache) == 2

        pool.clear_cache()
        assert len(pool._cache) == 0

    def test_repr(self):
        """__repr__ includes size and config info."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=10, include_random=True)
        pool.add_checkpoint("/ckpt/1")
        r = repr(pool)
        assert "1/10" in r
        assert "include_random=True" in r

    def test_len(self):
        """__len__ returns number of checkpoints (excludes random)."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=True)
        assert len(pool) == 0
        pool.add_checkpoint("/ckpt/1")
        assert len(pool) == 1


# ─── AMSDRLSelfPlay Integration Tests ───


class TestAMSDRLOpponentPoolIntegration:
    """Tests for opponent pool integration in AMSDRLSelfPlay."""

    def test_amsdrl_accepts_pool_size_zero(self):
        """AMSDRLSelfPlay with opponent_pool_size=0 creates no pools."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=0,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                verbose=0,
            )
            assert sp.pursuer_pool is None
            assert sp.evader_pool is None

    def test_amsdrl_accepts_pool_size_positive(self):
        """AMSDRLSelfPlay with opponent_pool_size>0 creates pools."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=5,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                verbose=0,
            )
            assert sp.pursuer_pool is not None
            assert sp.evader_pool is not None
            assert sp.pursuer_pool.max_size == 5
            assert sp.evader_pool.max_size == 5

    def test_wrap_opponent_model_none_returns_none(self):
        """_wrap_opponent_model with None model returns None (random)."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=0,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                verbose=0,
            )

            from envs.pursuit_evasion_env import PursuitEvasionEnv

            base_env = PursuitEvasionEnv()
            result = sp._wrap_opponent_model(None, "evader", base_env)
            assert result is None

    def test_wrap_opponent_model_with_model_full_obs(self):
        """_wrap_opponent_model wraps model for full-obs mode."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=0,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                fixed_speed=False,
                verbose=0,
            )

            from envs.pursuit_evasion_env import PursuitEvasionEnv

            base_env = PursuitEvasionEnv()
            mock_model = MagicMock()

            result = sp._wrap_opponent_model(mock_model, "evader", base_env)
            # Full-obs, no fixed speed: should return model directly
            assert result is mock_model

    def test_wrap_opponent_model_with_model_fixed_speed(self):
        """_wrap_opponent_model wraps model with FixedSpeedModelAdapter for fixed_speed."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=0,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                fixed_speed=True,
                verbose=0,
            )

            from envs.pursuit_evasion_env import PursuitEvasionEnv
            from envs.wrappers import FixedSpeedModelAdapter

            base_env = PursuitEvasionEnv()
            mock_model = MagicMock()

            result = sp._wrap_opponent_model(mock_model, "evader", base_env)
            assert isinstance(result, FixedSpeedModelAdapter)
