"""Tests for Opponent Pool for Self-Play Diversity.

Tests cover:
  - OpponentPool: add, sample, eviction (FIFO + reservoir), caching, repr
  - Pool with random policy included/excluded
  - Reservoir sampling: default strategy, size bounds, uniform coverage, cache eviction
  - Integration: AMSDRLSelfPlay accepts opponent_pool_size param
"""

from __future__ import annotations

import tempfile
from collections import Counter
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

    def test_fifo_eviction_strategy(self):
        """Oldest checkpoint is evicted when pool exceeds max_size (FIFO mode)."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=3, include_random=False, eviction_strategy="fifo")
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
        """Evicted checkpoints are removed from cache (FIFO mode)."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=2, eviction_strategy="fifo")
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
        """__repr__ includes size, strategy, and config info."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=10, include_random=True)
        pool.add_checkpoint("/ckpt/1")
        r = repr(pool)
        assert "1/10" in r
        assert "include_random=True" in r
        assert "strategy=reservoir" in r
        assert "total_added=1" in r

    def test_len(self):
        """__len__ returns number of checkpoints (excludes random)."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=True)
        assert len(pool) == 0
        pool.add_checkpoint("/ckpt/1")
        assert len(pool) == 1

    def test_invalid_eviction_strategy(self):
        """Invalid eviction strategy raises ValueError."""
        from training.opponent_pool import OpponentPool

        with pytest.raises(ValueError, match="eviction_strategy"):
            OpponentPool(max_size=5, eviction_strategy="invalid")


class TestReservoirSampling:
    """Tests for reservoir sampling eviction strategy."""

    def test_default_strategy_is_reservoir(self):
        """Default eviction strategy is reservoir."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5)
        assert pool.eviction_strategy == "reservoir"

    def test_reservoir_never_exceeds_max_size(self):
        """Reservoir sampling never lets pool exceed max_size."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=False,
                            eviction_strategy="reservoir")
        for i in range(200):
            pool.add_checkpoint(f"/ckpt/{i}")
            assert len(pool) <= 5

        assert len(pool) == 5
        assert pool._total_added == 200

    def test_reservoir_uniform_coverage(self):
        """Reservoir sampling produces approximately uniform coverage.

        Over 5000 trials, each of 100 checkpoints added to a pool of size 10
        should have roughly equal probability of being in the final pool.
        We check that early, middle, and late checkpoints all appear.
        """
        import random as rand
        from training.opponent_pool import OpponentPool

        rand.seed(42)

        n_trials = 5000
        n_checkpoints = 100
        pool_size = 10

        # Count how often each checkpoint index ends up in the final pool
        presence_count: Counter = Counter()

        for _ in range(n_trials):
            pool = OpponentPool(max_size=pool_size, include_random=False,
                                eviction_strategy="reservoir")
            for i in range(n_checkpoints):
                pool.add_checkpoint(f"/ckpt/{i}")
            for ckpt in pool.checkpoints:
                idx = int(ckpt.split("/")[-1])
                presence_count[idx] += 1

        # Expected frequency: pool_size / n_checkpoints * n_trials = 10/100 * 5000 = 500
        expected = pool_size / n_checkpoints * n_trials

        # Check early checkpoints (0-9) are represented
        early_total = sum(presence_count[i] for i in range(10))
        assert early_total > 0, "Early checkpoints should appear in pool"

        # Check middle checkpoints (45-54) are represented
        mid_total = sum(presence_count[i] for i in range(45, 55))
        assert mid_total > 0, "Middle checkpoints should appear in pool"

        # Check late checkpoints (90-99) are represented
        late_total = sum(presence_count[i] for i in range(90, 100))
        assert late_total > 0, "Late checkpoints should appear in pool"

        # Each group of 10 should have roughly expected_group = 10 * expected = 5000
        # Allow generous tolerance (within 50% of expected)
        expected_group = 10 * expected
        for group_name, total in [("early", early_total), ("mid", mid_total),
                                   ("late", late_total)]:
            assert total > expected_group * 0.3, (
                f"{group_name} group frequency {total} is too low "
                f"(expected ~{expected_group})"
            )
            assert total < expected_group * 1.7, (
                f"{group_name} group frequency {total} is too high "
                f"(expected ~{expected_group})"
            )

    def test_reservoir_eviction_clears_cache(self):
        """Reservoir eviction clears cached models for evicted checkpoints."""
        import random as rand
        from training.opponent_pool import OpponentPool

        rand.seed(0)

        pool = OpponentPool(max_size=3, include_random=False,
                            eviction_strategy="reservoir")
        # Fill pool
        for i in range(3):
            pool.add_checkpoint(f"/ckpt/{i}")
            pool._cache[f"/ckpt/{i}"] = MagicMock()

        assert len(pool._cache) == 3

        # Add many more to trigger evictions
        for i in range(3, 50):
            pool.add_checkpoint(f"/ckpt/{i}")

        # Cache should only contain entries still in the pool
        for cached_path in list(pool._cache.keys()):
            assert cached_path in pool.checkpoints, (
                f"Cached path {cached_path} not in pool checkpoints"
            )

    def test_total_added_tracks_all_offers(self):
        """_total_added counts all non-duplicate add_checkpoint calls."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=3, eviction_strategy="reservoir")
        for i in range(10):
            pool.add_checkpoint(f"/ckpt/{i}")
        assert pool._total_added == 10

        # Duplicate should not increment
        pool.add_checkpoint("/ckpt/0")
        # /ckpt/0 might or might not be in pool (reservoir), but if it is,
        # the duplicate check triggers first, so _total_added stays at 10
        # If /ckpt/0 was evicted, it's treated as new, incrementing to 11
        # Either way, the pool size should be <= 3
        assert len(pool) <= 3


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

    def test_amsdrl_pools_use_reservoir_strategy(self):
        """AMSDRLSelfPlay creates pools with reservoir eviction strategy."""
        from training.amsdrl import AMSDRLSelfPlay

        with tempfile.TemporaryDirectory() as tmp:
            sp = AMSDRLSelfPlay(
                output_dir=tmp,
                opponent_pool_size=10,
                max_phases=1,
                timesteps_per_phase=64,
                cold_start_timesteps=64,
                n_envs=1,
                full_obs=True,
                verbose=0,
            )
            assert sp.pursuer_pool.eviction_strategy == "reservoir"
            assert sp.evader_pool.eviction_strategy == "reservoir"

    def test_amsdrl_convergence_consecutive_default(self):
        """AMSDRLSelfPlay defaults to convergence_consecutive=5."""
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
            assert sp.convergence_consecutive == 5

    def test_amsdrl_convergence_consecutive_custom(self):
        """AMSDRLSelfPlay accepts custom convergence_consecutive."""
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
                convergence_consecutive=10,
            )
            assert sp.convergence_consecutive == 10

    def test_amsdrl_adaptive_ratio_defaults(self):
        """AMSDRLSelfPlay defaults to adaptive_ratio_threshold=0 (disabled)."""
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
            assert sp.adaptive_ratio_threshold == 0.0
            assert sp.adaptive_boost_phases == 20

    def test_amsdrl_lr_dampen_defaults(self):
        """AMSDRLSelfPlay defaults to lr_dampen_threshold=0 (disabled)."""
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
            assert sp.lr_dampen_threshold == 0.0

    def test_amsdrl_accepts_adaptive_params(self):
        """AMSDRLSelfPlay accepts custom adaptive ratio + LR dampen params."""
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
                adaptive_ratio_threshold=0.3,
                adaptive_boost_phases=15,
                lr_dampen_threshold=0.25,
            )
            assert sp.adaptive_ratio_threshold == 0.3
            assert sp.adaptive_boost_phases == 15
            assert sp.lr_dampen_threshold == 0.25

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


# ─── Collapse Rollback + PFSP Tests ───


class TestCollapseRollbackDefaults:
    """Tests for collapse rollback and PFSP-lite parameter defaults."""

    def test_collapse_rollback_disabled_by_default(self):
        """Collapse rollback is disabled by default (threshold=0.0)."""
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
            assert sp.collapse_threshold == 0.0
            assert sp.collapse_streak_limit == 3

    def test_pfsp_disabled_by_default(self):
        """PFSP-lite is disabled by default."""
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
            assert sp.pfsp_enabled is False

    def test_amsdrl_accepts_collapse_and_pfsp_params(self):
        """AMSDRLSelfPlay accepts custom collapse + PFSP params."""
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
                collapse_threshold=0.05,
                collapse_streak_limit=5,
                pfsp_enabled=True,
            )
            assert sp.collapse_threshold == 0.05
            assert sp.collapse_streak_limit == 5
            assert sp.pfsp_enabled is True


class TestPFSPSampling:
    """Tests for PFSP-lite sampling in OpponentPool."""

    def test_pfsp_sample_returns_correct_count(self):
        """sample_pfsp returns requested number of samples."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=False)
        for i in range(5):
            pool.add_checkpoint(f"/tmp/ckpt_{i}")

        samples = pool.sample_pfsp(4)
        assert len(samples) == 4

    def test_pfsp_sample_biases_toward_older(self):
        """sample_pfsp with high bias_strength favors older (lower-index) entries."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=10, include_random=False)
        for i in range(10):
            pool.add_checkpoint(f"/tmp/ckpt_{i}")

        # With strong bias, older checkpoints should be sampled more often
        counts = Counter()
        n_samples = 5000
        for _ in range(n_samples):
            s = pool.sample_pfsp(1, bias_strength=2.0)[0]
            counts[s] += 1

        # Oldest checkpoint should appear more than newest
        oldest_count = counts.get("/tmp/ckpt_0", 0)
        newest_count = counts.get("/tmp/ckpt_9", 0)
        assert oldest_count > newest_count, (
            f"Expected oldest ({oldest_count}) > newest ({newest_count})"
        )

    def test_pfsp_sample_zero_bias_is_uniform(self):
        """sample_pfsp with bias_strength=0 behaves like uniform."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=3, include_random=False)
        for i in range(3):
            pool.add_checkpoint(f"/tmp/ckpt_{i}")

        counts = Counter()
        n_samples = 3000
        for _ in range(n_samples):
            s = pool.sample_pfsp(1, bias_strength=0.0)[0]
            counts[s] += 1

        # Each should be roughly equal (within 30% of expected)
        expected = n_samples / 3
        for ckpt, count in counts.items():
            assert abs(count - expected) < expected * 0.3, (
                f"Expected ~{expected:.0f} for {ckpt}, got {count}"
            )

    def test_pfsp_sample_empty_pool_returns_none(self):
        """sample_pfsp on empty pool with include_random returns None values."""
        from training.opponent_pool import OpponentPool

        pool = OpponentPool(max_size=5, include_random=True)
        samples = pool.sample_pfsp(3)
        assert len(samples) == 3
        assert all(s is None for s in samples)


# ─── Warm-Seed Self-Play Tests ───


class TestWarmSeedSelfPlay:
    """Tests for warm-seeded self-play initialization."""

    def test_amsdrl_accepts_init_model_paths(self):
        """AMSDRLSelfPlay accepts init_pursuer_path and init_evader_path."""
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
                init_pursuer_path="/some/pursuer.zip",
                init_evader_path="/some/evader.zip",
            )
            assert sp.init_pursuer_path == "/some/pursuer.zip"
            assert sp.init_evader_path == "/some/evader.zip"

    def test_amsdrl_init_model_paths_default_none(self):
        """AMSDRLSelfPlay defaults init model paths to None."""
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
            assert sp.init_pursuer_path is None
            assert sp.init_evader_path is None
