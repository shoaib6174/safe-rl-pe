"""Opponent pool for diverse self-play training.

Stores historical opponent checkpoints and samples from them to prevent
overfitting to a single frozen opponent. Each sub-environment can get a
different opponent from the pool, increasing training diversity.

Design:
  - Stores checkpoint directory paths (not models) for memory efficiency.
  - Lazy-loads and caches PPO models on first use.
  - Optionally includes a random policy (None sentinel) in the pool.
  - FIFO eviction when pool exceeds max_size.
"""

from __future__ import annotations

import random
from pathlib import Path

from stable_baselines3 import PPO


class OpponentPool:
    """Pool of historical opponent models for diverse self-play.

    Each pool instance manages checkpoints for one role (pursuer or evader).
    During training, the trainer samples from the opposing role's pool to
    provide diverse frozen opponents.

    Args:
        max_size: Maximum number of checkpoints to retain.
        include_random: Whether to include random policy (None) as a candidate.
    """

    def __init__(self, max_size: int = 10, include_random: bool = True):
        self.max_size = max_size
        self.include_random = include_random
        self.checkpoints: list[str] = []  # checkpoint dir paths
        self._cache: dict[str, PPO] = {}  # path -> loaded PPO model

    def add_checkpoint(self, ckpt_path: str) -> None:
        """Add a milestone checkpoint path to the pool.

        If the pool is full, the oldest checkpoint is evicted (FIFO).

        Args:
            ckpt_path: Path to checkpoint directory containing ppo.zip.
        """
        # Normalize to string for consistent dict keys
        ckpt_path = str(ckpt_path)

        # Avoid duplicates
        if ckpt_path in self.checkpoints:
            return

        self.checkpoints.append(ckpt_path)

        # FIFO eviction
        if len(self.checkpoints) > self.max_size:
            old = self.checkpoints.pop(0)
            self._cache.pop(old, None)

    def sample(self, n: int) -> list[str | None]:
        """Sample n opponents from the pool.

        Each sample is either a checkpoint path (str) or None (random policy).

        Args:
            n: Number of opponents to sample (typically n_envs).

        Returns:
            List of checkpoint paths or None values.
        """
        candidates = list(self.checkpoints)
        if self.include_random:
            candidates.append(None)

        if not candidates:
            # Pool is empty and no random — return all None
            return [None] * n

        return [random.choice(candidates) for _ in range(n)]

    def get_model(self, ckpt_path: str, device: str = "cpu") -> PPO:
        """Load and cache a PPO model from a checkpoint path.

        Args:
            ckpt_path: Path to checkpoint directory containing ppo.zip.
            device: Device to load model onto.

        Returns:
            Loaded PPO model.
        """
        ckpt_path = str(ckpt_path)
        if ckpt_path not in self._cache:
            ppo_path = Path(ckpt_path) / "ppo.zip"
            self._cache[ckpt_path] = PPO.load(str(ppo_path), device=device)
        return self._cache[ckpt_path]

    def clear_cache(self) -> None:
        """Clear all cached models to free memory."""
        self._cache.clear()

    def __len__(self) -> int:
        """Number of checkpoints in the pool (excludes random)."""
        return len(self.checkpoints)

    def __repr__(self) -> str:
        return (
            f"OpponentPool(size={len(self.checkpoints)}/{self.max_size}, "
            f"include_random={self.include_random}, "
            f"cached={len(self._cache)})"
        )
