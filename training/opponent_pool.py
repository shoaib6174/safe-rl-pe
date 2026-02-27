"""Opponent pool for diverse self-play training.

Stores historical opponent checkpoints and samples from them to prevent
overfitting to a single frozen opponent. Each sub-environment can get a
different opponent from the pool, increasing training diversity.

Design:
  - Stores checkpoint directory paths (not models) for memory efficiency.
  - Lazy-loads and caches PPO models on first use.
  - Optionally includes a random policy (None sentinel) in the pool.
  - Eviction strategy: "reservoir" (default) preserves uniform coverage
    across full training history; "fifo" evicts oldest first.
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
        eviction_strategy: "reservoir" for reservoir sampling (uniform coverage
            across full history) or "fifo" for oldest-first eviction.
    """

    def __init__(
        self,
        max_size: int = 10,
        include_random: bool = True,
        eviction_strategy: str = "reservoir",
    ):
        if eviction_strategy not in ("reservoir", "fifo"):
            raise ValueError(
                f"eviction_strategy must be 'reservoir' or 'fifo', "
                f"got '{eviction_strategy}'"
            )
        self.max_size = max_size
        self.include_random = include_random
        self.eviction_strategy = eviction_strategy
        self.checkpoints: list[str] = []  # checkpoint dir paths
        self._cache: dict[str, PPO] = {}  # path -> loaded PPO model
        self._total_added: int = 0  # total checkpoints ever offered (for reservoir)

    def add_checkpoint(self, ckpt_path: str) -> None:
        """Add a milestone checkpoint path to the pool.

        When the pool is full:
          - "fifo": evicts the oldest checkpoint.
          - "reservoir": uses reservoir sampling (probability max_size / total_added)
            to decide whether to replace a random existing entry. This preserves
            uniform coverage across the full training history.

        Args:
            ckpt_path: Path to checkpoint directory containing ppo.zip.
        """
        # Normalize to string for consistent dict keys
        ckpt_path = str(ckpt_path)

        # Avoid duplicates
        if ckpt_path in self.checkpoints:
            return

        self._total_added += 1

        if len(self.checkpoints) < self.max_size:
            # Pool not full yet — always add
            self.checkpoints.append(ckpt_path)
        elif self.eviction_strategy == "fifo":
            # FIFO: evict oldest
            old = self.checkpoints.pop(0)
            self._cache.pop(old, None)
            self.checkpoints.append(ckpt_path)
        else:
            # Reservoir sampling: accept with probability max_size / total_added
            if random.random() < self.max_size / self._total_added:
                idx = random.randrange(self.max_size)
                evicted = self.checkpoints[idx]
                self._cache.pop(evicted, None)
                self.checkpoints[idx] = ckpt_path

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

    def sample_pfsp(self, n: int, bias_strength: float = 0.7) -> list[str | None]:
        """Sample n opponents with PFSP-lite bias toward older/weaker opponents.

        Older checkpoints (earlier in training) are more likely to be weaker,
        so biasing toward them gives a collapsing agent beatable opponents
        for recovery. Uses exponential decay weighting: older = higher weight.

        Args:
            n: Number of opponents to sample.
            bias_strength: How strongly to bias toward older opponents.
                0.0 = uniform (same as sample()), 1.0 = strong bias.

        Returns:
            List of checkpoint paths or None values.
        """
        candidates = list(self.checkpoints)
        if self.include_random:
            candidates.append(None)

        if not candidates:
            return [None] * n

        if len(candidates) <= 1 or bias_strength <= 0:
            return [random.choice(candidates) for _ in range(n)]

        # Exponential decay weights: older (lower index) gets higher weight
        # w_i = exp(-bias_strength * i / (len-1))
        import math
        num = len(candidates)
        weights = []
        for i in range(num):
            # i=0 is oldest checkpoint, i=num-1 is newest
            # None (random) is last — always gets some weight
            w = math.exp(-bias_strength * i / max(num - 1, 1))
            weights.append(w)

        # Normalize to probabilities
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Weighted sampling
        return random.choices(candidates, weights=probs, k=n)

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
            f"strategy={self.eviction_strategy}, "
            f"total_added={self._total_added}, "
            f"include_random={self.include_random}, "
            f"cached={len(self._cache)})"
        )
