import copy
import math
import random
import torch
import torch.nn as nn
from agents.maddpg import ActorNetwork


class PopulationManager:
    """Store actor network snapshots and sample random opponents on demand.

    Stores state dicts on CPU. Returns a fresh nn.Module on the target device
    per sample call. No gradient, eval mode.

    Supports:
    - Uniform random sampling (default)
    - Recency-biased sampling (exponential decay by age)
    """

    def __init__(
        self,
        max_size: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.actor_states: list[dict] = []
        self.episodes_added: list[int] = []

    def add(self, actor_state: dict, episode: int = 0) -> None:
        """Add a snapshot. Deep copy to CPU. Evict oldest if at capacity."""
        self.actor_states.append(copy.deepcopy(actor_state))
        self.episodes_added.append(episode)
        if len(self.actor_states) > self.max_size:
            self.actor_states.pop(0)
            self.episodes_added.pop(0)

    def sample(self) -> nn.Module:
        """Uniform random sample. Return fresh nn.Module on device, no-grad, eval."""
        if len(self.actor_states) == 0:
            raise RuntimeError("Population is empty — cannot sample.")
        idx = random.randint(0, len(self.actor_states) - 1)
        return self._load_actor(idx)

    def sample_recency(self, current_episode: int, lambda_decay: float = 0.15) -> nn.Module:
        """Recency-biased sample using exponential decay.

        Weights = exp(-lambda_decay * age) where age = current_episode - episode_added.
        Higher lambda_decay = stronger bias toward recent opponents.
        lambda_decay = 0 gives uniform sampling.
        """
        if len(self.actor_states) == 0:
            raise RuntimeError("Population is empty — cannot sample.")

        n = len(self.actor_states)
        if n == 1 or lambda_decay <= 0:
            return self.sample()

        ages = [current_episode - ep for ep in self.episodes_added]
        weights = [math.exp(-lambda_decay * max(0, age)) for age in ages]
        total = sum(weights)
        probs = [w / total for w in weights]

        idx = random.choices(range(n), weights=probs, k=1)[0]
        return self._load_actor(idx)

    def _load_actor(self, idx: int) -> nn.Module:
        """Load actor state at index into a fresh module."""
        state = self.actor_states[idx]
        actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        actor.load_state_dict(state)
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        return actor

    def __len__(self) -> int:
        return len(self.actor_states)

    def save(self, path: str) -> None:
        """Save population state dicts and episode metadata to disk."""
        torch.save({
            "actor_states": self.actor_states,
            "episodes_added": self.episodes_added,
        }, path)

    def load(self, path: str) -> None:
        """Load population state dicts and episode metadata from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.actor_states = checkpoint["actor_states"]
        self.episodes_added = checkpoint.get("episodes_added", [0] * len(self.actor_states))
