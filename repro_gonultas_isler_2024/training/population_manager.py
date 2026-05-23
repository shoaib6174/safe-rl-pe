import copy
import random
import torch
import torch.nn as nn
from agents.maddpg import ActorNetwork


class PopulationManager:
    """Store actor network snapshots and sample random opponents on demand.

    Stores state dicts on CPU. Returns a fresh nn.Module on the target device
    per sample call. No gradient, eval mode.
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

    def add(self, actor_state: dict) -> None:
        """Add a snapshot. Deep copy to CPU. Evict oldest if at capacity."""
        self.actor_states.append(copy.deepcopy(actor_state))
        if len(self.actor_states) > self.max_size:
            self.actor_states.pop(0)

    def sample(self) -> nn.Module:
        """Uniform random sample. Return fresh nn.Module on device, no-grad, eval."""
        if len(self.actor_states) == 0:
            raise RuntimeError("Population is empty — cannot sample.")
        idx = random.randint(0, len(self.actor_states) - 1)
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
        """Save population state dicts to disk."""
        torch.save({"actor_states": self.actor_states}, path)

    def load(self, path: str) -> None:
        """Load population state dicts from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.actor_states = checkpoint["actor_states"]
