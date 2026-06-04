import torch
import torch.nn as nn
import numpy as np


class GreedyActor(nn.Module):
    """
    Non-learnable actor that implements a greedy policy.

    For pursuer: moves toward opponent (chase).
    For evader: moves away from opponent (flee).

    Works with frame-stacked, normalized observations from PursuitEvasionEnv.
    """

    def __init__(
        self,
        arena_size: float = 16.0,
        is_pursuer: bool = True,
        frame_stack: int = 2,
        obs_dim: int = 11,
        action_dim: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.arena_half = arena_size / 2.0
        self.is_pursuer = is_pursuer
        self.frame_stack = frame_stack
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._device = device

        # Dummy parameter so PyTorch tracks device / state_dict correctly
        self.dummy = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim * frame_stack) normalized observations
        Returns:
            actions: (batch, action_dim) in [-1, 1]
        """
        batch_size = obs.shape[0]

        # Extract the LAST frame from the stacked observation
        last_frame = obs[:, (self.frame_stack - 1) * self.obs_dim :]

        # Denormalize positions (obs[0:2] = own_pos, obs[4:6] = opp_pos)
        own_x = last_frame[:, 0] * self.arena_half
        own_y = last_frame[:, 1] * self.arena_half
        opp_x = last_frame[:, 4] * self.arena_half
        opp_y = last_frame[:, 5] * self.arena_half

        # Greedy direction
        if self.is_pursuer:
            # Pursuer: move TOWARD opponent
            dx = opp_x - own_x
            dy = opp_y - own_y
        else:
            # Evader: move AWAY from opponent
            dx = own_x - opp_x
            dy = own_y - opp_y

        # Compute steering angle
        angle = torch.atan2(dy, dx)

        # Map to action space: steering ∈ [-1, 1], throttle ∈ [0, 1]
        # angle ∈ [-π, π]; scale by 0.5 to keep mostly within [-1, 1]
        # (env will clip anyway)
        steering = angle * 0.5
        throttle = torch.ones_like(angle)

        return torch.stack([steering, throttle], dim=1)

    def to(self, device):
        self._device = device
        return super().to(device)
