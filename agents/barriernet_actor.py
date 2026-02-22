"""BarrierNet actor: MLP policy with differentiable QP safety layer.

Architecture:
    observation -> MLP backbone -> nominal action u_nom
    (state, u_nom) -> DifferentiableVCPCBFQP -> safe action u*

The QP layer is differentiable, so gradients from the PPO loss flow
through the QP back to the MLP parameters. Unlike CBF-Beta (which
truncates the Beta distribution support), this produces a deterministic
safe action. For PPO exploration, learned Gaussian noise is added
BEFORE the QP layer.

Log-probability: We use log pi(u_nom | s) as an approximation to
log pi(u_safe | s). This is exact when the QP is inactive (u_safe == u_nom)
and introduces small bias when constraints are active. This is the same
approach used in the original BarrierNet paper (Xiao et al., T-RO 2023).

Reference: Xiao et al. 2023 (BarrierNet), Zhang & Yang 2025 (VCP-CBF).
"""

import math

import torch
import torch.nn as nn

from safety.differentiable_qp import DifferentiableVCPCBFQP


class BarrierNetActor(nn.Module):
    """BarrierNet actor: MLP backbone + differentiable QP safety layer.

    Args:
        obs_dim: Observation dimensionality (14 + 2*n_obstacle_obs).
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        n_constraints_max: Maximum CBF constraints (4 arena + n_obs + 1 collision).
        v_max: Maximum linear velocity.
        omega_max: Maximum angular velocity.
        w_v: QP weight on velocity deviation.
        w_omega: QP weight on angular velocity deviation.
        alpha: CBF class-K function parameter.
        d: VCP offset distance.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_constraints_max: int = 10,
        v_max: float = 1.0,
        omega_max: float = 2.84,
        w_v: float = 150.0,
        w_omega: float = 1.0,
        alpha: float = 1.0,
        d: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.v_max = v_max
        self.omega_max = omega_max

        # MLP backbone
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Nominal action head: outputs raw values, squashed via tanh
        self.action_mean = nn.Linear(hidden_dim, 2)  # [v, omega]

        # Learnable exploration noise (log std)
        # Initialize with smaller std for meaningful exploration:
        # v std=0.3 (log=-1.2), omega std=0.5 (log=-0.7)
        self.action_log_std = nn.Parameter(torch.tensor([-1.2, -0.7]))

        # Differentiable QP safety layer
        self.qp_layer = DifferentiableVCPCBFQP(
            n_constraints_max=n_constraints_max,
            v_min=0.0,
            v_max=v_max,
            omega_min=-omega_max,
            omega_max=omega_max,
            w_v=w_v,
            w_omega=w_omega,
            alpha=alpha,
            d=d,
        )

    def get_nominal_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get unconstrained nominal action from MLP.

        Args:
            obs: (B, obs_dim) observations.

        Returns:
            u_nom: (B, 2) nominal actions [v, omega] squashed to bounds.
        """
        features = self.backbone(obs)
        raw = self.action_mean(features)

        # Squash to action bounds using tanh
        v_nom = (torch.tanh(raw[:, 0]) + 1) / 2 * self.v_max      # [0, v_max]
        omega_nom = torch.tanh(raw[:, 1]) * self.omega_max          # [-omega_max, omega_max]

        return torch.stack([v_nom, omega_nom], dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        obstacles: list[dict] | None = None,
        opponent_states: torch.Tensor | None = None,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
        robot_radius: float = 0.15,
        r_min_separation: float = 0.35,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Forward pass: obs -> MLP -> nominal action -> QP -> safe action.

        Args:
            obs: (B, obs_dim) observations.
            states: (B, 3) robot states [x, y, theta] for CBF constraints.
            obstacles: List of obstacle dicts with 'x', 'y', 'radius'.
            opponent_states: (B, 3) opponent states, or None.
            arena_half_w: Half arena width.
            arena_half_h: Half arena height.
            robot_radius: Robot radius.
            r_min_separation: Minimum inter-robot separation.
            deterministic: If True, use mean action (no exploration noise).

        Returns:
            u_safe: (B, 2) safe actions.
            log_prob: (B,) log probability of nominal action (Approach A).
            entropy: scalar entropy of the Gaussian exploration distribution.
            info: Dict with diagnostic info.
        """
        B = obs.shape[0]
        device = obs.device

        # Step 1: Get nominal action from MLP
        u_nom_mean = self.get_nominal_action(obs)

        if deterministic:
            u_nom = u_nom_mean
            log_prob = torch.zeros(B, device=device)
        else:
            # Add exploration noise (Gaussian in action space)
            std = torch.exp(self.action_log_std).expand(B, -1)
            noise = torch.randn_like(u_nom_mean) * std
            u_nom = u_nom_mean + noise

            # Clamp to action bounds (pre-QP)
            u_nom = torch.stack([
                torch.clamp(u_nom[:, 0], 0.0, self.v_max),
                torch.clamp(u_nom[:, 1], -self.omega_max, self.omega_max),
            ], dim=-1)

            # Log probability of nominal action (Gaussian)
            # log N(u_nom | u_nom_mean, std) = -0.5 * sum((u-mean)^2/var + log(2*pi*var))
            var = std ** 2
            log_prob = -0.5 * (
                ((u_nom - u_nom_mean) ** 2 / var).sum(dim=-1)
                + torch.log(var).sum(dim=-1)
                + 2 * math.log(2 * math.pi)
            )

        # Step 2: Pass through differentiable QP safety layer
        u_safe, qp_info = self.qp_layer(
            u_nom, states,
            obstacles=obstacles,
            opponent_states=opponent_states,
            arena_half_w=arena_half_w,
            arena_half_h=arena_half_h,
            robot_radius=robot_radius,
            r_min_separation=r_min_separation,
        )

        # Entropy of exploration distribution
        std_detached = torch.exp(self.action_log_std)
        entropy = 0.5 * (1 + torch.log(2 * math.pi * std_detached ** 2)).sum()

        # Correction magnitude (for monitoring)
        with torch.no_grad():
            correction = (u_safe - u_nom).norm(dim=-1)

        info = {
            "u_nom": u_nom.detach(),
            "u_safe": u_safe.detach(),
            "u_nom_mean": u_nom_mean.detach(),
            "qp_correction": correction,
            "action_std": std_detached.detach(),
            "qp_feasible": qp_info["feasible"],
            "min_cbf_value": qp_info["min_cbf_value"],
        }

        return u_safe, log_prob, entropy, info


class BarrierNetCritic(nn.Module):
    """Standard MLP critic for value estimation (no QP layer needed).

    Args:
        obs_dim: Observation dimensionality.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            obs: (B, obs_dim) observations.

        Returns:
            value: (B,) estimated state values.
        """
        return self.net(obs).squeeze(-1)
