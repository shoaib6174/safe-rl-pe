"""Differentiable QP layer for VCP-CBF safety constraints.

Wraps qpth's QPFunction to provide a differentiable safety filter
that takes batched nominal actions and states, applies VCP-CBF constraints,
and outputs safe actions with correct gradients flowing through the
QP solution via implicit differentiation of KKT conditions.

This is the core of the BarrierNet architecture (Xiao et al., T-RO 2023)
adapted for our VCP-CBF pursuit-evasion problem.

QP formulation:
    min_{u}  (u - u_nom)^T W (u - u_nom)
    s.t.     -a_v_i * v - a_omega_i * omega <= alpha * h_i   (CBF constraints)
             -v <= -v_min                                     (lower bound v)
              v <=  v_max                                     (upper bound v)
             -omega <= -omega_min                             (lower bound omega)
              omega <=  omega_max                             (upper bound omega)

where W = diag(w_v, w_omega) encodes anisotropic cost (w_v >> w_omega
to prefer steering over braking).

The constraint format matches safety/vcp_cbf.py:
    a_v * v + a_omega * omega + alpha * h >= 0

Reference: Zhang & Yang 2025 (VCP-CBF), Xiao et al. 2023 (BarrierNet).
"""

import torch
import torch.nn as nn
import numpy as np


class DifferentiableVCPCBFQP(nn.Module):
    """Differentiable QP layer for VCP-CBF safety constraints using qpth.

    Solves batched QPs where each sample has its own constraint set derived
    from the robot's state and environment. Gradients flow through the QP
    solution via implicit differentiation of KKT conditions.

    The QP is:
        min_{u}  (u - u_nom)^T W (u - u_nom)
        s.t.     G @ u <= h

    where G and h encode VCP-CBF constraints and control bounds.

    Args:
        n_constraints_max: Maximum number of CBF constraints per sample
            (excluding the 4 control bound constraints which are always added).
        v_min: Minimum linear velocity.
        v_max: Maximum linear velocity.
        omega_min: Minimum angular velocity.
        omega_max: Maximum angular velocity.
        w_v: Weight on linear velocity deviation (higher = prefer maintaining v).
        w_omega: Weight on angular velocity deviation.
        alpha: CBF class-K function parameter.
        d: VCP offset distance.
    """

    def __init__(
        self,
        n_constraints_max: int = 10,
        v_min: float = 0.0,
        v_max: float = 1.0,
        omega_min: float = -2.84,
        omega_max: float = 2.84,
        w_v: float = 150.0,
        w_omega: float = 1.0,
        alpha: float = 1.0,
        d: float = 0.1,
    ):
        super().__init__()
        self.n_constraints_max = n_constraints_max
        self.v_min = v_min
        self.v_max = v_max
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.w_v = w_v
        self.w_omega = w_omega
        self.alpha = alpha
        self.d = d

        # Total inequality constraints: n_cbf + 4 control bounds
        self.n_total = n_constraints_max + 4

    def build_constraint_matrices(
        self,
        states: torch.Tensor,
        obstacles: list[dict] | None = None,
        opponent_states: torch.Tensor | None = None,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
        robot_radius: float = 0.15,
        r_min_separation: float = 0.35,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build batched constraint matrices G and h for the QP.

        Constructs the constraint matrices matching the existing VCP-CBF
        formulation in safety/vcp_cbf.py. Each constraint has the form:
            a_v * v + a_omega * omega + alpha * h >= 0
        which is converted to qpth standard form:
            -a_v * v - a_omega * omega <= alpha * h
            i.e., G_row = [-a_v, -a_omega], h_row = alpha * h

        Args:
            states: (B, 3) tensor of robot states [x, y, theta].
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
            opponent_states: (B, 3) tensor of opponent states, or None.
            arena_half_w: Half arena width.
            arena_half_h: Half arena height.
            robot_radius: Robot radius (added to obstacle radii).
            r_min_separation: Minimum inter-robot separation distance.

        Returns:
            G: (B, n_total, 2) constraint matrix.
            h: (B, n_total) constraint vector (RHS of Gu <= h).
            n_active: (B,) number of active CBF constraints per sample.
        """
        B = states.shape[0]
        device = states.device
        dtype = states.dtype

        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        d = self.d

        # VCP position
        qx = x + d * cos_t
        qy = y + d * sin_t

        G_rows = []
        h_rows = []

        # --- Arena boundary constraints (4) ---
        x_lim = arena_half_w - robot_radius
        y_lim = arena_half_h - robot_radius

        # Right wall: h = x_lim - qx, a_v = -cos_t, a_omega = d*sin_t
        h_right = x_lim - qx
        G_rows.append(torch.stack([cos_t, -d * sin_t], dim=-1))  # -[-cos_t, d*sin_t]
        h_rows.append(self.alpha * h_right)

        # Left wall: h = qx + x_lim, a_v = cos_t, a_omega = -d*sin_t
        h_left = qx + x_lim
        G_rows.append(torch.stack([-cos_t, d * sin_t], dim=-1))  # -[cos_t, -d*sin_t]
        h_rows.append(self.alpha * h_left)

        # Top wall: h = y_lim - qy, a_v = -sin_t, a_omega = -d*cos_t
        h_top = y_lim - qy
        G_rows.append(torch.stack([sin_t, d * cos_t], dim=-1))  # -[-sin_t, -d*cos_t]
        h_rows.append(self.alpha * h_top)

        # Bottom wall: h = qy + y_lim, a_v = sin_t, a_omega = d*cos_t
        h_bot = qy + y_lim
        G_rows.append(torch.stack([-sin_t, -d * cos_t], dim=-1))  # -[sin_t, d*cos_t]
        h_rows.append(self.alpha * h_bot)

        n_active_cbf = 4  # arena constraints always active

        # --- Obstacle constraints ---
        if obstacles:
            for obs in obstacles:
                ox = obs["x"]
                oy = obs["y"]
                effective_r = obs["radius"] + robot_radius

                dx = qx - ox
                dy = qy - oy
                h_obs = dx ** 2 + dy ** 2 - effective_r ** 2

                # a_v = 2*(dx*cos_t + dy*sin_t)
                # a_omega = 2*(-dx*d*sin_t + dy*d*cos_t)
                a_v = 2.0 * (dx * cos_t + dy * sin_t)
                a_omega = 2.0 * (-dx * d * sin_t + dy * d * cos_t)

                G_rows.append(torch.stack([-a_v, -a_omega], dim=-1))
                h_rows.append(self.alpha * h_obs)
                n_active_cbf += 1

        # --- Inter-robot collision constraint ---
        if opponent_states is not None:
            x_o = opponent_states[:, 0]
            y_o = opponent_states[:, 1]
            theta_o = opponent_states[:, 2]

            qx_o = x_o + d * torch.cos(theta_o)
            qy_o = y_o + d * torch.sin(theta_o)

            dx_c = qx - qx_o
            dy_c = qy - qy_o
            h_coll = dx_c ** 2 + dy_c ** 2 - r_min_separation ** 2

            a_v_c = 2.0 * (dx_c * cos_t + dy_c * sin_t)
            a_omega_c = 2.0 * (-dx_c * d * sin_t + dy_c * d * cos_t)

            G_rows.append(torch.stack([-a_v_c, -a_omega_c], dim=-1))
            h_rows.append(self.alpha * h_coll)
            n_active_cbf += 1

        # --- Pad to n_constraints_max with inactive constraints ---
        while len(G_rows) < self.n_constraints_max:
            G_rows.append(torch.zeros(B, 2, device=device, dtype=dtype))
            h_rows.append(torch.full((B,), 1e6, device=device, dtype=dtype))

        # Truncate if we exceeded max (shouldn't happen with proper sizing)
        G_rows = G_rows[:self.n_constraints_max]
        h_rows = h_rows[:self.n_constraints_max]

        # --- Control bound constraints (always 4) ---
        # -v <= -v_min
        G_rows.append(torch.tensor([[-1.0, 0.0]], device=device, dtype=dtype).expand(B, -1))
        h_rows.append(torch.full((B,), -self.v_min, device=device, dtype=dtype))
        # v <= v_max
        G_rows.append(torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).expand(B, -1))
        h_rows.append(torch.full((B,), self.v_max, device=device, dtype=dtype))
        # -omega <= -omega_min
        G_rows.append(torch.tensor([[0.0, -1.0]], device=device, dtype=dtype).expand(B, -1))
        h_rows.append(torch.full((B,), -self.omega_min, device=device, dtype=dtype))
        # omega <= omega_max
        G_rows.append(torch.tensor([[0.0, 1.0]], device=device, dtype=dtype).expand(B, -1))
        h_rows.append(torch.full((B,), self.omega_max, device=device, dtype=dtype))

        G = torch.stack(G_rows, dim=1)  # (B, n_total, 2)
        h = torch.stack(h_rows, dim=1)  # (B, n_total)

        n_active = torch.full((B,), n_active_cbf, device=device, dtype=torch.long)

        return G, h, n_active

    def forward(
        self,
        u_nominal: torch.Tensor,
        states: torch.Tensor,
        obstacles: list[dict] | None = None,
        opponent_states: torch.Tensor | None = None,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
        robot_radius: float = 0.15,
        r_min_separation: float = 0.35,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass: solve batched QP for safe actions.

        Args:
            u_nominal: (B, 2) nominal actions [v, omega].
            states: (B, 3) robot states [x, y, theta].
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' keys.
            opponent_states: (B, 3) opponent states, or None.
            arena_half_w: Half arena width.
            arena_half_h: Half arena height.
            robot_radius: Robot radius.
            r_min_separation: Minimum inter-robot separation.

        Returns:
            u_safe: (B, 2) safe actions with gradients.
            info: Dict with solver metadata.
        """
        from qpth.qp import QPFunction

        B = u_nominal.shape[0]
        device = u_nominal.device

        # Build constraint matrices
        G, h, n_active = self.build_constraint_matrices(
            states, obstacles, opponent_states,
            arena_half_w, arena_half_h, robot_radius, r_min_separation,
        )

        # Build weighted Q matrix: W = diag(w_v, w_omega)
        # QP: min 0.5 u^T Q u + p^T u
        # We want: min (u - u_nom)^T W (u - u_nom) = u^T W u - 2 u_nom^T W u + const
        # => Q = 2W, p = -2 W u_nom
        W = torch.tensor(
            [[self.w_v, 0.0], [0.0, self.w_omega]],
            device=device, dtype=torch.float64,
        )
        Q = (2.0 * W).unsqueeze(0).expand(B, -1, -1)  # (B, 2, 2)
        p = -2.0 * (W.unsqueeze(0) @ u_nominal.double().unsqueeze(-1)).squeeze(-1)  # (B, 2)

        # No equality constraints
        A_eq = torch.zeros(B, 0, 2, device=device, dtype=torch.float64)
        b_eq = torch.zeros(B, 0, device=device, dtype=torch.float64)

        # Convert to float64 for qpth numerical stability
        G_d = G.double()
        h_d = h.double()

        # Solve batched QP via qpth PDIPM
        try:
            u_safe = QPFunction(verbose=-1, maxIter=30, notImprovedLim=5)(
                Q, p, G_d, h_d, A_eq, b_eq,
            )
            u_safe = u_safe.float()
            feasible = True
        except Exception:
            # Infeasibility fallback: clamp nominal action, detach gradient
            u_safe = torch.clamp(
                u_nominal.detach(),
                min=torch.tensor([self.v_min, self.omega_min], device=device),
                max=torch.tensor([self.v_max, self.omega_max], device=device),
            )
            feasible = False

        # Clamp to bounds (qpth may have small numerical violations)
        u_safe = torch.clamp(
            u_safe,
            min=torch.tensor([self.v_min, self.omega_min], device=device),
            max=torch.tensor([self.v_max, self.omega_max], device=device),
        )

        # Compute constraint satisfaction info
        with torch.no_grad():
            Gu = torch.bmm(G, u_safe.unsqueeze(-1)).squeeze(-1)  # (B, n_total)
            violations = Gu - h  # positive = violation
            max_violation = violations.max(dim=1).values  # (B,)
            min_h_cbf = h[:, :self.n_constraints_max].min(dim=1).values / self.alpha

            intervention = not torch.allclose(u_safe, u_nominal, atol=1e-4)

        info = {
            "feasible": feasible,
            "max_constraint_violation": max_violation.max().item(),
            "min_cbf_value": min_h_cbf.min().item(),
            "n_active_constraints": n_active[0].item(),
            "intervention": intervention,
        }

        return u_safe, info

    def compute_constraint_values(
        self,
        u: torch.Tensor,
        states: torch.Tensor,
        obstacles: list[dict] | None = None,
        opponent_states: torch.Tensor | None = None,
        arena_half_w: float = 10.0,
        arena_half_h: float = 10.0,
        robot_radius: float = 0.15,
        r_min_separation: float = 0.35,
    ) -> torch.Tensor:
        """Compute CBF constraint values for given actions (for verification).

        Returns the CBF constraint margins: a_v*v + a_omega*omega + alpha*h.
        Positive = satisfied, negative = violated.

        Args:
            u: (B, 2) actions [v, omega].
            states: (B, 3) robot states.
            ... (same as build_constraint_matrices)

        Returns:
            margins: (B, n_constraints_max) CBF constraint margins.
        """
        G, h, _ = self.build_constraint_matrices(
            states, obstacles, opponent_states,
            arena_half_w, arena_half_h, robot_radius, r_min_separation,
        )

        # G_cbf @ u <= h_cbf  is equivalent to  -(a_v*v + a_omega*omega) <= alpha*h
        # So margin = alpha*h - G_cbf @ u = h_cbf - G_cbf @ u (positive = satisfied)
        G_cbf = G[:, :self.n_constraints_max, :]  # (B, n_cbf, 2)
        h_cbf = h[:, :self.n_constraints_max]      # (B, n_cbf)

        Gu = torch.bmm(G_cbf, u.unsqueeze(-1)).squeeze(-1)  # (B, n_cbf)
        margins = h_cbf - Gu  # positive = satisfied

        return margins


def verify_numpy_torch_consistency(
    state: np.ndarray,
    obstacles: list[dict] | None = None,
    opponent_state: np.ndarray | None = None,
    d: float = 0.1,
    alpha: float = 1.0,
    arena_half_w: float = 10.0,
    arena_half_h: float = 10.0,
    robot_radius: float = 0.15,
    r_min_separation: float = 0.35,
) -> dict:
    """Verify that torch constraint matrices match numpy VCP-CBF functions.

    This is a debugging utility that compares the constraint matrices built
    by DifferentiableVCPCBFQP with the scalar outputs from vcp_cbf.py.

    Args:
        state: Robot state [x, y, theta].
        obstacles: List of obstacle dicts.
        opponent_state: Opponent state [x, y, theta], or None.
        ... (same as VCPCBFFilter parameters)

    Returns:
        Dict with comparison results and max absolute differences.
    """
    from safety.vcp_cbf import (
        vcp_cbf_boundary,
        vcp_cbf_obstacle,
        vcp_cbf_collision,
    )

    # Get numpy constraints
    np_constraints = []
    boundary = vcp_cbf_boundary(state, arena_half_w, arena_half_h, robot_radius, d, alpha)
    np_constraints.extend(boundary)

    if obstacles:
        for obs in obstacles:
            obs_pos = np.array([obs["x"], obs["y"]])
            effective_r = obs["radius"] + robot_radius
            h, a_v, a_omega = vcp_cbf_obstacle(state, obs_pos, effective_r, d, alpha)
            np_constraints.append((h, a_v, a_omega))

    if opponent_state is not None:
        h, a_v, a_omega = vcp_cbf_collision(state, opponent_state, r_min_separation, d, alpha)
        np_constraints.append((h, a_v, a_omega))

    # Get torch constraints
    qp_layer = DifferentiableVCPCBFQP(
        n_constraints_max=len(np_constraints),
        alpha=alpha, d=d,
    )
    states_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    opponent_t = None
    if opponent_state is not None:
        opponent_t = torch.tensor(opponent_state, dtype=torch.float32).unsqueeze(0)

    G, h_vec, _ = qp_layer.build_constraint_matrices(
        states_t, obstacles, opponent_t,
        arena_half_w, arena_half_h, robot_radius, r_min_separation,
    )

    # Compare: for constraint i, G[0,i] = [-a_v, -a_omega], h_vec[0,i] = alpha * h
    max_diff_a_v = 0.0
    max_diff_a_omega = 0.0
    max_diff_h = 0.0
    results = []

    for i, (h_np, a_v_np, a_omega_np) in enumerate(np_constraints):
        a_v_torch = -G[0, i, 0].item()
        a_omega_torch = -G[0, i, 1].item()
        h_torch = h_vec[0, i].item() / alpha

        diff_a_v = abs(a_v_np - a_v_torch)
        diff_a_omega = abs(a_omega_np - a_omega_torch)
        diff_h = abs(h_np - h_torch)

        max_diff_a_v = max(max_diff_a_v, diff_a_v)
        max_diff_a_omega = max(max_diff_a_omega, diff_a_omega)
        max_diff_h = max(max_diff_h, diff_h)

        results.append({
            "constraint": i,
            "np": (h_np, a_v_np, a_omega_np),
            "torch": (h_torch, a_v_torch, a_omega_torch),
            "diff": (diff_h, diff_a_v, diff_a_omega),
        })

    return {
        "max_diff_h": max_diff_h,
        "max_diff_a_v": max_diff_a_v,
        "max_diff_a_omega": max_diff_a_omega,
        "all_close": max(max_diff_h, max_diff_a_v, max_diff_a_omega) < 1e-5,
        "n_constraints": len(np_constraints),
        "details": results,
    }
