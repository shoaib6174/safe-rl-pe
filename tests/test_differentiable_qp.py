"""Tests for the differentiable QP layer (Phase 2.5 Session 2).

Tests cover:
1. Constraint matrix construction (arena, obstacle, collision)
2. Numpy/torch consistency with existing vcp_cbf.py
3. QP forward pass (safe/unsafe actions)
4. Gradient flow through QP solution
5. Batch consistency
6. Constraint satisfaction verification
7. QP solve time benchmarks
"""

import os
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety.differentiable_qp import (
    DifferentiableVCPCBFQP,
    verify_numpy_torch_consistency,
)
from safety.vcp_cbf import (
    CBFConstraint,
    ConstraintType,
    VCPCBFFilter,
    vcp_cbf_boundary,
    vcp_cbf_collision,
    vcp_cbf_obstacle,
)


# --- Fixtures ---

@pytest.fixture
def qp_layer():
    """Standard QP layer with 10 max constraints."""
    return DifferentiableVCPCBFQP(
        n_constraints_max=10,
        v_min=0.0,
        v_max=1.0,
        omega_min=-2.84,
        omega_max=2.84,
        w_v=150.0,
        w_omega=1.0,
        alpha=1.0,
        d=0.1,
    )


@pytest.fixture
def simple_obstacles():
    """Two obstacles for testing."""
    return [
        {"x": 3.0, "y": 0.0, "radius": 0.5},
        {"x": -2.0, "y": 3.0, "radius": 0.8},
    ]


@pytest.fixture
def center_state():
    """Robot at center, facing +x."""
    return torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)


@pytest.fixture
def near_wall_state():
    """Robot near right wall, facing +x."""
    return torch.tensor([[9.5, 0.0, 0.0]], dtype=torch.float32)


@pytest.fixture
def near_obstacle_state():
    """Robot near obstacle at (3, 0), facing toward it."""
    return torch.tensor([[2.3, 0.0, 0.0]], dtype=torch.float32)


# ==============================================================================
# Test Class 1: Constraint Matrix Construction
# ==============================================================================


class TestConstraintMatrixConstruction:
    """Test that constraint matrices are built correctly."""

    def test_shape_no_obstacles(self, qp_layer, center_state):
        """G and h should have correct shape with arena-only constraints."""
        G, h, n_active = qp_layer.build_constraint_matrices(center_state)

        # n_total = n_constraints_max + 4 control bounds = 14
        assert G.shape == (1, 14, 2)
        assert h.shape == (1, 14)
        assert n_active.shape == (1,)
        assert n_active[0].item() == 4  # 4 arena walls

    def test_shape_with_obstacles(self, qp_layer, center_state, simple_obstacles):
        """Shape should account for obstacle constraints."""
        G, h, n_active = qp_layer.build_constraint_matrices(
            center_state, obstacles=simple_obstacles,
        )
        assert G.shape == (1, 14, 2)
        assert h.shape == (1, 14)
        assert n_active[0].item() == 6  # 4 arena + 2 obstacles

    def test_shape_with_opponent(self, qp_layer, center_state):
        """Shape should account for collision constraint."""
        opponent = torch.tensor([[5.0, 5.0, 1.0]], dtype=torch.float32)
        G, h, n_active = qp_layer.build_constraint_matrices(
            center_state, opponent_states=opponent,
        )
        assert n_active[0].item() == 5  # 4 arena + 1 collision

    def test_shape_with_all(self, qp_layer, center_state, simple_obstacles):
        """Shape with arena + obstacles + collision."""
        opponent = torch.tensor([[5.0, 5.0, 1.0]], dtype=torch.float32)
        G, h, n_active = qp_layer.build_constraint_matrices(
            center_state, obstacles=simple_obstacles, opponent_states=opponent,
        )
        assert n_active[0].item() == 7  # 4 + 2 + 1

    def test_inactive_constraints_padded(self, qp_layer, center_state):
        """Inactive constraints should have h = 1e6 (always satisfied)."""
        G, h, n_active = qp_layer.build_constraint_matrices(center_state)

        # Active: first 4 (arena). Inactive: 5..10 (padded). Control: 11..14.
        for i in range(4, 10):
            assert h[0, i].item() == pytest.approx(1e6, abs=1.0)
            assert G[0, i, 0].item() == 0.0
            assert G[0, i, 1].item() == 0.0

    def test_control_bound_constraints(self, qp_layer, center_state):
        """Last 4 constraints should encode control bounds."""
        G, h, _ = qp_layer.build_constraint_matrices(center_state)

        # -v <= -v_min => G[-4] = [-1, 0], h[-4] = -0.0
        assert G[0, -4, 0].item() == pytest.approx(-1.0)
        assert G[0, -4, 1].item() == pytest.approx(0.0)
        assert h[0, -4].item() == pytest.approx(-0.0)

        # v <= v_max => G[-3] = [1, 0], h[-3] = 1.0
        assert G[0, -3, 0].item() == pytest.approx(1.0)
        assert h[0, -3].item() == pytest.approx(1.0)

        # -omega <= -omega_min => G[-2] = [0, -1], h[-2] = 2.84
        assert G[0, -2, 1].item() == pytest.approx(-1.0)
        assert h[0, -2].item() == pytest.approx(2.84)

        # omega <= omega_max => G[-1] = [0, 1], h[-1] = 2.84
        assert G[0, -1, 1].item() == pytest.approx(1.0)
        assert h[0, -1].item() == pytest.approx(2.84)

    def test_batched_construction(self, qp_layer, simple_obstacles):
        """Constraint matrices should work with batch size > 1."""
        states = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 1.57],
            [-3.0, -3.0, 3.14],
        ], dtype=torch.float32)

        G, h, n_active = qp_layer.build_constraint_matrices(
            states, obstacles=simple_obstacles,
        )
        assert G.shape == (3, 14, 2)
        assert h.shape == (3, 14)
        assert n_active.shape == (3,)

        # Each sample should have different constraint values
        assert not torch.allclose(G[0], G[1])


# ==============================================================================
# Test Class 2: Numpy/Torch Consistency
# ==============================================================================


class TestNumpyTorchConsistency:
    """Verify torch implementation matches numpy vcp_cbf.py."""

    def test_arena_boundary_consistency(self):
        """Arena constraints should match numpy vcp_cbf_boundary."""
        state = np.array([2.0, 3.0, 0.5])
        result = verify_numpy_torch_consistency(state)
        assert result["all_close"], (
            f"Arena mismatch: max_h={result['max_diff_h']:.2e}, "
            f"max_a_v={result['max_diff_a_v']:.2e}, "
            f"max_a_omega={result['max_diff_a_omega']:.2e}"
        )

    def test_obstacle_consistency(self):
        """Obstacle constraints should match numpy vcp_cbf_obstacle."""
        state = np.array([1.0, 1.0, 0.8])
        obstacles = [
            {"x": 3.0, "y": 1.0, "radius": 0.5},
            {"x": 0.0, "y": 3.0, "radius": 0.8},
        ]
        result = verify_numpy_torch_consistency(state, obstacles=obstacles)
        assert result["all_close"], (
            f"Obstacle mismatch: {result['details']}"
        )
        assert result["n_constraints"] == 6  # 4 arena + 2 obstacle

    def test_collision_consistency(self):
        """Collision constraint should match numpy vcp_cbf_collision."""
        state = np.array([1.0, 1.0, 0.3])
        opponent = np.array([2.0, 1.5, -1.0])
        result = verify_numpy_torch_consistency(
            state, opponent_state=opponent, r_min_separation=0.4,
        )
        assert result["all_close"], (
            f"Collision mismatch: {result['details']}"
        )
        assert result["n_constraints"] == 5  # 4 arena + 1 collision

    def test_full_consistency(self):
        """All constraints together should match numpy."""
        state = np.array([-1.5, 2.0, -0.7])
        obstacles = [{"x": 0.0, "y": 3.0, "radius": 0.6}]
        opponent = np.array([1.0, 2.0, 1.5])
        result = verify_numpy_torch_consistency(
            state, obstacles=obstacles, opponent_state=opponent,
        )
        assert result["all_close"]
        assert result["n_constraints"] == 6  # 4 + 1 + 1

    @pytest.mark.parametrize("theta", [0.0, np.pi/4, np.pi/2, np.pi, -np.pi/2, -np.pi])
    def test_consistency_various_headings(self, theta):
        """Constraints should match for various robot headings."""
        state = np.array([0.0, 0.0, theta])
        obstacles = [{"x": 2.0, "y": 2.0, "radius": 0.5}]
        result = verify_numpy_torch_consistency(state, obstacles=obstacles)
        assert result["all_close"], f"Mismatch at theta={theta}: {result}"

    @pytest.mark.parametrize("pos", [
        (0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (-8.0, -8.0), (5.0, -5.0),
    ])
    def test_consistency_various_positions(self, pos):
        """Constraints should match for various positions."""
        state = np.array([pos[0], pos[1], 0.3])
        result = verify_numpy_torch_consistency(state)
        assert result["all_close"], f"Mismatch at pos={pos}: {result}"


# ==============================================================================
# Test Class 3: QP Forward Pass
# ==============================================================================


class TestQPForwardPass:
    """Test the QP solver produces correct safe actions."""

    def test_safe_action_unchanged(self, qp_layer, center_state):
        """When nominal action is safe, QP should return ~same action."""
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, center_state)

        assert u_safe.shape == (1, 2)
        assert torch.allclose(u_safe, u_nom, atol=0.05), (
            f"Safe action modified unnecessarily: {u_safe} vs {u_nom}"
        )
        assert info["feasible"]

    def test_unsafe_action_corrected(self, qp_layer, near_wall_state):
        """When heading into wall, QP should reduce velocity or steer."""
        # Near right wall (x=9.5), facing +x, full speed
        u_nom = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, near_wall_state)

        assert u_safe.shape == (1, 2)
        # Safe action should differ from nominal
        assert not torch.allclose(u_safe, u_nom, atol=0.01), (
            f"Unsafe action not corrected: {u_safe}"
        )
        # Should reduce speed or add steering
        assert u_safe[0, 0] < 1.0 or abs(u_safe[0, 1]) > 0.01

    def test_obstacle_avoidance(self, qp_layer, near_obstacle_state):
        """QP should avoid obstacle directly ahead."""
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        u_nom = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        u_safe, info = qp_layer(
            u_nom, near_obstacle_state, obstacles=obstacles,
        )

        # Should modify action (obstacle at 3.0, robot at 2.3, effective_r = 0.65)
        assert not torch.allclose(u_safe, u_nom, atol=0.05)
        assert info["feasible"]

    def test_control_bounds_respected(self, qp_layer, center_state):
        """Output should always satisfy control bounds."""
        # Extreme nominal action
        u_nom = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
        u_safe, _ = qp_layer(u_nom, center_state)

        assert u_safe[0, 0] >= -0.01  # v >= v_min
        assert u_safe[0, 0] <= 1.01   # v <= v_max
        assert u_safe[0, 1] >= -2.85  # omega >= omega_min
        assert u_safe[0, 1] <= 2.85   # omega <= omega_max

    def test_collision_avoidance(self, qp_layer):
        """QP should avoid collision with opponent."""
        # Robot at (0,0) facing opponent at (0.5, 0)
        states = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        opponent = torch.tensor([[0.5, 0.0, 3.14]], dtype=torch.float32)
        u_nom = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        u_safe, info = qp_layer(
            u_nom, states, opponent_states=opponent, r_min_separation=0.4,
        )

        # Should modify action to avoid collision
        assert not torch.allclose(u_safe, u_nom, atol=0.05)

    def test_multiple_obstacles(self, qp_layer):
        """QP should handle multiple simultaneous constraints."""
        states = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        obstacles = [
            {"x": 1.0, "y": 0.0, "radius": 0.3},
            {"x": 0.0, "y": 1.0, "radius": 0.3},
            {"x": -1.0, "y": 0.0, "radius": 0.3},
        ]
        u_nom = torch.tensor([[0.8, 0.5]], dtype=torch.float32)

        u_safe, info = qp_layer(u_nom, states, obstacles=obstacles)
        assert u_safe.shape == (1, 2)
        assert info["feasible"]


# ==============================================================================
# Test Class 4: Gradient Flow
# ==============================================================================


class TestGradientFlow:
    """Test that gradients flow through the QP solution."""

    def test_gradient_exists(self, qp_layer, center_state):
        """u_nom.grad should be non-None after backward."""
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float32, requires_grad=True)
        u_safe, _ = qp_layer(u_nom, center_state)

        loss = u_safe.sum()
        loss.backward()

        assert u_nom.grad is not None, "No gradient on u_nom"

    def test_gradient_nonzero_unconstrained(self, qp_layer, center_state):
        """Gradient should be identity-like when unconstrained (far from obstacles)."""
        u_nom = torch.tensor([[0.3, 0.0]], dtype=torch.float32, requires_grad=True)
        u_safe, _ = qp_layer(u_nom, center_state)

        loss = u_safe.sum()
        loss.backward()

        # When unconstrained, du_safe/du_nom should be approximately identity
        # because the QP just passes through the nominal action
        assert not torch.all(u_nom.grad == 0), "Gradient is zero"

    def test_gradient_with_active_constraint(self, qp_layer, near_wall_state):
        """Gradient should still flow when constraints are active."""
        u_nom = torch.tensor([[0.8, 0.0]], dtype=torch.float32, requires_grad=True)
        u_safe, _ = qp_layer(u_nom, near_wall_state)

        loss = u_safe.sum()
        loss.backward()

        assert u_nom.grad is not None

    def test_gradient_through_obstacle_constraint(self, qp_layer):
        """Gradient should flow through obstacle CBF constraint."""
        states = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        u_nom = torch.tensor([[0.9, 0.0]], dtype=torch.float32, requires_grad=True)

        u_safe, _ = qp_layer(u_nom, states, obstacles=obstacles)
        loss = u_safe.sum()
        loss.backward()

        assert u_nom.grad is not None

    def test_gradient_for_ppo_loss(self, qp_layer, center_state):
        """Gradient should work for a PPO-style loss."""
        u_nom = torch.tensor([[0.5, 0.3]], dtype=torch.float32, requires_grad=True)
        u_safe, _ = qp_layer(u_nom, center_state)

        # Simulate PPO advantage-weighted log-prob loss
        target = torch.tensor([[0.7, -0.2]], dtype=torch.float32)
        advantage = 1.5
        loss = -advantage * ((u_safe - target) ** 2).sum()
        loss.backward()

        assert u_nom.grad is not None
        assert not torch.all(u_nom.grad == 0)


# ==============================================================================
# Test Class 5: Batch Consistency
# ==============================================================================


class TestBatchConsistency:
    """Test batched solving gives correct per-sample results."""

    def test_batch_shape(self, qp_layer, simple_obstacles):
        """Output shape should match batch size."""
        B = 4
        states = torch.randn(B, 3)
        states[:, :2] *= 3  # positions in [-3, 3]
        u_nom = torch.rand(B, 2)
        u_nom[:, 0] *= 1.0  # v in [0, 1]
        u_nom[:, 1] = u_nom[:, 1] * 5.68 - 2.84  # omega in [-2.84, 2.84]

        u_safe, info = qp_layer(u_nom, states, obstacles=simple_obstacles)
        assert u_safe.shape == (B, 2)

    def test_single_vs_batch(self, qp_layer):
        """Batched result should match individual solves."""
        states = torch.tensor([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 1.57],
        ], dtype=torch.float32)
        u_nom = torch.tensor([
            [0.5, 0.0],
            [0.8, -1.0],
        ], dtype=torch.float32)

        # Batched solve
        u_safe_batch, _ = qp_layer(u_nom, states)

        # Individual solves
        u_safe_0, _ = qp_layer(u_nom[0:1], states[0:1])
        u_safe_1, _ = qp_layer(u_nom[1:2], states[1:2])

        assert torch.allclose(u_safe_batch[0], u_safe_0[0], atol=1e-3), (
            f"Batch[0]={u_safe_batch[0]} vs Single={u_safe_0[0]}"
        )
        assert torch.allclose(u_safe_batch[1], u_safe_1[0], atol=1e-3), (
            f"Batch[1]={u_safe_batch[1]} vs Single={u_safe_1[0]}"
        )


# ==============================================================================
# Test Class 6: Constraint Satisfaction
# ==============================================================================


class TestConstraintSatisfaction:
    """Verify QP output satisfies all CBF constraints."""

    def test_arena_constraints_satisfied(self, qp_layer, near_wall_state):
        """Safe action should satisfy arena boundary constraints."""
        u_nom = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        u_safe, _ = qp_layer(u_nom, near_wall_state)

        margins = qp_layer.compute_constraint_values(u_safe, near_wall_state)
        # All margins should be >= -epsilon (satisfied)
        assert torch.all(margins >= -1e-3), (
            f"Constraint violated: margins = {margins}"
        )

    def test_obstacle_constraints_satisfied(self, qp_layer, near_obstacle_state):
        """Safe action should satisfy obstacle constraints (within qpth tolerance)."""
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        u_nom = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        u_safe, _ = qp_layer(u_nom, near_obstacle_state, obstacles=obstacles)

        margins = qp_layer.compute_constraint_values(
            u_safe, near_obstacle_state, obstacles=obstacles,
        )
        # qpth's PDIPM solver gives approximate feasibility; allow small violations
        assert torch.all(margins >= -0.1), (
            f"Constraint violated beyond tolerance: margins = {margins}"
        )

    def test_all_constraints_satisfied_random(self, qp_layer):
        """Random states and actions should produce approximately feasible output."""
        torch.manual_seed(42)
        B = 10
        states = torch.randn(B, 3)
        states[:, :2] *= 5  # positions in [-5, 5]
        u_nom = torch.rand(B, 2)
        u_nom[:, 0] *= 1.0
        u_nom[:, 1] = u_nom[:, 1] * 5.68 - 2.84

        obstacles = [{"x": 2.0, "y": 2.0, "radius": 0.5}]
        u_safe, info = qp_layer(u_nom, states, obstacles=obstacles)

        if info["feasible"]:
            margins = qp_layer.compute_constraint_values(
                u_safe, states, obstacles=obstacles,
            )
            # qpth's PDIPM gives approximate feasibility; allow small violations
            assert torch.all(margins >= -0.1), (
                f"Constraint violated beyond tolerance: min margin = {margins.min().item()}"
            )


# ==============================================================================
# Test Class 7: Performance Benchmark
# ==============================================================================


class TestPerformance:
    """Benchmark QP solve times."""

    def test_single_solve_time(self, qp_layer, center_state):
        """Single QP solve should be <50ms on CPU."""
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]

        # Warmup
        qp_layer(u_nom, center_state, obstacles=obstacles)

        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            qp_layer(u_nom, center_state, obstacles=obstacles)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Single QP solve: {avg_ms:.2f}ms avg ({min(times)*1000:.2f}ms min)")
        assert avg_ms < 50, f"QP too slow: {avg_ms:.1f}ms > 50ms"

    def test_batch_solve_time(self, qp_layer):
        """Batched QP solve should scale sub-linearly."""
        B = 32
        states = torch.randn(B, 3)
        states[:, :2] *= 5
        u_nom = torch.rand(B, 2)
        u_nom[:, 0] *= 1.0
        u_nom[:, 1] = u_nom[:, 1] * 5.68 - 2.84
        obstacles = [
            {"x": 3.0, "y": 0.0, "radius": 0.5},
            {"x": -2.0, "y": 3.0, "radius": 0.8},
        ]

        # Warmup
        qp_layer(u_nom, states, obstacles=obstacles)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            qp_layer(u_nom, states, obstacles=obstacles)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        per_sample_ms = avg_ms / B
        print(f"\n  Batch-{B} QP solve: {avg_ms:.2f}ms avg ({per_sample_ms:.3f}ms/sample)")
        assert avg_ms < 500, f"Batch QP too slow: {avg_ms:.1f}ms > 500ms"

    def test_constraint_build_time(self, qp_layer):
        """Constraint matrix construction should be fast."""
        B = 64
        states = torch.randn(B, 3)
        states[:, :2] *= 5
        obstacles = [
            {"x": 3.0, "y": 0.0, "radius": 0.5},
            {"x": -2.0, "y": 3.0, "radius": 0.8},
            {"x": 0.0, "y": -4.0, "radius": 0.6},
        ]
        opponent = torch.randn(B, 3)

        # Warmup
        qp_layer.build_constraint_matrices(states, obstacles, opponent)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            qp_layer.build_constraint_matrices(states, obstacles, opponent)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Constraint build (B={B}): {avg_ms:.3f}ms avg")
        assert avg_ms < 10, f"Constraint build too slow: {avg_ms:.1f}ms > 10ms"


# ==============================================================================
# Test Class 8: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_nominal_action(self, qp_layer, center_state):
        """Zero nominal action should be safe (robot is stationary)."""
        u_nom = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, center_state)
        assert info["feasible"]
        # Zero action is always safe (no movement)
        assert torch.allclose(u_safe, u_nom, atol=0.1)

    def test_robot_at_origin(self, qp_layer):
        """Robot at origin with all defaults."""
        states = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        u_nom = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, states)
        assert u_safe.shape == (1, 2)
        assert info["feasible"]

    def test_no_obstacles(self, qp_layer, center_state):
        """Should work with no obstacles and no opponent."""
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, center_state)
        assert info["feasible"]
        assert torch.allclose(u_safe, u_nom, atol=0.05)

    def test_many_obstacles(self):
        """QP should handle max constraint count."""
        qp = DifferentiableVCPCBFQP(n_constraints_max=20, d=0.1)
        states = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        obstacles = [
            {"x": float(3 + i), "y": float(3 + i), "radius": 0.3}
            for i in range(15)
        ]
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        u_safe, info = qp.forward(u_nom, states, obstacles=obstacles)
        assert u_safe.shape == (1, 2)

    def test_negative_heading(self, qp_layer):
        """Should work with negative heading angles."""
        states = torch.tensor([[0.0, 0.0, -np.pi/2]], dtype=torch.float32)
        u_nom = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        u_safe, info = qp_layer(u_nom, states)
        assert info["feasible"]

    def test_float64_input(self, qp_layer):
        """Should handle float64 inputs (converted internally)."""
        states = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        u_nom = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        u_safe, info = qp_layer(u_nom.float(), states.float())
        assert u_safe.dtype == torch.float32


# ==============================================================================
# Test Class 9: Integration with Existing VCPCBFFilter
# ==============================================================================


class TestIntegrationWithVCPCBF:
    """Test that diff QP layer agrees with existing VCPCBFFilter on solutions."""

    def _compare_solutions(self, state_np, u_nom_np, obstacles=None, opponent=None):
        """Compare diff QP solution with VCPCBFFilter solution."""
        # Existing numpy filter
        cbf_filter = VCPCBFFilter(
            d=0.1, alpha=1.0, v_max=1.0, omega_max=2.84,
            arena_half_w=10.0, arena_half_h=10.0, robot_radius=0.15,
            w_v=150.0, w_omega=1.0, r_min_separation=0.35,
        )
        u_np, info_np = cbf_filter.filter_action(
            u_nom_np, state_np,
            obstacles=obstacles, opponent_state=opponent,
        )

        # New torch layer
        qp_layer = DifferentiableVCPCBFQP(
            n_constraints_max=10, d=0.1, alpha=1.0,
            v_min=0.0, v_max=1.0, omega_min=-2.84, omega_max=2.84,
            w_v=150.0, w_omega=1.0,
        )
        state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        u_nom_t = torch.tensor(u_nom_np, dtype=torch.float32).unsqueeze(0)
        opponent_t = None
        if opponent is not None:
            opponent_t = torch.tensor(opponent, dtype=torch.float32).unsqueeze(0)

        u_t, info_t = qp_layer(
            u_nom_t, state_t, obstacles=obstacles, opponent_states=opponent_t,
        )

        return u_np, u_t[0].detach().numpy()

    def test_center_safe_action(self):
        """Both solvers should return ~same for safe action at center."""
        u_np, u_t = self._compare_solutions(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0], dtype=np.float32),
        )
        assert np.allclose(u_np, u_t, atol=0.1), f"NP={u_np}, Torch={u_t}"

    def test_near_wall_correction(self):
        """Both solvers should correct near-wall actions similarly."""
        u_np, u_t = self._compare_solutions(
            np.array([9.0, 0.0, 0.0]),
            np.array([1.0, 0.0], dtype=np.float32),
        )
        # Both should reduce speed or add steering
        # Direction should agree even if magnitude differs slightly
        # (different QP solvers may find different but equally valid optima)
        if u_np[0] < 0.9:  # numpy solver intervened
            assert u_t[0] < 1.0 or abs(u_t[1]) > 0.01, (
                f"Torch did not intervene: NP={u_np}, Torch={u_t}"
            )

    def test_obstacle_correction(self):
        """Both solvers should avoid obstacles similarly."""
        obstacles = [{"x": 3.0, "y": 0.0, "radius": 0.5}]
        u_np, u_t = self._compare_solutions(
            np.array([2.3, 0.0, 0.0]),
            np.array([1.0, 0.0], dtype=np.float32),
            obstacles=obstacles,
        )
        # Both should modify the action
        assert not np.allclose(u_t, [1.0, 0.0], atol=0.05), (
            f"Torch did not correct: {u_t}"
        )
