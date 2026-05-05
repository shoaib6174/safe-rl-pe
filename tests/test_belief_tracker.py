"""Unit tests for BeliefTracker particle filter.

Test Index:
  A: test_visible_target_clusters
  B: test_hidden_target_diffuses
  C: test_reappearance_reconverges
  D: test_particles_clamped_to_arena
  E: test_safe_default_before_first_update
"""

import numpy as np
import pytest

from envs.belief_tracker import BeliefTracker


# --- Test A: Visible target clusters around true position ---

def test_visible_target_clusters():
    """When opponent is visible, particles should cluster near true position."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    tracker.update(own_state, opp_state, is_visible=True)
    stats = tracker.get_statistics()

    mean_x, mean_y = stats[0], stats[1]
    assert abs(mean_x - 8.0) < 0.1, f"mean_x should be ~8.0, got {mean_x}"
    assert abs(mean_y - 8.0) < 0.1, f"mean_y should be ~8.0, got {mean_y}"


# --- Test B: Hidden target diffuses over time ---

def test_hidden_target_diffuses():
    """When opponent is hidden, particle spread should grow with sqrt(t)."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=500,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    # First observe
    tracker.update(own_state, opp_state, is_visible=True)
    stats_visible = tracker.get_statistics()
    std_visible = (stats_visible[2] + stats_visible[3]) / 2

    # Hide for 200 steps (10 seconds at dt=0.05)
    for _ in range(200):
        tracker.update(own_state, opp_state, is_visible=False)

    stats_hidden = tracker.get_statistics()
    std_hidden = (stats_hidden[2] + stats_hidden[3]) / 2

    # Diffusion: std should grow from ~0.2 (resample noise) toward uniform (~5.8m)
    # With arena clamping, 200 steps spreads particles significantly
    assert std_visible < 0.3, f"Visible std too large: {std_visible}"
    assert std_hidden > std_visible, f"Hidden std should exceed visible std"
    assert std_hidden > 3.0, f"Hidden std should be significantly spread, got {std_hidden}"

    # time_since_seen should be 10.0 seconds
    assert abs(stats_hidden[5] - 10.0) < 0.01, f"time_since_seen should be 10.0, got {stats_hidden[5]}"


# --- Test C: Reappearance reconverges quickly ---

def test_reappearance_reconverges():
    """After diffusion, one visible update should reconverge to true position."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    # Observe, hide, then reappear at a different location
    tracker.update(own_state, opp_state, is_visible=True)
    for _ in range(100):
        tracker.update(own_state, opp_state, is_visible=False)

    # Opponent reappears at new position
    new_opp = np.array([12.0, 12.0, 0.0], dtype=np.float32)
    tracker.update(own_state, new_opp, is_visible=True)
    stats = tracker.get_statistics()

    assert abs(stats[0] - 12.0) < 0.1, f"mean_x should reconverge to 12.0, got {stats[0]}"
    assert abs(stats[1] - 12.0) < 0.1, f"mean_y should reconverge to 12.0, got {stats[1]}"


# --- Test D: Particles clamped to arena ---

def test_particles_clamped_to_arena():
    """Particles should never leave [0, arena_width] x [0, arena_height]."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=10.0,
        arena_height=10.0,
        opponent_v_max=5.0,  # High speed to accelerate diffusion
        n_particles=1000,
        seed=42,
    )
    own_state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    opp_state = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    tracker.update(own_state, opp_state, is_visible=True)
    for _ in range(500):
        tracker.update(own_state, opp_state, is_visible=False)

    # All particles should be inside arena
    assert np.all(tracker.particles[:, 0] >= 0.0)
    assert np.all(tracker.particles[:, 0] <= 10.0)
    assert np.all(tracker.particles[:, 1] >= 0.0)
    assert np.all(tracker.particles[:, 1] <= 10.0)


# --- Test E: Safe default before first update ---

def test_safe_default_before_first_update():
    """Before any update, get_statistics should return safe default."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    stats = tracker.get_statistics()

    # mean should be arena center
    assert abs(stats[0] - 10.0) < 0.01, f"default mean_x should be 10.0, got {stats[0]}"
    assert abs(stats[1] - 10.0) < 0.01, f"default mean_y should be 10.0, got {stats[1]}"
    # std should be arena diagonal / 2 ~ 14.14
    expected_std = np.sqrt(20.0**2 + 20.0**2) / 2
    assert abs(stats[2] - expected_std) < 0.1
    assert abs(stats[3] - expected_std) < 0.1
    # time_since_seen should be 0
    assert stats[5] == 0.0
