"""Demo visualization of the pursuit-evasion environment.

Shows the full rendering pipeline including obstacles, CBF overlay,
trajectory trails, and HUD — using greedy/random policies (no SB3 needed).

Usage:
    python scripts/demo_visualization.py
    python scripts/demo_visualization.py --with-cbf       # Enable CBF safety filter
    python scripts/demo_visualization.py --n-obstacles 6   # More obstacles
    python scripts/demo_visualization.py --episodes 5      # Run multiple episodes
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from envs.pursuit_evasion_env import PursuitEvasionEnv
from safety.vcp_cbf import VCPCBFFilter
from safety.infeasibility import SafeActionResolver


def greedy_pursuer_action(p_state, e_state, v_max=1.0, omega_max=2.84):
    """Greedy pursuer: turn toward evader, full speed."""
    dx = e_state[0] - p_state[0]
    dy = e_state[1] - p_state[1]
    desired_heading = np.arctan2(dy, dx)
    heading_error = (desired_heading - p_state[2] + np.pi) % (2 * np.pi) - np.pi
    omega = np.clip(3.0 * heading_error, -omega_max, omega_max)
    return np.array([v_max, omega], dtype=np.float32)


def greedy_evader_action(p_state, e_state, v_max=1.0, omega_max=2.84):
    """Greedy evader: turn away from pursuer, full speed."""
    dx = p_state[0] - e_state[0]
    dy = p_state[1] - e_state[1]
    danger_dir = np.arctan2(dy, dx)
    away_dir = danger_dir + np.pi
    heading_error = (away_dir - e_state[2] + np.pi) % (2 * np.pi) - np.pi
    omega = np.clip(3.0 * heading_error, -omega_max, omega_max)
    return np.array([v_max, omega], dtype=np.float32)


def run_demo(
    n_obstacles: int = 4,
    with_cbf: bool = False,
    n_episodes: int = 3,
    seed: int = 42,
    max_steps: int = 600,
    fps: int = 30,
):
    """Run visualization demo."""
    print(f"Starting demo: {n_obstacles} obstacles, CBF={'ON' if with_cbf else 'OFF'}, "
          f"{n_episodes} episodes")

    env = PursuitEvasionEnv(
        arena_width=20.0,
        arena_height=20.0,
        dt=0.05,
        max_steps=max_steps,
        capture_radius=0.5,
        collision_radius=0.3,
        robot_radius=0.15,
        pursuer_v_max=1.0,
        pursuer_omega_max=2.84,
        evader_v_max=1.0,
        evader_omega_max=2.84,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.3, 1.0),
        obstacle_margin=0.5,
        n_obstacle_obs=min(n_obstacles, 3),
        render_mode="human",
    )

    # Optional CBF safety filter
    resolver = None
    if with_cbf:
        cbf_filter = VCPCBFFilter(
            d=0.1,
            alpha=1.0,
            v_max=1.0,
            omega_max=2.84,
            arena_half_w=10.0,
            arena_half_h=10.0,
            robot_radius=0.15,
            w_v=150.0,
            w_omega=1.0,
            r_min_separation=0.4,
        )
        resolver = SafeActionResolver(cbf_filter=cbf_filter)
        print("  CBF safety filter enabled")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step = 0
        interventions = 0

        print(f"\n  Episode {ep + 1}/{n_episodes} — "
              f"{len(env.obstacles)} obstacles placed")

        while not done:
            # Greedy policies
            p_action = greedy_pursuer_action(env.pursuer_state, env.evader_state)
            e_action = greedy_evader_action(env.pursuer_state, env.evader_state)

            # Apply CBF filter if enabled
            if resolver is not None:
                p_safe, method, cbf_info = resolver.resolve(
                    p_action, env.pursuer_state,
                    obstacles=env.obstacles,
                    opponent_state=env.evader_state,
                )
                if method != "exact":
                    interventions += 1
                # Update render state with CBF info
                env.renderer._cbf_method = method
                p_action = p_safe

            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated
            step += 1

        # Episode summary
        metrics = info.get("episode_metrics", {})
        captured = metrics.get("captured", False)
        status = "CAPTURED" if captured else "TIMEOUT"
        cap_time = metrics.get("capture_time", step * 0.05)
        print(f"    {status} at t={cap_time:.1f}s ({step} steps), "
              f"min_dist={metrics.get('min_distance', 0):.2f}m"
              + (f", interventions={interventions}" if resolver else ""))

        # Brief pause between episodes
        if ep < n_episodes - 1:
            time.sleep(1.0)

    print("\nDemo complete. Close the window to exit.")

    # Keep window open until user closes it
    import pygame
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        time.sleep(0.1)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="PE Environment Demo Visualization")
    parser.add_argument("--n-obstacles", type=int, default=4, help="Number of obstacles")
    parser.add_argument("--with-cbf", action="store_true", help="Enable CBF safety filter")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    args = parser.parse_args()

    run_demo(
        n_obstacles=args.n_obstacles,
        with_cbf=args.with_cbf,
        n_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
