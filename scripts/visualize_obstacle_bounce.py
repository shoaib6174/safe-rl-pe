#!/usr/bin/env python3
"""Visualize obstacle collision enforcement.

Drives agents with constant velocity directly at obstacles to confirm
they bounce off (are projected to surface) instead of passing through.

No trained models needed — uses scripted constant-velocity actions.

Usage:
    ./venv/bin/python scripts/visualize_obstacle_bounce.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.pursuit_evasion_env import PursuitEvasionEnv


def run_constant_velocity_episode(
    env: PursuitEvasionEnv,
    pursuer_start: np.ndarray,
    evader_start: np.ndarray,
    pursuer_action: np.ndarray,
    evader_action: np.ndarray,
    obstacles: list[dict],
    n_steps: int = 400,
) -> dict:
    """Run episode with fixed actions and manually placed agents."""
    env.reset(seed=42)

    # Set obstacles AFTER reset (reset clears them since n_obstacles=0)
    env.obstacles = obstacles

    # Override initial positions
    env.pursuer_state = pursuer_start.copy()
    env.evader_state = evader_start.copy()
    env.prev_distance = np.sqrt(
        (pursuer_start[0] - evader_start[0])**2
        + (pursuer_start[1] - evader_start[1])**2
    )

    p_traj = [pursuer_start[:2].copy()]
    e_traj = [evader_start[:2].copy()]
    p_collisions = []
    e_collisions = []

    for step in range(n_steps):
        _, _, term, trunc, info = env.step(pursuer_action, evader_action)
        p_traj.append(env.pursuer_state[:2].copy())
        e_traj.append(env.evader_state[:2].copy())
        p_collisions.append(info.get("pursuer_obstacle_collision", False))
        e_collisions.append(info.get("evader_obstacle_collision", False))
        if term or trunc:
            break

    return {
        "pursuer": np.array(p_traj),
        "evader": np.array(e_traj),
        "p_collisions": p_collisions,
        "e_collisions": e_collisions,
        "obstacles": env.obstacles,
    }


def plot_scenario(ax, traj_data, title, arena_w, arena_h, robot_radius):
    """Plot one scenario on the given axes."""
    # Arena
    ax.add_patch(plt.Rectangle(
        (-arena_w / 2, -arena_h / 2), arena_w, arena_h,
        fill=False, edgecolor="black", linewidth=2
    ))

    # Obstacles with exclusion zone
    for obs in traj_data["obstacles"]:
        # Physical obstacle
        ax.add_patch(Circle(
            (obs["x"], obs["y"]), obs["radius"],
            fill=True, facecolor="#8B4513", edgecolor="black",
            linewidth=1.5, alpha=0.7, zorder=3
        ))
        # Exclusion zone (obstacle + robot_radius)
        ax.add_patch(Circle(
            (obs["x"], obs["y"]), obs["radius"] + robot_radius,
            fill=False, edgecolor="red", linewidth=1, linestyle="--",
            alpha=0.5, zorder=3
        ))

    p = traj_data["pursuer"]
    e = traj_data["evader"]

    # Trajectories
    ax.plot(p[:, 0], p[:, 1], "-", color="#d62728", linewidth=1.5, alpha=0.8, label="Pursuer")
    ax.plot(e[:, 0], e[:, 1], "-", color="#1f77b4", linewidth=1.5, alpha=0.8, label="Evader")

    # Start positions
    ax.plot(p[0, 0], p[0, 1], "o", color="#d62728", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.plot(e[0, 0], e[0, 1], "o", color="#1f77b4", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)

    # End positions
    ax.plot(p[-1, 0], p[-1, 1], "^", color="#d62728", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.plot(e[-1, 0], e[-1, 1], "^", color="#1f77b4", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)

    # Collision points
    for i, (pc, ec) in enumerate(zip(traj_data["p_collisions"], traj_data["e_collisions"])):
        if pc:
            ax.plot(p[i+1, 0], p[i+1, 1], "x", color="red", markersize=6, zorder=11)
        if ec:
            ax.plot(e[i+1, 0], e[i+1, 1], "x", color="red", markersize=6, zorder=11)

    # Time markers every 50 steps
    for i in range(0, len(p), 50):
        if i == 0:
            continue
        t_sec = i * 0.05
        ax.annotate(f"{t_sec:.1f}s", (p[i, 0], p[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="#d62728", alpha=0.7)
    for i in range(0, len(e), 50):
        if i == 0:
            continue
        t_sec = i * 0.05
        ax.annotate(f"{t_sec:.1f}s", (e[i, 0], e[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="#1f77b4", alpha=0.7)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def main():
    out_dir = Path("results/obstacle_bounce_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_w, arena_h = 20.0, 20.0
    robot_radius = 0.15

    # Create env with manually placed obstacles
    env = PursuitEvasionEnv(
        arena_width=arena_w, arena_height=arena_h,
        n_obstacles=0, render_mode=None,
        max_steps=600,
    )

    # Place agents on opposite sides so they don't capture each other quickly.
    # Each agent heads toward an obstacle between them.

    # ── Scenario 1: Head-on into obstacle ──
    # Pursuer close to obstacle, heading directly at it. Evader far away.
    obs1 = [{"x": 2.0, "y": 0.0, "radius": 1.0}]
    traj1 = run_constant_velocity_episode(
        env,
        pursuer_start=np.array([-1.0, 0.0, 0.0]),        # 3m from obstacle center, heading east
        evader_start=np.array([8.0, 5.0, np.pi / 2]),     # far away, heading north (away)
        pursuer_action=np.array([1.0, 0.0]),               # full speed east → obstacle
        evader_action=np.array([1.0, 0.0]),                # full speed north (away)
        obstacles=obs1,
        n_steps=400,
    )

    # ── Scenario 2: Diagonal approach ──
    # Both agents approach large central obstacle from opposite corners
    obs2 = [{"x": 0.0, "y": 0.0, "radius": 1.5}]
    theta_p = np.arctan2(0.0 - (-4.0), 0.0 - (-4.0))   # heading toward origin
    theta_e = np.arctan2(0.0 - 4.0, 0.0 - 4.0)         # heading toward origin
    traj2 = run_constant_velocity_episode(
        env,
        pursuer_start=np.array([-4.0, -4.0, theta_p]),
        evader_start=np.array([4.0, 4.0, theta_e]),
        pursuer_action=np.array([1.0, 0.0]),
        evader_action=np.array([1.0, 0.0]),
        obstacles=obs2,
        n_steps=400,
    )

    # ── Scenario 3: Grazing / sliding along obstacle ──
    # Agent approaches at a shallow angle — should slide along surface
    obs3 = [{"x": 3.0, "y": 0.0, "radius": 1.0}]
    traj3 = run_constant_velocity_episode(
        env,
        pursuer_start=np.array([0.0, -0.8, 0.05]),       # nearly east, slightly upward → will graze
        evader_start=np.array([0.0, 5.0, np.pi / 2]),     # far away heading north
        pursuer_action=np.array([1.0, 0.0]),
        evader_action=np.array([1.0, 0.0]),
        obstacles=obs3,
        n_steps=400,
    )

    # ── Scenario 4: Multiple obstacles gauntlet ──
    # Both agents run east through a line of obstacles
    obs4 = [
        {"x": -2.0, "y": 0.0, "radius": 1.0},
        {"x": 2.0, "y": 0.3, "radius": 0.8},
        {"x": 5.5, "y": -0.2, "radius": 1.0},
    ]
    traj4 = run_constant_velocity_episode(
        env,
        pursuer_start=np.array([-6.0, 0.0, 0.0]),        # heading east through gauntlet
        evader_start=np.array([-6.0, 3.0, 0.0]),          # heading east, offset above
        pursuer_action=np.array([1.0, 0.0]),
        evader_action=np.array([1.0, 0.0]),
        obstacles=obs4,
        n_steps=500,
    )

    # ── Plot all scenarios ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    scenarios = [
        (axes[0, 0], traj1, "1. Head-on collision (both agents → obstacle)"),
        (axes[0, 1], traj2, "2. Diagonal approach (obstacle at center)"),
        (axes[1, 0], traj3, "3. Grazing / tangential sliding"),
        (axes[1, 1], traj4, "4. Multi-obstacle gauntlet"),
    ]

    for ax, traj, title in scenarios:
        plot_scenario(ax, traj, title, arena_w, arena_h, robot_radius)
        # Zoom to relevant area
        all_pts = np.vstack([traj["pursuer"], traj["evader"]])
        for obs in traj["obstacles"]:
            all_pts = np.vstack([all_pts, [[obs["x"], obs["y"]]]])
        margin = 2.0
        x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
        y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
        # Ensure square aspect
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    fig.suptitle(
        "Obstacle Collision Enforcement — Agents Bounce Off Obstacles\n"
        "(Red dashed = exclusion zone, Red × = collision frame)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = out_dir / "obstacle_bounce_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    # ── Print collision statistics ──
    print("\n=== Collision Statistics ===")
    for i, (traj, label) in enumerate([(traj1, "Head-on"), (traj2, "Diagonal"),
                                        (traj3, "Grazing"), (traj4, "Gauntlet")]):
        n_p = sum(traj["p_collisions"])
        n_e = sum(traj["e_collisions"])
        p_final = traj["pursuer"][-1]
        e_final = traj["evader"][-1]

        # Check no agent ended up inside any obstacle
        p_inside = False
        e_inside = False
        for obs in traj["obstacles"]:
            dp = np.sqrt((p_final[0] - obs["x"])**2 + (p_final[1] - obs["y"])**2)
            de = np.sqrt((e_final[0] - obs["x"])**2 + (e_final[1] - obs["y"])**2)
            min_dist = obs["radius"] + robot_radius
            if dp < min_dist - 1e-6:
                p_inside = True
            if de < min_dist - 1e-6:
                e_inside = True

        # Check no trajectory point is inside obstacle
        p_violations = 0
        e_violations = 0
        for t in range(len(traj["pursuer"])):
            for obs in traj["obstacles"]:
                dp = np.sqrt((traj["pursuer"][t, 0] - obs["x"])**2 + (traj["pursuer"][t, 1] - obs["y"])**2)
                if dp < obs["radius"] + robot_radius - 1e-6:
                    p_violations += 1
            for obs in traj["obstacles"]:
                de = np.sqrt((traj["evader"][t, 0] - obs["x"])**2 + (traj["evader"][t, 1] - obs["y"])**2)
                if de < obs["radius"] + robot_radius - 1e-6:
                    e_violations += 1

        status = "PASS" if (not p_inside and not e_inside and p_violations == 0 and e_violations == 0) else "FAIL"
        print(f"\n  Scenario {i+1} ({label}): [{status}]")
        print(f"    Pursuer collisions: {n_p} frames, trajectory violations: {p_violations}")
        print(f"    Evader collisions:  {n_e} frames, trajectory violations: {e_violations}")
        if p_inside:
            print(f"    WARNING: Pursuer ended inside obstacle!")
        if e_inside:
            print(f"    WARNING: Evader ended inside obstacle!")

    env.close()


if __name__ == "__main__":
    main()
