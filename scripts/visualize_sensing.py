"""Visualize sensing mechanics: radius circles, LOS lines, visibility status.

Generates an animated GIF showing a single episode with detailed sensing
information. Useful for debugging and understanding how radius-based and
combined masking work.

Usage:
    # Radius-only sensing
    ./venv/bin/python scripts/visualize_sensing.py \
        --model results/S1v5_radius_sensing/evader_best.zip \
        --sensing_radius 3.0 --partial_obs

    # Combined masking (radius + LOS)
    ./venv/bin/python scripts/visualize_sensing.py \
        --model results/S1v5b_combined/evader_best.zip \
        --sensing_radius 3.0 --combined_masking --partial_obs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from stable_baselines3 import PPO

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer, line_of_sight_blocked
from envs.wrappers import SingleAgentPEWrapper, FixedSpeedWrapper
from training.baselines import GreedyPursuerPolicy


def run_episode(model, greedy_pursuer, env_kwargs, seed=None):
    """Run one episode recording full sensing state at each step."""
    arena_w = env_kwargs["arena_width"]
    arena_h = env_kwargs["arena_height"]
    diagonal = np.sqrt(arena_w**2 + arena_h**2)

    reward_computer = RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 0.1),
        survival_bonus=env_kwargs.get("survival_bonus", 0.03),
        timeout_penalty=env_kwargs.get("timeout_penalty", 0.0),
        capture_bonus=env_kwargs.get("capture_bonus", 5.0),
    )

    base_env = PursuitEvasionEnv(
        arena_width=arena_w,
        arena_height=arena_h,
        max_steps=env_kwargs.get("max_steps", 600),
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 2),
        pursuer_v_max=env_kwargs.get("pursuer_v_max", 1.0),
        evader_v_max=env_kwargs.get("evader_v_max", 1.0),
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs", 2),
        reward_computer=reward_computer,
        partial_obs=env_kwargs.get("partial_obs", False),
        n_obstacles_min=env_kwargs.get("n_obstacles_min"),
        n_obstacles_max=env_kwargs.get("n_obstacles_max"),
        asymmetric_obs=env_kwargs.get("asymmetric_obs", False),
        sensing_radius=env_kwargs.get("sensing_radius"),
        combined_masking=env_kwargs.get("combined_masking", False),
    )
    if seed is not None:
        base_env.np_random = np.random.default_rng(seed)

    single_env = SingleAgentPEWrapper(
        base_env, role="evader", opponent_policy=greedy_pursuer,
    )
    action_dim = model.action_space.shape[0]
    if action_dim == 1:
        env = FixedSpeedWrapper(single_env, v_max=base_env.evader_v_max)
    else:
        env = single_env

    obs, _ = env.reset()
    done = False

    sensing_radius = env_kwargs.get("sensing_radius")
    combined = env_kwargs.get("combined_masking", False)

    # Record per-step data
    frames = []
    obstacles = [(o["x"], o["y"], o["radius"]) for o in base_env.obstacles]

    def record_frame():
        px, py, ptheta = base_env.pursuer_state
        ex, ey, etheta = base_env.evader_state
        dist = float(np.sqrt((px - ex)**2 + (py - ey)**2))

        in_range = True
        if sensing_radius is not None:
            in_range = dist <= sensing_radius

        los_clear = not line_of_sight_blocked(
            base_env.pursuer_state, base_env.evader_state, base_env.obstacles,
        )

        if combined and sensing_radius is not None:
            visible = in_range and los_clear
        elif sensing_radius is not None:
            visible = in_range
        elif env_kwargs.get("partial_obs", False):
            visible = los_clear
        else:
            visible = True

        frames.append({
            "px": px, "py": py, "ptheta": ptheta,
            "ex": ex, "ey": ey, "etheta": etheta,
            "distance": dist,
            "in_range": in_range,
            "los_clear": los_clear,
            "visible": visible,
        })

    record_frame()

    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        record_frame()

    env.close()

    return {
        "frames": frames,
        "obstacles": obstacles,
        "steps": steps,
        "escaped": truncated,
        "captured": terminated,
        "arena_w": arena_w,
        "arena_h": arena_h,
        "sensing_radius": sensing_radius,
        "combined": combined,
        "partial_obs": env_kwargs.get("partial_obs", False),
    }


def make_sensing_gif(episode, output_path, fps=20, skip=2):
    """Create animated GIF with detailed sensing visualization."""
    frames = episode["frames"]
    obstacles = episode["obstacles"]
    arena_w = episode["arena_w"]
    arena_h = episode["arena_h"]
    half_w = arena_w / 2
    half_h = arena_h / 2
    sensing_r = episode["sensing_radius"]
    combined = episode["combined"]
    partial_obs = episode["partial_obs"]

    n_frames = len(frames)
    frame_indices = list(range(0, n_frames, skip))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Determine masking mode for title
    if sensing_r is not None and combined:
        mode_str = f"Combined (R={sensing_r:.1f}m + LOS)"
    elif sensing_r is not None:
        mode_str = f"Radius only (R={sensing_r:.1f}m)"
    elif partial_obs:
        mode_str = "LOS only"
    else:
        mode_str = "Full observability"

    # Collect all positions for trail
    all_px = [f["px"] for f in frames]
    all_py = [f["py"] for f in frames]
    all_ex = [f["ex"] for f in frames]
    all_ey = [f["ey"] for f in frames]

    def draw_frame(frame_idx):
        fi = frame_indices[frame_idx]
        f = frames[fi]
        ax.clear()

        # Arena
        ax.set_xlim(-half_w - 1.0, half_w + 1.0)
        ax.set_ylim(-half_h - 1.0, half_h + 1.0)
        arena_rect = patches.Rectangle(
            (-half_w, -half_h), arena_w, arena_h,
            linewidth=2, edgecolor="black", facecolor="#FFFDE7",
        )
        ax.add_patch(arena_rect)

        # Obstacles
        for ox, oy, r in obstacles:
            obs_circle = patches.Circle(
                (ox, oy), r, facecolor="#B0BEC5", edgecolor="#37474F",
                linewidth=1.5, alpha=0.8, zorder=3,
            )
            ax.add_patch(obs_circle)

        # Sensing radius circles (dashed)
        if sensing_r is not None:
            # Pursuer sensing radius
            p_sense = patches.Circle(
                (f["px"], f["py"]), sensing_r,
                facecolor="red", edgecolor="red",
                linewidth=1.5, linestyle="--", alpha=0.06, zorder=2,
            )
            ax.add_patch(p_sense)
            p_sense_edge = patches.Circle(
                (f["px"], f["py"]), sensing_r,
                facecolor="none", edgecolor="red",
                linewidth=1.5, linestyle="--", alpha=0.4, zorder=2,
            )
            ax.add_patch(p_sense_edge)

            # Evader sensing radius
            e_sense = patches.Circle(
                (f["ex"], f["ey"]), sensing_r,
                facecolor="blue", edgecolor="blue",
                linewidth=1.5, linestyle="--", alpha=0.06, zorder=2,
            )
            ax.add_patch(e_sense)
            e_sense_edge = patches.Circle(
                (f["ex"], f["ey"]), sensing_r,
                facecolor="none", edgecolor="blue",
                linewidth=1.5, linestyle="--", alpha=0.4, zorder=2,
            )
            ax.add_patch(e_sense_edge)

        # LOS line between agents
        if f["visible"]:
            ax.plot(
                [f["px"], f["ex"]], [f["py"], f["ey"]],
                "-", color="#4CAF50", linewidth=2.0, alpha=0.7, zorder=4,
            )
        else:
            ax.plot(
                [f["px"], f["ex"]], [f["py"], f["ey"]],
                "--", color="#F44336", linewidth=1.5, alpha=0.5, zorder=4,
            )

        # Trajectory trails (fading)
        trail_len = 150
        trail_start = max(0, fi - trail_len)
        if fi > 0:
            # Pursuer trail
            trail_px = all_px[trail_start:fi + 1]
            trail_py = all_py[trail_start:fi + 1]
            n_trail = len(trail_px)
            for j in range(n_trail - 1):
                alpha = 0.1 + 0.4 * (j / max(n_trail - 1, 1))
                ax.plot(
                    trail_px[j:j+2], trail_py[j:j+2],
                    "-", color="red", linewidth=1.0, alpha=alpha,
                )
            # Evader trail
            trail_ex = all_ex[trail_start:fi + 1]
            trail_ey = all_ey[trail_start:fi + 1]
            for j in range(n_trail - 1):
                alpha = 0.1 + 0.4 * (j / max(n_trail - 1, 1))
                ax.plot(
                    trail_ex[j:j+2], trail_ey[j:j+2],
                    "-", color="blue", linewidth=1.0, alpha=alpha,
                )

        # Agent bodies
        agent_r = 0.25
        pursuer_body = patches.Circle(
            (f["px"], f["py"]), agent_r,
            facecolor="#EF5350", edgecolor="#B71C1C",
            linewidth=2, zorder=6,
        )
        ax.add_patch(pursuer_body)
        evader_body = patches.Circle(
            (f["ex"], f["ey"]), agent_r,
            facecolor="#42A5F5", edgecolor="#0D47A1",
            linewidth=2, zorder=6,
        )
        ax.add_patch(evader_body)

        # Heading arrows
        arrow_len = 0.6
        ax.annotate(
            "", xy=(f["px"] + arrow_len * np.cos(f["ptheta"]),
                     f["py"] + arrow_len * np.sin(f["ptheta"])),
            xytext=(f["px"], f["py"]),
            arrowprops=dict(arrowstyle="-|>", color="#B71C1C", lw=2),
            zorder=7,
        )
        ax.annotate(
            "", xy=(f["ex"] + arrow_len * np.cos(f["etheta"]),
                     f["ey"] + arrow_len * np.sin(f["etheta"])),
            xytext=(f["ex"], f["ey"]),
            arrowprops=dict(arrowstyle="-|>", color="#0D47A1", lw=2),
            zorder=7,
        )

        # Capture X marker
        ep_done = fi == n_frames - 1
        if ep_done and episode["captured"]:
            ax.plot(
                f["ex"], f["ey"], "x", color="black",
                markersize=20, markeredgewidth=4, zorder=8,
            )

        # Agent labels
        ax.text(
            f["px"], f["py"] + agent_r + 0.25, "P",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#B71C1C", zorder=7,
        )
        ax.text(
            f["ex"], f["ey"] + agent_r + 0.25, "E",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#0D47A1", zorder=7,
        )

        # --- Info panel (text on plot) ---
        info_x = -half_w - 0.8
        info_y = half_h + 0.6

        # Step counter and outcome
        if ep_done:
            outcome = "ESCAPED" if episode["escaped"] else "CAPTURED"
            outcome_color = "#2E7D32" if episode["escaped"] else "#C62828"
        else:
            outcome = "..."
            outcome_color = "#333333"

        # Title
        ax.set_title(
            f"Sensing Visualization  |  {mode_str}\n"
            f"Step {fi}/{episode['steps']}  |  "
            f"d = {f['distance']:.2f}m  |  {outcome}",
            fontsize=13, fontweight="bold", color=outcome_color,
            pad=12,
        )

        # Status indicators (bottom of plot)
        status_y = -half_h - 0.7
        status_items = []

        if sensing_r is not None:
            range_str = "IN RANGE" if f["in_range"] else "OUT OF RANGE"
            range_color = "#2E7D32" if f["in_range"] else "#C62828"
            status_items.append((range_str, range_color))

        if partial_obs or combined:
            los_str = "LOS CLEAR" if f["los_clear"] else "LOS BLOCKED"
            los_color = "#2E7D32" if f["los_clear"] else "#C62828"
            status_items.append((los_str, los_color))

        vis_str = "VISIBLE" if f["visible"] else "MASKED"
        vis_color = "#2E7D32" if f["visible"] else "#C62828"
        status_items.append((vis_str, vis_color))

        n_items = len(status_items)
        total_width = arena_w + 2.0
        spacing = total_width / (n_items + 1)
        for idx, (text, color) in enumerate(status_items):
            x_pos = -half_w - 1.0 + spacing * (idx + 1)
            # Background box
            bbox = dict(
                boxstyle="round,pad=0.3", facecolor=color,
                edgecolor="none", alpha=0.15,
            )
            ax.text(
                x_pos, status_y, text,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color, bbox=bbox, zorder=10,
            )

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.12, linewidth=0.5)
        ax.tick_params(labelsize=8)

    print(f"Generating {len(frame_indices)} frames...")
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(frame_indices),
        interval=1000 // fps, repeat=False,
    )

    plt.tight_layout()
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize sensing mechanics (radius, LOS, combined)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained evader model (.zip)")
    parser.add_argument("--output", type=str, default="sensing_vis.gif",
                        help="Output GIF path (default: sensing_vis.gif)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--evader_v_max", type=float, default=1.0)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--n_obstacles_min", type=int, default=None)
    parser.add_argument("--n_obstacles_max", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--visibility_weight", type=float, default=0.5)
    parser.add_argument("--partial_obs", action="store_true",
                        help="Enable partial observability")
    parser.add_argument("--sensing_radius", type=float, default=None,
                        help="Sensing radius in meters")
    parser.add_argument("--combined_masking", action="store_true",
                        help="Combined masking: radius + LOS")
    parser.add_argument("--asymmetric_obs", action="store_true",
                        help="Asymmetric obs: only pursuer masked")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip", type=int, default=2,
                        help="Render every Nth step (default: 2)")
    args = parser.parse_args()

    n_obstacle_obs = args.n_obstacles_max if args.n_obstacles_max is not None else args.n_obstacles

    env_kwargs = {
        "arena_width": args.arena_width,
        "arena_height": args.arena_height,
        "max_steps": args.max_steps,
        "capture_radius": 0.5,
        "n_obstacles": args.n_obstacles,
        "pursuer_v_max": 1.0,
        "evader_v_max": args.evader_v_max,
        "distance_scale": 1.0,
        "use_visibility_reward": True,
        "visibility_weight": args.visibility_weight,
        "survival_bonus": 0.03,
        "timeout_penalty": 0.0,
        "capture_bonus": 5.0,
        "n_obstacle_obs": n_obstacle_obs,
        "partial_obs": args.partial_obs,
        "n_obstacles_min": args.n_obstacles_min,
        "n_obstacles_max": args.n_obstacles_max,
        "asymmetric_obs": args.asymmetric_obs,
        "sensing_radius": args.sensing_radius,
        "combined_masking": args.combined_masking,
    }

    greedy_pursuer = GreedyPursuerPolicy(
        v_max=1.0, omega_max=2.84, K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    print(f"Running episode (seed={args.seed})...")
    episode = run_episode(model, greedy_pursuer, env_kwargs, seed=args.seed)

    status = "ESCAPED" if episode["escaped"] else f"CAPTURED at step {episode['steps']}"
    print(f"  Result: {status}")

    # Stats summary
    n_visible = sum(1 for f in episode["frames"] if f["visible"])
    n_total = len(episode["frames"])
    print(f"  Visible: {n_visible}/{n_total} frames ({100*n_visible/n_total:.1f}%)")

    if episode["sensing_radius"] is not None:
        n_in_range = sum(1 for f in episode["frames"] if f["in_range"])
        print(f"  In range: {n_in_range}/{n_total} frames ({100*n_in_range/n_total:.1f}%)")

    if episode["partial_obs"] or episode["combined"]:
        n_los = sum(1 for f in episode["frames"] if f["los_clear"])
        print(f"  LOS clear: {n_los}/{n_total} frames ({100*n_los/n_total:.1f}%)")

    make_sensing_gif(episode, args.output, fps=args.fps, skip=args.skip)


if __name__ == "__main__":
    main()
