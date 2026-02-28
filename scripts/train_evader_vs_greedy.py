"""Diagnostic: train evader against fixed GreedyPursuerPolicy.

Tests whether the evader can learn evasion at all when the opponent
is stable (no co-evolution). If the evader can't beat a greedy pursuer,
the problem is fundamental (reward/obs/architecture). If it can, the
problem is purely self-play dynamics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper, FixedSpeedWrapper
from training.baselines import GreedyPursuerPolicy


def _make_reward_computer(env_kwargs):
    """Create RewardComputer from env_kwargs dict."""
    arena_w = env_kwargs["arena_width"]
    arena_h = env_kwargs["arena_height"]
    diagonal = np.sqrt(arena_w**2 + arena_h**2)
    return RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 0.1),
        survival_bonus=env_kwargs.get("survival_bonus", 0.0),
        timeout_penalty=env_kwargs.get("timeout_penalty", -50.0),
        capture_bonus=env_kwargs.get("capture_bonus", 50.0),
    )


def _make_base_env(env_kwargs):
    """Create PursuitEvasionEnv from env_kwargs dict."""
    reward_computer = _make_reward_computer(env_kwargs)
    return PursuitEvasionEnv(
        arena_width=env_kwargs["arena_width"],
        arena_height=env_kwargs["arena_height"],
        max_steps=env_kwargs.get("max_steps", 600),
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 2),
        pursuer_v_max=env_kwargs.get("pursuer_v_max", 1.0),
        evader_v_max=env_kwargs.get("evader_v_max", 1.0),
        n_obstacle_obs=env_kwargs.get("n_obstacle_obs", 2),
        reward_computer=reward_computer,
        partial_obs=env_kwargs.get("partial_obs", False),
    )


def make_env(seed, greedy_pursuer, env_kwargs, fixed_speed=False):
    """Create a single evader env with greedy pursuer opponent."""
    def _init():
        base_env = _make_base_env(env_kwargs)
        single_env = SingleAgentPEWrapper(
            base_env, role="evader", opponent_policy=greedy_pursuer,
        )
        env = single_env
        if fixed_speed:
            env = FixedSpeedWrapper(env, v_max=base_env.evader_v_max)
        return Monitor(env)
    return _init


def evaluate(model, greedy_pursuer, env_kwargs, n_episodes=100,
             fixed_speed=False):
    """Evaluate evader escape rate against greedy pursuer."""
    base_env = _make_base_env(env_kwargs)
    single_env = SingleAgentPEWrapper(
        base_env, role="evader", opponent_policy=greedy_pursuer,
    )
    env = single_env
    if fixed_speed:
        env = FixedSpeedWrapper(env, v_max=base_env.evader_v_max)

    escapes = 0
    total_steps_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1

        # truncated = timeout = escape
        if truncated:
            escapes += 1
        total_steps_list.append(ep_steps)

    env.close()
    escape_rate = escapes / n_episodes
    avg_steps = np.mean(total_steps_list)
    return escape_rate, avg_steps


def main():
    parser = argparse.ArgumentParser(
        description="Train evader against fixed greedy pursuer (diagnostic)")
    parser.add_argument("--output", type=str,
                        default="results/evader_vs_greedy",
                        help="Output directory")
    parser.add_argument("--total_steps", type=int, default=2_000_000,
                        help="Total training steps (default: 2M)")
    parser.add_argument("--eval_freq", type=int, default=50_000,
                        help="Evaluate every N steps (default: 50K)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--survival_bonus", type=float, default=0.03,
                        help="Evader survival bonus per step")
    parser.add_argument("--visibility_weight", type=float, default=0.1,
                        help="Visibility reward weight")
    parser.add_argument("--capture_bonus", type=float, default=50.0,
                        help="Terminal reward for capture (pursuer+, evader-)")
    parser.add_argument("--timeout_penalty", type=float, default=-50.0,
                        help="Terminal reward for timeout (pursuer-, evader+)")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="PPO entropy coefficient")
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--evader_v_max", type=float, default=1.05)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--fixed_speed", action="store_true",
                        help="Fix v=v_max, only learn omega (1D action). "
                             "Default: variable speed (2D action [v, omega]).")
    parser.add_argument("--partial_obs", action="store_true",
                        help="Enable LOS-based partial observability (mask opponent "
                             "when occluded by obstacle).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "survival_bonus": args.survival_bonus,
        "timeout_penalty": args.timeout_penalty,
        "capture_bonus": args.capture_bonus,
        "n_obstacle_obs": args.n_obstacles,
        "partial_obs": args.partial_obs,
    }

    # Create greedy pursuer opponent
    greedy_pursuer = GreedyPursuerPolicy(
        v_max=1.0,
        omega_max=2.84,
        K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    action_mode = "1D [omega] (fixed speed)" if args.fixed_speed else "2D [v, omega] (variable speed)"
    print("=" * 60)
    print("Diagnostic: Evader vs Greedy Pursuer")
    print(f"  Arena: {args.arena_width}x{args.arena_height}")
    print(f"  Evader speed: {args.evader_v_max}x")
    print(f"  Action mode: {action_mode}")
    print(f"  Obstacles: {args.n_obstacles}")
    print(f"  Visibility weight: {args.visibility_weight}")
    print(f"  Survival bonus: {args.survival_bonus}")
    print(f"  Capture bonus: {args.capture_bonus}")
    print(f"  Timeout penalty: {args.timeout_penalty}")
    print(f"  Entropy coef: {args.ent_coef}")
    if args.partial_obs:
        print(f"  Partial obs: LOS masking ENABLED")
    print(f"  Max episode steps: {args.max_steps}")
    print(f"  Total training: {args.total_steps:,} steps")
    print(f"  Eval every: {args.eval_freq:,} steps")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Baseline: random evader vs greedy pursuer
    print("\nBaseline: Random evader vs greedy pursuer...")
    base_env_test = _make_base_env(env_kwargs)
    rand_escapes = 0
    rand_steps = []
    for _ in range(100):
        obs, _ = base_env_test.reset()
        done = False
        ep_steps = 0
        while not done:
            p_action = greedy_pursuer.predict(obs["pursuer"])[0]
            e_action = np.array([args.evader_v_max,
                                 np.random.uniform(-2.84, 2.84)],
                                dtype=np.float32)
            obs, rewards, terminated, truncated, info = base_env_test.step(
                p_action, e_action)
            done = terminated or truncated
            ep_steps += 1
        if truncated:
            rand_escapes += 1
        rand_steps.append(ep_steps)
    base_env_test.close()
    print(f"  Random escape rate: {rand_escapes/100:.2f}, "
          f"avg survival: {np.mean(rand_steps):.0f} steps")

    # Create vectorized training env
    vec_env = DummyVecEnv([
        make_env(args.seed + i, greedy_pursuer, env_kwargs,
                 fixed_speed=args.fixed_speed)
        for i in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=0,
    )

    print(f"\nTraining evader ({args.total_steps:,} steps)...")
    print(f"{'Step':>10s} | {'Escape Rate':>11s} | {'Avg Ep Len':>10s} | "
          f"{'Time':>6s}")
    print("-" * 50)

    start_time = time.time()
    steps_trained = 0
    log_data = []

    while steps_trained < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        steps_trained += chunk

        escape_rate, avg_steps = evaluate(
            model, greedy_pursuer, env_kwargs, n_episodes=100,
            fixed_speed=args.fixed_speed)
        elapsed = time.time() - start_time

        log_data.append({
            "steps": steps_trained,
            "escape_rate": escape_rate,
            "avg_ep_len": avg_steps,
            "elapsed": elapsed,
        })

        print(f"{steps_trained:>10,} | {escape_rate:>11.3f} | "
              f"{avg_steps:>10.1f} | {elapsed/60:>5.1f}m")

    vec_env.close()

    # Save model and results
    model.save(str(output_dir / "evader_final"))
    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f}m")
    print(f"Final escape rate: {log_data[-1]['escape_rate']:.3f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
