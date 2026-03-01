"""Diagnostic: train pursuer against fixed GreedyEvaderPolicy.

Trains a pursuer to chase a greedy flee-from-pursuer evader under the
same observation config as self-play. The resulting model can be used
as a warm-seed for self-play to avoid the cold-start bootstrap problem.
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
from training.baselines import GreedyEvaderPolicy


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
        n_obstacles_min=env_kwargs.get("n_obstacles_min"),
        n_obstacles_max=env_kwargs.get("n_obstacles_max"),
        asymmetric_obs=env_kwargs.get("asymmetric_obs", False),
        sensing_radius=env_kwargs.get("sensing_radius"),
        combined_masking=env_kwargs.get("combined_masking", False),
    )


def make_env(seed, greedy_evader, env_kwargs, fixed_speed=False,
             greedy_full_obs=False):
    """Create a single pursuer env with greedy evader opponent."""
    def _init():
        base_env = _make_base_env(env_kwargs)
        single_env = SingleAgentPEWrapper(
            base_env, role="pursuer", opponent_policy=greedy_evader,
            greedy_full_obs=greedy_full_obs,
        )
        env = single_env
        if fixed_speed:
            env = FixedSpeedWrapper(env, v_max=base_env.pursuer_v_max)
        return Monitor(env)
    return _init


def evaluate(model, greedy_evader, env_kwargs, n_episodes=100,
             fixed_speed=False, greedy_full_obs=False):
    """Evaluate pursuer capture rate against greedy evader."""
    base_env = _make_base_env(env_kwargs)
    single_env = SingleAgentPEWrapper(
        base_env, role="pursuer", opponent_policy=greedy_evader,
        greedy_full_obs=greedy_full_obs,
    )
    env = single_env
    if fixed_speed:
        env = FixedSpeedWrapper(env, v_max=base_env.pursuer_v_max)

    captures = 0
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

        # terminated = capture (pursuer wins)
        if terminated:
            captures += 1
        total_steps_list.append(ep_steps)

    env.close()
    capture_rate = captures / n_episodes
    avg_steps = np.mean(total_steps_list)
    return capture_rate, avg_steps


def main():
    parser = argparse.ArgumentParser(
        description="Train pursuer against fixed greedy evader (diagnostic)")
    parser.add_argument("--output", type=str,
                        default="results/pursuer_vs_greedy",
                        help="Output directory")
    parser.add_argument("--total_steps", type=int, default=5_000_000,
                        help="Total training steps (default: 5M)")
    parser.add_argument("--eval_freq", type=int, default=50_000,
                        help="Evaluate every N steps (default: 50K)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--capture_bonus", type=float, default=50.0,
                        help="Terminal reward for capture (pursuer+)")
    parser.add_argument("--timeout_penalty", type=float, default=-50.0,
                        help="Terminal reward for timeout (pursuer-)")
    parser.add_argument("--ent_coef", type=float, default=0.03,
                        help="PPO entropy coefficient")
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--pursuer_v_max", type=float, default=1.0)
    parser.add_argument("--evader_v_max", type=float, default=1.2)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--fixed_speed", action="store_true",
                        help="Fix v=v_max, only learn omega (1D action).")
    parser.add_argument("--partial_obs", action="store_true",
                        help="Enable LOS-based partial observability.")
    parser.add_argument("--n_obstacles_min", type=int, default=None,
                        help="Minimum obstacle count (randomized).")
    parser.add_argument("--n_obstacles_max", type=int, default=None,
                        help="Maximum obstacle count (randomized).")
    parser.add_argument("--asymmetric_obs", action="store_true",
                        help="Asymmetric LOS: only pursuer is masked.")
    parser.add_argument("--sensing_radius", type=float, default=None,
                        help="Radius-based sensing distance.")
    parser.add_argument("--combined_masking", action="store_true",
                        help="Combined masking: radius + LOS.")
    parser.add_argument("--greedy_full_obs", action="store_true",
                        help="Give greedy evader unmasked observations.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_obstacle_obs = (args.n_obstacles_max if args.n_obstacles_max is not None
                      else args.n_obstacles)

    env_kwargs = {
        "arena_width": args.arena_width,
        "arena_height": args.arena_height,
        "max_steps": args.max_steps,
        "capture_radius": 0.5,
        "n_obstacles": args.n_obstacles,
        "pursuer_v_max": args.pursuer_v_max,
        "evader_v_max": args.evader_v_max,
        "distance_scale": 1.0,
        "timeout_penalty": args.timeout_penalty,
        "capture_bonus": args.capture_bonus,
        "n_obstacle_obs": n_obstacle_obs,
        "partial_obs": args.partial_obs,
        "n_obstacles_min": args.n_obstacles_min,
        "n_obstacles_max": args.n_obstacles_max,
        "asymmetric_obs": args.asymmetric_obs,
        "sensing_radius": args.sensing_radius,
        "combined_masking": args.combined_masking,
    }

    greedy_evader = GreedyEvaderPolicy(
        v_max=args.evader_v_max,
        omega_max=2.84,
        K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    action_mode = ("1D [omega] (fixed speed)" if args.fixed_speed
                   else "2D [v, omega] (variable speed)")
    print("=" * 60)
    print("Diagnostic: Pursuer vs Greedy Evader")
    print(f"  Arena: {args.arena_width}x{args.arena_height}")
    print(f"  Pursuer speed: {args.pursuer_v_max}x")
    print(f"  Evader speed: {args.evader_v_max}x (greedy)")
    print(f"  Action mode: {action_mode}")
    if args.n_obstacles_min is not None and args.n_obstacles_max is not None:
        print(f"  Obstacles: {args.n_obstacles_min}-{args.n_obstacles_max} "
              f"(randomized)")
    else:
        print(f"  Obstacles: {args.n_obstacles}")
    print(f"  Capture bonus: {args.capture_bonus}")
    print(f"  Timeout penalty: {args.timeout_penalty}")
    print(f"  Entropy coef: {args.ent_coef}")
    if args.partial_obs:
        if args.sensing_radius is not None and args.combined_masking:
            masking_type = (f"COMBINED (radius {args.sensing_radius:.1f}m "
                           f"+ LOS)")
        elif args.sensing_radius is not None:
            masking_type = f"RADIUS {args.sensing_radius:.1f}m"
        else:
            masking_type = "LOS"
        obs_mode = ("ASYMMETRIC (pursuer only)" if args.asymmetric_obs
                    else "SYMMETRIC (both agents)")
        print(f"  Partial obs: {masking_type} masking {obs_mode}")
        greedy_obs_mode = ("FULL (unmasked)" if args.greedy_full_obs
                           else "MASKED (same as pursuer)")
        print(f"  Greedy evader obs: {greedy_obs_mode}")
    print(f"  Max episode steps: {args.max_steps}")
    print(f"  Total training: {args.total_steps:,} steps")
    print(f"  Eval every: {args.eval_freq:,} steps")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Baseline: random pursuer vs greedy evader
    print("\nBaseline: Random pursuer vs greedy evader...")
    base_env_test = _make_base_env(env_kwargs)
    rand_captures = 0
    rand_steps = []
    for _ in range(100):
        obs, _ = base_env_test.reset()
        done = False
        ep_steps = 0
        while not done:
            # Give greedy evader unmasked obs if greedy_full_obs is set
            if args.greedy_full_obs:
                e_obs = base_env_test.obs_builder.build(
                    self_state=base_env_test.evader_state,
                    self_action=base_env_test.evader_action,
                    opp_state=base_env_test.pursuer_state,
                    opp_action=base_env_test.pursuer_action,
                    obstacles=base_env_test.obstacles,
                    los_blocked=False,
                )
            else:
                e_obs = obs["evader"]
            e_action = greedy_evader.predict(e_obs)[0]
            p_action = np.array([args.pursuer_v_max,
                                 np.random.uniform(-2.84, 2.84)],
                                dtype=np.float32)
            obs, rewards, terminated, truncated, info = base_env_test.step(
                p_action, e_action)
            done = terminated or truncated
            ep_steps += 1
        if terminated:
            rand_captures += 1
        rand_steps.append(ep_steps)
    base_env_test.close()
    print(f"  Random capture rate: {rand_captures/100:.2f}, "
          f"avg ep len: {np.mean(rand_steps):.0f} steps")

    # Create vectorized training env
    vec_env = DummyVecEnv([
        make_env(args.seed + i, greedy_evader, env_kwargs,
                 fixed_speed=args.fixed_speed,
                 greedy_full_obs=args.greedy_full_obs)
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

    print(f"\nTraining pursuer ({args.total_steps:,} steps)...")
    print(f"{'Step':>10s} | {'Capture Rate':>12s} | {'Avg Ep Len':>10s} | "
          f"{'Time':>6s}")
    print("-" * 50)

    start_time = time.time()
    steps_trained = 0
    log_data = []
    best_capture_rate = 0.0
    best_capture_step = 0

    while steps_trained < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        steps_trained += chunk

        capture_rate, avg_steps = evaluate(
            model, greedy_evader, env_kwargs, n_episodes=100,
            fixed_speed=args.fixed_speed,
            greedy_full_obs=args.greedy_full_obs)
        elapsed = time.time() - start_time

        # Best-model checkpointing
        is_best = capture_rate > best_capture_rate
        if is_best:
            best_capture_rate = capture_rate
            best_capture_step = steps_trained
            model.save(str(output_dir / "pursuer_best"))

        log_data.append({
            "steps": steps_trained,
            "capture_rate": capture_rate,
            "avg_ep_len": avg_steps,
            "elapsed": elapsed,
            "best_capture_rate": best_capture_rate,
            "best_capture_step": best_capture_step,
        })

        best_marker = " *BEST*" if is_best else ""
        print(f"{steps_trained:>10,} | {capture_rate:>12.3f} | "
              f"{avg_steps:>10.1f} | {elapsed/60:>5.1f}m{best_marker}")

    vec_env.close()

    # Save model and results
    model.save(str(output_dir / "pursuer_final"))
    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f}m")
    print(f"Final capture rate: {log_data[-1]['capture_rate']:.3f}")
    print(f"Best capture rate:  {best_capture_rate:.3f} "
          f"(at step {best_capture_step:,})")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
