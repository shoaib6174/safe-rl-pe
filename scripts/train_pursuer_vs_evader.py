"""Diagnostic: train pursuer against fixed learned evader (S1 model).

Tests whether a learned pursuer can counter obstacle-hugging evasion
at equal speed. The S1 evader orbits obstacles tightly; the pursuer
must learn to navigate around them to capture.
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


class PPOOpponentPolicy:
    """Wraps a PPO model trained with FixedSpeedWrapper as an opponent.

    The PPO model outputs 1D [omega] actions, but SingleAgentPEWrapper
    expects 2D [v, omega]. This adapter converts between them.
    """

    def __init__(self, model_path: str, v_max: float = 1.0):
        self.model = PPO.load(model_path, device="cpu")
        self.v_max = v_max

    def predict(self, obs, deterministic=False):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        # Model outputs [omega], convert to [v_max, omega]
        full_action = np.array([self.v_max, action[0]], dtype=np.float32)
        return full_action, None


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
        timeout_penalty=env_kwargs.get("timeout_penalty", 0.0),
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
    )


def make_env(seed, evader_opponent, env_kwargs):
    """Create a single pursuer env with fixed evader opponent."""
    def _init():
        base_env = _make_base_env(env_kwargs)
        single_env = SingleAgentPEWrapper(
            base_env, role="pursuer", opponent_policy=evader_opponent,
        )
        # Fixed speed: pursuer only controls omega
        v_max = base_env.pursuer_v_max
        env = FixedSpeedWrapper(single_env, v_max=v_max)
        return Monitor(env)
    return _init


def evaluate(model, evader_opponent, env_kwargs, n_episodes=100):
    """Evaluate pursuer capture rate against fixed evader."""
    base_env = _make_base_env(env_kwargs)
    single_env = SingleAgentPEWrapper(
        base_env, role="pursuer", opponent_policy=evader_opponent,
    )
    env = FixedSpeedWrapper(single_env, v_max=base_env.pursuer_v_max)

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

        # terminated = capture
        if terminated:
            captures += 1
        total_steps_list.append(ep_steps)

    env.close()
    capture_rate = captures / n_episodes
    avg_steps = np.mean(total_steps_list)
    return capture_rate, avg_steps


def main():
    parser = argparse.ArgumentParser(
        description="Train pursuer against fixed learned evader (diagnostic)")
    parser.add_argument("--evader_model", type=str, required=True,
                        help="Path to trained evader model (.zip)")
    parser.add_argument("--output", type=str,
                        default="results/pursuer_vs_evader",
                        help="Output directory")
    parser.add_argument("--total_steps", type=int, default=3_000_000,
                        help="Total training steps (default: 3M)")
    parser.add_argument("--eval_freq", type=int, default=50_000,
                        help="Evaluate every N steps (default: 50K)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--capture_bonus", type=float, default=50.0,
                        help="Pursuer reward for capture")
    parser.add_argument("--timeout_penalty", type=float, default=-10.0,
                        help="Pursuer penalty for timeout (failed capture)")
    parser.add_argument("--survival_bonus", type=float, default=0.1,
                        help="Evader survival bonus (for evader reward only)")
    parser.add_argument("--visibility_weight", type=float, default=0.5,
                        help="Visibility reward weight (for evader reward)")
    parser.add_argument("--ent_coef", type=float, default=0.03,
                        help="PPO entropy coefficient")
    parser.add_argument("--arena_width", type=float, default=10.0)
    parser.add_argument("--arena_height", type=float, default=10.0)
    parser.add_argument("--pursuer_v_max", type=float, default=1.0)
    parser.add_argument("--evader_v_max", type=float, default=1.0)
    parser.add_argument("--n_obstacles", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "arena_width": args.arena_width,
        "arena_height": args.arena_height,
        "max_steps": args.max_steps,
        "capture_radius": 0.5,
        "n_obstacles": args.n_obstacles,
        "pursuer_v_max": args.pursuer_v_max,
        "evader_v_max": args.evader_v_max,
        "distance_scale": 1.0,
        "use_visibility_reward": True,
        "visibility_weight": args.visibility_weight,
        "survival_bonus": args.survival_bonus,
        "timeout_penalty": args.timeout_penalty,
        "capture_bonus": args.capture_bonus,
        "n_obstacle_obs": args.n_obstacles,
    }

    # Load fixed evader opponent
    print(f"Loading evader model: {args.evader_model}")
    evader_opponent = PPOOpponentPolicy(
        args.evader_model, v_max=args.evader_v_max,
    )

    print("=" * 60)
    print("Diagnostic: Pursuer vs Fixed Learned Evader")
    print(f"  Arena: {args.arena_width}x{args.arena_height}")
    print(f"  Pursuer speed: {args.pursuer_v_max}x")
    print(f"  Evader speed: {args.evader_v_max}x (fixed S1 model)")
    print(f"  Obstacles: {args.n_obstacles}")
    print(f"  Capture bonus: {args.capture_bonus}")
    print(f"  Timeout penalty: {args.timeout_penalty}")
    print(f"  Entropy coef: {args.ent_coef}")
    print(f"  Max episode steps: {args.max_steps}")
    print(f"  Total training: {args.total_steps:,} steps")
    print(f"  Eval every: {args.eval_freq:,} steps")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Baseline: greedy pursuer vs S1 evader
    print("\nBaseline: Greedy pursuer vs S1 evader...")
    from training.baselines import GreedyPursuerPolicy
    greedy_pursuer = GreedyPursuerPolicy(
        v_max=args.pursuer_v_max,
        omega_max=2.84,
        K_p=3.0,
        arena_half_w=args.arena_width / 2,
        arena_half_h=args.arena_height / 2,
    )

    base_env_test = _make_base_env(env_kwargs)
    greedy_captures = 0
    greedy_steps = []
    for _ in range(100):
        obs, _ = base_env_test.reset()
        done = False
        ep_steps = 0
        while not done:
            p_action = greedy_pursuer.predict(obs["pursuer"])[0]
            e_action = evader_opponent.predict(obs["evader"], deterministic=True)[0]
            obs, rewards, terminated, truncated, info = base_env_test.step(
                p_action, e_action)
            done = terminated or truncated
            ep_steps += 1
        if terminated:
            greedy_captures += 1
        greedy_steps.append(ep_steps)
    base_env_test.close()
    print(f"  Greedy capture rate: {greedy_captures/100:.2f}, "
          f"avg ep len: {np.mean(greedy_steps):.0f} steps")

    # Baseline: random pursuer vs S1 evader
    print("Baseline: Random pursuer vs S1 evader...")
    base_env_test = _make_base_env(env_kwargs)
    rand_captures = 0
    rand_steps = []
    for _ in range(100):
        obs, _ = base_env_test.reset()
        done = False
        ep_steps = 0
        while not done:
            p_action = np.array([args.pursuer_v_max,
                                 np.random.uniform(-2.84, 2.84)],
                                dtype=np.float32)
            e_action = evader_opponent.predict(obs["evader"], deterministic=True)[0]
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
        make_env(args.seed + i, evader_opponent, env_kwargs)
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

    while steps_trained < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        steps_trained += chunk

        capture_rate, avg_steps = evaluate(
            model, evader_opponent, env_kwargs, n_episodes=100)
        elapsed = time.time() - start_time

        log_data.append({
            "steps": steps_trained,
            "capture_rate": capture_rate,
            "avg_ep_len": avg_steps,
            "elapsed": elapsed,
        })

        # Save best checkpoint
        if capture_rate > best_capture_rate:
            best_capture_rate = capture_rate
            model.save(str(output_dir / "pursuer_best"))

        marker = " *" if capture_rate >= best_capture_rate else ""
        print(f"{steps_trained:>10,} | {capture_rate:>12.3f} | "
              f"{avg_steps:>10.1f} | {elapsed/60:>5.1f}m{marker}")

    vec_env.close()

    # Save final model and results
    model.save(str(output_dir / "pursuer_final"))
    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f}m")
    print(f"Final capture rate: {log_data[-1]['capture_rate']:.3f}")
    print(f"Best capture rate: {best_capture_rate:.3f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
