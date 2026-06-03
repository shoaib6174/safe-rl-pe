"""Warm-start BR training: load pursuer from checkpoint, freeze evader,
continue training under full partial obs (no curriculum).

This addresses the post-curriculum drift by warm-starting the pursuer
from a curriculum-trained checkpoint instead of training from scratch.

Usage:
    PYTHONPATH=. python scripts/train_br_warmstart.py \\
        --pursuer_checkpoint results/SP17b_s50/checkpoints/pursuer/milestone_phase2550_pursuer \\
        --evader_checkpoint results/SP17b_s50/checkpoints/evader/milestone_phase2550_evader \\
        --total_steps 3000000 --seed 500 --output_dir results/BR_ws_s50_pursuer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper


COHORT_ENV_KWARGS = {
    "arena_width": 20.0,
    "arena_height": 20.0,
    "max_steps": 1200,
    "capture_radius": 0.5,
    "n_obstacles": 0,
    "n_obstacles_min": 0,
    "n_obstacles_max": 3,
    "n_obstacle_obs": 0,
    "pursuer_v_max": 1.0,
    "evader_v_max": 1.0,
    "distance_scale": 1.0,
    "use_visibility_reward": False,
    "visibility_weight": 1.0,
    "survival_bonus": 0.0,
    "timeout_penalty": -100.0,
    "capture_bonus": 100.0,
    "partial_obs": True,
    "sensing_radius": 8.0,
    "combined_masking": True,
    "fov_angle": 90.0,
    "w_search": 0.0001,
    "t_stale": 100,
    "w_vis_pursuer": 0.2,
    "history_length": 10,
}

COHORT_SAC_KWARGS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "learning_starts": 10_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 0.03,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": {"net_arch": [256, 256]},
}


@dataclass
class WarmStartSetup:
    base_env: PursuitEvasionEnv
    wrapper: Monitor
    frozen_model: SAC
    learner: SAC
    output_dir: Path


def _make_reward_computer(env_kwargs: dict) -> RewardComputer:
    diagonal = float(np.sqrt(env_kwargs["arena_width"] ** 2
                             + env_kwargs["arena_height"] ** 2))
    return RewardComputer(
        distance_scale=env_kwargs.get("distance_scale", 1.0),
        d_max=diagonal,
        use_visibility_reward=env_kwargs.get("use_visibility_reward", False),
        visibility_weight=env_kwargs.get("visibility_weight", 1.0),
        survival_bonus=env_kwargs.get("survival_bonus", 0.0),
        timeout_penalty=env_kwargs.get("timeout_penalty", -100.0),
        capture_bonus=env_kwargs.get("capture_bonus", 100.0),
        w_vis_pursuer=env_kwargs.get("w_vis_pursuer", 0.0),
        sensing_radius=env_kwargs.get("sensing_radius", None),
    )


def _make_base_env(env_kwargs: dict) -> PursuitEvasionEnv:
    rc = _make_reward_computer(env_kwargs)
    return PursuitEvasionEnv(
        arena_width=env_kwargs["arena_width"],
        arena_height=env_kwargs["arena_height"],
        max_steps=env_kwargs["max_steps"],
        capture_radius=env_kwargs.get("capture_radius", 0.5),
        n_obstacles=env_kwargs.get("n_obstacles", 0),
        pursuer_v_max=env_kwargs["pursuer_v_max"],
        evader_v_max=env_kwargs["evader_v_max"],
        reward_computer=rc,
        partial_obs=env_kwargs["partial_obs"],
        n_obstacles_min=env_kwargs.get("n_obstacles_min"),
        n_obstacles_max=env_kwargs.get("n_obstacles_max"),
        asymmetric_obs=False,
        sensing_radius=env_kwargs.get("sensing_radius"),
        combined_masking=env_kwargs.get("combined_masking", False),
        fov_angle=env_kwargs.get("fov_angle", None),
        w_search=env_kwargs.get("w_search", 0.0),
        t_stale=env_kwargs.get("t_stale", 50),
    )


def build_warmstart_setup(
    pursuer_checkpoint: str,
    evader_checkpoint: str,
    seed: int,
    output_dir: str,
    env_kwargs: dict | None = None,
    clear_buffer: bool = True,
) -> WarmStartSetup:
    """Load warm-start pursuer, freeze evader, build training env.

    Args:
        pursuer_checkpoint: Dir containing ppo.zip of the pursuer to continue training.
        evader_checkpoint: Dir containing ppo.zip of the frozen evader.
        seed: Random seed.
        output_dir: Where to write logs and checkpoints.
        env_kwargs: Override env params (test-only).
        clear_buffer: Whether to clear the replay buffer after loading.
            Default True — old transitions from curriculum phase have different
            observation distribution and may confuse Q-networks under full PO.
    """
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else dict(COHORT_ENV_KWARGS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build base env.
    base_env = _make_base_env(env_kwargs)

    # 2) Load frozen evader.
    evader_zip = Path(evader_checkpoint) / "ppo.zip"
    if not evader_zip.exists():
        raise FileNotFoundError(f"evader checkpoint missing: {evader_zip}")
    frozen_evader = SAC.load(evader_zip)

    # 3) Wrap frozen evader into opponent adapter.
    evader_adapter = PartialObsOpponentAdapter(
        model=frozen_evader,
        role="evader",
        base_env=base_env,
        deterministic=False,
    )

    # 4) Build SingleAgentPEWrapper (pursuer trains vs frozen evader).
    single_env = SingleAgentPEWrapper(
        base_env,
        role="pursuer",
        opponent_policy=evader_adapter,
        p_full_obs=0.0,  # NO curriculum
    )

    # 5) Wrap with PartialObsWrapper.
    env = PartialObsWrapper(
        single_env,
        role="pursuer",
        history_length=env_kwargs.get("history_length", 10),
    )
    monitored = Monitor(env)

    # 6) Load warm-start pursuer.
    pursuer_zip = Path(pursuer_checkpoint) / "ppo.zip"
    if not pursuer_zip.exists():
        raise FileNotFoundError(f"pursuer checkpoint missing: {pursuer_zip}")
    learner = SAC.load(pursuer_zip, env=monitored)

    if clear_buffer:
        # Clear replay buffer — old transitions collected under p_full > 0
        # have different obs distribution than full PO.
        if hasattr(learner, "replay_buffer") and learner.replay_buffer is not None:
            learner.replay_buffer.reset()
            print("  [warm-start] Replay buffer cleared (old curriculum transitions removed)")

    # Ensure seed is set for reproducibility.
    learner.set_random_seed(seed)

    return WarmStartSetup(
        base_env=base_env,
        wrapper=monitored,
        frozen_model=frozen_evader,
        learner=learner,
        output_dir=out,
    )


def evaluate(setup: WarmStartSetup, n_episodes: int = 200) -> tuple[float, float]:
    """Roll out n_episodes deterministic; return (capture_rate, avg_steps)."""
    eval_env_kwargs = dict(COHORT_ENV_KWARGS)
    eval_base = _make_base_env(eval_env_kwargs)
    eval_opp = PartialObsOpponentAdapter(
        model=setup.frozen_model,
        role="evader",
        base_env=eval_base,
        deterministic=True,
    )
    eval_single = SingleAgentPEWrapper(
        eval_base,
        role="pursuer",
        opponent_policy=eval_opp,
        p_full_obs=0.0,
    )
    eval_env = PartialObsWrapper(
        eval_single,
        role="pursuer",
        history_length=eval_env_kwargs.get("history_length", 10),
    )

    captures = 0
    steps_list: list[int] = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        terminated = truncated = False
        ep_steps = 0
        while not (terminated or truncated):
            action, _ = setup.learner.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = eval_env.step(action)
            ep_steps += 1
        if terminated:
            captures += 1
        steps_list.append(ep_steps)
    eval_env.close()
    return captures / n_episodes, float(np.mean(steps_list))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pursuer_checkpoint", required=True,
                   help="Dir containing ppo.zip of pursuer to warm-start from.")
    p.add_argument("--evader_checkpoint", required=True,
                   help="Dir containing ppo.zip of frozen evader.")
    p.add_argument("--total_steps", type=int, default=3_000_000)
    p.add_argument("--eval_freq", type=int, default=250_000)
    p.add_argument("--n_eval_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--keep_buffer", action="store_true",
                   help="Keep replay buffer from checkpoint (default: clear it)")
    args = p.parse_args()

    setup = build_warmstart_setup(
        pursuer_checkpoint=args.pursuer_checkpoint,
        evader_checkpoint=args.evader_checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        clear_buffer=not args.keep_buffer,
    )

    print("=" * 70)
    print("Warm-Start BR Training")
    print(f"  Pursuer checkpoint: {args.pursuer_checkpoint}")
    print(f"  Frozen evader:      {args.evader_checkpoint}")
    print(f"  Total steps:        {args.total_steps:,}")
    print(f"  Eval cadence:       every {args.eval_freq:,} steps")
    print(f"  Seed:               {args.seed}")
    print(f"  Output:             {setup.output_dir}")
    print("=" * 70)

    history: list[dict] = []
    best_cr = -1.0
    best_step = 0
    start = time.time()
    steps_done = 0

    while steps_done < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_done)
        setup.learner.learn(total_timesteps=chunk,
                            reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk

        cr, avg_ep = evaluate(setup, n_episodes=args.n_eval_episodes)
        is_best = cr > best_cr
        if is_best:
            best_cr = cr
            best_step = steps_done
            setup.learner.save(setup.output_dir / "best_model")

        elapsed = time.time() - start
        history.append({
            "phase": f"WS{steps_done}",
            "total_steps": steps_done,
            "capture_rate": cr,
            "escape_rate": 1.0 - cr,
            "avg_ep_len": avg_ep,
            "frozen": "evader",
            "trained": "pursuer",
            "p_full_obs": 0.0,
            "elapsed_seconds": elapsed,
            "best_cr": best_cr,
            "best_step": best_step,
        })
        with open(setup.output_dir / "history.json", "w") as f:
            json.dump({"history": history,
                       "converged": False,
                       "elapsed_seconds": elapsed}, f, indent=2)

        flag = " *BEST*" if is_best else ""
        print(f"  step={steps_done:>10,} | CR={cr:.3f} | "
              f"avg_ep={avg_ep:>5.0f} | elapsed={elapsed/60:>5.1f}m{flag}")

    setup.learner.save(setup.output_dir / "final_model")
    print(f"\nDONE. best_CR={best_cr:.3f} at step {best_step:,}")


if __name__ == "__main__":
    main()
