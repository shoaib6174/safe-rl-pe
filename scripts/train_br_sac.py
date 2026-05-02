"""Train a fresh SAC learner against a FROZEN SAC opponent for the BR
exploitability test (CoRL 2026 paper).

Spec: docs/superpowers/specs/2026-05-02-paper-strengthening-design.md

The frozen opponent is loaded from a `milestone_phase*` snapshot dir
(containing `ppo.zip` despite the legacy filename — it is SAC weights).
Wrapped with `PartialObsOpponentAdapter` so the frozen side observes the
world through its own sensors, identical to how the cohort trained it.

The trained side runs SAC with cohort-identical hyperparameters, p_full
fixed at 0.0 (full partial obs, no curriculum), no PFSP, no alternating
freeze, no forced switch — vanilla single-side training against a
fixed adversary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow imports when script is run from repo root as `python scripts/train_br_sac.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.opponent_adapter import PartialObsOpponentAdapter
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.rewards import RewardComputer
from envs.wrappers import SingleAgentPEWrapper


# Cohort-identical hyperparameters (from scripts/launch_sp17b.sh).
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
class BRSetup:
    """Aggregate of the live objects the BR trainer needs."""
    base_env: PursuitEvasionEnv
    wrapper: SingleAgentPEWrapper
    frozen_model: SAC
    learner: SAC
    output_dir: Path
    trained_role: str
    frozen_role: str


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
    """Construct the raw PursuitEvasionEnv matching cohort training."""
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


def build_br_setup(
    frozen_opponent_path: str,
    frozen_role: str,
    seed: int,
    output_dir: str,
    env_kwargs: dict | None = None,
) -> BRSetup:
    """Construct env + frozen opponent + fresh learner. No training yet.

    Args:
        frozen_opponent_path: Directory containing `ppo.zip` (cohort snapshot).
        frozen_role: 'pursuer' or 'evader' — the role the FROZEN side plays.
        seed: Random seed for the trainer.
        output_dir: Where eval log + checkpoints are written.
        env_kwargs: Override of COHORT_ENV_KWARGS (test/smoke-only). None => cohort.
    """
    if frozen_role not in ("pursuer", "evader"):
        raise ValueError(f"frozen_role must be 'pursuer' or 'evader', got {frozen_role!r}")
    trained_role = "evader" if frozen_role == "pursuer" else "pursuer"

    env_kwargs = dict(env_kwargs) if env_kwargs is not None else dict(COHORT_ENV_KWARGS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build base env (raw PursuitEvasionEnv).
    base_env = _make_base_env(env_kwargs)

    # 2) Load frozen SAC opponent.
    zip_path = Path(frozen_opponent_path) / "ppo.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"frozen opponent zip missing: {zip_path}")
    frozen_model = SAC.load(zip_path)

    # 3) Wrap frozen model into PartialObsOpponentAdapter.
    opp_adapter = PartialObsOpponentAdapter(
        model=frozen_model,
        role=frozen_role,
        base_env=base_env,
        deterministic=False,
    )

    # 4) Build SingleAgentPEWrapper (trained side + frozen opponent).
    #    p_full_obs=0.0 => always partial obs. NO CURRICULUM (per spec §4).
    single_env = SingleAgentPEWrapper(
        base_env,
        role=trained_role,
        opponent_policy=opp_adapter,
        p_full_obs=0.0,
    )

    # 5) Wrap with PartialObsWrapper (identical to cohort training stack).
    env = PartialObsWrapper(
        single_env,
        role=trained_role,
        history_length=env_kwargs.get("history_length", 10),
    )
    monitored = Monitor(env)

    # 6) Fresh SAC learner.
    learner = SAC("MultiInputPolicy", monitored, seed=seed, verbose=0,
                  **COHORT_SAC_KWARGS)

    return BRSetup(
        base_env=base_env,
        wrapper=monitored,
        frozen_model=frozen_model,
        learner=learner,
        output_dir=out,
        trained_role=trained_role,
        frozen_role=frozen_role,
    )


def run_one_update(setup: BRSetup, n_train_steps: int = 200) -> None:
    """Run a small `learn()` call. Used by tests to verify frozen invariance."""
    setup.learner.learn(total_timesteps=n_train_steps,
                        reset_num_timesteps=False, progress_bar=False)


def evaluate(setup: BRSetup, n_episodes: int = 200) -> tuple[float, float]:
    """Roll out n_episodes deterministic episodes; return (capture_rate, avg_steps).

    Build a SEPARATE eval env so the replay buffer is not polluted by eval rollouts.
    """
    eval_env_kwargs = dict(COHORT_ENV_KWARGS)
    eval_base = _make_base_env(eval_env_kwargs)
    eval_opp = PartialObsOpponentAdapter(
        model=setup.frozen_model,
        role=setup.frozen_role,
        base_env=eval_base,
        deterministic=True,
    )
    eval_single = SingleAgentPEWrapper(
        eval_base,
        role=setup.trained_role,
        opponent_policy=eval_opp,
        p_full_obs=0.0,
    )
    eval_env = PartialObsWrapper(
        eval_single,
        role=setup.trained_role,
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
    p.add_argument("--frozen_opponent_path", required=True,
                   help="Dir containing ppo.zip of the frozen SAC opponent.")
    p.add_argument("--frozen_role", required=True, choices=("pursuer", "evader"),
                   help="Role of the FROZEN side.")
    p.add_argument("--total_steps", type=int, default=1_500_000)
    p.add_argument("--eval_freq",   type=int, default=250_000)
    p.add_argument("--n_eval_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    setup = build_br_setup(
        frozen_opponent_path=args.frozen_opponent_path,
        frozen_role=args.frozen_role,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("=" * 70)
    print("BR Exploitability Test")
    print(f"  Frozen role:    {setup.frozen_role}")
    print(f"  Trained role:   {setup.trained_role}")
    print(f"  Frozen path:    {args.frozen_opponent_path}")
    print(f"  Total steps:    {args.total_steps:,}")
    print(f"  Eval cadence:   every {args.eval_freq:,} steps, "
          f"{args.n_eval_episodes} episodes")
    print(f"  Seed:           {args.seed}")
    print(f"  Output:         {setup.output_dir}")
    print("=" * 70)

    history: list[dict] = []
    best_metric = -1.0  # higher is better for the trained side
    best_step = 0
    start = time.time()
    steps_done = 0

    while steps_done < args.total_steps:
        chunk = min(args.eval_freq, args.total_steps - steps_done)
        setup.learner.learn(total_timesteps=chunk,
                            reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk

        cr, avg_ep = evaluate(setup, n_episodes=args.n_eval_episodes)
        # For pursuer BR: high CR good. For evader BR: low CR good.
        metric = cr if setup.trained_role == "pursuer" else (1.0 - cr)
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            best_step = steps_done
            setup.learner.save(setup.output_dir / "best_model")

        elapsed = time.time() - start
        history.append({
            "phase": f"BR{steps_done}",
            "total_steps": steps_done,
            "capture_rate": cr,
            "escape_rate": 1.0 - cr,
            "avg_ep_len": avg_ep,
            "frozen": setup.frozen_role,
            "trained": setup.trained_role,
            "p_full_obs": 0.0,
            "elapsed_seconds": elapsed,
            "best_metric": best_metric,
            "best_step": best_step,
        })
        # Persist after every eval — never lose progress on crash.
        with open(setup.output_dir / "history.json", "w") as f:
            json.dump({"history": history,
                       "converged": False,
                       "elapsed_seconds": elapsed}, f, indent=2)

        flag = " *BEST*" if is_best else ""
        print(f"  step={steps_done:>10,} | CR={cr:.3f} | "
              f"avg_ep={avg_ep:>5.0f} | elapsed={elapsed/60:>5.1f}m{flag}")

    setup.learner.save(setup.output_dir / "final_model")
    print(f"\nDONE. best_metric={best_metric:.3f} at step {best_step:,}")


if __name__ == "__main__":
    main()
