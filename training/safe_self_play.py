"""Safety-constrained self-play training loop.

Extends Phase 1 alternating self-play with:
- CBF-Beta policy for both agents
- SafeActionResolver for 3-tier infeasibility handling
- Safety metrics tracking
- Obstacle-augmented environment
- Safety reward shaping (w5 * CBF margin)

Comparison runs (logged with group tags):
- Run A: Full safe (CBF-Beta + obstacles + w5)
- Run B: Unsafe (no CBF, Phase 1 baseline)
- Run C: CBF-QP filter only (no Beta, post-hoc filtering)
- Run D: Safe without w5 (safety reward ablation)

Usage:
    from training.safe_self_play import SafeSelfPlayConfig, safe_self_play_rollout

    config = SafeSelfPlayConfig()
    results = safe_self_play_rollout(config)

SB3 integration is lazy-imported for training:
    from training.safe_self_play import run_safe_self_play
    run_safe_self_play(hydra_config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from envs.dynamics import unicycle_step
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.rewards import SafetyRewardComputer
from safety.infeasibility import SafeActionResolver
from safety.vcp_cbf import VCPCBFFilter
from training.safety_metrics import SafetyMetricsTracker


@dataclass
class SafeSelfPlayConfig:
    """Configuration for safe self-play training.

    All defaults match the Phase 2 specification.
    """

    # Self-play
    n_phases: int = 10
    timesteps_per_phase: int = 300_000
    eval_episodes: int = 50
    seed: int = 42

    # Environment
    arena_width: float = 20.0
    arena_height: float = 20.0
    dt: float = 0.05
    max_steps: int = 1200
    capture_radius: float = 0.5
    collision_radius: float = 0.3
    robot_radius: float = 0.15
    v_max: float = 1.0
    omega_max: float = 2.84

    # Obstacles
    n_obstacles: int = 4
    obstacle_radius_range: tuple[float, float] = (0.3, 1.0)
    obstacle_margin: float = 0.5
    n_obstacle_obs: int = 3

    # CBF
    cbf_alpha: float = 1.0
    vcp_d: float = 0.1
    safety_margin: float = 0.1
    w_v: float = 150.0
    w_omega: float = 1.0

    # Safety reward
    w_safety: float = 0.05
    h_ref: float = 1.0

    # Feasibility
    use_feasibility_classifier: bool = False
    n_feasibility_iterations: int = 3

    # Comparison run type
    run_type: str = "A_full_safe"  # A, B, C, D
    use_cbf: bool = True
    use_beta_policy: bool = True
    use_safety_reward: bool = True


def make_safe_env(config: SafeSelfPlayConfig) -> PursuitEvasionEnv:
    """Create a PursuitEvasionEnv configured for safe training.

    Args:
        config: SafeSelfPlayConfig.

    Returns:
        PursuitEvasionEnv with obstacles and safety reward.
    """
    # Build reward computer based on config
    arena_diag = np.sqrt(config.arena_width**2 + config.arena_height**2)
    if config.use_safety_reward:
        reward_computer = SafetyRewardComputer(
            w_safety=config.w_safety,
            h_ref=config.h_ref,
            d_max=arena_diag,
        )
    else:
        reward_computer = None  # default RewardComputer

    return PursuitEvasionEnv(
        arena_width=config.arena_width,
        arena_height=config.arena_height,
        dt=config.dt,
        max_steps=config.max_steps,
        capture_radius=config.capture_radius,
        collision_radius=config.collision_radius,
        robot_radius=config.robot_radius,
        pursuer_v_max=config.v_max,
        pursuer_omega_max=config.omega_max,
        evader_v_max=config.v_max,
        evader_omega_max=config.omega_max,
        n_obstacles=config.n_obstacles,
        obstacle_radius_range=config.obstacle_radius_range,
        obstacle_margin=config.obstacle_margin,
        n_obstacle_obs=config.n_obstacle_obs,
        reward_computer=reward_computer,
        render_mode=None,
    )


def make_cbf_filter(config: SafeSelfPlayConfig) -> VCPCBFFilter:
    """Create VCPCBFFilter from config."""
    return VCPCBFFilter(
        d=config.vcp_d,
        alpha=config.cbf_alpha,
        v_max=config.v_max,
        omega_max=config.omega_max,
        arena_half_w=config.arena_width / 2.0,
        arena_half_h=config.arena_height / 2.0,
        robot_radius=config.robot_radius,
        w_v=config.w_v,
        w_omega=config.w_omega,
        r_min_separation=config.collision_radius + config.safety_margin,
    )


def make_resolver(
    config: SafeSelfPlayConfig,
    cbf_filter: VCPCBFFilter,
) -> SafeActionResolver:
    """Create SafeActionResolver from config."""
    return SafeActionResolver(
        cbf_filter=cbf_filter,
        use_feasibility_classifier=config.use_feasibility_classifier,
    )


def safe_self_play_rollout(
    config: SafeSelfPlayConfig,
    n_episodes: int = 10,
    pursuer_policy=None,
    evader_policy=None,
) -> dict:
    """Run safe self-play rollout (SB3-independent).

    This is a lightweight rollout loop for testing and evaluation.
    For full training with SB3, use run_safe_self_play().

    Args:
        config: SafeSelfPlayConfig.
        n_episodes: Number of episodes to simulate.
        pursuer_policy: Callable(obs) -> action, or None for random.
        evader_policy: Callable(obs) -> action, or None for random.

    Returns:
        Dict with rollout results and safety metrics.
    """
    env = make_safe_env(config)
    cbf_filter = make_cbf_filter(config)
    resolver = make_resolver(config, cbf_filter)
    tracker = SafetyMetricsTracker()

    episode_results = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=config.seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Get nominal actions
            if pursuer_policy is not None:
                p_action = pursuer_policy(obs["pursuer"])
            else:
                p_action = env.pursuer_action_space.sample()

            if evader_policy is not None:
                e_action = evader_policy(obs["evader"])
            else:
                e_action = env.evader_action_space.sample()

            # Apply CBF safety filter (if enabled)
            if config.use_cbf:
                p_safe, p_method, p_info = resolver.resolve(
                    p_action, env.pursuer_state,
                    obstacles=env.obstacles,
                    opponent_state=env.evader_state,
                )
                tracker.record_step(
                    method=p_method,
                    min_h=p_info.get("min_h", 0.0),
                    intervention=p_info.get("intervention", False),
                )
                p_action = p_safe

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(p_action, e_action)
            done = terminated or truncated
            ep_reward += rewards["pursuer"]
            ep_steps += 1

            # Check for violations
            if info.get("pursuer_obstacle_collision", False):
                tracker.record_step(
                    method="violation",
                    min_h=-1.0,
                    violation=True,
                )

        episode_results.append({
            "reward": ep_reward,
            "steps": ep_steps,
            "captured": info.get("episode_metrics", {}).get("captured", False),
        })

    # End phase for tracker
    phase_metrics = tracker.end_phase()
    safety_targets = tracker.check_safety_targets()

    env.close()

    return {
        "episodes": episode_results,
        "safety_metrics": phase_metrics,
        "safety_targets": safety_targets,
        "summary": tracker.get_summary(),
        "resolver_metrics": resolver.get_metrics(),
    }


# =============================================================================
# Comparison run configurations
# =============================================================================

def get_run_configs() -> dict[str, SafeSelfPlayConfig]:
    """Get configs for all 4 comparison runs.

    Returns:
        Dict mapping run label to SafeSelfPlayConfig.
    """
    configs = {}

    # Run A: Full safe (CBF-Beta + obstacles + w5)
    configs["A_full_safe"] = SafeSelfPlayConfig(
        run_type="A_full_safe",
        use_cbf=True,
        use_beta_policy=True,
        use_safety_reward=True,
    )

    # Run B: Unsafe (no CBF, Phase 1 baseline)
    configs["B_unsafe"] = SafeSelfPlayConfig(
        run_type="B_unsafe",
        use_cbf=False,
        use_beta_policy=False,
        use_safety_reward=False,
        n_obstacles=0,
        n_obstacle_obs=0,
    )

    # Run C: CBF-QP filter only (no Beta policy)
    configs["C_cbf_qp_only"] = SafeSelfPlayConfig(
        run_type="C_cbf_qp_only",
        use_cbf=True,
        use_beta_policy=False,
        use_safety_reward=False,
    )

    # Run D: Safe without w5 (safety reward ablation)
    configs["D_no_safety_reward"] = SafeSelfPlayConfig(
        run_type="D_no_safety_reward",
        use_cbf=True,
        use_beta_policy=True,
        use_safety_reward=False,
    )

    return configs
