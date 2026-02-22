"""Comparison framework for BarrierNet vs CBF-Beta evaluation.

Provides unified evaluation for both approaches:
1. BarrierNet PPO (differentiable QP, end-to-end)
2. CBF-Beta PPO (truncated Beta sampling) + optional RCBF-QP deployment filter

Metrics computed:
- Safety: violation rate, CBF margins, infeasibility
- Task performance: capture rate, capture time, episode reward
- Computational: inference time, QP correction magnitude
- Train-deploy gap: performance difference between training and deployment safety

Reference: Phase 2.5 spec (docs/phases/phase2_5_barriernet_experiment.md)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from scipy import stats

from safety.vcp_cbf import (
    vcp_cbf_boundary,
    vcp_cbf_obstacle,
    vcp_cbf_collision,
)


@dataclass
class EvaluationResult:
    """Results for one approach across N evaluation episodes."""

    approach_name: str
    n_episodes: int

    # Safety metrics
    safety_violation_rate: float  # fraction of steps with any CBF violation
    mean_min_cbf_margin: float  # mean of per-episode min h_i(x)
    cbf_margin_values: np.ndarray  # per-episode min CBF margin

    # Task performance
    capture_rate: float  # fraction of episodes with capture
    mean_capture_time: float  # mean time to capture (captured eps only)
    mean_episode_reward: float  # mean total reward
    episode_rewards: np.ndarray  # per-episode rewards

    # Computational
    mean_inference_time_ms: float  # mean per-step inference time (ms)
    training_wall_clock_hours: float  # total training time

    # QP / safety metrics
    qp_infeasibility_rate: float  # fraction of steps with infeasible QP
    mean_qp_correction: float  # mean ||u_safe - u_nom||
    intervention_rate: float  # fraction of steps where action was modified

    # Episode details
    episode_lengths: np.ndarray
    capture_times: np.ndarray  # capture time per captured episode
    min_distances: np.ndarray  # min pursuer-evader distance per episode


@dataclass
class ComparisonReport:
    """Comparison between two approaches."""

    barriernet: EvaluationResult
    cbf_beta_train: EvaluationResult  # CBF-Beta under training conditions
    cbf_beta_deploy: EvaluationResult  # CBF-Beta with RCBF-QP filter

    # Train-deploy gap for CBF-Beta
    train_deploy_gap_capture: float  # |train_cap - deploy_cap| / train_cap
    train_deploy_gap_reward: float  # |train_rew - deploy_rew| / |train_rew|

    # Statistical tests
    capture_rate_p_value: float  # BarrierNet vs CBF-Beta deploy
    reward_p_value: float  # BarrierNet vs CBF-Beta deploy
    safety_p_value: float  # CBF margin comparison


def compute_cbf_margins(
    state: np.ndarray,
    obstacles: list[dict],
    opponent_state: np.ndarray | None,
    arena_half_w: float = 10.0,
    arena_half_h: float = 10.0,
    d: float = 0.1,
    alpha: float = 1.0,
    robot_radius: float = 0.15,
    r_min: float = 0.35,
) -> list[float]:
    """Compute all CBF margin values for a given state.

    Returns list of h values (positive = safe, negative = violated).
    """
    margins = []

    # Arena boundary constraints
    for constraint in vcp_cbf_boundary(state, arena_half_w, arena_half_h, d, alpha):
        margins.append(constraint[0])  # h value

    # Obstacle constraints
    for obs in obstacles:
        obs_pos = np.array([obs["x"], obs["y"]])
        obs_r = obs["radius"] + robot_radius
        h, _, _ = vcp_cbf_obstacle(state, obs_pos, obs_r, d, alpha)
        margins.append(h)

    # Collision constraint
    if opponent_state is not None:
        h, _, _ = vcp_cbf_collision(state, opponent_state, d, alpha, r_min)
        margins.append(h)

    return margins


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for evaluation-compatible agents."""

    def get_eval_action(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        opponent_state: np.ndarray | None,
        arena_half_w: float,
        arena_half_h: float,
    ) -> tuple[np.ndarray, dict]:
        """Get deterministic action for evaluation.

        Returns:
            action: (2,) numpy array [v, omega]
            info: dict with optional keys: qp_correction, qp_feasible
        """
        ...


class BarrierNetEvalAgent:
    """Wrapper around BarrierNetPPO for evaluation."""

    def __init__(self, agent):
        """
        Args:
            agent: BarrierNetPPO instance.
        """
        self.agent = agent

    def get_eval_action(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        opponent_state: np.ndarray | None,
        arena_half_w: float,
        arena_half_h: float,
    ) -> tuple[np.ndarray, dict]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        opp_t = None
        if opponent_state is not None:
            opp_t = torch.tensor(opponent_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            u_safe, _, _, info = self.agent.get_action(
                obs_t, state_t,
                obstacles=obstacles,
                opponent_states=opp_t,
                deterministic=True,
            )

        action = u_safe.squeeze(0).numpy()
        eval_info = {
            "qp_correction": info["qp_correction"][0].item() if "qp_correction" in info else 0.0,
            "qp_feasible": info.get("qp_feasible", True),
        }
        return action, eval_info


class SB3EvalAgent:
    """Wrapper around SB3 PPO model for evaluation."""

    def __init__(self, model, safety_filter=None):
        """
        Args:
            model: SB3 PPO model.
            safety_filter: Optional VCPCBFFilter for deployment safety.
        """
        self.model = model
        self.safety_filter = safety_filter
        # Determine expected obs dim from model's observation space
        self._expected_obs_dim = model.observation_space.shape[0]

    def get_eval_action(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        obstacles: list[dict],
        opponent_state: np.ndarray | None,
        arena_half_w: float,
        arena_half_h: float,
    ) -> tuple[np.ndarray, dict]:
        # Truncate obs if model expects fewer dims (e.g. baseline without obstacles)
        model_obs = obs[:self._expected_obs_dim] if len(obs) > self._expected_obs_dim else obs
        action, _ = self.model.predict(model_obs, deterministic=True)
        eval_info = {"qp_correction": 0.0, "qp_feasible": True}

        if self.safety_filter is not None:
            action_pre = action.copy()
            action = self.safety_filter.filter_action(
                action, state,
                obstacles=obstacles,
                opponent_state=opponent_state,
            )
            correction = float(np.linalg.norm(action - action_pre))
            eval_info["qp_correction"] = correction
            # Check feasibility from filter metrics
            metrics = self.safety_filter.get_metrics()
            eval_info["qp_feasible"] = not metrics.get("last_infeasible", False)

        return action, eval_info


def evaluate_approach(
    agent,
    env,
    n_episodes: int = 200,
    approach_name: str = "unknown",
    arena_half_w: float = 10.0,
    arena_half_h: float = 10.0,
    d: float = 0.1,
    alpha: float = 1.0,
    robot_radius: float = 0.15,
    r_min: float = 0.35,
    training_wall_clock_hours: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate an agent over n_episodes.

    Args:
        agent: Agent implementing get_eval_action().
        env: SingleAgentPEWrapper environment.
        n_episodes: Number of evaluation episodes.
        approach_name: Name for logging.
        arena_half_w/h: Arena half-dimensions.
        d: VCP offset.
        alpha: CBF class-K parameter.
        robot_radius: Robot radius.
        r_min: Minimum separation distance.
        training_wall_clock_hours: Wall-clock training time to report.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        EvaluationResult with all metrics.
    """
    # Unwrap to get PE env internals
    pe_env = env
    while hasattr(pe_env, "env"):
        if hasattr(pe_env, "pursuer_state"):
            break
        pe_env = pe_env.env

    # Accumulators
    all_episode_rewards = []
    all_episode_lengths = []
    all_min_distances = []
    all_capture_times = []
    all_min_cbf_margins = []
    all_inference_times = []
    all_qp_corrections = []

    n_captures = 0
    n_violations = 0
    n_interventions = 0
    n_infeasible = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0.0
        min_cbf_margin = float("inf")
        done = False

        while not done:
            # Get state from env
            p_state = pe_env.pursuer_state.copy()
            e_state = pe_env.evader_state.copy()
            obstacles = pe_env.obstacles

            # Get action
            t0 = time.perf_counter()
            action, act_info = agent.get_eval_action(
                obs, p_state, obstacles, e_state,
                arena_half_w, arena_half_h,
            )
            inference_time = (time.perf_counter() - t0) * 1000
            all_inference_times.append(inference_time)

            # Track QP metrics
            correction = act_info.get("qp_correction", 0.0)
            all_qp_corrections.append(correction)
            if correction > 0.01:
                n_interventions += 1
            if not act_info.get("qp_feasible", True):
                n_infeasible += 1

            # Compute CBF margins at current state
            margins = compute_cbf_margins(
                p_state, obstacles, e_state,
                arena_half_w, arena_half_h,
                d, alpha, robot_radius, r_min,
            )
            if margins:
                min_margin = min(margins)
                min_cbf_margin = min(min_cbf_margin, min_margin)
                if min_margin < -1e-4:
                    n_violations += 1

            # Step environment
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            total_steps += 1

        # Episode stats
        all_episode_rewards.append(episode_reward)
        all_min_cbf_margins.append(min_cbf_margin if min_cbf_margin != float("inf") else 0.0)

        ep_metrics = step_info.get("episode_metrics", {})
        all_episode_lengths.append(ep_metrics.get("episode_length", 0))
        all_min_distances.append(ep_metrics.get("min_distance", 0.0))

        if ep_metrics.get("captured", False):
            n_captures += 1
            all_capture_times.append(ep_metrics.get("capture_time", 0.0))

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(
                f"  [{approach_name}] {ep + 1}/{n_episodes} episodes, "
                f"capture={n_captures / (ep + 1):.2%}, "
                f"violations={n_violations}/{total_steps}"
            )

    return EvaluationResult(
        approach_name=approach_name,
        n_episodes=n_episodes,
        safety_violation_rate=n_violations / max(total_steps, 1),
        mean_min_cbf_margin=float(np.mean(all_min_cbf_margins)),
        cbf_margin_values=np.array(all_min_cbf_margins),
        capture_rate=n_captures / n_episodes,
        mean_capture_time=float(np.mean(all_capture_times)) if all_capture_times else float("inf"),
        mean_episode_reward=float(np.mean(all_episode_rewards)),
        episode_rewards=np.array(all_episode_rewards),
        mean_inference_time_ms=float(np.mean(all_inference_times)),
        training_wall_clock_hours=training_wall_clock_hours,
        qp_infeasibility_rate=n_infeasible / max(total_steps, 1),
        mean_qp_correction=float(np.mean(all_qp_corrections)),
        intervention_rate=n_interventions / max(total_steps, 1),
        episode_lengths=np.array(all_episode_lengths),
        capture_times=np.array(all_capture_times),
        min_distances=np.array(all_min_distances),
    )


def compute_comparison(
    barriernet_result: EvaluationResult,
    cbf_beta_train_result: EvaluationResult,
    cbf_beta_deploy_result: EvaluationResult,
) -> ComparisonReport:
    """Compute comparison metrics between approaches.

    Args:
        barriernet_result: BarrierNet evaluation result.
        cbf_beta_train_result: CBF-Beta under training conditions.
        cbf_beta_deploy_result: CBF-Beta with RCBF-QP filter.

    Returns:
        ComparisonReport with statistical tests.
    """
    # Train-deploy gap
    train_cap = cbf_beta_train_result.capture_rate
    deploy_cap = cbf_beta_deploy_result.capture_rate
    gap_capture = abs(train_cap - deploy_cap) / max(train_cap, 1e-6)

    train_rew = cbf_beta_train_result.mean_episode_reward
    deploy_rew = cbf_beta_deploy_result.mean_episode_reward
    gap_reward = abs(train_rew - deploy_rew) / max(abs(train_rew), 1e-6)

    # Statistical tests (Welch's t-test)
    # Capture rate comparison via episode rewards (proxy for performance)
    _, reward_p = stats.ttest_ind(
        barriernet_result.episode_rewards,
        cbf_beta_deploy_result.episode_rewards,
        equal_var=False,
    )

    # CBF margin comparison
    _, safety_p = stats.ttest_ind(
        barriernet_result.cbf_margin_values,
        cbf_beta_deploy_result.cbf_margin_values,
        equal_var=False,
    )

    # Capture rate: use proportions z-test
    n1, n2 = barriernet_result.n_episodes, cbf_beta_deploy_result.n_episodes
    p1 = barriernet_result.capture_rate
    p2 = cbf_beta_deploy_result.capture_rate
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 and p_pool < 1 else 1e-6
    z = (p1 - p2) / se
    capture_p = 2 * (1 - stats.norm.cdf(abs(z)))

    return ComparisonReport(
        barriernet=barriernet_result,
        cbf_beta_train=cbf_beta_train_result,
        cbf_beta_deploy=cbf_beta_deploy_result,
        train_deploy_gap_capture=gap_capture,
        train_deploy_gap_reward=gap_reward,
        capture_rate_p_value=capture_p,
        reward_p_value=reward_p,
        safety_p_value=safety_p,
    )


def format_comparison_table(report: ComparisonReport) -> str:
    """Format a comparison table as a string.

    Returns markdown-formatted table.
    """
    bn = report.barriernet
    ct = report.cbf_beta_train
    cd = report.cbf_beta_deploy

    lines = [
        "| Metric | CBF-Beta (Train) | CBF-Beta+RCBF-QP (Deploy) | BarrierNet E2E | Winner |",
        "|--------|------------------|---------------------------|----------------|--------|",
    ]

    # Safety
    def winner_lower(a, b, c, metric_name):
        vals = {"CBF-Beta Train": a, "CBF-Beta Deploy": b, "BarrierNet": c}
        w = min(vals, key=vals.get)
        return w

    def winner_higher(a, b, c, metric_name):
        vals = {"CBF-Beta Train": a, "CBF-Beta Deploy": b, "BarrierNet": c}
        w = max(vals, key=vals.get)
        return w

    lines.append(
        f"| Safety violations (%) | {ct.safety_violation_rate:.4%} | {cd.safety_violation_rate:.4%} | "
        f"{bn.safety_violation_rate:.4%} | {winner_lower(ct.safety_violation_rate, cd.safety_violation_rate, bn.safety_violation_rate, 'safety')} |"
    )
    lines.append(
        f"| Mean min CBF margin | {ct.mean_min_cbf_margin:.4f} | {cd.mean_min_cbf_margin:.4f} | "
        f"{bn.mean_min_cbf_margin:.4f} | {winner_higher(ct.mean_min_cbf_margin, cd.mean_min_cbf_margin, bn.mean_min_cbf_margin, 'margin')} |"
    )
    lines.append(
        f"| Capture rate (%) | {ct.capture_rate:.2%} | {cd.capture_rate:.2%} | "
        f"{bn.capture_rate:.2%} | {winner_higher(ct.capture_rate, cd.capture_rate, bn.capture_rate, 'capture')} |"
    )
    lines.append(
        f"| Mean capture time (s) | {ct.mean_capture_time:.2f} | {cd.mean_capture_time:.2f} | "
        f"{bn.mean_capture_time:.2f} | {winner_lower(ct.mean_capture_time, cd.mean_capture_time, bn.mean_capture_time, 'time')} |"
    )
    lines.append(
        f"| Mean episode reward | {ct.mean_episode_reward:.2f} | {cd.mean_episode_reward:.2f} | "
        f"{bn.mean_episode_reward:.2f} | {winner_higher(ct.mean_episode_reward, cd.mean_episode_reward, bn.mean_episode_reward, 'reward')} |"
    )
    lines.append(
        f"| Mean inference time (ms) | {ct.mean_inference_time_ms:.2f} | {cd.mean_inference_time_ms:.2f} | "
        f"{bn.mean_inference_time_ms:.2f} | {winner_lower(ct.mean_inference_time_ms, cd.mean_inference_time_ms, bn.mean_inference_time_ms, 'inference')} |"
    )
    lines.append(
        f"| Training time (hours) | {ct.training_wall_clock_hours:.1f} | {cd.training_wall_clock_hours:.1f} | "
        f"{bn.training_wall_clock_hours:.1f} | {winner_lower(ct.training_wall_clock_hours, cd.training_wall_clock_hours, bn.training_wall_clock_hours, 'training_time')} |"
    )
    lines.append(
        f"| QP infeasibility (%) | {ct.qp_infeasibility_rate:.4%} | {cd.qp_infeasibility_rate:.4%} | "
        f"{bn.qp_infeasibility_rate:.4%} | {winner_lower(ct.qp_infeasibility_rate, cd.qp_infeasibility_rate, bn.qp_infeasibility_rate, 'infeas')} |"
    )
    lines.append(
        f"| Mean QP correction | {ct.mean_qp_correction:.4f} | {cd.mean_qp_correction:.4f} | "
        f"{bn.mean_qp_correction:.4f} | {winner_lower(ct.mean_qp_correction, cd.mean_qp_correction, bn.mean_qp_correction, 'qp_corr')} |"
    )
    lines.append(
        f"| Intervention rate (%) | {ct.intervention_rate:.2%} | {cd.intervention_rate:.2%} | "
        f"{bn.intervention_rate:.2%} | {winner_lower(ct.intervention_rate, cd.intervention_rate, bn.intervention_rate, 'interv')} |"
    )

    lines.append("")
    lines.append(f"**Train-Deploy Gap (CBF-Beta)**: Capture rate gap = {report.train_deploy_gap_capture:.2%}, Reward gap = {report.train_deploy_gap_reward:.2%}")
    lines.append(f"**BarrierNet Train-Deploy Gap**: 0% (by construction)")
    lines.append("")
    lines.append("**Statistical Significance (BarrierNet vs CBF-Beta Deploy)**:")
    lines.append(f"- Capture rate: p = {report.capture_rate_p_value:.4f} {'(significant)' if report.capture_rate_p_value < 0.05 else '(not significant)'}")
    lines.append(f"- Episode reward: p = {report.reward_p_value:.4f} {'(significant)' if report.reward_p_value < 0.05 else '(not significant)'}")
    lines.append(f"- CBF margin: p = {report.safety_p_value:.4f} {'(significant)' if report.safety_p_value < 0.05 else '(not significant)'}")

    return "\n".join(lines)
