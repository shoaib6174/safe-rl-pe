"""SB3 callbacks for AMS-DRL self-play health monitoring.

Three callback classes:
  - EntropyMonitorCallback: Tracks entropy, clamps log_std floor
  - SelfPlayHealthMonitorCallback: Entropy + capture rate collapse detection + rollback
  - FixedBaselineEvalCallback: Evaluate against scripted baselines + Elo tracking
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.checkpoint_manager import CheckpointManager


class EntropyMonitorCallback(BaseCallback):
    """Standalone entropy monitor with optional log_std clamping.

    Monitors the policy's action distribution entropy and applies a floor
    on log_std to prevent entropy collapse during training.

    For a Gaussian distribution, entropy = 0.5 * log(2*pi*e) + log_std
    per action dimension. 0.5 * log(2*pi*e) ≈ 1.4189.

    Args:
        check_freq: How often to check entropy (in steps).
        log_std_floor: Minimum allowed log_std value.
        enable_clamp: Whether to actually clamp log_std.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        check_freq: int = 2048,
        log_std_floor: float = -2.0,
        enable_clamp: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_std_floor = log_std_floor
        self.enable_clamp = enable_clamp

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        policy = self.model.policy
        if not hasattr(policy, "log_std"):
            return True

        import torch
        log_std = policy.log_std.detach().cpu().numpy()
        total_entropy = sum(1.4189 + ls for ls in log_std)
        mean_sigma = float(np.mean(np.exp(log_std)))

        self.logger.record("entropy/total", float(total_entropy))
        self.logger.record("entropy/mean_sigma", mean_sigma)
        self.logger.record("entropy/mean_log_std", float(np.mean(log_std)))

        # Clamp log_std to prevent collapse
        if self.enable_clamp:
            with torch.no_grad():
                policy.log_std.clamp_(min=self.log_std_floor)

        return True


class SelfPlayHealthMonitorCallback(BaseCallback):
    """Health monitor for AMS-DRL self-play with rollback support.

    Integrates entropy monitoring, capture rate tracking, and checkpoint
    rollback in a single SB3 callback. Operates per-phase: tracks metrics
    within each self-play training phase.

    Rollback conditions:
    - Total entropy drops below entropy_collapse threshold
    - Capture rate hits collapse (< 0.02) or domination (> 0.98) for
      capture_window consecutive episodes

    Args:
        checkpoint_manager: CheckpointManager instance for save/rollback.
        checkpoint_freq: Save rolling checkpoint every N steps.
        entropy_yellow: Warning threshold for total entropy.
        entropy_red: Danger threshold for total entropy.
        entropy_collapse: Rollback trigger threshold for total entropy.
        capture_collapse: Rollback if capture rate below this.
        capture_domination: Rollback if capture rate above this.
        capture_window: Window of recent episodes for capture rate.
        cooldown_steps: Minimum steps between rollbacks.
        encoder: Optional BiMDN encoder for checkpointing.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_freq: int = 10_000,
        entropy_yellow: float = 0.5,
        entropy_red: float = -0.5,
        entropy_collapse: float = -2.0,
        capture_collapse: float = 0.02,
        capture_domination: float = 0.98,
        capture_window: int = 200,
        cooldown_steps: int = 50_000,
        encoder=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ckpt_mgr = checkpoint_manager
        self.checkpoint_freq = checkpoint_freq
        self.entropy_yellow = entropy_yellow
        self.entropy_red = entropy_red
        self.entropy_collapse = entropy_collapse
        self.capture_collapse = capture_collapse
        self.capture_domination = capture_domination
        self.capture_window = capture_window
        self.cooldown_steps = cooldown_steps
        self.encoder = encoder

        self.capture_history: deque = deque(maxlen=capture_window)
        self.last_rollback_step = -cooldown_steps
        self.rollback_count = 0

    def _on_step(self) -> bool:
        # Periodic checkpoint
        if self.num_timesteps % self.checkpoint_freq == 0:
            self.ckpt_mgr.save_rolling(
                model=self.model,
                encoder=self.encoder,
                step=self.num_timesteps,
            )

        # Track capture events from episode info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                # SB3 VecEnv stores terminal info under "terminal_info" or
                # wraps it in episode dict. Check for our custom metrics.
                ep_info = info
                if "episode_metrics" in ep_info:
                    captured = ep_info["episode_metrics"].get("captured", False)
                    self.capture_history.append(1.0 if captured else 0.0)
                elif "terminal_info" in ep_info and "episode_metrics" in ep_info["terminal_info"]:
                    captured = ep_info["terminal_info"]["episode_metrics"].get("captured", False)
                    self.capture_history.append(1.0 if captured else 0.0)

        # Log entropy
        self._log_entropy()

        # Log capture rate
        if len(self.capture_history) > 0:
            rate = float(np.mean(self.capture_history))
            self.logger.record("health/capture_rate_rolling", rate)

        # Check rollback conditions
        if self._should_rollback():
            self._trigger_rollback()

        return True

    def _log_entropy(self):
        """Log entropy metrics and status."""
        policy = self.model.policy
        if not hasattr(policy, "log_std"):
            return

        log_std = policy.log_std.detach().cpu().numpy()
        total_entropy = sum(1.4189 + ls for ls in log_std)

        self.logger.record("health/total_entropy", float(total_entropy))
        self.logger.record("health/mean_log_std", float(np.mean(log_std)))

        # Status indicators
        if total_entropy < self.entropy_collapse:
            status = "COLLAPSE"
        elif total_entropy < self.entropy_red:
            status = "RED"
        elif total_entropy < self.entropy_yellow:
            status = "YELLOW"
        else:
            status = "OK"
        self.logger.record("health/entropy_status", status)

    def _should_rollback(self) -> bool:
        """Check if rollback conditions are met."""
        # Cooldown check
        if self.num_timesteps - self.last_rollback_step < self.cooldown_steps:
            return False

        # Entropy collapse
        policy = self.model.policy
        if hasattr(policy, "log_std"):
            log_std = policy.log_std.detach().cpu().numpy()
            total_entropy = sum(1.4189 + ls for ls in log_std)
            if total_entropy < self.entropy_collapse:
                if self.verbose:
                    print(f"[HEALTH] Entropy collapse: {total_entropy:.2f} "
                          f"< {self.entropy_collapse}")
                return True

        # Capture rate collapse/domination
        if len(self.capture_history) >= self.capture_window:
            rate = float(np.mean(self.capture_history))
            if rate <= self.capture_collapse:
                if self.verbose:
                    print(f"[HEALTH] Capture collapse: {rate:.3f} "
                          f"<= {self.capture_collapse}")
                return True
            if rate >= self.capture_domination:
                if self.verbose:
                    print(f"[HEALTH] Capture domination: {rate:.3f} "
                          f">= {self.capture_domination}")
                return True

        return False

    def _trigger_rollback(self):
        """Perform rollback to earlier checkpoint."""
        self.rollback_count += 1
        self.last_rollback_step = self.num_timesteps

        try:
            if self.encoder is not None:
                model, _, step = self.ckpt_mgr.perform_rollback(
                    type(self.model), encoder=self.encoder, rollback_steps=3
                )
            else:
                model, step = self.ckpt_mgr.perform_rollback(
                    type(self.model), rollback_steps=3
                )

            # Restore policy weights
            self.model.policy.load_state_dict(model.policy.state_dict())
            self.capture_history.clear()

            print(f"[ROLLBACK #{self.rollback_count}] Restored to step {step} "
                  f"(from step {self.num_timesteps})")
        except ValueError as e:
            print(f"[ROLLBACK FAILED] {e}")

    def reset_phase(self):
        """Reset tracking state for a new self-play phase."""
        self.capture_history.clear()


class FixedBaselineEvalCallback(BaseCallback):
    """Evaluate training agent against fixed scripted baselines with Elo tracking.

    Periodically runs evaluation episodes against scripted opponents
    (random, pure pursuit, flee to corner) and tracks performance via Elo ratings.

    Uses two evaluation frequencies:
    - Lightweight: Fast evaluation against random only (more frequent)
    - Full: All baselines (less frequent, more expensive)

    Args:
        eval_env: Base PursuitEvasionEnv for evaluation.
        baselines: Dict mapping name -> callable(env) -> action, or None for random.
        role: Role of the training agent.
        lightweight_eval_freq: Steps between lightweight evals (random only).
        full_eval_freq: Steps between full evals (all baselines).
        n_eval_episodes: Number of evaluation episodes per baseline.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_env,
        baselines: dict,
        role: str = "pursuer",
        lightweight_eval_freq: int = 10_000,
        full_eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        agent_adapter=None,
        fixed_speed_v_max: float | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.baselines = baselines
        self.role = role
        self.lightweight_eval_freq = lightweight_eval_freq
        self.full_eval_freq = full_eval_freq
        self.n_eval = n_eval_episodes
        self.agent_adapter = agent_adapter
        self.fixed_speed_v_max = fixed_speed_v_max

        # Elo tracking
        self.elo: dict[str, float] = {"training_agent": 1200.0}
        for name in baselines:
            self.elo[name] = 1200.0

    def _on_step(self) -> bool:
        is_lightweight = (
            self.num_timesteps % self.lightweight_eval_freq == 0
            and self.num_timesteps % self.full_eval_freq != 0
        )
        is_full = self.num_timesteps % self.full_eval_freq == 0

        if not is_lightweight and not is_full:
            return True

        # Keep adapter's model reference in sync with the training model
        if self.agent_adapter is not None:
            self.agent_adapter.model = self.model

        for name, baseline_fn in self.baselines.items():
            # Lightweight: only evaluate against random (None)
            if is_lightweight and baseline_fn is not None:
                continue

            win_rate = self._evaluate_against(baseline_fn)
            self.logger.record(f"baseline/{name}_win_rate", win_rate)

            # Update Elo
            ra = self.elo["training_agent"]
            rb = self.elo[name]
            expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
            self.elo["training_agent"] += 32 * (win_rate - expected)

        self.logger.record("elo/training_agent", self.elo["training_agent"])
        return True

    def _evaluate_against(self, baseline_fn) -> float:
        """Evaluate the training agent against a baseline policy.

        Args:
            baseline_fn: Callable(env) -> action, or None for random.

        Returns:
            Win rate (0.0 to 1.0).
        """
        wins = 0
        env = self.eval_env

        for ep in range(self.n_eval):
            obs, info = env.reset()
            if self.agent_adapter is not None:
                self.agent_adapter.reset()
            done = False

            while not done:
                # Training agent's action
                if self.agent_adapter is not None:
                    # Use adapter to convert full-state obs to partial-obs Dict
                    agent_action, _ = self.agent_adapter.predict(
                        obs[self.role], deterministic=True
                    )
                else:
                    agent_obs = obs[self.role]
                    agent_action, _ = self.model.predict(agent_obs, deterministic=True)

                # Opponent action
                opp_role = "evader" if self.role == "pursuer" else "pursuer"
                if baseline_fn is not None:
                    opp_action = baseline_fn(env)
                else:
                    if opp_role == "pursuer":
                        opp_action = env.pursuer_action_space.sample()
                    else:
                        opp_action = env.evader_action_space.sample()

                # Expand 1D fixed-speed actions to 2D [v_max, omega]
                if self.fixed_speed_v_max is not None and agent_action.shape[-1] == 1:
                    agent_action = np.array(
                        [self.fixed_speed_v_max, agent_action[0]], dtype=np.float32
                    )

                # Step environment
                if self.role == "pursuer":
                    obs, _, terminated, truncated, info = env.step(
                        agent_action, opp_action
                    )
                else:
                    obs, _, terminated, truncated, info = env.step(
                        opp_action, agent_action
                    )
                done = terminated or truncated

            # Determine outcome
            if "episode_metrics" in info:
                captured = info["episode_metrics"].get("captured", False)
                if self.role == "pursuer" and captured:
                    wins += 1
                elif self.role == "evader" and not captured:
                    wins += 1

        return wins / max(self.n_eval, 1)


# ─── Scripted Baseline Policies ───


def pure_pursuit_policy(env) -> np.ndarray:
    """Pursuer heads directly toward evader at full speed."""
    dx = env.evader_state[0] - env.pursuer_state[0]
    dy = env.evader_state[1] - env.pursuer_state[1]
    angle_to_target = np.arctan2(dy, dx)
    angle_diff = angle_to_target - env.pursuer_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    return np.array([env.pursuer_v_max, np.clip(angle_diff * 3.0,
                     -env.pursuer_omega_max, env.pursuer_omega_max)])


def flee_away_policy(env) -> np.ndarray:
    """Evader flees directly away from pursuer at full speed."""
    dx = env.evader_state[0] - env.pursuer_state[0]
    dy = env.evader_state[1] - env.pursuer_state[1]
    flee_angle = np.arctan2(dy, dx)
    angle_diff = flee_angle - env.evader_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    return np.array([env.evader_v_max, np.clip(angle_diff * 3.0,
                     -env.evader_omega_max, env.evader_omega_max)])


def flee_to_corner_policy(env) -> np.ndarray:
    """Evader flees to nearest arena corner."""
    corners = [
        (0.5, 0.5),
        (env.arena_width - 0.5, 0.5),
        (0.5, env.arena_height - 0.5),
        (env.arena_width - 0.5, env.arena_height - 0.5),
    ]
    nearest = min(
        corners,
        key=lambda c: np.hypot(c[0] - env.evader_state[0], c[1] - env.evader_state[1]),
    )
    dx = nearest[0] - env.evader_state[0]
    dy = nearest[1] - env.evader_state[1]
    angle = np.arctan2(dy, dx)
    diff = (angle - env.evader_state[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([env.evader_v_max, np.clip(diff * 3.0,
                     -env.evader_omega_max, env.evader_omega_max)])
