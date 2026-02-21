"""Self-play health monitoring: entropy, diversity, rollback.

Monitors training health and triggers alerts on detected failures:
- Policy entropy collapse (mode collapse)
- Trajectory diversity collapse (degenerate behavior)
- Capture rate imbalance (one agent dominating)
- Greedy baseline regression (mutual degradation)
"""

from pathlib import Path

import numpy as np


class SelfPlayHealthMonitor:
    """Monitors self-play training health and manages checkpoints."""

    def __init__(
        self,
        min_entropy: float = 0.1,
        max_capture_rate: float = 0.90,
        min_capture_rate: float = 0.10,
        greedy_eval_interval: int = 2,
        max_checkpoints: int = 5,
    ):
        self.min_entropy = min_entropy
        self.max_capture_rate = max_capture_rate
        self.min_capture_rate = min_capture_rate
        self.greedy_eval_interval = greedy_eval_interval
        self.max_checkpoints = max_checkpoints
        self._checkpoint_phases = []

    def check_entropy(self, model) -> float | None:
        """Estimate policy entropy from the model's action distribution.

        For SB3 PPO with Gaussian policy:
            entropy = 0.5 * log(2 * pi * e * sigma^2)

        Returns the mean entropy across action dimensions, or None if
        entropy cannot be computed (e.g., no rollout buffer data).
        """
        try:
            # SB3 stores log_std as a learnable parameter
            policy = model.policy
            if hasattr(policy, "log_std"):
                log_std = policy.log_std.detach().cpu().numpy()
                # Gaussian entropy: 0.5 * log(2 * pi * e) + log_std
                entropy_per_dim = 0.5 * np.log(2 * np.pi * np.e) + log_std
                return float(np.mean(entropy_per_dim))
            return None
        except Exception:
            return None

    def check_trajectory_diversity(
        self,
        trajectories: list,
        n_clusters: int = 3,
    ) -> int:
        """Check trajectory diversity via simple endpoint clustering.

        Uses a lightweight distance-based clustering instead of sklearn
        to avoid the dependency. Groups endpoints that are within 2m of
        each other.

        Args:
            trajectories: List of trajectories, each a list of (x, y) positions.
            n_clusters: Not used directly; kept for API compatibility.

        Returns:
            Number of distinct endpoint groups (should be >= 2 for healthy diversity).
        """
        if len(trajectories) < 2:
            return 1

        # Extract final positions
        endpoints = []
        for traj in trajectories:
            if len(traj) > 0:
                endpoints.append(traj[-1])

        if len(endpoints) < 2:
            return 1

        endpoints = np.array(endpoints)

        # Simple greedy clustering: group endpoints within 2m radius
        cluster_threshold = 2.0
        clusters = []
        assigned = [False] * len(endpoints)

        for i in range(len(endpoints)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(endpoints)):
                if not assigned[j]:
                    dist = np.sqrt(
                        (endpoints[i][0] - endpoints[j][0]) ** 2
                        + (endpoints[i][1] - endpoints[j][1]) ** 2
                    )
                    if dist < cluster_threshold:
                        cluster.append(j)
                        assigned[j] = True
            clusters.append(cluster)

        return len(clusters)

    def should_rollback(self, capture_rate_history: list[float]) -> bool:
        """Check if capture rate is imbalanced for 2 consecutive phases.

        Returns True if the last 2 phases have capture rate outside
        [min_capture_rate, max_capture_rate].
        """
        if len(capture_rate_history) < 2:
            return False

        last_two = capture_rate_history[-2:]
        return all(
            cr > self.max_capture_rate or cr < self.min_capture_rate
            for cr in last_two
        )

    def save_checkpoint(
        self,
        pursuer_model,
        evader_model,
        checkpoint_dir: Path,
        phase: int,
    ):
        """Save model checkpoint with rolling window.

        Keeps only the last max_checkpoints checkpoints to avoid disk bloat.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pursuer_model.save(str(checkpoint_dir / f"pursuer_phase{phase}"))
        evader_model.save(str(checkpoint_dir / f"evader_phase{phase}"))
        self._checkpoint_phases.append(phase)

        # Clean old checkpoints
        while len(self._checkpoint_phases) > self.max_checkpoints:
            old_phase = self._checkpoint_phases.pop(0)
            p_path = checkpoint_dir / f"pursuer_phase{old_phase}.zip"
            e_path = checkpoint_dir / f"evader_phase{old_phase}.zip"
            if p_path.exists():
                p_path.unlink()
            if e_path.exists():
                e_path.unlink()

    def get_latest_checkpoint_phase(self) -> int | None:
        """Return the most recent checkpoint phase number."""
        if self._checkpoint_phases:
            return self._checkpoint_phases[-1]
        return None
