"""Safety metrics tracking for CBF-constrained training.

SB3-independent tracker for all safety-related metrics during training.
Integrates with SafeActionResolver to record per-step CBF outcomes.

Tracked metrics:
- Safety violations (obstacle/wall collisions)
- CBF feasibility rate per phase
- CBF intervention rate (should decrease over training)
- Backup controller activation rate (target < 1%)
- CBF margin statistics (min, mean, distribution)
- Resolution method breakdown (exact, relaxed, backup)
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SafetyMetricsTracker:
    """Tracks safety metrics during CBF-constrained training.

    SB3-independent — collects data via record_step() and provides
    aggregate statistics via get_phase_metrics() and get_summary().

    Args:
        window_size: Rolling window for recent statistics.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Lifetime counters
        self.total_steps = 0
        self.total_violations = 0
        self.total_interventions = 0
        self.total_backups = 0
        self.total_infeasible = 0

        # Phase-level tracking
        self._current_phase = 0
        self._phase_steps = 0
        self._phase_violations = 0
        self._phase_interventions = 0
        self._phase_backups = 0
        self._phase_margins: list[float] = []
        self._phase_methods: dict[str, int] = {}

        # History per phase
        self.phase_metrics: list[dict] = []

        # Rolling window
        self._recent_margins = deque(maxlen=window_size)
        self._recent_methods = deque(maxlen=window_size)
        self._recent_violations = deque(maxlen=window_size)

    def record_step(
        self,
        method: str,
        min_h: float = 0.0,
        intervention: bool = False,
        violation: bool = False,
    ):
        """Record a single step's safety outcomes.

        Args:
            method: Resolution method ('exact', 'relaxed_obstacles',
                    'relaxed_arena', 'backup', etc.).
            min_h: Minimum CBF value across all constraints.
            intervention: Whether CBF modified the action.
            violation: Whether a safety violation occurred (collision).
        """
        self.total_steps += 1
        self._phase_steps += 1

        if violation:
            self.total_violations += 1
            self._phase_violations += 1
        if intervention:
            self.total_interventions += 1
            self._phase_interventions += 1
        if method == "backup":
            self.total_backups += 1
            self._phase_backups += 1
        if method in ("relaxed_obstacles", "relaxed_arena", "unconstrained"):
            self.total_infeasible += 1

        self._phase_margins.append(min_h)
        self._phase_methods[method] = self._phase_methods.get(method, 0) + 1

        self._recent_margins.append(min_h)
        self._recent_methods.append(method)
        self._recent_violations.append(violation)

    def end_phase(self) -> dict:
        """Finalize current phase and return its metrics.

        Call this at the end of each self-play phase to archive metrics
        and reset phase-level counters.

        Returns:
            Dict with all phase-level safety metrics.
        """
        metrics = self._compute_phase_metrics()
        self.phase_metrics.append(metrics)

        # Reset phase counters
        self._current_phase += 1
        self._phase_steps = 0
        self._phase_violations = 0
        self._phase_interventions = 0
        self._phase_backups = 0
        self._phase_margins = []
        self._phase_methods = {}

        return metrics

    def _compute_phase_metrics(self) -> dict:
        """Compute metrics for the current phase."""
        n = max(self._phase_steps, 1)
        margins = self._phase_margins if self._phase_margins else [0.0]

        return {
            "phase": self._current_phase,
            "steps": self._phase_steps,
            "violations": self._phase_violations,
            "violation_rate": self._phase_violations / n,
            "intervention_rate": self._phase_interventions / n,
            "backup_rate": self._phase_backups / n,
            "feasibility_rate": 1.0 - sum(
                1 for m in self._phase_methods
                if m in ("relaxed_obstacles", "relaxed_arena", "unconstrained", "backup")
            ) / n,
            "exact_rate": self._phase_methods.get("exact", 0) / n,
            "min_cbf_margin": float(np.min(margins)),
            "mean_cbf_margin": float(np.mean(margins)),
            "std_cbf_margin": float(np.std(margins)),
            "method_counts": dict(self._phase_methods),
        }

    def get_recent_metrics(self) -> dict:
        """Get metrics over the rolling window.

        Returns:
            Dict with recent safety statistics.
        """
        n = max(len(self._recent_margins), 1)
        margins = list(self._recent_margins) if self._recent_margins else [0.0]
        methods = list(self._recent_methods)
        violations = list(self._recent_violations)

        return {
            "window_size": len(self._recent_margins),
            "violation_rate": sum(violations) / n,
            "intervention_rate": sum(1 for m in methods if m != "exact") / n,
            "backup_rate": sum(1 for m in methods if m == "backup") / n,
            "exact_rate": sum(1 for m in methods if m == "exact") / n,
            "min_cbf_margin": float(np.min(margins)),
            "mean_cbf_margin": float(np.mean(margins)),
        }

    def get_summary(self) -> dict:
        """Get overall summary across all phases.

        Returns:
            Dict with lifetime safety statistics.
        """
        n = max(self.total_steps, 1)
        return {
            "total_steps": self.total_steps,
            "total_violations": self.total_violations,
            "violation_rate": self.total_violations / n,
            "total_interventions": self.total_interventions,
            "intervention_rate": self.total_interventions / n,
            "total_backups": self.total_backups,
            "backup_rate": self.total_backups / n,
            "total_infeasible": self.total_infeasible,
            "infeasibility_rate": self.total_infeasible / n,
            "n_phases_completed": len(self.phase_metrics),
        }

    def check_safety_targets(self) -> dict:
        """Check if safety targets are met.

        Returns:
            Dict with boolean checks for each target:
            - zero_violations: No safety violations
            - feasibility_above_99: CBF-QP feasibility > 99%
            - backup_below_1pct: Backup activations < 1%
        """
        summary = self.get_summary()
        return {
            "zero_violations": summary["total_violations"] == 0,
            "feasibility_above_99": summary["infeasibility_rate"] < 0.01,
            "backup_below_1pct": summary["backup_rate"] < 0.01,
            "all_targets_met": (
                summary["total_violations"] == 0
                and summary["infeasibility_rate"] < 0.01
                and summary["backup_rate"] < 0.01
            ),
        }


def _get_safety_metrics_callback_class():
    """Lazy import of SB3 BaseCallback and return SafetyMetricsCallback."""
    from stable_baselines3.common.callbacks import BaseCallback

    class SafetyMetricsCallback(BaseCallback):
        """SB3 callback that logs safety metrics to TensorBoard/wandb.

        Args:
            tracker: SafetyMetricsTracker instance.
            log_frequency: How often to log (in steps).
            verbose: Verbosity level.
        """

        def __init__(
            self,
            tracker: SafetyMetricsTracker,
            log_frequency: int = 1024,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.tracker = tracker
            self.log_frequency = log_frequency

        def _on_step(self) -> bool:
            # Log to TensorBoard at frequency
            if self.n_calls % self.log_frequency == 0:
                recent = self.tracker.get_recent_metrics()
                self.logger.record("safety/intervention_rate", recent["intervention_rate"])
                self.logger.record("safety/backup_rate", recent["backup_rate"])
                self.logger.record("safety/violation_rate", recent["violation_rate"])
                self.logger.record("safety/mean_cbf_margin", recent["mean_cbf_margin"])
                self.logger.record("safety/min_cbf_margin", recent["min_cbf_margin"])
                self.logger.record("safety/exact_rate", recent["exact_rate"])
            return True

    return SafetyMetricsCallback
