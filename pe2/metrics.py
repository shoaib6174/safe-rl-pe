"""Step 6: shield metrics — violation/intervention rates, CBF margins,
infeasibility, fallback-tier histogram and a deadlock diagnostic.

A violation is an env-level event (wall_contact or obstacle_contact in the env
info dict), NOT h < 0 in the shield — the two coincide when constraint radii
include agent_radius, but the env flag is the ground truth the gate counts.

Infeasibility: the exact all-constraints QP had no solution (method != "exact").
Deadlock: CBF controllers can settle at a force-equilibrium and stall short of
the goal (Grover, Liu & Sycara, WAFR 2020 / IJRR 2023). We detect the
behavioral signature: the shield holds executed speed near zero while the
nominal policy wants to move, for DEADLOCK_STEPS consecutive steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TIERS = ("exact", "relaxed", "backup")
DEADLOCK_STEPS = 40  # 2.0 s at dt = 0.05


@dataclass
class ShieldMetrics:
    """Accumulates per-step shield stats for one episode (or a whole sweep)."""

    n_steps: int = 0
    n_violations: int = 0
    n_interventions: int = 0
    n_infeasible: int = 0
    n_backup: int = 0
    n_stalled: int = 0
    longest_stall_run: int = 0
    _current_stall_run: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)
    min_h_floor: float = math.inf  # worst margin seen anywhere
    min_h_sum: float = 0.0  # for the mean margin

    def record(self, shield_info: dict, violation: bool) -> None:
        """One step: shield_info from CBFShield.filter, violation from env info."""
        self.n_steps += 1
        self.n_violations += int(violation)
        self.n_interventions += int(shield_info["intervention"])
        method = shield_info["method"]
        self.method_counts[method] = self.method_counts.get(method, 0) + 1
        self.n_infeasible += int(method != "exact")
        self.n_backup += int(method == "backup")
        self.min_h_floor = min(self.min_h_floor, shield_info["min_h"])
        self.min_h_sum += shield_info["min_h"]
        if shield_info["stalled"]:
            self.n_stalled += 1
            self._current_stall_run += 1
            self.longest_stall_run = max(
                self.longest_stall_run, self._current_stall_run
            )
        else:
            self._current_stall_run = 0

    def summary(self) -> dict:
        """Rates + margins. min_h_floor is None if no steps were recorded."""
        n = self.n_steps
        return {
            "n_steps": n,
            "violation_rate": self.n_violations / n if n else 0.0,
            "intervention_rate": self.n_interventions / n if n else 0.0,
            "infeasibility_rate": self.n_infeasible / n if n else 0.0,
            "backup_rate": self.n_backup / n if n else 0.0,
            "method_counts": dict(self.method_counts),
            "tier_histogram": {t: self.method_counts.get(t, 0) for t in TIERS},
            "min_h_floor": None if n == 0 else self.min_h_floor,
            "min_h_mean": self.min_h_sum / n if n else 0.0,
            "stall_step_rate": self.n_stalled / n if n else 0.0,
            "longest_stall_run": self.longest_stall_run,
            "deadlock": self.longest_stall_run >= DEADLOCK_STEPS,
        }
