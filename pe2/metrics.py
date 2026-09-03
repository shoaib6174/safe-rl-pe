"""Step 6: shield metrics — violation/intervention rates, CBF margins.

A violation is an env-level event (wall_contact or obstacle_contact in the env
info dict), NOT h < 0 in the shield — the two coincide when constraint radii
include agent_radius, but the env flag is the ground truth the gate counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ShieldMetrics:
    """Accumulates per-step shield stats for one episode or a whole sweep."""

    n_steps: int = 0
    n_violations: int = 0
    n_interventions: int = 0
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
        self.min_h_floor = min(self.min_h_floor, shield_info["min_h"])
        self.min_h_sum += shield_info["min_h"]

    def summary(self) -> dict:
        """Rates + margins. min_h_floor is None if no steps were recorded."""
        return {
            "n_steps": self.n_steps,
            "violation_rate": self.n_violations / self.n_steps if self.n_steps else 0.0,
            "intervention_rate": (
                self.n_interventions / self.n_steps if self.n_steps else 0.0
            ),
            "method_counts": dict(self.method_counts),
            "min_h_floor": None if self.n_steps == 0 else self.min_h_floor,
            "min_h_mean": self.min_h_sum / self.n_steps if self.n_steps else 0.0,
        }
