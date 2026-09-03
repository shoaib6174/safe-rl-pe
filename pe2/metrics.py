"""Step 6: shield metrics — violation/intervention rates, CBF margins,
infeasibility, fallback-tier histogram and a deadlock diagnostic.

A violation is an env-level event (wall_contact or obstacle_contact in the env
info dict), NOT h < 0 in the shield. The two do NOT coincide: the shield's
barriers are evaluated at a virtual control point d = 0.1 m ahead of the body,
so a barrier is a conservative early warning that leads contact by roughly d
in the heading direction (on top of the agent_radius folded into the
constraint radii). h < 0 does not imply contact; only the env flag does, and
it is the ground truth the gate counts.

Units: `min_h` is the shield's worst quadratic-program margin over constraints
of both kinds and is therefore MIXED-UNIT (obstacle barriers are m², boundary
barriers are m). It must never be reported as a distance. `min_clearance` is
the worst signed distance in metres and is the reportable safety margin;
`min_h_obstacle_floor` / `min_h_boundary_floor` are filter-internal
diagnostics.

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


def _floor(value: float) -> float | None:
    """None when a floor was never observed (still at +inf)."""
    return None if value == math.inf else value


def _min_or_none(values) -> float | None:
    """Minimum of the non-None values; None when every value is None.

    Only for the PER-KIND barrier floors, where None legitimately means "this
    constraint kind never appeared" (an obstacle-free map has no obstacle
    barrier). It is deliberately NOT used for `min_h_floor` /
    `min_clearance_floor`, where a None comes from an episode with no recorded
    steps and must surface as an error rather than be skipped.
    """
    present = [v for v in values if v is not None]
    return min(present) if present else None


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
    min_h_floor: float = math.inf  # worst mixed-unit QP margin seen anywhere
    min_h_sum: float = 0.0  # for the mean margin
    min_clearance_floor: float = math.inf  # worst signed distance, metres
    min_h_obstacle_floor: float = math.inf  # worst obstacle barrier (m²)
    min_h_boundary_floor: float = math.inf  # worst boundary barrier (m)

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
        self.min_clearance_floor = min(
            self.min_clearance_floor, shield_info["min_clearance"]
        )
        # A per-kind barrier is absent when that kind had no constraints this
        # step (no obstacles in the map); leave its floor untouched.
        for key, attr in (
            ("min_h_obstacle", "min_h_obstacle_floor"),
            ("min_h_boundary", "min_h_boundary_floor"),
        ):
            value = shield_info[key]
            if value is not None:
                setattr(self, attr, min(getattr(self, attr), value))
        if shield_info["stalled"]:
            self.n_stalled += 1
            self._current_stall_run += 1
            self.longest_stall_run = max(
                self.longest_stall_run, self._current_stall_run
            )
        else:
            self._current_stall_run = 0

    def summary(self) -> dict:
        """Rates + margins. Every floor is None if it was never observed.

        `min_h_floor` / `min_h_mean` are mixed-unit QP margins (see the module
        docstring) and must not be reported as distances; `min_clearance_floor`
        is the reportable margin in metres.
        """
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
            "min_clearance_floor": _floor(self.min_clearance_floor),
            "min_h_obstacle_floor": _floor(self.min_h_obstacle_floor),
            "min_h_boundary_floor": _floor(self.min_h_boundary_floor),
            "stall_step_rate": self.n_stalled / n if n else 0.0,
            "longest_stall_run": self.longest_stall_run,
            "deadlock": self.longest_stall_run >= DEADLOCK_STEPS,
        }


def summarize_shield(logs: list[dict]) -> dict:
    """Aggregate `run_shielded_duel` logs into the sweep-level gate numbers.

    Step-level rates are pooled over all steps (weighted by episode length),
    matching how G1 counts "violation steps over N episodes". Per-episode
    minima are kept as a list so the gate can report their distribution.
    """
    n_episodes = len(logs)
    n_steps = sum(log["shield"]["n_steps"] for log in logs)
    tiers = {t: 0 for t in TIERS}
    interventions = 0.0
    infeasible = 0.0
    backup = 0.0
    for log in logs:
        sh = log["shield"]
        for t in TIERS:
            tiers[t] += sh["tier_histogram"].get(t, 0)
        interventions += sh["intervention_rate"] * sh["n_steps"]
        infeasible += sh["infeasibility_rate"] * sh["n_steps"]
        backup += sh["backup_rate"] * sh["n_steps"]
    min_h = [log["shield"]["min_h_floor"] for log in logs]
    min_clearance = [log["shield"]["min_clearance_floor"] for log in logs]
    deadlocks = sum(bool(log["shield"]["deadlock"]) for log in logs)
    with_violation = sum(log["violations"] > 0 for log in logs)
    return {
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "violation_steps": sum(log["violations"] for log in logs),
        "episodes_with_violation": with_violation,
        "violation_episode_rate": with_violation / n_episodes if n_episodes else 0.0,
        "intervention_rate": interventions / n_steps if n_steps else 0.0,
        "infeasibility_rate": infeasible / n_steps if n_steps else 0.0,
        "backup_rate": backup / n_steps if n_steps else 0.0,
        "tier_histogram": tiers,
        "min_h_per_episode": min_h,
        "min_h_floor": min(min_h) if min_h else None,
        "min_clearance_per_episode": min_clearance,
        "min_clearance_floor": min(min_clearance) if min_clearance else None,
        "min_h_obstacle_floor": _min_or_none(
            log["shield"]["min_h_obstacle_floor"] for log in logs
        ),
        "min_h_boundary_floor": _min_or_none(
            log["shield"]["min_h_boundary_floor"] for log in logs
        ),
        "deadlock_episodes": deadlocks,
        "deadlock_episode_rate": deadlocks / n_episodes if n_episodes else 0.0,
        "timeout_rate": (
            sum(log["outcome"] == "timeout" for log in logs) / n_episodes
            if n_episodes
            else 0.0
        ),
        "capture_rate": (
            sum(log["outcome"] == "capture" for log in logs) / n_episodes
            if n_episodes
            else 0.0
        ),
    }
