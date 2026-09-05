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

Three immobility diagnostics, because no single one covers every mechanism:
- `deadlock`: DEADLOCK_STEPS consecutive `stalled` steps (executed speed in
  the stall band AND the nominal wanted to move).
- `immobile_step_rate` / `longest_immobile_run`: the same speed test with no
  condition on the nominal, so a policy that lowers its own commanded speed
  cannot hide shield-induced immobility.
- `pinned`: DEADLOCK_STEPS consecutive BACKUP-tier steps. The backup tier
  means no candidate action could keep every hard (body) row, i.e. the state
  is outside the invariant set — at delta = 0 this cannot happen after a valid
  reset, on the robust arm it is the signature of an episode that started
  inside the inflated set and cannot get out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Every method `record` accepts. "none" is NoOpShield — the pre-registered
# unshielded negative control — and is kept distinct from "exact" so that
# control's zero infeasibility rate can never be read as an all-exact QP
# measurement. A method outside this tuple is an error, not a silent drop:
# a new tier that only half-registered would produce a histogram that does
# not sum to the step count beside a contradictory infeasibility rate.
TIERS = ("exact", "relaxed", "backup", "none")
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
    n_immobile: int = 0
    longest_stall_run: int = 0
    longest_immobile_run: int = 0
    longest_backup_run: int = 0
    # In-flight run counters: reset by the next non-matching step, so they are
    # not part of the reported state. repr/compare off so a gate script that
    # serializes or compares the dataclass instead of calling summary() cannot
    # pick them up; init off so they cannot be seeded through the constructor
    # either (ShieldMetrics(_current_stall_run=7) would otherwise report a
    # non-empty accumulator as already stalled).
    _current_stall_run: int = field(default=0, repr=False, compare=False, init=False)
    _current_immobile_run: int = field(default=0, repr=False, compare=False, init=False)
    _current_backup_run: int = field(default=0, repr=False, compare=False, init=False)
    method_counts: dict[str, int] = field(default_factory=dict)
    min_h_floor: float = math.inf  # worst mixed-unit QP margin seen anywhere
    min_h_sum: float = 0.0  # for the mean margin
    min_clearance_floor: float = math.inf  # worst signed distance, metres
    min_h_obstacle_floor: float = math.inf  # worst obstacle barrier (m²)
    min_h_boundary_floor: float = math.inf  # worst boundary barrier (m)

    def record(self, shield_info: dict, violation: bool) -> None:
        """One step: shield_info from CBFShield.filter, violation from env info."""
        method = shield_info["method"]
        if method not in TIERS:
            raise ValueError(
                f"unknown shield method {method!r}; add it to pe2.metrics.TIERS "
                "(and decide how it counts toward feasibility) before recording it"
            )
        self.n_steps += 1
        self.n_violations += int(violation)
        self.n_interventions += int(shield_info["intervention"])
        self.method_counts[method] = self.method_counts.get(method, 0) + 1
        # The shield's own `feasible` flag is the single source of truth; do
        # NOT re-derive it from the method string here, or a new tier can be
        # feasible in one module and infeasible in the other.
        self.n_infeasible += int(not shield_info["feasible"])
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
        if shield_info["immobile"]:
            self.n_immobile += 1
            self._current_immobile_run += 1
            self.longest_immobile_run = max(
                self.longest_immobile_run, self._current_immobile_run
            )
        else:
            self._current_immobile_run = 0
        # Tier-3 pinning: consecutive steps on which the hard rows were
        # unsatisfiable. Same threshold as deadlock.
        if method == "backup":
            self._current_backup_run += 1
            self.longest_backup_run = max(
                self.longest_backup_run, self._current_backup_run
            )
        else:
            self._current_backup_run = 0

    def summary(self) -> dict:
        """Rates + margins. Every floor is None if it was never observed.

        `min_h_floor` / `min_h_mean` are mixed-unit QP margins (see the module
        docstring) and must not be reported as distances; `min_clearance_floor`
        is the reportable margin in metres.
        """
        n = self.n_steps
        return {
            "n_steps": n,
            # Integer counts alongside the rates: the sweep aggregator sums
            # these rather than reconstructing counts from floating-point
            # rates, so a pooled count is exact and cannot disagree with the
            # pooled tier histogram.
            "n_interventions": self.n_interventions,
            "n_infeasible": self.n_infeasible,
            "n_backup": self.n_backup,
            "n_stalled": self.n_stalled,
            "violation_rate": self.n_violations / n if n else 0.0,
            # NOT a safety-effort proxy: on the relaxed tier the shield can
            # return the nominal unmodified while an obstacle barrier is
            # negative, so a step can log no intervention while being among
            # the least safe in the episode. Read it with the tier histogram.
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
            # Still stalled on the last recorded step. A genuine
            # force-equilibrium never recovers; a slow turn away does, so this
            # separates the two at equal stall-run length.
            "stalled_at_end": self._current_stall_run > 0,
            "n_immobile": self.n_immobile,
            "immobile_step_rate": self.n_immobile / n if n else 0.0,
            "longest_immobile_run": self.longest_immobile_run,
            "longest_backup_run": self.longest_backup_run,
            "pinned": self.longest_backup_run >= DEADLOCK_STEPS,
        }


def summarize_shield(logs: list[dict]) -> dict:
    """Aggregate `run_shielded_duel` logs into the sweep-level gate numbers.

    Step-level rates are pooled over all steps (weighted by episode length),
    matching how G1 counts "violation steps over N episodes". Per-episode
    minima are kept as a list so the gate can report their distribution.

    `min_h_per_episode` / `min_h_floor` are MIXED-UNIT quadratic-program
    margins (obstacle barriers m², boundary barriers m) and must never be
    reported as distances; `min_clearance_per_episode` / `min_clearance_floor`
    are the reportable metres.

    REFUSES (rather than tolerates) three malformed inputs, because all three
    would silently corrupt a gate number:
    - a log with no `shield` block or no `shielded_role` — i.e. a `run_duel`
      log. Use `run_shielded_duel` with a `pe2.shield.NoOpShield` for the
      unshielded control so its shield block is measured, not invented.
    - a mix of shielded roles. The stall and immobility thresholds are
      fractions of the per-role speed cap (1.0 for a shielded pursuer, 1.2
      for a shielded evader), so the two arms are not poolable.
    - a mix of a real shielded arm and a NoOpShield control arm. Both are
      valid `run_shielded_duel` logs of the same role, so nothing above
      catches this; only a partially-'none' pooled tier histogram does.
    """
    for i, log in enumerate(logs):
        if "shield" not in log:
            raise ValueError(
                f"log {i} has no 'shield' block: summarize_shield consumes "
                "run_shielded_duel logs only. For the unshielded control, run "
                "run_shielded_duel with pe2.shield.NoOpShield — never a "
                "hand-rolled pass-through, whose shield block would be fabricated."
            )
        if "shielded_role" not in log:
            raise ValueError(
                f"log {i} has no 'shielded_role': summarize_shield consumes "
                "run_shielded_duel logs only."
            )
    roles = sorted({log["shielded_role"] for log in logs})
    if len(roles) > 1:
        raise ValueError(
            f"logs mix shielded roles {roles}; summarize_shield pools one "
            "shielded role at a time because the stall and immobility "
            "thresholds are fractions of the per-role speed cap"
        )
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
    if 0 < tiers["none"] < n_steps:
        raise ValueError(
            f"{tiers['none']}/{n_steps} pooled steps are tier 'none' and the "
            "rest are not: logs mix a real shielded arm with a NoOpShield "
            "control arm. The two are the arms of a comparison and must be "
            "summarized separately, never pooled into one summary."
        )
    min_h = [log["shield"]["min_h_floor"] for log in logs]
    min_clearance = [log["shield"]["min_clearance_floor"] for log in logs]
    deadlocks = sum(bool(log["shield"]["deadlock"]) for log in logs)
    pinned = sum(bool(log["shield"]["pinned"]) for log in logs)
    # Exact pooled counts, summed from the per-episode integers. The rate keys
    # below keep their original pooling (rate x n_steps) for continuity.
    counts = {
        key: sum(log["shield"][key] for log in logs)
        for key in ("n_interventions", "n_infeasible", "n_backup", "n_stalled")
    }
    n_immobile = sum(log["shield"]["n_immobile"] for log in logs)
    with_violation = sum(log["violations"] > 0 for log in logs)
    return {
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "shielded_roles": roles,
        **counts,
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
        # The distribution behind the deadlock boolean, so DEADLOCK_STEPS is a
        # reported cut on visible data rather than a hidden constant. A benign
        # nose-first wall approach with a full-rate turn away takes roughly 28
        # steps to clear the stall band; FovLimitedEvader's 0.3 * omega_max
        # scan takes about 37, one short of the cut, and a slower learned
        # policy crosses it — so the per-episode runs and the
        # still-stalled-at-the-end flag are what separate a slow recovery
        # from a real standoff.
        "stall_step_rate": counts["n_stalled"] / n_steps if n_steps else 0.0,
        "longest_stall_run_per_episode": [
            log["shield"]["longest_stall_run"] for log in logs
        ],
        "stalled_at_episode_end": [
            bool(log["shield"]["stalled_at_end"]) for log in logs
        ],
        "episodes_stalled_at_end": sum(
            bool(log["shield"]["stalled_at_end"]) for log in logs
        ),
        "n_immobile": n_immobile,
        "immobile_step_rate": n_immobile / n_steps if n_steps else 0.0,
        "longest_immobile_run_per_episode": [
            log["shield"]["longest_immobile_run"] for log in logs
        ],
        "longest_backup_run_per_episode": [
            log["shield"]["longest_backup_run"] for log in logs
        ],
        "pinned_episodes": pinned,
        "pinned_episode_rate": pinned / n_episodes if n_episodes else 0.0,
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
