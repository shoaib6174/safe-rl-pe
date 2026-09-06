"""Step 6 gate: the pre-registered shield gates G1-G5.

Criteria are PRE-REGISTERED (D83, written 2026-08-25 in
`docs/worklogs/2026-08-24_S119.md` before any implementation) and are
reproduced verbatim below. Nothing here may adjust a threshold, a seed range,
an episode count or a definition; a criterion that cannot be met is reported,
not amended.

- G1 (kill criterion): shielded GreedyPursuer vs GreedyEvader, N=1000 seeded
  episodes (seeds 800000+) -> 0 violation steps. Any violation -> shield math
  wrong; stop.
- G2 (capture intact): shielded capture rate >= 0.5 x unshielded capture rate
  on the same paired seeds; paired McNemar reported as the primary comparison
  with effect size (descriptive, no p-threshold).
- G3 (adversarial probe): `ObstacleHuggingEvader` vs shielded GreedyPursuer,
  N=1000 (seeds 810000+) -> 0 violations; timeout rate reported.
- G4 (negative control): the identical probe WITHOUT the shield ->
  episodes-with->=1-violation must be >= 5%. If fewer, the PROBE is too weak
  and gets fixed -- never the gate.
- G5 (robust-margin arm, D87): the same shield with every constraint radius
  and boundary limit inflated by delta = 0.05 m, gated on the G1 and G3
  criteria, plus its intervention rate reported against the unmargined arm's.
  delta is the hardware localization bound, not a tuned value:
  `docs/phases/phase4_sim_to_real_transfer.md` requires localization error
  < 5 cm at the 95th percentile (`test_localization_accuracy`) and its §8.7
  prescribes adding that position uncertainty to the obstacle radii in the
  CBF -- the measurement-robust CBF construction of D87.

v2 (2026-09-05): criteria unchanged, seed streams fresh, shield redesigned --
see `docs/pathway/step6_shield_v2_prereg_2026-09-05.md`. The violation
definition is unchanged and magnitude-blind; the contact-depth census is
reported, never gated.

A violation is an ENV contact step (wall or obstacle) of the SHIELDED role,
never a barrier sign (pre-registered violation semantics; D89 corrected the
docstrings that claimed h < 0 coincides with contact -- the VCP leads the body
by 0.1 m, so a barrier is a conservative early warning).

Reported alongside the gated numbers, per plan v3 and D89, never gated:
infeasibility rate, `min_clearance_floor` (the reportable margin, signed
metres -- `min_h` is a mixed-unit QP margin and is reported only under that
name), `min_clearance_hard_floor` (the body-to-surface distance alone, the
minimum over hard rows only -- never separately measured before this field
existed; see D97's correction), the fallback-tier histogram, and the stall /
immobility / pinning rates with their per-episode run-length distributions.

Three review notes from S120 are honoured here: the shielded and control
sweeps go through SEPARATE `summarize_shield` calls (the aggregator refuses a
pooled shielded+control sweep); the control arms' stall / deadlock /
immobility / pinning fields are STRUCTURALLY ZERO rather than measured, since
`NoOpShield` returns the nominal unchanged and the scripted nominal is always
full speed; and G1's Wilson bound comes from `pe2.eval.wilson_ci`, which
`summarize_shield` deliberately does not compute.

Usage: ./venv/bin/python scripts/pe2_step06_gate.py [--n 1000]
Exits non-zero if any gate fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.eval import mcnemar_pvalue, wilson_ci
from pe2.metrics import DEADLOCK_STEPS, summarize_shield
from pe2.scripted import (
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
    run_shielded_duel,
)
from pe2.shield import CBFShield, NoOpShield

OUT_DIR = pathlib.Path("results/visual_gates/step_06")
OUT_JSON = OUT_DIR / "step_06_gate.json"

# --- pre-registered constants (D83 / D87). Do not tune. ---
MAIN_SEED0 = 800000  # v2 stream (v1's 760000+ consumed, spec Part C)
PROBE_SEED0 = 810000
N_EPISODES = 1000
G2_CAPTURE_FACTOR = 0.5  # shielded CR >= 0.5 x unshielded CR
G4_MIN_VIOLATION_EPISODE_RATE = 0.05  # negative control floor
ROBUST_DELTA = 0.05  # metres; phase4 §8.7 + test_localization_accuracy (D87)
ONE_SIDED_Z = 1.645  # 95% one-sided, as the N justification is stated

SHIELDED_ROLE = "pursuer"
# Structurally-zero fields on a NoOpShield arm: the filter returns the nominal
# unchanged, so with a full-speed scripted nominal the speed-band flags can
# never fire and no fallback tier can ever be entered.
CONTROL_STRUCTURAL_ZEROS = (
    "stall_step_rate",
    "immobile_step_rate",
    "deadlock_episode_rate",
    "pinned_episode_rate",
    "backup_rate",
    "infeasibility_rate",
)


# ---------------------------------------------------------------- pure parts


def config_fingerprint(cfg: EnvConfig) -> str:
    """Stable content hash of a config (Python's hash() is salted per process)."""
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def shield_spec(shield) -> dict:
    """Every parameter that defines a shield arm, for the JSON provenance."""
    return {
        "class": type(shield).__name__,
        "gamma": shield.gamma,
        "vcp_d": shield.d,
        "w_v": shield.w_v,
        "w_omega": shield.w_omega,
        "margin": shield.margin,
    }


def violation_upper_bound(k: int, n: int) -> float:
    """One-sided 95% Wilson upper bound on the per-episode violation rate.

    `summarize_shield` deliberately does not compute confidence intervals, so
    the bound the N justification quotes comes from `pe2.eval.wilson_ci`
    (S120 review note). At k=0, n=1000 this is 0.0027, matching the
    pre-registered "rule of three" figure of ~0.003.
    """
    return wilson_ci(k, n, z=ONE_SIDED_Z)[1]


def g2_capture_verdict(shielded_cr: float, control_cr: float) -> bool:
    """G2: the shield must not cost more than half the captures."""
    return shielded_cr >= G2_CAPTURE_FACTOR * control_cr


def run_length_summary(runs: list[int]) -> dict:
    """Distribution of a per-episode run-length list (D89 auditability).

    The 40-step deadlock cut is reported ON this distribution rather than as a
    hidden constant, so a benign slow turn away (~28-37 steps measured) can be
    told apart from a genuine standoff.
    """
    if not runs:
        return {
            "n": 0,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "n_ge_deadlock": 0,
            "histogram": {},
        }
    arr = np.asarray(runs, dtype=float)
    edges = [0, 1, 5, 10, 20, 30, 40, 80, 160, 320]
    histogram = {}
    for lo, hi in zip(edges, edges[1:] + [None], strict=True):
        label = f"{lo}" if hi == lo + 1 else (f"{lo}-{hi - 1}" if hi else f">={lo}")
        histogram[label] = int(
            np.sum((arr >= lo) & (arr < hi)) if hi else np.sum(arr >= lo)
        )
    return {
        "n": len(runs),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n_ge_deadlock": int(np.sum(arr >= DEADLOCK_STEPS)),
        "histogram": histogram,
    }


def episode_row(log: dict) -> dict:
    """Compact per-episode record: every gate number traces back to one seed."""
    sh = log["shield"]
    return {
        "seed": log["seed"],
        "outcome": log["outcome"],
        "steps": log["steps"],
        "violations": log["violations"],
        "pursuer_violations": log["pursuer_violations"],
        "evader_violations": log["evader_violations"],
        "n_steps": sh["n_steps"],
        "n_interventions": sh["n_interventions"],
        "n_infeasible": sh["n_infeasible"],
        "n_backup": sh["n_backup"],
        "n_stalled": sh["n_stalled"],
        "n_immobile": sh["n_immobile"],
        "min_clearance_floor": sh["min_clearance_floor"],
        "min_h_floor": sh["min_h_floor"],
        "longest_stall_run": sh["longest_stall_run"],
        "longest_immobile_run": sh["longest_immobile_run"],
        "longest_backup_run": sh["longest_backup_run"],
        "deadlock": sh["deadlock"],
        "pinned": sh["pinned"],
        "stalled_at_end": sh["stalled_at_end"],
        "start_in_barrier_set": log["start_in_barrier_set"],
        "start_hard_feasible": log["start_hard_feasible"],
    }


def reported_not_gated(summary: dict, is_control: bool) -> dict:
    """The plan-v3 / D89 fields that are REPORTED, never gated."""
    block = {
        "infeasibility_rate": summary["infeasibility_rate"],
        "min_clearance_floor_m": summary["min_clearance_floor"],
        "min_clearance_hard_floor_m": summary["min_clearance_hard_floor"],
        "min_h_floor_qp_margin_mixed_units": summary["min_h_floor"],
        "tier_histogram": summary["tier_histogram"],
        "intervention_rate": summary["intervention_rate"],
        "backup_rate": summary["backup_rate"],
        "stall_step_rate": summary["stall_step_rate"],
        "immobile_step_rate": summary["immobile_step_rate"],
        "deadlock_episode_rate": summary["deadlock_episode_rate"],
        "pinned_episode_rate": summary["pinned_episode_rate"],
        "episodes_stalled_at_end": summary["episodes_stalled_at_end"],
        "stall_run_distribution": run_length_summary(
            summary["longest_stall_run_per_episode"]
        ),
        "immobile_run_distribution": run_length_summary(
            summary["longest_immobile_run_per_episode"]
        ),
        "backup_run_distribution": run_length_summary(
            summary["longest_backup_run_per_episode"]
        ),
    }
    if is_control:
        block["structurally_zero_fields"] = list(CONTROL_STRUCTURAL_ZEROS)
        block["structurally_zero_note"] = (
            "NoOpShield returns the nominal action unchanged and solves no "
            "program, so on this arm the stall / immobility / deadlock / "
            "pinning / backup / infeasibility fields are zero BY CONSTRUCTION "
            "(the scripted nominal is always full speed and no fallback tier "
            "exists), not measured. Read them only beside the 'none' tier."
        )
    return block


# ------------------------------------------------------------------- sweeps


def shielded_pose(env: PursuitEvasionEnv) -> np.ndarray:
    """The pose of whichever role `SHIELDED_ROLE` names.

    `run_shielded_duel` selects the shielded agent's pose by its
    `shielded_role` argument; this mirrors that selection so `start_flags`
    and `forensic_replay` describe the same agent the sweep actually shielded
    rather than hardcoding the pursuer."""
    return env.pursuer_pose if SHIELDED_ROLE == "pursuer" else env.evader_pose


def start_flags(cfg, shield, seed: int) -> dict:
    """Whether the RESET state already satisfies every row / every hard row.

    RC1 in the v2 spec: 1.45% of resets start with a VCP row < 0; the report
    must be able to separate those episodes without re-running."""
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=seed)
    cons = shield.constraints(np.array(shielded_pose(env)), env.obstacles)
    return {
        "start_in_barrier_set": bool(all(c.h >= 0.0 for c in cons)),
        "start_hard_feasible": bool(all(c.h >= 0.0 for c in cons if c.hard)),
    }


def run_sweep(cfg, pursuer_factory, evader_factory, shield, seed0: int, n: int) -> list:
    """N seeded shielded duels; one env reused, one shield reused (stateless)."""
    env = PursuitEvasionEnv(cfg)
    logs = [
        run_shielded_duel(
            env,
            pursuer_factory(),
            evader_factory(),
            shield,
            seed0 + i,
            shielded_role=SHIELDED_ROLE,
        )
        for i in range(n)
    ]
    for i, log in enumerate(logs):
        log.update(start_flags(cfg, shield, seed0 + i))
    return logs


def contact_geometry(cfg, pose_before, action, obstacles: list[dict]) -> dict:
    """How far the executed step would have gone past a surface, in metres.

    Replicates the env's own arithmetic (`_advance_agent` then
    `_project_out_of_obstacles`): one Euler step from `pose_before` under
    `action`, the wall overshoot measured against the same limits the env
    clips to, then the deepest obstacle overlap of the wall-clipped point.
    Both are zero when nothing was resolved, so a contact step's numbers say
    HOW MUCH the env had to correct -- the difference between a shield that
    lets a robot drive into a wall and one that grazes the surface at
    float-noise scale.
    """
    lim = cfg.arena_size / 2.0 - cfg.agent_radius
    x = float(pose_before[0]) + action[0] * np.cos(pose_before[2]) * cfg.dt
    y = float(pose_before[1]) + action[0] * np.sin(pose_before[2]) * cfg.dt
    wall_overshoot = max(0.0, x - lim, -lim - x, y - lim, -lim - y)
    xc, yc = float(np.clip(x, -lim, lim)), float(np.clip(y, -lim, lim))
    penetration = 0.0
    for obs in obstacles:
        dist = float(np.hypot(xc - obs["x"], yc - obs["y"]))
        penetration = max(penetration, obs["radius"] + cfg.agent_radius - dist)
    return {
        "wall_overshoot_m": float(wall_overshoot),
        "obstacle_penetration_m": float(penetration),
    }


def distribution(values: list[float]) -> dict:
    """min / median / p95 / max plus counts above decade thresholds."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "n_above_1e-9_m": int(np.sum(arr > 1e-9)),
        "n_above_1e-6_m": int(np.sum(arr > 1e-6)),
        "n_above_1e-3_m": int(np.sum(arr > 1e-3)),
        "n_above_1e-2_m": int(np.sum(arr > 1e-2)),
    }


def forensic_replay(cfg, pursuer_factory, evader_factory, shield, seed: int) -> dict:
    """Re-run one seed step by step and return EVERY contact step it contains.

    The pre-registration requires the state, the tier and the barrier values
    at an offending step to be reported, and the sweep runner keeps only
    aggregates. Returns the first offending step in full plus a compact row
    per contact step, so the kill report can say whether the arm drove into a
    wall or grazed one. The replay is exact -- `reset(seed)` and the scripted
    policies are deterministic.
    """
    env = PursuitEvasionEnv(cfg)
    pursuer, evader = pursuer_factory(), evader_factory()
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    first: dict | None = None
    rows: list[dict] = []
    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        state_before = np.array(shielded_pose(env), dtype=float)
        obstacles = [dict(o) for o in env.obstacles]
        cons = shield.constraints(state_before, env.obstacles)
        (v_p, w_p), sinfo = shield.filter(
            (v_p, w_p), shielded_pose(env), env.obstacles, v_max=cfg.v_max
        )
        executed = (float(v_p), float(w_p))
        geometry = contact_geometry(cfg, state_before, executed, obstacles)
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        if info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]:
            rows.append(
                {
                    "step": env.current_step,
                    "wall_contact": bool(info["pursuer_wall_contact"]),
                    "obstacle_contact": bool(info["pursuer_obstacle_contact"]),
                    "tier": sinfo["method"],
                    "feasible": sinfo["feasible"],
                    "min_clearance": sinfo["min_clearance"],
                    **geometry,
                }
            )
            if first is None:
                first = {
                    "seed": seed,
                    "step": env.current_step,
                    "pursuer_state_before": state_before.tolist(),
                    "pursuer_state_after": np.array(env.pursuer_pose).tolist(),
                    "evader_state_after": np.array(env.evader_pose).tolist(),
                    "obstacles": obstacles,
                    "wall_contact": bool(info["pursuer_wall_contact"]),
                    "obstacle_contact": bool(info["pursuer_obstacle_contact"]),
                    "tier": sinfo["method"],
                    "feasible": sinfo["feasible"],
                    "nominal_action": [sinfo["v_nom"], sinfo["v_exec"]],
                    "executed_action": list(executed),
                    "min_h": sinfo["min_h"],
                    "min_clearance": sinfo["min_clearance"],
                    "h_obstacle": sinfo["h_obstacle"],
                    "h_boundary": sinfo["h_boundary"],
                    **geometry,
                    "constraints": [
                        {
                            "kind": c.kind,
                            "h": c.h,
                            "clearance": c.clearance,
                            "a_v": c.a_v,
                            "a_omega": c.a_omega,
                            "hard": c.hard,
                            "scale": c.scale,
                        }
                        for c in cons
                    ],
                }
        if terminated or truncated:
            break
    if first is None:
        raise RuntimeError(
            f"forensic replay of seed {seed} found no contact step -- the "
            "replay does not reproduce the sweep, which is a determinism bug"
        )
    return {"first_offending_step": first, "contact_steps": rows}


def violation_census(arm_name: str, cfg, arm, logs: list) -> dict:
    """Replay every offending episode of an arm and describe its violations.

    The pre-registration's kill report is "the state, the tier and the barrier
    values at the offending step"; this is that, plus the distribution over
    all of them, because a single step cannot say whether the arm drove into a
    wall or grazed one. It measures, it does not excuse: the verdict is
    decided by the violation count alone, upstream of here.
    """
    offenders = sorted(
        (log for log in logs if log["violations"] > 0), key=lambda log: log["seed"]
    )
    replays = [
        forensic_replay(cfg, arm["pursuer"], arm["evader"], arm["shield"], log["seed"])
        for log in offenders
    ]
    rows = [row for replay in replays for row in replay["contact_steps"]]
    tiers: dict[str, int] = {}
    for row in rows:
        tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
    return {
        "arm": arm_name,
        "episodes_with_violation": len(offenders),
        "n_episodes": len(logs),
        "violation_steps": sum(log["violations"] for log in logs),
        "contact_steps_replayed": len(rows),
        "offending_seeds": [log["seed"] for log in offenders],
        "wall_contact_steps": sum(row["wall_contact"] for row in rows),
        "obstacle_contact_steps": sum(row["obstacle_contact"] for row in rows),
        "tier_at_contact": tiers,
        "wall_overshoot_m": distribution(
            [row["wall_overshoot_m"] for row in rows if row["wall_contact"]]
        ),
        "obstacle_penetration_m": distribution(
            [row["obstacle_penetration_m"] for row in rows if row["obstacle_contact"]]
        ),
        "min_clearance_at_contact_m": distribution(
            [row["min_clearance"] for row in rows]
        ),
        "first_offending_step": replays[0]["first_offending_step"],
    }


def print_census(gate: str, census: dict) -> None:
    """Print the pre-registered kill report for one arm."""
    print(f"\n{'=' * 72}")
    print(f"{gate} KILL CRITERION TRIPPED on arm {census['arm']}")
    print("=" * 72)
    print(
        f"{census['episodes_with_violation']} of {census['n_episodes']} episodes "
        f"contain a shielded-role contact step; {census['violation_steps']} "
        f"violation steps in total "
        f"({census['wall_contact_steps']} wall, "
        f"{census['obstacle_contact_steps']} obstacle), "
        f"tiers at contact {census['tier_at_contact']}."
    )
    for key in ("wall_overshoot_m", "obstacle_penetration_m"):
        dist = census[key]
        if dist.get("n"):
            print(
                f"  {key}: n={dist['n']} min={dist['min']:.3e} "
                f"p50={dist['p50']:.3e} p95={dist['p95']:.3e} max={dist['max']:.3e} "
                f"(>1e-9 m: {dist['n_above_1e-9_m']}, >1e-3 m: "
                f"{dist['n_above_1e-3_m']}, >1e-2 m: {dist['n_above_1e-2_m']})"
            )
    print("First offending episode, first offending step:")
    print(json.dumps(census["first_offending_step"], indent=2))
    print(
        "\nPre-registration: any shielded violation means the shield math is "
        "wrong. Stopping. This is a finding about the shield, not a gate to "
        "loosen."
    )


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_EPISODES)
    args = parser.parse_args()
    n = args.n

    cfg = EnvConfig()
    fingerprint = config_fingerprint(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arms = {
        # G1 + the shielded half of G2.
        "shielded_main": {
            "pursuer": GreedyPursuer,
            "evader": GreedyEvader,
            "shield": CBFShield(cfg),
            "seed0": MAIN_SEED0,
            "control": False,
        },
        # G2's paired control: same seeds, same policies, no shield.
        "control_main": {
            "pursuer": GreedyPursuer,
            "evader": GreedyEvader,
            "shield": NoOpShield(cfg),
            "seed0": MAIN_SEED0,
            "control": True,
        },
        # G3: the adversarial probe against the shielded pursuer.
        "shielded_probe": {
            "pursuer": GreedyPursuer,
            "evader": ObstacleHuggingEvader,
            "shield": CBFShield(cfg),
            "seed0": PROBE_SEED0,
            "control": False,
        },
        # G4: the same probe with no shield -- the negative control.
        "control_probe": {
            "pursuer": GreedyPursuer,
            "evader": ObstacleHuggingEvader,
            "shield": NoOpShield(cfg),
            "seed0": PROBE_SEED0,
            "control": True,
        },
        # G5: the robust-margin arm on both streams.
        "robust_main": {
            "pursuer": GreedyPursuer,
            "evader": GreedyEvader,
            "shield": CBFShield(cfg, margin=ROBUST_DELTA),
            "seed0": MAIN_SEED0,
            "control": False,
        },
        "robust_probe": {
            "pursuer": GreedyPursuer,
            "evader": ObstacleHuggingEvader,
            "shield": CBFShield(cfg, margin=ROBUST_DELTA),
            "seed0": PROBE_SEED0,
            "control": False,
        },
    }

    print(f"Step 6 gate: N={n} per arm, config {fingerprint}")
    print(
        f"seed streams: main {MAIN_SEED0}..{MAIN_SEED0 + n - 1}, "
        f"probe {PROBE_SEED0}..{PROBE_SEED0 + n - 1}"
    )

    logs, summaries, arm_blocks = {}, {}, {}
    for name, arm in arms.items():
        t0 = time.time()
        logs[name] = run_sweep(
            cfg, arm["pursuer"], arm["evader"], arm["shield"], arm["seed0"], n
        )
        # SEPARATE summarize_shield call per arm: the aggregator refuses a
        # pooled shielded + control sweep, and rightly so (S120).
        summaries[name] = summarize_shield(logs[name])
        s = summaries[name]
        arm_blocks[name] = {
            "pursuer": arm["pursuer"].__name__,
            "evader": arm["evader"].__name__,
            "shield": shield_spec(arm["shield"]),
            "shielded_role": SHIELDED_ROLE,
            "is_unshielded_control": arm["control"],
            "seed0": arm["seed0"],
            "seeds": [arm["seed0"], arm["seed0"] + n - 1],
            "n_episodes": s["n_episodes"],
            "n_steps": s["n_steps"],
            "violation_steps": s["violation_steps"],
            "episodes_with_violation": s["episodes_with_violation"],
            "violation_episode_rate": s["violation_episode_rate"],
            "capture_rate": s["capture_rate"],
            "timeout_rate": s["timeout_rate"],
            "reported_not_gated": reported_not_gated(s, arm["control"]),
            "episodes_starting_outside_barrier_set": sum(
                not log["start_in_barrier_set"] for log in logs[name]
            ),
            "episodes_starting_hard_infeasible": sum(
                not log["start_hard_feasible"] for log in logs[name]
            ),
            "episodes": [episode_row(log) for log in logs[name]],
        }
        print(
            f"  {name:<15} {time.time() - t0:6.1f}s  "
            f"violation steps {s['violation_steps']:>6}  "
            f"eps w/ violation {s['episodes_with_violation']:>4}/{n}  "
            f"CR {s['capture_rate']:.3f}  interv {s['intervention_rate']:.4f}"
        )

    # Every arm that produced a violation is replayed and described BEFORE any
    # verdict is announced, so a kill stops the step with its evidence written
    # rather than with it lost. The criteria themselves are untouched: an arm
    # with a single violation step fails, whatever the census says about it.
    censuses = {
        name: violation_census(name, cfg, arms[name], logs[name])
        for name in arms
        if not arms[name]["control"] and summaries[name]["violation_steps"] > 0
    }

    # --- G1 (kill criterion) ---
    g1 = summaries["shielded_main"]
    g1_upper = violation_upper_bound(g1["episodes_with_violation"], n)
    g1_pass = g1["violation_steps"] == 0

    # --- G2 (capture intact; McNemar primary, descriptive) ---
    ctrl = summaries["control_main"]
    p_mc, b, c = mcnemar_pvalue(logs["shielded_main"], logs["control_main"], "pursuer")
    g2_pass = g2_capture_verdict(g1["capture_rate"], ctrl["capture_rate"])

    # --- G3 (adversarial probe, kill criterion) ---
    g3 = summaries["shielded_probe"]
    g3_pass = g3["violation_steps"] == 0

    # --- G4 (negative control: the probe must bite an unshielded pursuer) ---
    g4 = summaries["control_probe"]
    g4_pass = g4["violation_episode_rate"] >= G4_MIN_VIOLATION_EPISODE_RATE

    # --- G5 (robust-margin arm: G1's and G3's criteria) ---
    r_main, r_probe = summaries["robust_main"], summaries["robust_probe"]
    g5_pass = r_main["violation_steps"] == 0 and r_probe["violation_steps"] == 0

    gates = {
        "G1": {
            "criterion": (
                "shielded GreedyPursuer vs GreedyEvader, N=1000 seeds 800000+ "
                "-> 0 violation steps"
            ),
            "arm": "shielded_main",
            "seeds": [MAIN_SEED0, MAIN_SEED0 + n - 1],
            "violation_steps": g1["violation_steps"],
            "episodes_with_violation": g1["episodes_with_violation"],
            "n_episodes": g1["n_episodes"],
            "n_steps": g1["n_steps"],
            "violation_episode_rate_one_sided_95_upper": g1_upper,
            "wilson_z": ONE_SIDED_Z,
            "pass": g1_pass,
        },
        "G2": {
            "criterion": (
                "shielded capture rate >= 0.5 x unshielded capture rate on the "
                "same paired seeds; paired McNemar reported, no p-threshold"
            ),
            "arms": ["shielded_main", "control_main"],
            "seeds": [MAIN_SEED0, MAIN_SEED0 + n - 1],
            "shielded_capture_rate": g1["capture_rate"],
            "unshielded_capture_rate": ctrl["capture_rate"],
            "required_minimum": G2_CAPTURE_FACTOR * ctrl["capture_rate"],
            "capture_rate_delta": g1["capture_rate"] - ctrl["capture_rate"],
            "mcnemar_p": p_mc,
            "mcnemar_b_shielded_only_captures": b,
            "mcnemar_c_unshielded_only_captures": c,
            "mcnemar_discordant_pairs": b + c,
            "pass": g2_pass,
        },
        "G3": {
            "criterion": (
                "ObstacleHuggingEvader vs shielded GreedyPursuer, N=1000 seeds "
                "810000+ -> 0 violation steps; timeout rate reported"
            ),
            "arm": "shielded_probe",
            "seeds": [PROBE_SEED0, PROBE_SEED0 + n - 1],
            "violation_steps": g3["violation_steps"],
            "episodes_with_violation": g3["episodes_with_violation"],
            "n_episodes": g3["n_episodes"],
            "n_steps": g3["n_steps"],
            "timeout_rate": g3["timeout_rate"],
            "violation_episode_rate_one_sided_95_upper": violation_upper_bound(
                g3["episodes_with_violation"], n
            ),
            "pass": g3_pass,
        },
        "G4": {
            "criterion": (
                "the identical probe WITHOUT the shield -> "
                "episodes-with->=1-violation >= 5%"
            ),
            "arm": "control_probe",
            "seeds": [PROBE_SEED0, PROBE_SEED0 + n - 1],
            "episodes_with_violation": g4["episodes_with_violation"],
            "n_episodes": g4["n_episodes"],
            "violation_episode_rate": g4["violation_episode_rate"],
            "required_minimum": G4_MIN_VIOLATION_EPISODE_RATE,
            "violation_steps": g4["violation_steps"],
            "pass": g4_pass,
        },
        "G5": {
            "criterion": (
                "robust-margin arm (delta = 0.05 m, phase4 §8.7 + "
                "test_localization_accuracy, D87) under G1's and G3's "
                "criteria; intervention rate reported against the unmargined arm"
            ),
            "arms": ["robust_main", "robust_probe"],
            "delta_m": ROBUST_DELTA,
            "delta_provenance": (
                "docs/phases/phase4_sim_to_real_transfer.md: "
                "test_localization_accuracy requires error < 5 cm at the 95th "
                "percentile; §8.7 prescribes adding position uncertainty to "
                "the obstacle radii in the CBF (measurement-robust CBF, D87)"
            ),
            "main": {
                "seeds": [MAIN_SEED0, MAIN_SEED0 + n - 1],
                "violation_steps": r_main["violation_steps"],
                "episodes_with_violation": r_main["episodes_with_violation"],
                "capture_rate": r_main["capture_rate"],
                "intervention_rate": r_main["intervention_rate"],
                "unmargined_intervention_rate": g1["intervention_rate"],
                "intervention_rate_delta": (
                    r_main["intervention_rate"] - g1["intervention_rate"]
                ),
                "min_clearance_floor_m": r_main["min_clearance_floor"],
                "unmargined_min_clearance_floor_m": g1["min_clearance_floor"],
            },
            "probe": {
                "seeds": [PROBE_SEED0, PROBE_SEED0 + n - 1],
                "violation_steps": r_probe["violation_steps"],
                "episodes_with_violation": r_probe["episodes_with_violation"],
                "timeout_rate": r_probe["timeout_rate"],
                "intervention_rate": r_probe["intervention_rate"],
                "unmargined_intervention_rate": g3["intervention_rate"],
                "intervention_rate_delta": (
                    r_probe["intervention_rate"] - g3["intervention_rate"]
                ),
                "min_clearance_floor_m": r_probe["min_clearance_floor"],
                "unmargined_min_clearance_floor_m": g3["min_clearance_floor"],
            },
            "pass": g5_pass,
        },
    }

    payload = {
        "step": 6,
        "n_per_arm": n,
        "config_fingerprint": fingerprint,
        "config": dataclasses.asdict(cfg),
        "seed_streams": {"main": MAIN_SEED0, "probe": PROBE_SEED0},
        "pre_registration": (
            "docs/worklogs/2026-08-24_S119.md, 'Pre-registered gate criteria "
            "(D83 - written before any implementation, 2026-08-25)'; G5 per "
            "D87; reported-not-gated definitions per D89 and the S120 worklog"
        ),
        "v2_spec": "docs/pathway/step6_shield_v2_prereg_2026-09-05.md",
        "gates": gates,
        "arms": arm_blocks,
        "violation_census": censuses,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n--- reported, NOT gated ---")
    for name in arms:
        r = arm_blocks[name]["reported_not_gated"]
        floor = r["min_clearance_floor_m"]
        print(
            f"  {name:<15} infeas {r['infeasibility_rate']:.5f}  "
            f"clearance floor {floor:+.4f} m  tiers {r['tier_histogram']}"
        )
        print(
            f"  {'':<15} stall {r['stall_step_rate']:.5f} "
            f"(max run {r['stall_run_distribution']['max']})  "
            f"immobile {r['immobile_step_rate']:.5f} "
            f"(max run {r['immobile_run_distribution']['max']})  "
            f"pinned eps {r['pinned_episode_rate']:.4f} "
            f"(max backup run {r['backup_run_distribution']['max']})"
            + ("  [control: structurally zero]" if arms[name]["control"] else "")
        )

    print("\n--- verdicts ---")
    print(
        f"G1 {'PASS' if g1_pass else 'FAIL'}: {g1['violation_steps']} violation "
        f"steps in {g1['n_episodes']} episodes ({g1['n_steps']} steps), seeds "
        f"{MAIN_SEED0}..{MAIN_SEED0 + n - 1}; one-sided 95% upper on the "
        f"per-episode violation rate = {g1_upper:.4f}"
    )
    print(
        f"G2 {'PASS' if g2_pass else 'FAIL'}: shielded CR {g1['capture_rate']:.3f} "
        f">= 0.5 x unshielded CR {ctrl['capture_rate']:.3f} = "
        f"{G2_CAPTURE_FACTOR * ctrl['capture_rate']:.3f}; McNemar p={p_mc:.4g} "
        f"(b={b} shielded-only, c={c} unshielded-only), delta CR "
        f"{g1['capture_rate'] - ctrl['capture_rate']:+.3f}"
    )
    print(
        f"G3 {'PASS' if g3_pass else 'FAIL'}: {g3['violation_steps']} violation "
        f"steps in {g3['n_episodes']} probe episodes, seeds "
        f"{PROBE_SEED0}..{PROBE_SEED0 + n - 1}; timeout rate "
        f"{g3['timeout_rate']:.3f}"
    )
    print(
        f"G4 {'PASS' if g4_pass else 'FAIL'}: unshielded probe "
        f"{g4['episodes_with_violation']}/{g4['n_episodes']} = "
        f"{g4['violation_episode_rate']:.3f} episodes with >=1 violation "
        f">= {G4_MIN_VIOLATION_EPISODE_RATE}"
    )
    print(
        f"G5 {'PASS' if g5_pass else 'FAIL'}: robust delta={ROBUST_DELTA} m -> "
        f"{r_main['violation_steps']} main + {r_probe['violation_steps']} probe "
        f"violation steps; intervention rate main "
        f"{r_main['intervention_rate']:.4f} vs unmargined "
        f"{g1['intervention_rate']:.4f}, probe "
        f"{r_probe['intervention_rate']:.4f} vs unmargined "
        f"{g3['intervention_rate']:.4f}"
    )
    print(f"\nwrote {OUT_JSON}")

    for gate_name, arm_name in (
        ("G1", "shielded_main"),
        ("G3", "shielded_probe"),
        ("G5", "robust_main"),
        ("G5", "robust_probe"),
    ):
        if arm_name in censuses:
            print_census(gate_name, censuses[arm_name])

    verdicts = {
        "G1": g1_pass,
        "G2": g2_pass,
        "G3": g3_pass,
        "G4": g4_pass,
        "G5": g5_pass,
    }
    failed = [k for k, v in verdicts.items() if not v]
    if failed:
        print(f"GATE FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
