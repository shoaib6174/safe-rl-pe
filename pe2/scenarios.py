"""Scenario-stratified evaluation inits (D70).

The D57 random paired battery answers "how well does the agent do overall";
this module answers "WHERE does it fail" — a fixed set of canonical start
geometries (scenarios), each with seeded jittered variants, so performance
is reported per stratum: open-space pursuit at three ranges, occluded
start, shadow-adjacent (evader one move from breaking LOS), corner trap.

Design:
- CANONICAL OBSTACLE LAYOUT (fixed LAYOUT_SEED) for the whole suite, so
  scenario geometry is comparable across models and runs. Complements the
  random-seed battery, never replaces it.
- Variants: per-variant jitter via derive_seed(root, f"{name}/{i}") (D66,
  insertion-stable), VALIDATED against the scenario predicate (visibility
  flag, distance band, not overlapping an obstacle). A jitter draw that
  breaks the qualitative geometry is resampled — deterministically.
- Evaluation reuses evaluate_sb3_policy via its init_poses parameter, so
  scenario logs carry the same fields (and los_breaks) as everything else.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.rng import derive_seed

LAYOUT_SEED = 730000  # canonical obstacle layout for the scenario suite
SCENARIO_SEED_ROOT = 740000  # root for per-variant jitter streams

_POS_JITTER = 0.4  # m, per-axis
_THETA_JITTER = 0.3  # rad


@dataclass(frozen=True)
class Scenario:
    """A canonical start geometry on the canonical obstacle layout."""

    name: str
    pursuer_pose: tuple[float, float, float]
    evader_pose: tuple[float, float, float]
    expect_visible: bool  # pursuer-view los_visible at episode start
    d_band: tuple[float, float]  # admissible start distance after jitter


def _heading_to(x: float, y: float, tx: float, ty: float) -> float:
    return math.atan2(ty - y, tx - x)


def default_scenarios(cfg: EnvConfig) -> list[Scenario]:
    """The six canonical scenarios, computed from the canonical layout."""
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=LAYOUT_SEED)
    ox, oy, or_ = (
        env.obstacles[0]["x"],
        env.obstacles[0]["y"],
        env.obstacles[0]["radius"],
    )

    # line through obstacle 0: pursuer side (+u) / evader side (-u)
    u = np.array([1.0, 0.0]) if ox < 0 else np.array([-1.0, 0.0])

    def obs_line(d_p: float, d_e: float) -> tuple[tuple, tuple]:
        p = np.array([ox, oy]) + u * d_p
        e = np.array([ox, oy]) - u * d_e
        return (
            (p[0], p[1], _heading_to(*p, *e)),
            (e[0], e[1], _heading_to(*e, *(e - u))),
        )

    p_occ, e_occ = obs_line(or_ + 2.5, or_ + 1.0)  # obstacle blocks LOS
    _, e_sh = obs_line(or_ + 2.5, or_ + 0.6)  # evader at the shadow mouth
    # shadow-adjacent: pursuer anchored NW of obstacle 0 so LOS clears the
    # disk (segment-to-disk 1.07 > r=0.81 on the canonical layout) while the
    # evader sits 0.6m from the shadow mouth
    p_sh = (-3.0, 0.5, _heading_to(-3.0, 0.5, *e_sh[:2]))
    e_sh = (e_sh[0], e_sh[1], 0.0)

    return [
        Scenario(
            "close_open",
            (-3.5, 2.5, 0.0),
            (-2.0, 2.5, 0.0),
            True,
            (1.0, 2.2),
        ),
        Scenario(
            "mid_open",
            (-3.0, 1.5, 0.0),
            (0.5, 1.5, 0.0),
            True,
            (2.8, 4.4),
        ),
        Scenario(
            "far_open",
            (-3.5, 3.5, 0.0),
            (3.0, 3.5, 0.0),
            True,
            (5.5, 7.5),
        ),
        Scenario(
            "occluded_start",
            p_occ,
            e_occ,
            False,
            (or_ + 2.2, or_ + 4.5),
        ),
        Scenario(
            "shadow_adjacent",
            p_sh,
            e_sh,
            True,
            (1.5, 4.5),
        ),
        Scenario(
            "corner_trap",
            (1.5, 1.5, _heading_to(1.5, 1.5, 3.8, 3.8)),
            (3.8, 3.8, _heading_to(3.8, 3.8, 4.5, 4.5)),
            True,
            (2.4, 4.2),
        ),
    ]


def _valid(env: PursuitEvasionEnv, scn: Scenario) -> bool:
    """Predicate: visibility flag, distance band, outside all obstacles."""
    obs = env._get_obs()["pursuer"]
    if (obs[6] > 0.5) != scn.expect_visible:
        return False
    d = float(np.hypot(*(env.pursuer_pose[:2] - env.evader_pose[:2])))
    if not (scn.d_band[0] <= d <= scn.d_band[1]):
        return False
    cfg = env.cfg
    for o in env.obstacles:
        for pose in (env.pursuer_pose, env.evader_pose):
            if np.hypot(pose[0] - o["x"], pose[1] - o["y"]) < (
                o["radius"] + cfg.agent_radius + 0.05
            ):
                return False
    return True


def scenario_inits(
    scn: Scenario, n: int, cfg: EnvConfig, seed_root: int = SCENARIO_SEED_ROOT
) -> list[tuple[tuple, tuple]]:
    """N validated jittered variants of a scenario (deterministic)."""
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=LAYOUT_SEED)
    inits = []
    for i in range(n):
        rng = np.random.default_rng(derive_seed(seed_root, f"{scn.name}/{i}"))
        for _ in range(100):
            jittered = []
            for pose in (scn.pursuer_pose, scn.evader_pose):
                dx, dy = rng.uniform(-_POS_JITTER, _POS_JITTER, 2)
                dth = rng.uniform(-_THETA_JITTER, _THETA_JITTER)
                jittered.append((pose[0] + dx, pose[1] + dy, pose[2] + dth))
            env.pursuer_pose = np.array(jittered[0])
            env.evader_pose = np.array(jittered[1])
            if _valid(env, scn):
                inits.append((jittered[0], jittered[1]))
                break
        else:
            raise RuntimeError(
                f"scenario {scn.name}: no valid jitter in 100 draws (variant {i})"
            )
    return inits


def evaluate_scenarios(
    model,
    cfg: EnvConfig,
    role: str,
    opponent_factory: Callable[[], object],
    scenarios: list[Scenario],
    n_variants: int,
    seed_root: int = SCENARIO_SEED_ROOT,
) -> dict[str, list[dict]]:
    """Per-scenario eval logs (same fields as evaluate_sb3_policy + name).

    Every episode resets to the canonical layout (seed=LAYOUT_SEED) with the
    scenario's validated init poses, so episodes differ only by jitter.
    """
    from pe2.exploitability import evaluate_sb3_policy

    results = {}
    for scn in scenarios:
        inits = scenario_inits(scn, n_variants, cfg, seed_root)
        logs = evaluate_sb3_policy(
            model,
            cfg,
            role,
            opponent_factory,
            n_episodes=n_variants,
            seed0=LAYOUT_SEED,
            init_poses=inits,
        )
        for log in logs:
            log["scenario"] = scn.name
        results[scn.name] = logs
    return results
