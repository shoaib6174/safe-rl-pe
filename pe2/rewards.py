"""Tier 0 (T0.2): the three PROVEN dense training rewards, ported from the
root chassis, plus their non-degeneracy checks.

Why this module exists: every Step-5 arm (D68-D82) trained on the terminal
+/-1 alone, while lessons §3 (D20) says a per-step pursuer visibility
reward is mandatory under partial observability (0.57 vs 0.10 CR) and D24
names search-staleness as the missing search skill. Neither was ever tried
in pe2 (D92). Everything here is TRAINING-ONLY: evaluation always uses the
unshaped SingleAgentWrapper so reported win rates are true outcomes.

Terms (all per-step bounded or potential-based, so terminal outcomes are
unchanged):
- pursuer visibility: +w_vis per step the opponent is visible to the
  learning role (obs[6] == 1: FOV cone AND LOS clear, D6 combined masking).
- search staleness: +w_search * mean staleness of the grid cells the role
  can currently see (SearchStalenessTracker; D24 w=1e-4, D4: >= 3e-4
  teaches patrol-not-hunt).
- evader visibility (root "Mode B"): +w_evis if hidden, -w_evis if visible.
  KNOWN PLATEAU (lessons §5b: flat -1 in open space), so it is only enabled
  together with the PBRS obstacle-seeking potential Phi = -distance to the
  nearest obstacle surface, F = gamma*Phi(s') - Phi(s).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.obs import build_observation
from pe2.sensing import is_visible


class SearchStalenessTracker:
    """Grid-cell staleness for the search reward (root envs/search_staleness.py,
    with observation defined by pe2's FOV+LOS sensing instead of a radius)."""

    def __init__(self, cfg: EnvConfig, grid_size: int = 10, t_stale: int = 50):
        self.cfg = cfg
        self.grid_size = grid_size
        self.t_stale = t_stale
        half = cfg.arena_size / 2.0
        self.cell = cfg.arena_size / grid_size
        centres = -half + (np.arange(grid_size) + 0.5) * self.cell
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        self.centres = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (G*G, 2)
        self.last_observed = np.full(grid_size * grid_size, -t_stale, dtype=np.int64)

    def cell_index(self, x: float, y: float) -> int:
        half = self.cfg.arena_size / 2.0
        ix = int(np.clip((x + half) // self.cell, 0, self.grid_size - 1))
        iy = int(np.clip((y + half) // self.cell, 0, self.grid_size - 1))
        return ix * self.grid_size + iy

    def reset(self) -> None:
        self.last_observed[:] = -self.t_stale

    def observe_and_reward(
        self, own_pose: np.ndarray, obstacles: list[dict], current_step: int
    ) -> float:
        seen = [
            i
            for i, c in enumerate(self.centres)
            if is_visible(
                own_pose, (float(c[0]), float(c[1])), obstacles, self.cfg.sensing
            )
        ]
        if not seen:
            return 0.0
        steps_since = current_step - self.last_observed[seen]
        staleness = np.clip(steps_since / self.t_stale, 0.0, 1.0)
        self.last_observed[seen] = current_step
        return float(staleness.mean())


def _nearest_obstacle_surface(xy: np.ndarray, obstacles: list[dict]) -> float:
    if not obstacles:
        return 0.0
    return float(
        min(np.hypot(xy[0] - o["x"], xy[1] - o["y"]) - o["radius"] for o in obstacles)
    )


def visibility_term(w_vis: float, obs: np.ndarray) -> float:
    """The `vis` dense term for one step: +w_vis iff the observation flags
    the opponent visible.

    `obs[6]` is the combined FOV-cone + clear-LOS flag (D6) that
    `pe2.obs.build_observation` writes, so this one function is the whole
    definition of the term. `DenseRewardWrapper` pays it and
    `visibility_monotone_report` checks it; there is no second copy of the
    geometry that could drift from the one that trains.
    """
    return w_vis * float(obs[6] > 0.5)


class DenseRewardWrapper(gym.Wrapper):
    """Training-only dense reward on top of SingleAgentWrapper (T0.2).

    `step` returns the combined reward (env terminal + dense) as the actual
    reward, since that is what training must optimize, but also reports the
    dense term alone under `info["dense_reward"]` so a non-degeneracy check
    can measure the dense signal in isolation instead of being contaminated
    by the rare terminal +/-1, and the individual terms under
    `info["dense_terms"]` because the plateau check is per term.
    """

    def __init__(
        self,
        env: SingleAgentWrapper,
        *,
        w_vis: float = 0.0,
        w_search: float = 0.0,
        w_evis: float = 0.0,
        w_pbrs: float = 0.0,
        gamma: float = 0.999,
        grid_size: int = 10,
        t_stale: int = 50,
    ):
        super().__init__(env)
        if w_evis > 0 and w_pbrs == 0:
            raise ValueError(
                "evader visibility reward is a known plateau (lessons §5b); "
                "enable it only with the PBRS obstacle potential (w_pbrs > 0)"
            )
        self.w_vis, self.w_search, self.w_evis, self.w_pbrs = (
            w_vis,
            w_search,
            w_evis,
            w_pbrs,
        )
        self.gamma = gamma
        self.tracker = (
            SearchStalenessTracker(env.cfg, grid_size, t_stale) if w_search else None
        )

    # ---- helpers on the learning role's TRUE state (training-only: the
    # policy never sees these; the reward may) ----
    def _own_pose(self) -> np.ndarray:
        e = self.env.env
        return e.pursuer_pose if self.env.role == "pursuer" else e.evader_pose

    def _own_obs(self) -> np.ndarray:
        return self.env._last_joint_obs[self.env.role]

    def _visible(self) -> bool:
        return bool(self._own_obs()[6] > 0.5)

    def _phi(self) -> float:
        return -_nearest_obstacle_surface(self._own_pose()[:2], self.env.env.obstacles)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if self.tracker is not None:
            self.tracker.reset()
        return out

    def step(self, action):
        phi = self._phi() if self.w_pbrs else 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        terms: dict[str, float] = {}
        visible = self._visible()
        if self.w_vis:
            # Through `visibility_term`, never inline: the T0.2 monotone
            # criterion drives this same function, so the criterion cannot
            # pass while the reward that trains pays something else.
            terms["vis"] = visibility_term(self.w_vis, self._own_obs())
        if self.tracker is not None:
            terms["search"] = self.w_search * self.tracker.observe_and_reward(
                self._own_pose(), self.env.env.obstacles, self.env.env.current_step
            )
        if self.w_evis:
            terms["evis"] = self.w_evis * (-1.0 if visible else 1.0)
        if self.w_pbrs:
            terms["pbrs"] = self.w_pbrs * (self.gamma * self._phi() - phi)
        dense = sum(terms.values())
        # The terms are reported SEPARATELY as well as summed: the T0.2
        # plateau check is a per-term question (see nondegeneracy_report),
        # and it cannot be asked of a sum of terms on different scales.
        info["dense_terms"] = terms
        info["dense_reward"] = dense
        return obs, reward + dense, terminated, truncated, info


def signal_nondegeneracy(rewards: list[float]) -> dict:
    """Descriptive statistics of one dense term over a rollout.

    `frac_modal` is the fraction of steps carrying the single most common
    value: the scale-free way to see a flat region in a continuous term,
    where `std > 0` alone is satisfied by a term that is constant on 96% of
    steps. The verdicts live in `nondegeneracy_report`.
    """
    r = np.asarray(rewards, dtype=float)
    if not r.size:
        return {"frac_positive": 0.0, "std": 0.0, "mean": 0.0, "frac_modal": 1.0}
    _, counts = np.unique(r, return_counts=True)
    return {
        "frac_positive": float((r > 0).mean()),
        "std": float(r.std()),
        "mean": float(r.mean()),
        "frac_modal": float(counts.max() / r.size),
    }


# Which plateau check each dense term is owed. The derisk spec (T0.2) states
# a DIFFERENT check per term, because the terms are different shapes:
#
#   vis        "monotone": the spec's own words are "reward-vs-bearing curve
#              must be monotone into the cone" — a property of the reward's
#              GEOMETRY, checked analytically by `visibility_monotone_report`
#              and not estimated from a rollout at all. It replaced a
#              step-wise positive-fraction band (amendment A7, S121): the
#              band was never in the spec, and the term's true positive
#              fraction is 0.0528 (95% CI [0.0493, 0.0563] over 2,500
#              episodes / 2,566,786 steps at five disjoint seeds), sitting ON
#              the 0.05 edge, so no affordable rollout could decide it. The
#              rollout statistics are still recorded for `vis`, as
#              OBSERVATIONS (`stats_role`), and decide nothing.
#   evis       "indicator": the term is on/off (+w / -w). A flat region shows
#              up as the term being on, or off, on nearly every step -> the
#              0.05-0.95 band on frac_positive. UNCHANGED by A7.
#   search,    "continuous": the term takes many values on a bounded range.
#   pbrs       Staleness is NON-NEGATIVE by construction (mean staleness of
#              the observed cells, 0 only when nothing is visible), so its
#              frac_positive is ~1 whatever it does and the band says nothing
#              about it; the spec's check is that it be "non-constant along a
#              random trajectory" -> std > 0 AND no single value on >= 95% of
#              steps. Measured on the shipped P-A configuration over 6211
#              clone steps: frac_positive 0.969, but ~700 distinct values and
#              frac_modal 0.49 — a live signal, not a plateau.
#
# Applying the band to the SUM (frac_positive 0.971 for that configuration)
# is not interpretable: the sum mixes a 0.1-scale indicator with a 1e-4-scale
# continuous term, and its positivity is decided by whichever term is
# non-negative.
TERM_KINDS = {
    "vis": "monotone",
    "evis": "indicator",
    "search": "continuous",
    "pbrs": "continuous",
}
# What each kind's verdict is made of, so a results file names its own
# criterion instead of leaving the reader to infer it from a number.
CRITERIA = {
    "monotone": "monotone_into_cone",
    "indicator": "positive_fraction_band",
    "continuous": "non_constancy",
}
# Kinds whose verdict is analytic: exact, deterministic, no sampling error,
# so resampling a rollout says nothing about them (the runner reports them
# decisive by construction instead of bootstrapping them).
DETERMINISTIC_KINDS = frozenset({"monotone"})
BAND = (0.05, 0.95)
MAX_MODAL_FRACTION = 0.95
# Sweep resolution for the monotone check: 361 bearings over [-pi, pi] is a
# 1 deg grid, on which the default pi/4 half-angle is a grid point exactly.
MONOTONE_BEARINGS = 361


def visibility_monotone_report(
    cfg: EnvConfig,
    *,
    w_vis: float,
    n_bearings: int = MONOTONE_BEARINGS,
    range_m: float | None = None,
    term_fn=visibility_term,
) -> dict:
    """T0.2's criterion for the pursuer visibility term: "reward-vs-bearing
    curve must be monotone into the cone".

    Hold the opponent at a fixed range inside sensing range in an
    obstacle-free world and sweep its bearing relative to the agent's
    heading. The term must be

      1. non-decreasing as the bearing MAGNITUDE falls (turning toward the
         opponent can only help — a reward that fell here would be one the
         agent is punished for following),
      2. not constant (it must actually carry information), and
      3. stepping at the configured half-angle `fov_angle / 2`, so the
         reward and the sensing configuration agree.

    It is a step, not a slope — that is expected of an indicator reward; the
    property that matters is that the step points the right way and sits
    where the configuration says. The check is analytic and deterministic:
    same configuration, same verdict, with no sampling error to bootstrap.

    `term_fn` exists so a negative control can be swept through the identical
    machinery; the default IS the term `DenseRewardWrapper` pays.
    """
    half_angle = cfg.sensing.fov_angle / 2.0
    r = cfg.sensing.fov_range / 2.0 if range_m is None else float(range_m)
    if not 0.0 < r < cfg.sensing.fov_range:
        raise ValueError(
            f"sweep range {r} must lie strictly inside the sensing range "
            f"{cfg.sensing.fov_range}: outside it the range gate, not the "
            "cone, decides every sample and the sweep says nothing"
        )
    own_pose = np.zeros(3)  # origin, heading +x -> bearing IS relative bearing
    bearings = np.linspace(-np.pi, np.pi, n_bearings)
    rewards = np.array(
        [
            term_fn(
                w_vis,
                build_observation(
                    own_pose,
                    np.array([r * np.cos(b), r * np.sin(b), 0.0]),
                    [],  # no obstacles: LOS is clear at every bearing
                    cfg,
                ),
            )
            for b in bearings
        ],
        dtype=float,
    )
    # "Into the cone" = the bearing magnitude falling. Each side of the
    # heading is its own sweep, so a term that misbehaves on only one side
    # is still caught.
    left = rewards[bearings <= 0.0]  # -pi -> 0: magnitude falling
    right = rewards[bearings >= 0.0][::-1]  # +pi -> 0: magnitude falling
    monotone = bool(np.all(np.diff(left) >= 0.0) and np.all(np.diff(right) >= 0.0))
    grid_step = float(bearings[1] - bearings[0])
    positive = np.flatnonzero(rewards > 0.0)
    transition = float(np.abs(bearings[positive]).max()) if positive.size else None
    # One grid step: the sweep can only locate the edge to its own resolution.
    matches = transition is not None and abs(transition - half_angle) <= grid_step
    varies = bool(rewards.max() > rewards.min())
    return {
        "criterion": CRITERIA["monotone"],
        "bearings": [float(b) for b in bearings],
        "rewards": [float(v) for v in rewards],
        "monotone_into_cone": monotone,
        "varies": varies,
        "transition_bearing": transition,
        "expected_transition_bearing": float(half_angle),
        "transition_matches": bool(matches),
        "transition_tolerance": grid_step,
        "grid_step": grid_step,
        "n_bearings": int(n_bearings),
        "range_m": r,
        "fov_angle": float(cfg.sensing.fov_angle),
        "fov_range": float(cfg.sensing.fov_range),
        "obstacles": 0,
        "w_vis": float(w_vis),
        "passes": bool(monotone and varies and matches),
    }


def nondegeneracy_report(
    terms: dict[str, list[float]],
    cfg: EnvConfig | None = None,
    w_vis: float | None = None,
) -> dict:
    """T0.2 plateau check, applied PER TERM (see TERM_KINDS).

    `terms` maps a term name to its per-step values over a rollout, as
    `DenseRewardWrapper` reports them in `info["dense_terms"]`. Returns
    `{"terms": {name: {...stats, "kind", "criterion", "stats_role",
    "passes"}}, "all_pass": bool}`. A term with no declared kind raises
    rather than being checked by a criterion that may not apply to it.

    A term of the "monotone" kind is decided by `visibility_monotone_report`
    on `cfg` — a property of the reward, not of this rollout — so `cfg` and
    `w_vis` are required whenever such a term is present, and its rollout
    statistics are carried with `stats_role="observation"` to say plainly
    that they are recorded and not applied.
    """
    out = {}
    for name, values in terms.items():
        if name not in TERM_KINDS:
            raise ValueError(
                f"dense term {name!r} has no declared kind; add it to "
                "TERM_KINDS so the right plateau check is applied to it"
            )
        kind = TERM_KINDS[name]
        entry = {
            **signal_nondegeneracy(values),
            "kind": kind,
            "criterion": CRITERIA[kind],
            "stats_role": "observation" if kind == "monotone" else "criterion",
        }
        if kind == "monotone":
            if cfg is None or w_vis is None:
                raise ValueError(
                    f"dense term {name!r} is checked by the analytic "
                    f"{CRITERIA[kind]} criterion, which needs `cfg` and "
                    "`w_vis`; its rollout statistics are recorded as "
                    "observations and decide nothing (amendment A7)"
                )
            geometry = visibility_monotone_report(cfg, w_vis=w_vis)
            entry |= {"monotone": geometry, "passes": geometry["passes"]}
        elif kind == "indicator":
            entry["passes"] = bool(
                BAND[0] < entry["frac_positive"] < BAND[1] and entry["std"] > 0.0
            )
        else:
            entry["passes"] = bool(
                entry["std"] > 0.0 and entry["frac_modal"] < MAX_MODAL_FRACTION
            )
        out[name] = entry
    return {"terms": out, "all_pass": all(t["passes"] for t in out.values())}
