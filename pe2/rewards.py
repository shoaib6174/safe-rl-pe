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

    def _visible(self) -> bool:
        return bool(self.env._last_joint_obs[self.env.role][6] > 0.5)

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
            terms["vis"] = self.w_vis * float(visible)
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
#   vis, evis  "indicator": the term is on/off (+w / 0, or +w / -w). A flat
#              region shows up as the term being on, or off, on nearly every
#              step -> the 0.05-0.95 band on frac_positive.
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
    "vis": "indicator",
    "evis": "indicator",
    "search": "continuous",
    "pbrs": "continuous",
}
BAND = (0.05, 0.95)
MAX_MODAL_FRACTION = 0.95


def nondegeneracy_report(terms: dict[str, list[float]]) -> dict:
    """T0.2 plateau check, applied PER TERM (see TERM_KINDS).

    `terms` maps a term name to its per-step values over a rollout, as
    `DenseRewardWrapper` reports them in `info["dense_terms"]`. Returns
    `{"terms": {name: {...stats, "kind", "passes"}}, "all_pass": bool}`.
    A term with no declared kind raises rather than being checked by a
    criterion that may not apply to it.
    """
    out = {}
    for name, values in terms.items():
        if name not in TERM_KINDS:
            raise ValueError(
                f"dense term {name!r} has no declared kind; add it to "
                "TERM_KINDS so the right plateau check is applied to it"
            )
        kind = TERM_KINDS[name]
        stats = signal_nondegeneracy(values)
        if kind == "indicator":
            passes = BAND[0] < stats["frac_positive"] < BAND[1] and stats["std"] > 0.0
        else:
            passes = stats["std"] > 0.0 and stats["frac_modal"] < MAX_MODAL_FRACTION
        out[name] = {**stats, "kind": kind, "passes": bool(passes)}
    return {"terms": out, "all_pass": all(t["passes"] for t in out.values())}
