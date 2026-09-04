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
    by the rare terminal +/-1.
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
        dense = 0.0
        visible = self._visible()
        if self.w_vis:
            dense += self.w_vis * float(visible)
        if self.tracker is not None:
            dense += self.w_search * self.tracker.observe_and_reward(
                self._own_pose(), self.env.env.obstacles, self.env.env.current_step
            )
        if self.w_evis:
            dense += self.w_evis * (-1.0 if visible else 1.0)
        if self.w_pbrs:
            dense += self.w_pbrs * (self.gamma * self._phi() - phi)
        info["dense_reward"] = dense
        return obs, reward + dense, terminated, truncated, info


def signal_nondegeneracy(rewards: list[float]) -> dict:
    """T0.2 plateau check: a dense signal must vary (std > 0) and be neither
    always-on nor never-on (0.05 < frac_positive < 0.95) over a rollout."""
    r = np.asarray(rewards, dtype=float)
    return {
        "frac_positive": float((r > 0).mean()) if r.size else 0.0,
        "std": float(r.std()) if r.size else 0.0,
        "mean": float(r.mean()) if r.size else 0.0,
    }
