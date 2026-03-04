"""Staleness-based search reward for pursuer.

Discretizes the arena into grid cells and tracks time since each cell was
last observed by the pursuer. Rewards the pursuer for visiting stale
(not recently observed) cells, incentivizing active search behavior.

Based on radar scan staleness (arXiv:2502.13584) and patrol idleness
(Santana et al., AAMAS 2004).
"""

import numpy as np

from envs.rewards import line_of_sight_blocked


class SearchStalenessTracker:
    """Tracks grid-cell staleness for pursuer search reward."""

    def __init__(
        self,
        arena_width: float,
        arena_height: float,
        grid_size: int = 10,
        t_stale: int = 50,
        sensing_radius: float = 3.0,
    ):
        self.grid_size = grid_size
        self.t_stale = t_stale
        self.sensing_radius = sensing_radius
        self.cell_w = arena_width / grid_size
        self.cell_h = arena_height / grid_size
        self.half_w = arena_width / 2
        self.half_h = arena_height / 2

        # Pre-compute cell centers (grid_size x grid_size x 2)
        cx = -self.half_w + (np.arange(grid_size) + 0.5) * self.cell_w
        cy = -self.half_h + (np.arange(grid_size) + 0.5) * self.cell_h
        xx, yy = np.meshgrid(cx, cy, indexing='ij')
        self.cell_centers = np.stack([xx, yy], axis=-1)  # (G, G, 2)
        self.cell_centers_flat = self.cell_centers.reshape(-1, 2)  # (G*G, 2)

        self.last_observed = None  # initialized in reset()

    def reset(self):
        """Reset all cells to fully stale."""
        self.last_observed = np.full(
            self.grid_size * self.grid_size, -self.t_stale, dtype=np.int32
        )

    def observe_and_reward(
        self,
        pursuer_pos: np.ndarray,
        obstacles: list[dict],
        current_step: int,
    ) -> float:
        """Observe visible cells, return mean staleness of observed cells.

        Returns float in [0, 1]. 0 = all observed cells fresh, 1 = all fully stale.
        Returns 0.0 if no cells observed.
        """
        px, py = pursuer_pos[0], pursuer_pos[1]

        # 1. Distance filter: cells within sensing_radius
        dists = np.sqrt(
            (self.cell_centers_flat[:, 0] - px) ** 2
            + (self.cell_centers_flat[:, 1] - py) ** 2
        )
        in_range_mask = dists <= self.sensing_radius

        if not np.any(in_range_mask):
            return 0.0

        # 2. LOS filter (only if obstacles exist)
        in_range_indices = np.where(in_range_mask)[0]
        visible_indices = []

        if obstacles:
            for idx in in_range_indices:
                cell_pos = self.cell_centers_flat[idx]
                if not line_of_sight_blocked(pursuer_pos, cell_pos, obstacles):
                    visible_indices.append(idx)
        else:
            visible_indices = in_range_indices.tolist()

        if not visible_indices:
            return 0.0

        # 3. Compute staleness of visible cells
        steps_since = current_step - self.last_observed[visible_indices]
        staleness = np.clip(steps_since / self.t_stale, 0.0, 1.0)

        # 4. Update last_observed for visible cells
        self.last_observed[visible_indices] = current_step

        return float(np.mean(staleness))
