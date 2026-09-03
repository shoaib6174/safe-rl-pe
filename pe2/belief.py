"""Step 4: belief tracking — last-seen baseline with dead reckoning.

Contract: the tracker consumes ONLY the 11-D observation vector. It never
reads env state, so belief cannot leak truth during occlusion (enforced by
the API — `update(obs)` takes nothing else — and tested in test_belief.py).

Baseline design (plan Step 4 decision point: BiMDN deferred if this
suffices):
- VISIBLE: mean = observed opponent position (observations are noiseless in
  sim, so in-FOV error ~ 0); velocity = low-passed finite difference between
  consecutive visible steps; sigma = SIGMA_OBS floor.
- OCCLUDED after having been seen ("just lost"): dead reckoning — mean
  propagates along the velocity estimate, velocity decays (the target may
  have turned), sigma grows linearly with time-since-seen.
- NEVER SEEN: `seen` stays False; mean/velocity are meaningless, sigma=inf.
  L4: never-seen targets are unpredictable in principle — metrics split
  "just lost" vs "never seen" (pe2_step04_belief_metrics.py).
"""

from __future__ import annotations

import numpy as np

from pe2.config import EnvConfig
from pe2.obs import opponent_world_from_obs

SIGMA_OBS = 0.05  # m — noiseless-observation floor
SIGMA_GROWTH = 0.5  # m/s — uncertainty growth while occluded
VELOCITY_DECAY = 0.966  # per step — halves in ~1s at dt=0.05
VEL_LOWPASS = 0.5  # weight on the new finite-difference velocity


class LastSeenBelief:
    """Tracks the opponent from one agent's observation stream."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.seen = False
        self.mean = np.zeros(2)
        self.velocity = np.zeros(2)
        self.sigma = float("inf")
        self.t_since_seen: int | None = None
        self._prev_visible = False

    def update(self, obs: np.ndarray) -> None:
        visible = bool(obs[6] > 0.5)
        if visible:
            pos = opponent_world_from_obs(obs, self.cfg)
            if self.seen and self._prev_visible:
                inst = (pos - self.mean) / self.cfg.dt
                self.velocity = (1 - VEL_LOWPASS) * self.velocity + VEL_LOWPASS * inst
            # Re-acquisition after a gap: snap position to the observation,
            # keep the (decayed) prior velocity — the gap finite-difference
            # would be meaningless.
            self.mean = pos
            self.sigma = SIGMA_OBS
            self.t_since_seen = 0
            self.seen = True
        elif self.seen:
            self.t_since_seen += 1
            self.mean = self.mean + self.velocity * self.cfg.dt
            self.velocity = self.velocity * VELOCITY_DECAY
            self.sigma = SIGMA_OBS + SIGMA_GROWTH * self.t_since_seen * self.cfg.dt
        # never-seen: leave everything at reset values (sigma inf)
        self._prev_visible = visible

    @property
    def just_lost(self) -> bool:
        """Seen before but not visible this step (L4 metrics split)."""
        return self.seen and not self._prev_visible
