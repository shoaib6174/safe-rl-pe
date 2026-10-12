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


# ---------------------------------------------------------------------------
# S24.0 Step 2 — the occlusion-aware particle filter
# ---------------------------------------------------------------------------
# Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1 ("Memory") and the
# S20 design in `docs/pathway/agent_intelligence_plan_2026-10-05.md` section 3
# (written there, never built, because S19's gate withheld it).
#
# What makes it occlusion-aware is the negative information: when the opponent
# is NOT visible, every particle inside the observer's CURRENT visibility
# region (FOV cone AND clear LOS, `pe2.sensing.is_visible`) is killed — "I can
# see there and it is not there". Dead reckoning above cannot use that
# information at all; it only ever propagates the last sighting.
#
# Binding rule (plan section 1, speed-agnostic): the opponent's speed bound is
# a CONSTRUCTOR ARGUMENT. This class never reads `cfg.evader_speed_ratio` or
# `cfg.evader_v_max` — the caller supplies the bound it believes in (the
# pursuer may legitimately know the opponent's speed bound; it may not read the
# config field, and `tests/pe2/test_particle_belief.py` asserts that on the
# source).
#
# Readings taken where the brief left a choice (each stated, none tuned):
# - The predict step is a CORRELATED random walk: every particle carries a
#   heading, perturbed by N(0, omega_max * dt) each step, and moves
#   `U(0, 1) * v_max_opp * dt` along it — so one step's displacement is bounded
#   by the speed bound (the brief's requirement) while the SUPPORT expands
#   LINEARLY in time, i.e. it covers the reachable set the S20 design means by
#   "survivors propagate with the evader's known speed bound".
#   This replaces an i.i.d.-uniform-in-a-disc displacement, which was built
#   first and is mis-specified: i.i.d. displacements diffuse as sqrt(t), so the
#   cloud measured 0.87 m wide 400 occluded steps after a sighting while the
#   opponent's reachable radius was 24 m. A support that cannot contain the
#   truth makes both the mean and the visibility kill meaningless, whichever
#   median wins. The defect was diagnosed on the dispersion above (a property
#   of the model, not of any gate number) and fixed before the gate ran; the
#   gate itself ran once, on this model.
#   `omega_max` is the arena's own symmetric turn-rate limit (`EnvConfig`), not
#   the opponent's speed — the speed-agnostic rule is about the speed bound,
#   which stays a constructor argument.
# - An invalid predict proposal (inside an inflated obstacle or outside the
#   inset arena) is RESAMPLED with a FRESH uniform heading, up to
#   `MAX_PROPOSALS` attempts per particle; a particle whose attempts all fail
#   STAYS where it was. Not projected to the boundary: projection out of an
#   inflated disc can land farther than `v_max_opp * dt` from the particle's
#   previous position (up to twice it), which would break the speed bound this
#   filter is supposed to respect. A fresh heading rather than a fresh
#   displacement because a target that runs into a wall turns.
# - "Survivors are renormalised" is implemented as the bootstrap filter's
#   standard step: renormalise, and systematically RESAMPLE to `n` equally
#   weighted particles when the effective sample size falls below `n / 2`.
#   Without it the binary visibility kill monotonically shrinks the support
#   (killed particles can never recover weight) and the cloud dies out.
# - The visibility test is VECTORISED here rather than looping over
#   `is_visible` (500 particles x 1200 steps x 400 episodes of Python calls is
#   not affordable). It is asserted EXACTLY equal to `is_visible`, pose by pose
#   and point by point, in the tests — including the tangency convention.
# - Inflation and the inset arena follow `pe2.planner` / `pe2.env.pursuer_step`
#   exactly: obstacles by `radius + agent_radius` (the env's projection
#   distance), the arena by `agent_radius`. So the valid set is exactly the set
#   of positions the env can leave an agent in, and the true opponent position
#   is always inside it.

MAX_PROPOSALS = 8  # predict: resample attempts before a particle stays put
UNIFORM_TRIES = 32  # reset / fallback: rejection-sampling rounds
MODE_CELL = 0.5  # m — grid cell for the strongest-mode (densest-cell) readout


class ParticleBelief:
    """Occlusion-aware particle filter over the opponent's position.

    `update` consumes only what one agent can know: its own pose, the
    visibility flag, the observed opponent position when visible, and the
    static map. The TRUE opponent position never enters (the gate script
    reads it for the error metric only, outside this class).
    """

    def __init__(
        self, cfg: EnvConfig, v_max_opp: float, n: int = 500, seed: int = 0
    ) -> None:
        self.cfg = cfg
        self.v_max_opp = float(v_max_opp)
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self._lim = cfg.arena_size / 2.0 - cfg.agent_radius
        self.reset()

    # -- static map ------------------------------------------------------
    def reset(self, env=None) -> None:
        """Read the static map (known, as everywhere in pe2) and spread the
        cloud uniformly over the free space."""
        self.obstacles: list[dict] = list(env.obstacles) if env is not None else []
        self._centres = np.array(
            [[o["x"], o["y"]] for o in self.obstacles], dtype=float
        ).reshape(-1, 2)
        self._radii = np.array([o["radius"] for o in self.obstacles], dtype=float)
        self._inflated = self._radii + self.cfg.agent_radius
        self.particles = self._uniform_free(self.n)
        self.headings = self._uniform_headings(self.n)
        self.weights = np.full(self.n, 1.0 / self.n)
        self.ever_seen = False
        self.steps_since_seen = 0
        self._own_pose = np.zeros(3)
        self._visible_mass = 0.0

    # -- geometry (vectorised; asserted equal to pe2.sensing in the tests) --
    def _free(self, pts: np.ndarray) -> np.ndarray:
        """Inside the inset arena and outside every inflated obstacle."""
        ok = (np.abs(pts[:, 0]) <= self._lim) & (np.abs(pts[:, 1]) <= self._lim)
        if len(self._inflated):
            d = np.linalg.norm(pts[:, None, :] - self._centres[None, :, :], axis=2)
            ok &= np.all(d >= self._inflated[None, :], axis=1)
        return ok

    def visible_mask(self, own_pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """`is_visible(own_pose, p, self.obstacles, cfg.sensing)` for every row
        of `pts`, vectorised. Same tangency convention (strict inequality)."""
        s = self.cfg.sensing
        d = pts - np.asarray(own_pose[:2], dtype=float)
        rng_ok = np.hypot(d[:, 0], d[:, 1]) <= s.fov_range
        rel = (np.arctan2(d[:, 1], d[:, 0]) - own_pose[2] + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        mask = rng_ok & (np.abs(rel) <= s.fov_angle / 2.0)
        if not s.los_occlusion or not len(self._radii):
            return mask
        # segment own->p vs each obstacle disc: closest point on the segment,
        # clamped to [0, 1]; blocked iff strictly inside the disc.
        f = np.asarray(own_pose[:2], dtype=float)[None, :] - self._centres  # (m,2)
        seg_sq = np.maximum((d * d).sum(axis=1), 1e-18)  # (k,)
        fdotd = f[None, :, 0] * d[:, None, 0] + f[None, :, 1] * d[:, None, 1]
        t = np.clip(-fdotd / seg_sq[:, None], 0.0, 1.0)  # (k,m)
        cx = f[None, :, 0] + t * d[:, None, 0]
        cy = f[None, :, 1] + t * d[:, None, 1]
        blocked = (cx * cx + cy * cy) < (self._radii[None, :] ** 2)
        return mask & ~np.any(blocked, axis=1)

    def _uniform_free(self, k: int, own_pose: np.ndarray | None = None) -> np.ndarray:
        """`k` points uniform over the free space; when `own_pose` is given,
        over the free space that is NOT currently visible from it."""
        out = np.empty((k, 2))
        filled = 0
        for _ in range(UNIFORM_TRIES):
            cand = self.rng.uniform(-self._lim, self._lim, size=(k, 2))
            ok = self._free(cand)
            if own_pose is not None:
                ok &= ~self.visible_mask(own_pose, cand)
            take = cand[ok][: k - filled]
            out[filled : filled + len(take)] = take
            filled += len(take)
            if filled == k:
                return out
        if own_pose is not None and filled < k:
            # Hidden free space too small to fill by rejection: fall back to the
            # whole free space (stated fallback; the cloud is wrong rather than
            # empty, and `visible_mass` in the summary says so).
            out[filled:] = self._uniform_free(k - filled)
            return out
        out[filled:] = 0.0  # degenerate map only (no free space at all)
        return out

    # -- filter ----------------------------------------------------------
    def _uniform_headings(self, k: int) -> np.ndarray:
        return self.rng.uniform(-np.pi, np.pi, size=k)

    def _propose(self, pts: np.ndarray, ang: np.ndarray) -> np.ndarray:
        """Move along `ang` by `U(0, 1) * v_max_opp * dt` — never farther."""
        rad = self.v_max_opp * self.cfg.dt * self.rng.uniform(0.0, 1.0, size=len(ang))
        return pts + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    def _predict(self) -> None:
        pts = self.particles
        ang = self.headings + self.rng.normal(
            0.0, self.cfg.omega_max * self.cfg.dt, size=len(pts)
        )
        new = pts.copy()
        pending = np.ones(len(pts), dtype=bool)
        for attempt in range(MAX_PROPOSALS):
            k = int(pending.sum())
            if k == 0:
                break
            if attempt:  # a blocked particle turns: fresh uniform heading
                ang[pending] = self._uniform_headings(k)
            cand = self._propose(pts[pending], ang[pending])
            ok = self._free(cand)
            idx = np.flatnonzero(pending)[ok]
            new[idx] = cand[ok]
            pending[idx] = False
        self.particles = new  # particles still pending stayed where they were
        self.headings = ang

    def _resample(self) -> None:
        """Systematic resampling to `n` equally weighted particles."""
        edges = np.cumsum(self.weights)
        u = (self.rng.uniform() + np.arange(self.n)) / self.n
        idx = np.searchsorted(edges, u * edges[-1])
        self.particles = self.particles[idx]
        self.headings = self.headings[idx]
        self.weights = np.full(self.n, 1.0 / self.n)

    def update(
        self,
        own_pose: np.ndarray,
        obs_visible: bool,
        opp_world_xy_if_visible,
        obstacles: list[dict] | None = None,
    ) -> None:
        """One filter step: predict, then the occlusion-aware correction."""
        if obstacles is not None and obstacles is not self.obstacles:
            self.obstacles = list(obstacles)
            self._centres = np.array(
                [[o["x"], o["y"]] for o in self.obstacles], dtype=float
            ).reshape(-1, 2)
            self._radii = np.array([o["radius"] for o in self.obstacles], dtype=float)
            self._inflated = self._radii + self.cfg.agent_radius
        own_pose = np.asarray(own_pose, dtype=float)
        self._own_pose = own_pose
        self._predict()
        if obs_visible:
            # A sighting dominates everything: resample the whole cloud at the
            # observation (noiseless in sim) with the SIGMA_OBS floor, rejecting
            # proposals outside the free space.
            pos = np.asarray(opp_world_xy_if_visible, dtype=float)[:2]
            cand = pos[None, :] + self.rng.normal(0.0, SIGMA_OBS, size=(self.n, 2))
            bad = ~self._free(cand)
            cand[bad] = pos  # the observation itself is always free
            self.particles = cand
            # direction of travel is unknown at a fresh sighting
            self.headings = self._uniform_headings(self.n)
            self.weights = np.full(self.n, 1.0 / self.n)
            self.ever_seen = True
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
            seen_region = self.visible_mask(own_pose, self.particles)
            w = np.where(seen_region, 0.0, self.weights)
            total = float(w.sum())
            if total <= 0.0:
                # Every particle was inside the visibility region: the cloud was
                # wrong. Fallback (stated): resample uniformly over the free
                # space that is NOT currently visible.
                self.particles = self._uniform_free(self.n, own_pose=own_pose)
                self.headings = self._uniform_headings(self.n)
                self.weights = np.full(self.n, 1.0 / self.n)
            else:
                self.weights = w / total
                ess = 1.0 / float((self.weights**2).sum())
                if ess < self.n / 2.0:
                    self._resample()
        self._visible_mass = float(
            self.weights[self.visible_mask(own_pose, self.particles)].sum()
        )

    # -- readout ---------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.particles

    def _mode(self) -> np.ndarray:
        """Strongest mode: the densest `MODE_CELL`-metre grid cell, reported as
        the weighted mean of the particles in it."""
        cell = np.floor(self.particles / MODE_CELL).astype(np.int64)
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        mass = np.bincount(inv, weights=self.weights, minlength=len(keys))
        sel = inv == int(np.argmax(mass))
        w = self.weights[sel]
        return (w @ self.particles[sel]) / max(float(w.sum()), 1e-12)

    def summary(self) -> dict:
        mean = self.mean
        d = self.particles - mean[None, :]
        return {
            "mean": mean,
            "spread": float(np.sqrt(self.weights @ (d * d).sum(axis=1))),
            "mode": self._mode(),
            "visible_mass": self._visible_mass,
            "steps_since_seen": int(self.steps_since_seen),
            "ever_seen": bool(self.ever_seen),
        }
