"""Particle filter belief tracker for opponent position in pursuit-evasion.

Maintains a distribution over the opponent's (x, y) position using a
simplified particle filter. Outputs geometric statistics (mean, std, entropy,
time-since-seen) as a 6-D belief vector for the policy network.
"""

from __future__ import annotations

import numpy as np


class BeliefTracker:
    """Track opponent position belief via particle filter.

    Args:
        role: 'pursuer' or 'evader' — which agent this tracker serves.
        arena_width: Arena width in meters.
        arena_height: Arena height in meters.
        opponent_v_max: Max speed of the opponent being tracked (m/s).
        n_particles: Number of particles (default 100).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        role: str,
        arena_width: float,
        arena_height: float,
        opponent_v_max: float,
        n_particles: int = 100,
        seed: int = 0,
    ):
        self.role = role
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.opponent_v_max = opponent_v_max
        self.n_particles = n_particles
        self.dt = 0.05  # Simulation timestep (matches PursuitEvasionEnv)

        self.rng = np.random.default_rng(seed)

        # Particle state
        self.particles = None
        self.weights = None
        self.steps_since_seen = 0
        self._ever_updated = False

        self.reset()

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        # Initialize particles uniformly over arena
        self.particles = np.stack([
            self.rng.uniform(0.0, self.arena_width, size=self.n_particles),
            self.rng.uniform(0.0, self.arena_height, size=self.n_particles),
        ], axis=1).astype(np.float32)
        self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles
        self.steps_since_seen = 0
        self._ever_updated = False

    def update(
        self,
        own_state: np.ndarray,
        opponent_state: np.ndarray,
        is_visible: bool,
    ) -> None:
        """Update belief given current visibility.

        Args:
            own_state: [x, y, theta] of the tracking agent.
            opponent_state: [x, y, theta] of the opponent.
            is_visible: Whether opponent is currently visible.
        """
        self._ever_updated = True

        if is_visible:
            # Resample all particles around observed position
            true_pos = opponent_state[:2]
            self.particles = self.rng.normal(
                loc=true_pos,
                scale=0.2,
                size=(self.n_particles, 2),
            ).astype(np.float32)
            self._clamp_particles()
            self.steps_since_seen = 0
        else:
            # Propagate particles with diffusion
            self.steps_since_seen += 1
            sigma = self.opponent_v_max * self.dt * np.sqrt(self.steps_since_seen)
            noise = self.rng.normal(loc=0.0, scale=sigma, size=(self.n_particles, 2))
            self.particles += noise.astype(np.float32)
            self._clamp_particles()

    def get_statistics(self) -> np.ndarray:
        """Return 6-D belief statistics vector.

        Returns:
            [mean_x, mean_y, std_x, std_y, entropy, time_since_seen]
        """
        if not self._ever_updated:
            # Safe default before first update
            mean_x = self.arena_width / 2.0
            mean_y = self.arena_height / 2.0
            std_x = np.sqrt(self.arena_width**2 + self.arena_height**2) / 2.0
            std_y = std_x
            entropy = np.log(std_x * std_y + 1e-6)
            time_since_seen = 0.0
            return np.array([mean_x, mean_y, std_x, std_y, entropy, time_since_seen], dtype=np.float32)

        mean_x = float(np.mean(self.particles[:, 0]))
        mean_y = float(np.mean(self.particles[:, 1]))
        std_x = float(np.std(self.particles[:, 0]))
        std_y = float(np.std(self.particles[:, 1]))

        # Entropy: log(det(cov) + eps)
        # For independent axes: det(cov) = std_x^2 * std_y^2
        det_cov = (std_x ** 2) * (std_y ** 2)
        entropy = float(np.log(det_cov + 1e-6))

        time_since_seen = float(self.steps_since_seen * self.dt)

        return np.array([mean_x, mean_y, std_x, std_y, entropy, time_since_seen], dtype=np.float32)

    def _clamp_particles(self) -> None:
        """Clamp all particles to arena bounds."""
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, self.arena_width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, self.arena_height)
