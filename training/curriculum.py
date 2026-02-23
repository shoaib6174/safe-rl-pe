"""Curriculum manager for progressive difficulty in AMS-DRL self-play.

Implements a 4-level curriculum that gradually increases task difficulty:
  Level 1: Close range, no obstacles (easiest)
  Level 2: Medium range, no obstacles
  Level 3: Close range, with obstacles
  Level 4: Full scenario — variable range and obstacles (hardest)

Difficulty is controlled by initial agent separation distance and obstacle count.
FOV (partial observability) is always active — the BiMDN encoder needs consistent
partial-obs data across all levels to avoid training discontinuities.

Advancement criterion: pursuer capture rate > threshold (default 70%) at the
current level. The evader's quality is implicitly validated by the self-play
equilibrium; NE convergence is checked separately by AMS-DRL.
"""

from __future__ import annotations

import numpy as np


class CurriculumManager:
    """Manages progressive difficulty levels for self-play training.

    Each level specifies environment parameter overrides (init distance range,
    obstacle count) that are applied when creating training environments.
    Advancement is checked after each self-play evaluation phase.

    Args:
        arena_width: Arena width in meters (for distance clamping).
        arena_height: Arena height in meters (for distance clamping).
        advancement_threshold: Capture rate required to advance (default 0.70).
        eval_window: Not used directly — advancement is based on the evaluation
            capture rate reported by AMS-DRL after each phase.
    """

    def __init__(
        self,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        advancement_threshold: float = 0.70,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.advancement_threshold = advancement_threshold

        # Clamp max distances to 80% of arena diagonal to ensure valid spawns
        max_possible = 0.8 * np.hypot(arena_width, arena_height)

        self.levels = {
            1: {
                "min_init_distance": 2.0,
                "max_init_distance": min(5.0, max_possible),
                "n_obstacles": 0,
                "description": "Close, no obstacles",
            },
            2: {
                "min_init_distance": min(5.0, max_possible * 0.4),
                "max_init_distance": min(15.0, max_possible),
                "n_obstacles": 0,
                "description": "Medium distance, no obstacles",
            },
            3: {
                "min_init_distance": 2.0,
                "max_init_distance": min(5.0, max_possible),
                "n_obstacles": 3,
                "description": "Close, with obstacles",
            },
            4: {
                "min_init_distance": 2.0,
                "max_init_distance": min(15.0, max_possible),
                "n_obstacles": 3,
                "description": "Full scenario (variable distance + obstacles)",
            },
        }

        self.current_level = 1
        self.level_history: list[dict] = []

    @property
    def max_level(self) -> int:
        return max(self.levels.keys())

    @property
    def at_max_level(self) -> bool:
        return self.current_level >= self.max_level

    def get_env_overrides(self) -> dict:
        """Return env_kwargs overrides for the current curriculum level.

        Returns a dict with keys that can be passed to PursuitEvasionEnv:
            - min_init_distance
            - max_init_distance
        And a separate key for the obstacle count:
            - n_obstacles
        """
        level_config = self.levels[self.current_level]
        return {
            "min_init_distance": level_config["min_init_distance"],
            "max_init_distance": level_config["max_init_distance"],
            "n_obstacles": level_config["n_obstacles"],
        }

    def check_advancement(self, capture_rate: float) -> bool:
        """Check if the pursuer should advance to the next level.

        Args:
            capture_rate: Pursuer capture rate from evaluation (0.0 to 1.0).

        Returns:
            True if advanced to next level, False otherwise.
        """
        self.level_history.append({
            "level": self.current_level,
            "capture_rate": capture_rate,
            "advanced": False,
        })

        if self.at_max_level:
            return False

        if capture_rate > self.advancement_threshold:
            old_level = self.current_level
            self.current_level += 1
            self.level_history[-1]["advanced"] = True
            print(
                f"[CURRICULUM] Advanced: Level {old_level} → {self.current_level} "
                f"(capture_rate={capture_rate:.2f} > {self.advancement_threshold:.2f})"
            )
            print(
                f"[CURRICULUM] Level {self.current_level}: "
                f"{self.levels[self.current_level]['description']}"
            )
            return True

        return False

    def get_status(self) -> dict:
        """Return current curriculum status for logging."""
        return {
            "curriculum_level": self.current_level,
            "curriculum_description": self.levels[self.current_level]["description"],
            "curriculum_at_max": self.at_max_level,
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumManager(level={self.current_level}/{self.max_level}, "
            f"threshold={self.advancement_threshold})"
        )
