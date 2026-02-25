"""Curriculum manager for progressive difficulty in AMS-DRL self-play.

Implements a 6-level curriculum that gradually increases task difficulty:
  Level 1: Close range, no obstacles (easiest)
  Level 2: Medium range, no obstacles
  Level 3: Close range, with obstacles
  Level 4: Medium range, with obstacles
  Level 5: Far range, with obstacles
  Level 6: Full scenario — variable range and obstacles (hardest)

Difficulty is controlled by initial agent separation distance and obstacle count.
FOV (partial observability) is always active — the BiMDN encoder needs consistent
partial-obs data across all levels to avoid training discontinuities.

Advancement criteria (all must be met):
  1. Pursuer capture_rate > advancement_threshold (default 0.70)
  2. Evader escape_rate >= min_escape_rate (default 0.0 for backward compat)
  3. phases_at_current_level >= min_phases_per_level (default 1 for backward compat)

Regression: If evader escape_rate stays below regression_floor for
regression_patience consecutive phases at a non-L1 level, regress one level.

History: The original 4-level curriculum (L1-L4) caused a Level 4 collapse where
the pursuer achieved 100% capture rate. The 6-level curriculum adds intermediate
distance+obstacle levels. Runs H-P showed the curriculum racing from L1→L6 in
5 phases before the evader trained with obstacles — dual-criteria gates and
regression were added in S48 to prevent this.
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
        min_escape_rate: Minimum evader escape rate to advance (default 0.0).
        min_phases_per_level: Minimum phases at each level before advancing (default 1).
        regression_floor: If escape_rate stays below this for regression_patience
            consecutive phases, regress one level (default 0.02).
        regression_patience: Number of consecutive low-escape phases before
            regression triggers (default 3).
    """

    def __init__(
        self,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        advancement_threshold: float = 0.70,
        min_escape_rate: float = 0.0,
        min_phases_per_level: int = 1,
        regression_floor: float = 0.02,
        regression_patience: int = 3,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.advancement_threshold = advancement_threshold
        self.min_escape_rate = min_escape_rate
        self.min_phases_per_level = min_phases_per_level
        self.regression_floor = regression_floor
        self.regression_patience = regression_patience

        # Phase counting and regression state
        self.phases_at_level: int = 0
        self.consecutive_floor_phases: int = 0

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
                "min_init_distance": min(5.0, max_possible * 0.4),
                "max_init_distance": min(8.0, max_possible),
                "n_obstacles": 3,
                "description": "Medium distance, with obstacles",
            },
            5: {
                "min_init_distance": min(5.0, max_possible * 0.4),
                "max_init_distance": min(12.0, max_possible),
                "n_obstacles": 3,
                "description": "Far distance, with obstacles",
            },
            6: {
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

    def check_advancement(self, capture_rate: float, escape_rate: float = 0.0) -> bool:
        """Check if agents should advance to the next curriculum level.

        All three criteria must be met to advance:
        1. capture_rate > advancement_threshold
        2. escape_rate >= min_escape_rate
        3. phases_at_level >= min_phases_per_level

        Args:
            capture_rate: Pursuer capture rate from evaluation (0.0 to 1.0).
            escape_rate: Evader escape rate from evaluation (0.0 to 1.0).

        Returns:
            True if advanced to next level, False otherwise.
        """
        self.phases_at_level += 1

        self.level_history.append({
            "level": self.current_level,
            "capture_rate": capture_rate,
            "escape_rate": escape_rate,
            "phases_at_level": self.phases_at_level,
            "advanced": False,
        })

        if self.at_max_level:
            return False

        # All three criteria must be met
        cr_ok = capture_rate > self.advancement_threshold
        er_ok = escape_rate >= self.min_escape_rate
        phases_ok = self.phases_at_level >= self.min_phases_per_level

        if cr_ok and er_ok and phases_ok:
            old_level = self.current_level
            self.current_level += 1
            self.phases_at_level = 0
            self.consecutive_floor_phases = 0
            self.level_history[-1]["advanced"] = True
            print(
                f"[CURRICULUM] Advanced: Level {old_level} → {self.current_level} "
                f"(capture_rate={capture_rate:.2f}, escape_rate={escape_rate:.2f}, "
                f"phases={self.level_history[-1]['phases_at_level']})"
            )
            print(
                f"[CURRICULUM] Level {self.current_level}: "
                f"{self.levels[self.current_level]['description']}"
            )
            return True

        if not cr_ok:
            reason = f"capture_rate={capture_rate:.2f} <= {self.advancement_threshold:.2f}"
        elif not er_ok:
            reason = f"escape_rate={escape_rate:.2f} < {self.min_escape_rate:.2f}"
        else:
            reason = f"phases_at_level={self.phases_at_level} < {self.min_phases_per_level}"
        print(f"[CURRICULUM] No advancement at Level {self.current_level}: {reason}")

        return False

    def check_regression(self, escape_rate: float) -> bool:
        """Check if curriculum should regress due to sustained evader collapse.

        If escape_rate stays below regression_floor for regression_patience
        consecutive phases at a non-L1 level, regress one level.

        Args:
            escape_rate: Evader escape rate from evaluation (0.0 to 1.0).

        Returns:
            True if regressed to previous level, False otherwise.
        """
        if self.current_level <= 1:
            return False

        if escape_rate < self.regression_floor:
            self.consecutive_floor_phases += 1
        else:
            self.consecutive_floor_phases = 0

        if self.consecutive_floor_phases >= self.regression_patience:
            old_level = self.current_level
            self.current_level -= 1
            self.phases_at_level = 0
            self.consecutive_floor_phases = 0
            print(
                f"[CURRICULUM] Regression: Level {old_level} → {self.current_level} "
                f"(escape_rate < {self.regression_floor:.2f} for "
                f"{self.regression_patience} consecutive phases)"
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
