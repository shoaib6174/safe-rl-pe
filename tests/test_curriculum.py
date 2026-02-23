"""Tests for curriculum learning (Session 5).

Tests CurriculumManager level transitions, env overrides, arena-aware
distance clamping, and integration with AMSDRLSelfPlay.
"""

import pytest
import numpy as np

from training.curriculum import CurriculumManager


# ─── CurriculumManager Unit Tests ───


class TestCurriculumManagerInit:
    """Test initialization and configuration."""

    def test_default_init(self):
        cm = CurriculumManager()
        assert cm.current_level == 1
        assert cm.advancement_threshold == 0.70
        assert not cm.at_max_level
        assert cm.max_level == 4

    def test_custom_threshold(self):
        cm = CurriculumManager(advancement_threshold=0.80)
        assert cm.advancement_threshold == 0.80

    def test_starts_at_level_1(self):
        cm = CurriculumManager()
        assert cm.current_level == 1

    def test_four_levels_defined(self):
        cm = CurriculumManager()
        assert set(cm.levels.keys()) == {1, 2, 3, 4}

    def test_level_descriptions(self):
        cm = CurriculumManager()
        for level, config in cm.levels.items():
            assert "description" in config
            assert isinstance(config["description"], str)

    def test_repr(self):
        cm = CurriculumManager()
        r = repr(cm)
        assert "level=1/4" in r
        assert "threshold=0.7" in r


class TestCurriculumLevels:
    """Test level configuration and overrides."""

    def test_level_1_no_obstacles(self):
        cm = CurriculumManager()
        overrides = cm.get_env_overrides()
        assert overrides["n_obstacles"] == 0
        assert overrides["min_init_distance"] == 2.0
        assert overrides["max_init_distance"] <= 5.0

    def test_level_2_no_obstacles_medium_distance(self):
        cm = CurriculumManager()
        cm.current_level = 2
        overrides = cm.get_env_overrides()
        assert overrides["n_obstacles"] == 0
        assert overrides["min_init_distance"] >= 4.0  # scaled by arena

    def test_level_3_has_obstacles(self):
        cm = CurriculumManager()
        cm.current_level = 3
        overrides = cm.get_env_overrides()
        assert overrides["n_obstacles"] > 0

    def test_level_4_has_obstacles(self):
        cm = CurriculumManager()
        cm.current_level = 4
        overrides = cm.get_env_overrides()
        assert overrides["n_obstacles"] > 0

    def test_overrides_have_required_keys(self):
        cm = CurriculumManager()
        for level in range(1, 5):
            cm.current_level = level
            overrides = cm.get_env_overrides()
            assert "min_init_distance" in overrides
            assert "max_init_distance" in overrides
            assert "n_obstacles" in overrides

    def test_min_lt_max_distance(self):
        cm = CurriculumManager()
        for level in range(1, 5):
            cm.current_level = level
            overrides = cm.get_env_overrides()
            assert overrides["min_init_distance"] < overrides["max_init_distance"]


class TestArenaAwareClamping:
    """Test that distances are clamped based on arena size."""

    def test_small_arena_clamps_distances(self):
        """10x10 arena diagonal ≈ 14.14, 80% ≈ 11.3. Max dist should be clamped."""
        cm = CurriculumManager(arena_width=10.0, arena_height=10.0)
        max_possible = 0.8 * np.hypot(10.0, 10.0)
        for level in range(1, 5):
            cm.current_level = level
            overrides = cm.get_env_overrides()
            assert overrides["max_init_distance"] <= max_possible + 1e-6

    def test_large_arena_uses_full_distances(self):
        """20x20 arena diagonal ≈ 28.28, 80% ≈ 22.6. 15m fits easily."""
        cm = CurriculumManager(arena_width=20.0, arena_height=20.0)
        # Level 2 should use 15.0 (not clamped)
        cm.current_level = 2
        overrides = cm.get_env_overrides()
        assert overrides["max_init_distance"] == 15.0

    def test_tiny_arena_still_valid(self):
        """5x5 arena should still produce valid distance ranges."""
        cm = CurriculumManager(arena_width=5.0, arena_height=5.0)
        for level in range(1, 5):
            cm.current_level = level
            overrides = cm.get_env_overrides()
            assert overrides["min_init_distance"] < overrides["max_init_distance"]
            assert overrides["min_init_distance"] > 0
            assert overrides["max_init_distance"] > 0


class TestCurriculumAdvancement:
    """Test advancement logic."""

    def test_advance_on_high_capture_rate(self):
        cm = CurriculumManager()
        assert cm.current_level == 1
        advanced = cm.check_advancement(0.80)
        assert advanced
        assert cm.current_level == 2

    def test_no_advance_on_low_capture_rate(self):
        cm = CurriculumManager()
        advanced = cm.check_advancement(0.50)
        assert not advanced
        assert cm.current_level == 1

    def test_no_advance_at_threshold(self):
        """Advancement requires EXCEEDING the threshold, not meeting it."""
        cm = CurriculumManager(advancement_threshold=0.70)
        advanced = cm.check_advancement(0.70)
        assert not advanced
        assert cm.current_level == 1

    def test_advance_through_all_levels(self):
        cm = CurriculumManager()
        for expected_level in range(2, 5):
            advanced = cm.check_advancement(0.90)
            assert advanced
            assert cm.current_level == expected_level

    def test_no_advance_past_max(self):
        cm = CurriculumManager()
        cm.current_level = 4
        advanced = cm.check_advancement(0.99)
        assert not advanced
        assert cm.current_level == 4

    def test_at_max_level_property(self):
        cm = CurriculumManager()
        assert not cm.at_max_level
        cm.current_level = 4
        assert cm.at_max_level

    def test_level_history_recorded(self):
        cm = CurriculumManager()
        cm.check_advancement(0.50)  # No advance
        cm.check_advancement(0.80)  # Advance
        assert len(cm.level_history) == 2
        assert cm.level_history[0]["level"] == 1
        assert cm.level_history[0]["advanced"] is False
        assert cm.level_history[1]["level"] == 1
        assert cm.level_history[1]["advanced"] is True

    def test_level_history_capture_rate(self):
        cm = CurriculumManager()
        cm.check_advancement(0.65)
        assert cm.level_history[0]["capture_rate"] == 0.65


class TestCurriculumStatus:
    """Test status reporting."""

    def test_get_status_keys(self):
        cm = CurriculumManager()
        status = cm.get_status()
        assert "curriculum_level" in status
        assert "curriculum_description" in status
        assert "curriculum_at_max" in status

    def test_status_reflects_current_level(self):
        cm = CurriculumManager()
        assert cm.get_status()["curriculum_level"] == 1
        cm.check_advancement(0.80)
        assert cm.get_status()["curriculum_level"] == 2

    def test_status_at_max(self):
        cm = CurriculumManager()
        cm.current_level = 4
        assert cm.get_status()["curriculum_at_max"] is True


class TestCurriculumIntegration:
    """Test curriculum integration with AMSDRLSelfPlay (import-level)."""

    def test_amsdrl_accepts_curriculum_param(self):
        """AMSDRLSelfPlay should accept curriculum=True without error."""
        from training.amsdrl import AMSDRLSelfPlay
        # Just verify the constructor accepts the parameter
        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_curriculum_amsdrl",
            max_phases=2,
            timesteps_per_phase=100,
            curriculum=True,
            n_envs=1,
        )
        assert amsdrl.curriculum is not None
        assert amsdrl.curriculum.current_level == 1

    def test_amsdrl_curriculum_disabled_by_default(self):
        from training.amsdrl import AMSDRLSelfPlay
        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_curriculum_amsdrl_off",
            max_phases=2,
            timesteps_per_phase=100,
            n_envs=1,
        )
        assert amsdrl.curriculum is None

    def test_curriculum_sets_initial_env_kwargs(self):
        """When curriculum is enabled, Level 1 overrides should be applied."""
        from training.amsdrl import AMSDRLSelfPlay
        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_curriculum_env_kwargs",
            max_phases=2,
            timesteps_per_phase=100,
            curriculum=True,
            n_envs=1,
        )
        # Level 1: close range (2-5m), no obstacles
        assert amsdrl.env_kwargs["min_init_distance"] == 2.0
        assert amsdrl.env_kwargs["max_init_distance"] <= 5.0
        assert amsdrl.n_obstacles == 0
