"""Tests for pe2.config — frozen, hashable, lesson-locked defaults."""

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from pe2.config import EnvConfig


class TestEnvConfig:
    def test_is_dataclass(self):
        assert is_dataclass(EnvConfig)

    def test_frozen(self):
        cfg = EnvConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.dt = 0.1  # type: ignore[misc]

    def test_hashable(self):
        assert hash(EnvConfig()) == hash(EnvConfig())
        assert len({EnvConfig(), EnvConfig(), EnvConfig(arena_size=20.0)}) == 2

    def test_equality(self):
        assert EnvConfig() == EnvConfig()
        assert EnvConfig(n_obstacles=3) != EnvConfig(n_obstacles=2)

    def test_locked_defaults(self):
        """Defaults must match lessons_learned.md §4 / plan Step 1.

        If this test fails because a default changed intentionally, the change
        needs a logged decision in docs/decisions.md.
        """
        cfg = EnvConfig()
        assert cfg.dt == 0.05
        assert cfg.v_max == 1.0
        assert cfg.omega_max == 2.84
        assert cfg.arena_size == 10.0
        assert cfg.n_obstacles == 2
        assert cfg.evader_speed_ratio == 1.2
        assert cfg.agent_radius == 0.15
        assert cfg.capture_distance == pytest.approx(0.30)
        # Step 1 additions (proven root-codebase defaults)
        assert cfg.max_steps == 1200
        assert cfg.prep_steps == 0
        assert cfg.min_init_distance == 3.0
        assert cfg.obstacle_radius_min == 0.3
        assert cfg.obstacle_radius_max == 1.0
        assert cfg.obstacle_margin == 0.5
        assert cfg.evader_v_max == pytest.approx(1.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
