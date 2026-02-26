"""Tests for S52 fixes: cold-start checkpoint, phase warmup, NE-gap advancement.

Fix 1: Save evader checkpoint after cold-start (bilateral rollback works in S1)
Fix 2: Phase length warmup (shorter phases early, ramp to full)
Fix 3: NE-gap-based curriculum advancement (advance when balanced, not dominated)
"""

import numpy as np
import pytest

from training.curriculum import SmoothCurriculumManager


# ─── Fix 1: Cold-start checkpoint ───


class TestColdStartCheckpoint:
    """Verify cold-start saves evader rolling + milestone checkpoints."""

    def _make_amsdrl(self, tmp_path):
        """Create a minimal AMSDRLSelfPlay for testing partial-obs cold-start."""
        from training.amsdrl import AMSDRLSelfPlay

        return AMSDRLSelfPlay(
            output_dir=str(tmp_path / "test_run"),
            max_phases=0,
            cold_start_timesteps=512,
            timesteps_per_phase=512,
            n_envs=1,
            use_dcbf=False,
            full_obs=False,
            fixed_speed=True,
            seed=42,
            n_steps=64,
            batch_size=32,
            arena_width=10.0,
            arena_height=10.0,
            verbose=0,
        )

    def test_cold_start_saves_evader_rolling_checkpoint(self, tmp_path):
        """After cold-start, evader should have at least one rolling checkpoint."""
        amsdrl = self._make_amsdrl(tmp_path)
        amsdrl._cold_start()

        assert len(amsdrl.evader_ckpt.rolling) >= 1, \
            "Evader should have at least 1 rolling checkpoint after cold-start"

    def test_cold_start_saves_evader_milestone(self, tmp_path):
        """After cold-start, evader milestone_phase0_evader should exist."""
        amsdrl = self._make_amsdrl(tmp_path)
        amsdrl._cold_start()

        milestone_dir = amsdrl.evader_ckpt.checkpoint_dir / "milestone_phase0_evader"
        assert milestone_dir.exists(), \
            "milestone_phase0_evader should exist after cold-start"
        assert (milestone_dir / "ppo.zip").exists(), \
            "PPO model should be saved in milestone"

    def test_cold_start_rolling_has_meta(self, tmp_path):
        """Rolling checkpoint meta should indicate cold_start source."""
        amsdrl = self._make_amsdrl(tmp_path)
        amsdrl._cold_start()

        import json
        _, ckpt_path = amsdrl.evader_ckpt.rolling[0]
        meta_path = f"{ckpt_path}/meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta.get("source") == "cold_start"
        assert meta.get("role") == "evader"


# ─── Fix 2: Phase Warmup ───


class TestPhaseWarmup:
    """Test phase length warmup scheduling."""

    def test_default_schedule(self):
        """Default phase warmup schedule should have S1-S6 entries."""
        from training.amsdrl import AMSDRLSelfPlay

        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_warmup",
            phase_warmup=True,
            verbose=0,
        )

        assert 1 in amsdrl.phase_warmup_schedule
        assert 6 in amsdrl.phase_warmup_schedule
        # S1-S2 should be shortest
        assert amsdrl.phase_warmup_schedule[1] == 100_000
        assert amsdrl.phase_warmup_schedule[2] == 100_000
        # S3-S4 should be medium
        assert amsdrl.phase_warmup_schedule[3] == 200_000
        assert amsdrl.phase_warmup_schedule[4] == 200_000
        # S5-S6 should be longer
        assert amsdrl.phase_warmup_schedule[5] == 300_000
        assert amsdrl.phase_warmup_schedule[6] == 300_000

    def test_custom_schedule(self):
        """Custom schedule should override defaults."""
        from training.amsdrl import AMSDRLSelfPlay

        custom = [(1, 50_000), (2, 50_000)]
        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_warmup",
            phase_warmup=True,
            phase_warmup_schedule=custom,
            verbose=0,
        )

        assert amsdrl.phase_warmup_schedule[1] == 50_000
        assert amsdrl.phase_warmup_schedule[2] == 50_000
        assert 3 not in amsdrl.phase_warmup_schedule

    def test_no_warmup_empty_schedule(self):
        """With phase_warmup=False, schedule should be empty."""
        from training.amsdrl import AMSDRLSelfPlay

        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_warmup",
            phase_warmup=False,
            verbose=0,
        )

        assert amsdrl.phase_warmup_schedule == {}

    def test_phase_beyond_schedule_uses_full(self):
        """Phases beyond warmup schedule should use full timesteps_per_phase."""
        from training.amsdrl import AMSDRLSelfPlay

        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_warmup",
            phase_warmup=True,
            timesteps_per_phase=500_000,
            verbose=0,
        )

        # Phase 7 is NOT in schedule, should use full 500K
        assert 7 not in amsdrl.phase_warmup_schedule
        # Phase 1 IS in schedule
        assert 1 in amsdrl.phase_warmup_schedule
        assert amsdrl.phase_warmup_schedule[1] < amsdrl.timesteps_per_phase


# ─── Fix 3: NE-gap Advancement ───


class TestNEGapAdvancement:
    """Test NE-gap-based curriculum advancement."""

    def test_ne_gap_params_stored(self):
        """NE-gap params should be stored correctly."""
        from training.amsdrl import AMSDRLSelfPlay

        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_ne",
            ne_gap_advancement=True,
            ne_gap_threshold=0.20,
            ne_gap_consecutive=3,
            verbose=0,
        )

        assert amsdrl.ne_gap_advancement is True
        assert amsdrl.ne_gap_threshold == 0.20
        assert amsdrl.ne_gap_consecutive == 3
        assert amsdrl._ne_gap_streak == 0

    def test_ne_gap_disabled_by_default(self):
        """NE-gap advancement should be disabled by default."""
        from training.amsdrl import AMSDRLSelfPlay

        amsdrl = AMSDRLSelfPlay(
            output_dir="/tmp/test_ne",
            verbose=0,
        )

        assert amsdrl.ne_gap_advancement is False


# ─── SmoothCurriculumManager._advance() ───


class TestSmoothCurriculumAdvance:
    """Test the _advance() method on SmoothCurriculumManager."""

    def test_advance_increases_distance(self):
        """_advance() should increase max_init_distance by increment."""
        cm = SmoothCurriculumManager(
            initial_max_distance=5.0,
            distance_increment=1.0,
            final_max_distance=15.0,
        )
        old = cm.max_init_distance
        cm._advance()
        assert cm.max_init_distance == old + 1.0

    def test_advance_capped_at_final(self):
        """_advance() should not exceed final_max_distance."""
        cm = SmoothCurriculumManager(
            initial_max_distance=14.5,
            distance_increment=1.0,
            final_max_distance=15.0,
        )
        cm._advance()
        assert cm.max_init_distance == 15.0

    def test_advance_increments_count(self):
        """_advance() should increment advancement_count."""
        cm = SmoothCurriculumManager(
            initial_max_distance=5.0,
            distance_increment=1.0,
        )
        assert cm.advancement_count == 0
        cm._advance()
        assert cm.advancement_count == 1
        cm._advance()
        assert cm.advancement_count == 2

    def test_advance_resets_phase_counters(self):
        """_advance() should reset phases_at_level and consecutive_floor_phases."""
        cm = SmoothCurriculumManager(
            initial_max_distance=5.0,
            distance_increment=1.0,
        )
        cm.phases_at_level = 5
        cm.consecutive_floor_phases = 3
        cm._advance()
        assert cm.phases_at_level == 0
        assert cm.consecutive_floor_phases == 0

    def test_advance_noop_at_max(self):
        """_advance() should do nothing if already at max."""
        cm = SmoothCurriculumManager(
            initial_max_distance=15.0,
            final_max_distance=15.0,
            distance_increment=1.0,
        )
        cm._advance()
        assert cm.max_init_distance == 15.0
        assert cm.advancement_count == 0

    def test_obstacles_activate_on_advance(self):
        """Obstacles should activate when max_distance crosses threshold."""
        cm = SmoothCurriculumManager(
            initial_max_distance=7.5,
            distance_increment=1.0,
            final_max_distance=15.0,
            obstacles_after_distance=8.0,
            n_obstacles=3,
        )
        assert cm.n_obstacles == 0  # Below threshold
        cm._advance()  # 7.5 -> 8.5
        assert cm.n_obstacles == 3  # Now above threshold

    def test_check_advancement_calls_advance(self):
        """check_advancement should call _advance when criteria met."""
        cm = SmoothCurriculumManager(
            initial_max_distance=5.0,
            distance_increment=1.0,
            advancement_threshold=0.50,
            min_escape_rate=0.0,
            min_phases_per_level=1,
        )
        old_max = cm.max_init_distance
        advanced = cm.check_advancement(capture_rate=0.60, escape_rate=0.40)
        assert advanced is True
        assert cm.max_init_distance == old_max + 1.0


# ─── CLI args ───


class TestCLIArgs:
    """Verify new CLI arguments are accepted by train_amsdrl.py."""

    def test_phase_warmup_arg(self):
        """--phase_warmup should be recognized."""
        import sys
        sys.path.insert(0, "scripts")
        # Just check that parse_args accepts the new args
        import importlib
        import scripts.train_amsdrl as train_mod
        importlib.reload(train_mod)

        import argparse
        parser = argparse.ArgumentParser()
        # Manually verify the arg exists in the script
        import inspect
        source = inspect.getsource(train_mod.parse_args)
        assert "--phase_warmup" in source
        assert "--ne_gap_advancement" in source
        assert "--ne_gap_threshold" in source
        assert "--ne_gap_consecutive" in source
