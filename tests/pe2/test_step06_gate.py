"""Tests for scripts/pe2_step06_gate.py — the Step 6 gate's own decisions.

The 6000-episode sweep is not exercised here; what is tested is what the
script DECIDES with the numbers and how it fingerprints a run, which is where
a wrongly passed or wrongly failed gate would come from. The script imports
only pe2 + numpy, so no optional-dependency skip is needed.
"""

import importlib.util
import pathlib

import pytest

from pe2.config import EnvConfig

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "pe2_step06_gate.py"


def _gate():
    spec = importlib.util.spec_from_file_location("pe2_step06_gate", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _gate()


class TestPreRegisteredConstants:
    """The pre-registration fixes these; a diff here is a criterion change."""

    def test_seed_streams_and_episode_count(self):
        assert GATE.MAIN_SEED0 == 800000
        assert GATE.PROBE_SEED0 == 810000
        assert GATE.N_EPISODES == 1000

    def test_thresholds(self):
        assert GATE.G2_CAPTURE_FACTOR == 0.5
        assert GATE.G4_MIN_VIOLATION_EPISODE_RATE == 0.05
        assert GATE.ROBUST_DELTA == 0.05


class TestStartFlags:
    def test_flags_are_booleans_and_hard_implies_nothing_about_soft(self):
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        flags = GATE.start_flags(cfg, CBFShield(cfg), 800000)
        assert set(flags) == {"start_in_barrier_set", "start_hard_feasible"}
        assert isinstance(flags["start_in_barrier_set"], bool)
        assert flags["start_hard_feasible"] is True  # δ = 0: the env guarantees it

    def test_a_known_outside_start_is_flagged(self):
        """Seed 760372 started with the VCP inside an obstacle in the v1 run
        (robust_main first offence at step 1); at δ = 0.05 its BODY is inside."""
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        flags = GATE.start_flags(cfg, CBFShield(cfg, margin=0.05), 760372)
        assert flags["start_hard_feasible"] is False


class TestConfigFingerprint:
    def test_is_stable_across_equal_configs(self):
        assert GATE.config_fingerprint(EnvConfig()) == GATE.config_fingerprint(
            EnvConfig()
        )

    def test_changes_with_any_parameter(self):
        base = GATE.config_fingerprint(EnvConfig())
        assert GATE.config_fingerprint(EnvConfig(n_obstacles=3)) != base
        assert GATE.config_fingerprint(EnvConfig(evader_speed_ratio=1.3)) != base

    def test_is_16_hex_characters(self):
        fp = GATE.config_fingerprint(EnvConfig())
        assert len(fp) == 16
        assert set(fp) <= set("0123456789abcdef")


class TestViolationUpperBound:
    def test_zero_of_one_thousand_matches_the_rule_of_three(self):
        """Wilson one-sided 95% (z = 1.645) at k=0, n=1000.

        p = 0; denom = 1 + 1.645^2/1000 = 1.00270602;
        centre = (0 + 1.645^2/2000)/denom = 0.00135301/1.00270602 = 0.00134936;
        half = 1.645*sqrt(0 + 1.645^2/(4e6))/denom
             = 1.645*0.00082250/1.00270602 = 0.00134936;
        upper = 0.00269873 — i.e. 0.0027, matching the pre-registered
        "rule of three" figure of ~0.003 (3/1000).
        """
        assert GATE.violation_upper_bound(0, 1000) == pytest.approx(
            0.00269873, abs=1e-7
        )

    def test_bound_grows_with_observed_violations(self):
        assert GATE.violation_upper_bound(1, 1000) > GATE.violation_upper_bound(0, 1000)


class TestG2Verdict:
    def test_passes_at_exactly_half(self):
        # 0.30 >= 0.5 * 0.60 = 0.30 — the boundary is inclusive.
        assert GATE.g2_capture_verdict(0.30, 0.60) is True

    def test_fails_just_below_half(self):
        # 0.299 < 0.5 * 0.60 = 0.30
        assert GATE.g2_capture_verdict(0.299, 0.60) is False

    def test_passes_when_the_shield_helps(self):
        assert GATE.g2_capture_verdict(0.62, 0.60) is True


class TestRunLengthSummary:
    def test_empty(self):
        s = GATE.run_length_summary([])
        assert s["n"] == 0
        assert s["max"] is None
        assert s["n_ge_deadlock"] == 0

    def test_hand_computed_distribution(self):
        # runs = [0, 0, 3, 12, 40, 45]: n = 6, max = 45,
        # mean = (0+0+3+12+40+45)/6 = 100/6 = 16.666...,
        # p50 (linear interpolation between the 3rd and 4th of the sorted
        # list, i.e. between 3 and 12) = 7.5,
        # two runs are >= DEADLOCK_STEPS (40).
        s = GATE.run_length_summary([0, 0, 3, 12, 40, 45])
        assert s["n"] == 6
        assert s["max"] == 45
        assert s["mean"] == pytest.approx(100 / 6)
        assert s["p50"] == pytest.approx(7.5)
        assert s["n_ge_deadlock"] == 2

    def test_histogram_buckets_are_exhaustive(self):
        runs = [0, 0, 1, 4, 5, 9, 10, 19, 20, 29, 30, 39, 40, 79, 80, 400]
        s = GATE.run_length_summary(runs)
        assert sum(s["histogram"].values()) == len(runs)
        # 0 -> "0" (twice); 1 and 4 -> "1-4"; 5 and 9 -> "5-9"; 400 -> ">=320"
        assert s["histogram"]["0"] == 2
        assert s["histogram"]["1-4"] == 2
        assert s["histogram"]["5-9"] == 2
        assert s["histogram"][">=320"] == 1

    def test_deadlock_cut_is_the_metrics_constant(self):
        from pe2.metrics import DEADLOCK_STEPS

        assert GATE.run_length_summary([DEADLOCK_STEPS - 1])["n_ge_deadlock"] == 0
        assert GATE.run_length_summary([DEADLOCK_STEPS])["n_ge_deadlock"] == 1


class TestShieldSpec:
    def test_records_the_margin_that_defines_the_arm(self):
        from pe2.shield import CBFShield, NoOpShield

        cfg = EnvConfig()
        assert GATE.shield_spec(CBFShield(cfg))["margin"] == 0.0
        assert GATE.shield_spec(CBFShield(cfg, margin=0.05))["margin"] == 0.05
        assert GATE.shield_spec(NoOpShield(cfg))["class"] == "NoOpShield"
        assert GATE.shield_spec(CBFShield(cfg))["gamma"] == 0.2


class TestReportedNotGated:
    def _summary(self):
        return {
            "infeasibility_rate": 0.001,
            "min_clearance_floor": -0.0088,
            "min_clearance_hard_floor": 0.0004,
            "min_h_floor": -0.02,
            "tier_histogram": {"exact": 9, "relaxed": 1, "backup": 0, "none": 0},
            "intervention_rate": 0.23,
            "backup_rate": 0.0,
            "stall_step_rate": 0.0,
            "immobile_step_rate": 0.0,
            "deadlock_episode_rate": 0.0,
            "pinned_episode_rate": 0.0,
            "episodes_stalled_at_end": 0,
            "longest_stall_run_per_episode": [0, 2],
            "longest_immobile_run_per_episode": [0, 3],
            "longest_backup_run_per_episode": [0, 0],
        }

    def test_shielded_arm_has_no_structural_zero_note(self):
        block = GATE.reported_not_gated(self._summary(), is_control=False)
        assert "structurally_zero_fields" not in block
        assert block["min_clearance_floor_m"] == -0.0088

    def test_min_clearance_hard_is_reported_never_gated(self):
        # The body-only floor (D97's fix for the false "floor = 0" claim)
        # rides alongside min_clearance_floor_m in the reported-not-gated
        # block, distinct from the VCP-inclusive number.
        block = GATE.reported_not_gated(self._summary(), is_control=False)
        assert block["min_clearance_hard_floor_m"] == 0.0004
        assert block["min_clearance_hard_floor_m"] != block["min_clearance_floor_m"]

    def test_control_arm_flags_its_structurally_zero_fields(self):
        block = GATE.reported_not_gated(self._summary(), is_control=True)
        assert "stall_step_rate" in block["structurally_zero_fields"]
        assert "pinned_episode_rate" in block["structurally_zero_fields"]
        assert "infeasibility_rate" in block["structurally_zero_fields"]
        assert "NoOpShield" in block["structurally_zero_note"]

    def test_reports_the_qp_margin_under_a_name_that_is_not_a_distance(self):
        # D89: min_h mixes m and m^2 and must never be published as a
        # distance; only min_clearance carries metres.
        block = GATE.reported_not_gated(self._summary(), is_control=False)
        assert "min_h_floor_qp_margin_mixed_units" in block
        assert (
            block["min_clearance_floor_m"] != block["min_h_floor_qp_margin_mixed_units"]
        )


class TestContactGeometry:
    """How far past a surface the executed step would have gone, in metres.

    Locked defaults: dt = 0.05, agent_radius = 0.15, arena 10 -> the env
    clips the body centre at +-4.85.
    """

    def test_no_contact_is_two_zeros(self):
        g = GATE.contact_geometry(EnvConfig(), (0.0, 0.0, 0.0), (1.0, 0.0), [])
        assert g["wall_overshoot_m"] == 0.0
        assert g["obstacle_penetration_m"] == 0.0

    def test_wall_overshoot(self):
        # x = 4.83 + 1.0 * cos(0) * 0.05 = 4.88; 4.88 - 4.85 = 0.03 m.
        g = GATE.contact_geometry(EnvConfig(), (4.83, 0.0, 0.0), (1.0, 0.0), [])
        assert g["wall_overshoot_m"] == pytest.approx(0.03)

    def test_obstacle_penetration(self):
        # Heading -x: x = 0.68 - 0.05 = 0.63. Obstacle at the origin, r = 0.5,
        # so the env's minimum centre distance is 0.5 + 0.15 = 0.65 and the
        # overlap is 0.65 - 0.63 = 0.02 m.
        import math

        g = GATE.contact_geometry(
            EnvConfig(),
            (0.68, 0.0, math.pi),
            (1.0, 0.0),
            [{"x": 0.0, "y": 0.0, "radius": 0.5}],
        )
        assert g["wall_overshoot_m"] == 0.0
        assert g["obstacle_penetration_m"] == pytest.approx(0.02)

    def test_obstacle_overlap_is_measured_after_the_wall_clip(self):
        # x = 4.84 + 0.05 = 4.89 -> overshoot 0.04, then the env clips to
        # 4.85 BEFORE resolving obstacles. Obstacle at (5.3, 0) r = 0.4:
        # from the clipped 4.85 the distance is 0.45 and the minimum is 0.55,
        # so the overlap is 0.10 m. Measured from the unclipped 4.89 it would
        # read 0.14 m, which is not what the env does.
        g = GATE.contact_geometry(
            EnvConfig(),
            (4.84, 0.0, 0.0),
            (1.0, 0.0),
            [{"x": 5.3, "y": 0.0, "radius": 0.4}],
        )
        assert g["wall_overshoot_m"] == pytest.approx(0.04)
        assert g["obstacle_penetration_m"] == pytest.approx(0.10)

    def test_a_stopped_robot_moves_nowhere(self):
        g = GATE.contact_geometry(
            EnvConfig(),
            (4.85, 0.0, 0.0),
            (0.0, 2.0),
            [{"x": 0.0, "y": 0.0, "radius": 0.5}],
        )
        assert g["wall_overshoot_m"] == 0.0
        assert g["obstacle_penetration_m"] == 0.0


class TestDistribution:
    def test_empty(self):
        assert GATE.distribution([]) == {"n": 0}

    def test_hand_computed(self):
        # sorted [0.0, 1e-10, 1e-4, 0.05, 0.2]: p50 is the 3rd value 1e-4;
        # p95 interpolates at index 0.95*4 = 3.8, i.e. 0.05 + 0.8*(0.2-0.05)
        # = 0.17. Three values exceed 1e-9 and 1e-6; two exceed 1e-3 and 1e-2.
        d = GATE.distribution([0.0, 1e-10, 1e-4, 0.05, 0.2])
        assert d["n"] == 5
        assert d["min"] == 0.0
        assert d["max"] == pytest.approx(0.2)
        assert d["p50"] == pytest.approx(1e-4)
        assert d["p95"] == pytest.approx(0.17)
        assert d["n_above_1e-9_m"] == 3
        assert d["n_above_1e-6_m"] == 3
        assert d["n_above_1e-3_m"] == 2
        assert d["n_above_1e-2_m"] == 2
