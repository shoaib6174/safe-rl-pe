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
        assert GATE.MAIN_SEED0 == 760000
        assert GATE.PROBE_SEED0 == 770000
        assert GATE.N_EPISODES == 1000

    def test_thresholds(self):
        assert GATE.G2_CAPTURE_FACTOR == 0.5
        assert GATE.G4_MIN_VIOLATION_EPISODE_RATE == 0.05
        assert GATE.ROBUST_DELTA == 0.05


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
