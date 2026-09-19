"""scripts/derisk_s13_speed.py — the S13 speed-ratio round's own decisions.

Spec: docs/pathway/speed_ratio_prereg_2026-09-19.md.

No test here loads a checkpoint. The enumeration, the verdict rules and the
cell bookkeeping are driven on fabricated records; `run_cell`'s one binding to
the environment — that a cell's `EnvConfig` carries THAT cell's speed ratio and
that the structured opponent is built on the same config — is checked with the
evaluation stubbed out.

The transfer caveat (spec section 2) is what makes H-SPD-1's comparator the
scripted probe at the SAME ratio; `TestSpd1.test_the_probe_is_measured_at_the
_same_ratio` is the test that pins it.
"""

import importlib.util
import json
import math
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s13_speed.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s13_speed", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n):
    """n paired indicators whose first round(rate * n) entries are True, so a
    higher-rate list strictly contains a lower-rate one (every discordant pair
    points the same way and the paired test sees the direction under test)."""
    k = round(rate * n)
    return [i < k for i in range(n)]


def _record(name, deadlock=0.5, capture=0.2, violations=0, n_ep=300):
    spec = R.CELLS[name]
    ind = _ind(deadlock, n_ep)
    cap = _ind(capture, n_ep)
    return {
        "name": name,
        "pursuer": spec["pursuer"],
        "opponent": spec["opponent"],
        "ratio": spec["ratio"],
        "delta": spec["delta"],
        "n": spec["n"],
        "gamma": spec["gamma"],
        "shield_obs": R.PURSUERS[spec["pursuer"]]["shield_obs"],
        "n_episodes": n_ep,
        "seed0": R.SEED0,
        "deadlock_ind": ind,
        "capture_ind": cap,
        "deadlock_rate": sum(ind) / n_ep,
        "capture_rate": sum(cap) / n_ep,
        "escape_rate": 1.0 - sum(cap) / n_ep,
        "outcomes": {"capture": sum(cap), "timeout": n_ep - sum(cap)},
        "stall_step_rate": 0.3,
        "intervention_rate": 0.4,
        "violations": violations,
        "min_clearance_floor": 0.01,
        "min_clearance_hard_floor": 0.02,
        "wilson_deadlock": R._ci(sum(ind), n_ep),
        "wilson_capture": R._ci(sum(cap), n_ep),
    }


def _records(rate_fn, capture_fn=None, violations_fn=None, n_ep=300):
    """A full 64-cell record set whose deadlock rate is rate_fn(spec)."""
    return {
        name: _record(
            name,
            rate_fn(spec),
            0.2 if capture_fn is None else capture_fn(spec),
            0 if violations_fn is None else violations_fn(spec),
            n_ep,
        )
        for name, spec in R.CELLS.items()
    }


def _paired_ind(b, c, n):
    """Two length-n indicator lists with exactly `b` a-only and `c` b-only
    discordant pairs (and no concordant True pairs), so H-SPD-2's Wald
    interval can be pinned against a hand computation."""
    a = [i < b for i in range(n)]
    d = [b <= i < b + c for i in range(n)]
    return a, d


def _set_pair(recs, pursuer, attacker, ind_lo, ind_hi):
    """Overwrite the 0.8 and 1.2 cells of one (pursuer, attacker) with the
    given paired indicators."""
    for ratio, ind in ((R.SPD2_RATIOS[0], ind_lo), (R.SPD2_RATIOS[1], ind_hi)):
        rec = recs[R.cell_name(pursuer, attacker, ratio)]
        rec["deadlock_ind"] = list(ind)
        rec["n_episodes"] = len(ind)
        rec["deadlock_rate"] = sum(ind) / len(ind)
        rec["wilson_deadlock"] = R._ci(sum(ind), len(ind))


# A structured attacker sits above every scripted probe at every ratio, and its
# rate does not move with the ratio: the whole round CONFIRMED.
_ORDER = {"RCM28": 0.00, "FT-B-41": 0.06, "RC19": 0.12, "STR14": 0.12}


def _all_confirm(spec):
    base = 0.30 if spec["opponent"] in R.STRUCTURED else 0.05
    return base + _ORDER[spec["pursuer"]]


# ---------------------------------------------------------------- enumeration
class TestCells:
    def test_four_pursuers_four_opponents_four_ratios_is_sixty_four_cells(self):
        assert len(R.PURSUERS) == 4
        assert len(R.OPPONENTS) == 4
        assert R.RATIOS == (0.8, 1.0, 1.2, 1.4)
        assert len(R.CELLS) == 64

    def test_every_pursuer_opponent_pair_has_all_four_ratios(self):
        for p in R.PURSUERS:
            for o in R.OPPONENTS:
                rs = sorted(
                    s["ratio"]
                    for s in R.CELLS.values()
                    if s["pursuer"] == p and s["opponent"] == o
                )
                assert rs == sorted(R.RATIOS)

    def test_cell_names_carry_the_ratio(self):
        assert R.cell_name("RC19", "ATT25", 0.8) == "RC19__ATT25__r0.8"
        assert R.cell_name("FT-B-41", "GREEDY", 1.4) == "FT-B-41__GREEDY__r1.4"
        for name, spec in R.CELLS.items():
            assert name == R.cell_name(spec["pursuer"], spec["opponent"], spec["ratio"])

    def test_every_cell_is_the_base_shield_setting(self):
        for s in R.CELLS.values():
            assert (s["delta"], s["n"], s["gamma"]) == (0.0, 2, 0.2)

    def test_pre_registered_constants(self):
        assert R.SEED0 == 1_360_000
        assert R.N_EPISODES == 300
        assert R.P_GATE == 0.05
        assert R.EQ_MARGIN == 0.10
        assert R.MAX_WORKERS == 6
        assert R.SPD2_RATIOS == (0.8, 1.2)
        assert R.CONFIRM_MIN == 3
        assert R.STRUCTURED == ("ATT25", "ATK4-45")
        assert R.SCRIPTED == ("HUG7", "GREEDY")
        assert R.SPD3_OPPONENT == "GREEDY"
        assert R.SATURATION == 0.95

    def test_the_four_frozen_pursuers_and_their_shield_obs(self):
        assert R.PURSUERS["RC19"]["ckpt"].name == "RC19_10000000.zip"
        assert R.PURSUERS["RCM28"]["ckpt"].name == "RCM28_10000000.zip"
        assert R.PURSUERS["FT-B-41"]["ckpt"].name == "FT-B-41_5000000.zip"
        assert R.PURSUERS["STR14"]["ckpt"].name == "STR14_10000000.zip"
        assert R.PURSUERS["STR14"]["shield_obs"] is False  # trained unshielded
        for p in ("RC19", "RCM28", "FT-B-41"):
            assert R.PURSUERS[p]["shield_obs"] is True

    def test_the_two_learned_attackers_are_att25_and_atk4_45(self):
        assert R.OPPONENTS["ATT25"]["ckpt"].name == "CTL-D225_10000000.zip"
        assert R.OPPONENTS["ATK4-45"]["ckpt"].name == "ATK4-45_10000000.zip"
        for o in R.SCRIPTED:
            assert "scripted" in R.OPPONENTS[o]

    def test_every_pursuer_votes(self):
        # unlike S11, the spec scores all four pursuers (>= 3 of 4)
        assert tuple(R.SCORING) == tuple(R.PURSUERS)


# ------------------------------------------------------------------- cell cfg
class _Stop(Exception):
    """Raised by the evaluation stub once it has captured its arguments."""


class TestCellConfig:
    def _call(self, monkeypatch, name):
        seen = {}

        def fake_eval(model, cfg, opponent_factory, n_episodes, seed0, **kw):
            seen.update(
                cfg=cfg,
                opponent=opponent_factory(),
                n_episodes=n_episodes,
                seed0=seed0,
                kw=kw,
            )
            raise _Stop

        def fake_opponent(opponent, cfg_cell, device):
            seen["opponent_cfg"] = cfg_cell
            return lambda: ("opponent", opponent)

        monkeypatch.setattr("pe2.structured.evaluate_structured", fake_eval)
        monkeypatch.setattr(R, "_model", lambda ckpt, device: ("model", ckpt))
        monkeypatch.setattr(R, "opponent_factory", fake_opponent)
        with pytest.raises(_Stop):
            R.run_cell(name)
        return seen

    @pytest.mark.parametrize("ratio", [0.8, 1.0, 1.2, 1.4])
    def test_the_cell_config_carries_that_cells_ratio(self, monkeypatch, ratio):
        from pe2.config import EnvConfig

        seen = self._call(monkeypatch, R.cell_name("RC19", "ATT25", ratio))
        cfg = seen["cfg"]
        assert cfg.evader_speed_ratio == ratio
        assert cfg.evader_v_max == ratio * cfg.v_max
        assert cfg.v_max == EnvConfig().v_max  # the pursuer's cap never moves
        assert cfg.n_obstacles == 2

    def test_the_structured_opponent_is_built_on_the_same_cell_config(
        self, monkeypatch
    ):
        seen = self._call(monkeypatch, R.cell_name("STR14", "ATK4-45", 0.8))
        assert seen["opponent_cfg"] is seen["cfg"]
        assert seen["opponent"] == ("opponent", "ATK4-45")

    def test_the_stream_and_the_shield_obs_flag_reach_the_evaluation(self, monkeypatch):
        seen = self._call(monkeypatch, R.cell_name("STR14", "GREEDY", 1.4))
        assert seen["seed0"] == R.SEED0
        assert seen["n_episodes"] == R.N_EPISODES
        assert seen["kw"]["shield_obs"] is False  # STR14
        shield = seen["kw"]["shield_factory"]()
        assert (shield.gamma, shield.margin) == (R.GAMMA, R.DELTA)

    def test_the_structured_opponent_factory_passes_the_cell_config_through(
        self, monkeypatch
    ):
        import dataclasses

        from pe2.config import EnvConfig
        from pe2.evade import D_POS, StructuredEvaderPolicy

        monkeypatch.setattr(R.S9, "_model", lambda ckpt, device: object())
        cfg = dataclasses.replace(EnvConfig(), evader_speed_ratio=0.8)
        policy = R.opponent_factory("ATT25", cfg, "cpu")()
        assert isinstance(policy, StructuredEvaderPolicy)
        assert policy.cfg is cfg
        assert (policy.hold, policy.d_pos) == (10, D_POS)

    def test_a_scripted_opponent_factory_is_the_class_itself(self):
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        cfg = None
        assert R.opponent_factory("GREEDY", cfg, "cpu") is GreedyEvader
        assert R.opponent_factory("HUG7", cfg, "cpu") is ObstacleHuggingEvader

    def test_the_ratio_moves_no_rng_draw_so_the_ratios_stay_seed_paired(self):
        # H-SPD-2 is a PAIRED test across two ratios; it is only meaningful if
        # the layout and the initial poses of episode i are the same at both.
        check = R.check_ratio_pairing()
        assert check["pass"] is True
        assert check["ratios"] == list(R.RATIOS)


# -------------------------------------------------------------------- H-SPD-1
class TestSpd1:
    def test_attackers_above_the_scripted_probes_confirm_every_pursuer(self):
        s = R.spd1_verdicts(_records(_all_confirm))
        assert s["hypothesis"] == "H-SPD-1"
        assert s["verdict"] == "CONFIRMED"
        assert s["n_confirmed"] == 4
        for p in R.PURSUERS:
            assert s["pursuers"][p]["verdict"] == "CONFIRMED", p
            for r in R.RATIOS:
                block = s["pursuers"][p]["ratios"][R._rkey(r)]
                assert block["verdict"] == "CONFIRMED"
                for a in R.STRUCTURED:
                    assert block["attackers"][a]["pass"] is True
        assert all(s["ratios_holding"].values())

    def test_the_probe_is_measured_at_the_same_ratio(self):
        """Spec section 2: every comparison is against the scripted probe at
        the SAME ratio, so an attacker that has failed to transfer cannot be
        rescued (or condemned) by a probe measured somewhere else."""

        def rate(spec):
            # the scripted probes are strong at 0.8 and weak elsewhere
            if spec["opponent"] in R.SCRIPTED:
                return 0.50 if spec["ratio"] == 0.8 else 0.05
            return 0.30

        s = R.spd1_verdicts(_records(rate))
        lo = s["pursuers"]["RC19"]["ratios"][R._rkey(0.8)]
        hi = s["pursuers"]["RC19"]["ratios"][R._rkey(1.2)]
        assert lo["best_scripted_cell"].endswith("r0.8")
        assert hi["best_scripted_cell"].endswith("r1.2")
        assert lo["attackers"]["ATT25"]["rate_b"] == pytest.approx(0.50)
        assert hi["attackers"]["ATT25"]["rate_b"] == pytest.approx(0.05)
        assert lo["verdict"] == "NOT CONFIRMED"
        assert hi["verdict"] == "CONFIRMED"

    def test_the_probe_is_the_higher_scripted_rate_of_the_two(self):
        def rate(spec):
            if spec["opponent"] == "GREEDY":
                return 0.25
            if spec["opponent"] == "HUG7":
                return 0.05
            return 0.40

        s = R.spd1_verdicts(_records(rate))
        block = s["pursuers"]["RC19"]["ratios"][R._rkey(1.2)]
        assert block["best_scripted_opponent"] == "GREEDY"
        assert block["attackers"]["ATT25"]["rate_b"] == pytest.approx(0.25)
        assert block["scripted_rates"]["HUG7"] == pytest.approx(0.05)

    def test_attackers_at_the_scripted_rate_fail(self):
        s = R.spd1_verdicts(_records(lambda spec: 0.30))
        assert s["verdict"] == "NOT CONFIRMED"
        assert s["n_confirmed"] == 0

    def test_one_failing_ratio_fails_the_pursuer(self):
        def rate(spec):
            if (
                spec["pursuer"] == "RC19"
                and spec["opponent"] == "ATT25"
                and spec["ratio"] == 0.8
            ):
                return 0.05  # down to the scripted level
            return _all_confirm(spec)

        s = R.spd1_verdicts(_records(rate))
        assert s["pursuers"]["RC19"]["verdict"] == "NOT CONFIRMED"
        assert s["pursuers"]["RC19"]["ratios"][R._rkey(0.8)]["verdict"] == (
            "NOT CONFIRMED"
        )
        assert s["pursuers"]["RC19"]["ratios"][R._rkey(1.2)]["verdict"] == "CONFIRMED"
        # three of four pursuers still confirm -> overall CONFIRMED
        assert s["verdict"] == "CONFIRMED"
        assert s["n_confirmed"] == 3

    def test_both_attackers_are_required(self):
        def rate(spec):
            if spec["opponent"] == "ATK4-45":
                return 0.05
            return _all_confirm(spec)

        s = R.spd1_verdicts(_records(rate))
        assert s["verdict"] == "NOT CONFIRMED"
        assert s["n_confirmed"] == 0
        block = s["pursuers"]["RC19"]["ratios"][R._rkey(1.0)]
        assert block["attackers"]["ATT25"]["pass"] is True
        assert block["attackers"]["ATK4-45"]["pass"] is False

    def test_two_failing_pursuers_fail_the_round(self):
        def rate(spec):
            if spec["pursuer"] in ("RC19", "STR14") and spec["opponent"] == "ATT25":
                return 0.05
            return _all_confirm(spec)

        s = R.spd1_verdicts(_records(rate))
        assert s["n_confirmed"] == 2
        assert s["verdict"] == "NOT CONFIRMED"

    def test_a_ratio_holds_when_three_of_four_pursuers_confirm_there(self):
        def rate(spec):
            if spec["pursuer"] == "RCM28" and spec["ratio"] == 0.8:
                return 0.05
            return _all_confirm(spec)

        s = R.spd1_verdicts(_records(rate))
        assert s["ratios_holding"][R._rkey(0.8)] is True
        assert s["n_confirmed_by_ratio"][R._rkey(0.8)] == 3

    def test_a_ratio_fails_when_two_pursuers_fail_there(self):
        def rate(spec):
            if spec["pursuer"] in ("RCM28", "STR14") and spec["ratio"] == 0.8:
                return 0.05
            return _all_confirm(spec)

        s = R.spd1_verdicts(_records(rate))
        assert s["ratios_holding"][R._rkey(0.8)] is False
        assert s["ratios_holding"][R._rkey(1.2)] is True

    def test_missing_cells_make_it_incomplete(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RC19", "HUG7", 1.0)]
        s = R.spd1_verdicts(recs)
        assert s["pursuers"]["RC19"]["ratios"][R._rkey(1.0)]["verdict"] == "INCOMPLETE"
        assert s["pursuers"]["RC19"]["verdict"] == "INCOMPLETE"
        assert s["verdict"] == "INCOMPLETE"
        assert s["ratios_holding"][R._rkey(1.0)] is None


# -------------------------------------------------------------------- H-SPD-2
class TestSpd2:
    def test_the_wald_interval_matches_a_hand_computation(self):
        # b = 6, c = 2, n = 100:
        #   d  = (6 - 2) / 100                          = 0.04
        #   hw = 1.96 * sqrt(6 + 2 - 16 / 100) / 100
        #      = 1.96 * sqrt(7.84) / 100 = 1.96 * 2.8 / 100 = 0.05488
        a, b = _paired_ind(6, 2, 100)
        out = R.wald_equivalence(a, b)
        assert out["b"] == 6 and out["c"] == 2 and out["n"] == 100
        assert out["d"] == pytest.approx(0.04)
        assert out["half_width"] == pytest.approx(0.05488)
        assert out["interval"][0] == pytest.approx(0.04 - 0.05488)
        assert out["interval"][1] == pytest.approx(0.04 + 0.05488)
        assert out["margin"] == 0.10
        assert out["pass"] is True

    def test_the_half_width_is_the_pre_registered_formula(self):
        for b, c, n in ((0, 0, 300), (17, 5, 300), (3, 41, 120), (60, 60, 200)):
            a, d = _paired_ind(b, c, n)
            out = R.wald_equivalence(a, d)
            want = 1.96 * math.sqrt(b + c - (b - c) ** 2 / n) / n
            assert out["half_width"] == pytest.approx(want)
            assert out["d"] == pytest.approx((b - c) / n)

    def test_a_wide_interval_straddling_the_margin_does_not_confirm(self):
        # b = c = 30, n = 100: d = 0, hw = 1.96 * sqrt(60) / 100 = 0.1518
        a, b = _paired_ind(30, 30, 100)
        out = R.wald_equivalence(a, b)
        assert out["d"] == pytest.approx(0.0)
        assert out["half_width"] == pytest.approx(1.96 * math.sqrt(60) / 100)
        assert out["interval"][0] < -0.10 < 0.10 < out["interval"][1]
        assert out["pass"] is False

    def test_a_large_difference_does_not_confirm(self):
        a, b = _paired_ind(60, 2, 300)
        assert R.wald_equivalence(a, b)["pass"] is False

    def test_identical_ratios_confirm_every_pursuer(self):
        s = R.spd2_verdicts(_records(_all_confirm))
        assert s["hypothesis"] == "H-SPD-2"
        assert s["ratios"] == [0.8, 1.2]
        assert s["margin"] == 0.10
        assert s["verdict"] == "CONFIRMED"
        assert s["n_confirmed"] == 4
        for p in R.PURSUERS:
            for a in R.STRUCTURED:
                block = s["pursuers"][p]["attackers"][a]
                assert block["pass"] is True
                assert block["d"] == pytest.approx(0.0)
                assert block["cells"] == [
                    R.cell_name(p, a, 0.8),
                    R.cell_name(p, a, 1.2),
                ]

    def test_the_difference_is_zero_point_eight_minus_one_point_two(self):
        recs = _records(_all_confirm)
        _set_pair(recs, "RC19", "ATT25", *_paired_ind(6, 2, 100))
        block = R.spd2_verdicts(recs)["pursuers"]["RC19"]["attackers"]["ATT25"]
        assert block["d"] == pytest.approx(0.04)
        assert block["rate_a"] == pytest.approx(0.06)  # the 0.8 cell
        assert block["rate_b"] == pytest.approx(0.02)  # the 1.2 cell
        assert block["pass"] is True

    def test_a_wide_pursuer_fails_and_three_others_still_carry_the_round(self):
        recs = _records(_all_confirm)
        _set_pair(recs, "RC19", "ATT25", *_paired_ind(30, 30, 100))
        s = R.spd2_verdicts(recs)
        assert s["pursuers"]["RC19"]["verdict"] == "NOT CONFIRMED"
        assert s["pursuers"]["RC19"]["attackers"]["ATK4-45"]["pass"] is True
        assert s["n_confirmed"] == 3
        assert s["verdict"] == "CONFIRMED"

    def test_two_failing_pursuers_fail_the_round(self):
        recs = _records(_all_confirm)
        for p in ("RC19", "STR14"):
            _set_pair(recs, p, "ATK4-45", *_paired_ind(30, 30, 100))
        s = R.spd2_verdicts(recs)
        assert s["n_confirmed"] == 2
        assert s["verdict"] == "NOT CONFIRMED"

    def test_only_the_learned_attackers_are_read(self):
        s = R.spd2_verdicts(_records(_all_confirm))
        assert set(s["pursuers"]["RC19"]["attackers"]) == set(R.STRUCTURED)

    def test_missing_cells_make_it_incomplete(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("STR14", "ATK4-45", 0.8)]
        s = R.spd2_verdicts(recs)
        assert s["pursuers"]["STR14"]["attackers"]["ATK4-45"]["verdict"] == (
            "INCOMPLETE"
        )
        assert s["pursuers"]["STR14"]["verdict"] == "INCOMPLETE"
        assert s["verdict"] == "INCOMPLETE"


# -------------------------------------------------------------------- H-SPD-3
class TestSpd3:
    def test_capture_against_greedy_is_reported_with_no_verdict(self):
        report = R.spd3_report(_records(_all_confirm))
        assert report["hypothesis"] == "H-SPD-3"
        assert report["opponent"] == "GREEDY"
        assert "verdict" not in report
        assert set(report["capture_rates"]) == set(R.PURSUERS)
        for p in R.PURSUERS:
            assert set(report["capture_rates"][p]) == {R._rkey(r) for r in R.RATIOS}

    def test_the_saturation_flag_fires_above_the_threshold(self):
        report = R.spd3_report(
            _records(
                _all_confirm,
                capture_fn=lambda spec: 0.96 if spec["ratio"] == 0.8 else 0.4,
            )
        )
        assert report["saturated"] is True
        assert report["saturation_threshold"] == 0.95
        for p in R.PURSUERS:
            assert report["capture_at_0.8"][p] == pytest.approx(0.96)

    def test_one_pursuer_below_the_threshold_clears_the_flag(self):
        def capture(spec):
            if spec["ratio"] != 0.8:
                return 0.4
            return 0.90 if spec["pursuer"] == "STR14" else 0.96

        report = R.spd3_report(_records(_all_confirm, capture_fn=capture))
        assert report["saturated"] is False

    def test_exactly_the_threshold_is_not_saturation(self):
        report = R.spd3_report(
            _records(
                _all_confirm,
                capture_fn=lambda spec: 0.95 if spec["ratio"] == 0.8 else 0.4,
                n_ep=100,
            )
        )
        assert report["saturated"] is False

    def test_capture_falling_with_the_ratio_is_reported_as_a_negative_rho(self):
        report = R.spd3_report(
            _records(
                _all_confirm,
                capture_fn=lambda spec: 0.9 - 0.2 * R.RATIOS.index(spec["ratio"]),
            )
        )
        assert report["rho_capture_vs_ratio"]["RC19"] < 0

    def test_missing_cells_are_named_and_the_flag_is_unset(self):
        recs = _records(_all_confirm)
        missing = R.cell_name("RCM28", "GREEDY", 0.8)
        del recs[missing]
        report = R.spd3_report(recs)
        assert report["missing"] == [missing]
        assert report["saturated"] is None
        assert report["capture_rates"]["RCM28"][R._rkey(0.8)] is None


# --------------------------------------------------------------------- H-SAFE
class TestHSafe:
    def test_holds_on_a_clean_full_round(self):
        h = R.hsafe(_records(_all_confirm))
        assert h["verdict"] == "HOLDS"
        assert h["violation_steps"] == 0
        assert h["offenders"] == []

    def test_fails_and_names_the_offending_cell(self):
        bad = R.cell_name("STR14", "GREEDY", 0.8)
        h = R.hsafe(
            _records(_all_confirm, violations_fn=lambda s: 3 if s["name"] == bad else 0)
        )
        assert h["verdict"] == "FAILS"
        assert h["violation_steps"] == 3
        assert h["offenders"] == [bad]

    def test_incomplete_when_cells_are_missing_and_none_offend(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RC19", "ATT25", 1.4)]
        assert R.hsafe(recs)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------ direction
def _spoil_spd2(recs, pursuers):
    """Widen the 0.8-vs-1.2 interval past the margin while leaving H-SPD-1
    intact: b = c = 140 of 300 gives d = 0 and hw = 1.96 * sqrt(280) / 300 =
    0.109, and both cells still deadlock far above the scripted probes. The
    length stays 300 so every other paired test in the round remains valid."""
    for p in pursuers:
        _set_pair(recs, p, "ATT25", *_paired_ind(140, 140, 300))
    return recs


class TestDirection:
    def test_both_confirmed_closes_the_reviewer_objection(self):
        d = R.direction(R.verdicts(_records(_all_confirm)))
        assert "not come from the speed asymmetry" in d
        assert "closed" in d

    def test_spd1_confirmed_and_spd2_not_reports_the_curve(self):
        recs = _spoil_spd2(_records(_all_confirm), ("RC19", "RCM28"))
        v = R.verdicts(recs)
        assert v["spd1"]["verdict"] == "CONFIRMED"
        assert v["spd2"]["verdict"] == "NOT CONFIRMED"
        d = R.direction(v)
        assert "report the curve" in d.lower()
        assert "independent of speed" in d

    def test_failing_at_zero_point_eight_only_narrows_the_scope(self):
        def rate(spec):
            if spec["ratio"] == 0.8 and spec["opponent"] in R.STRUCTURED:
                return 0.05
            return _all_confirm(spec)

        v = R.verdicts(_records(rate))
        assert v["spd1"]["verdict"] == "NOT CONFIRMED"
        d = R.direction(v)
        assert "cannot be run down" in d
        assert "scope" in d

    def test_failing_at_one_point_two_is_a_harness_failure(self):
        def rate(spec):
            if spec["ratio"] == 1.2 and spec["opponent"] in R.STRUCTURED:
                return 0.05
            return _all_confirm(spec)

        d = R.direction(R.verdicts(_records(rate)))
        assert "harness" in d.lower()
        assert "report nothing" in d

    def test_a_violation_stops_the_round_before_anything_else_is_read(self):
        bad = R.cell_name("RC19", "ATT25", 1.0)
        recs = _records(
            _all_confirm, violations_fn=lambda s: 1 if s["name"] == bad else 0
        )
        d = R.direction(R.verdicts(recs))
        assert "H-SAFE" in d
        assert "Stop" in d
        assert bad in d

    def test_an_incomplete_round_reads_nothing(self):
        recs = _records(_all_confirm)
        del recs[R.cell_name("RC19", "ATT25", 1.0)]
        assert R.direction(R.verdicts(recs)).startswith("incomplete")


# ------------------------------------------------------------- cell bookkeeping
class TestRunCells:
    def test_skips_cells_that_already_have_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        done = R.cell_name("RC19", "ATT25", 0.8)
        todo = R.cell_name("RC19", "ATT25", 1.0)
        R.cells_dir().mkdir(parents=True)
        (R.cells_dir() / f"{done}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells([done, todo], workers=1)
        assert called == [todo]
        assert json.loads((R.cells_dir() / f"{todo}.json").read_text())["name"] == todo
        assert (R.cells_dir() / f"{done}.json").read_text() == "{}"

    def test_no_names_runs_every_missing_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        done = list(R.CELLS)[:5]
        for name in done:
            (R.cells_dir() / f"{name}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells(None, workers=1)
        assert len(called) == len(R.CELLS) - 5
        assert set(called).isdisjoint(done)

    def test_unknown_cell_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit):
            R.run_cells(["NOPE"], workers=1)

    def test_cells_refuse_to_run_when_the_ratios_are_not_seed_paired(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R, "check_ratio_pairing", lambda: {"pass": False, "ratios": list(R.RATIOS)}
        )
        monkeypatch.setattr(R, "run_cell", lambda name, device="cpu": None)
        with pytest.raises(SystemExit, match="paired"):
            R.run_cells(None, workers=1)

    def test_records_are_written_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        name = R.cell_name("RC19", "ATT25", 1.2)
        R.cells_dir().mkdir(parents=True)
        R.write_cell(_record(name))
        assert json.loads((R.cells_dir() / f"{name}.json").read_text())["name"] == name
        assert list(R.cells_dir().glob("*.tmp")) == []

    def test_the_cells_directory_carries_the_s13_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.cells_dir() == tmp_path / "s13_cells"
        assert R.ARTEFACT_PREFIX == "s13"


# -------------------------------------------------------------------- summarize
class TestSummarize:
    def test_writes_round_json_and_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        for rec in _records(_all_confirm).values():
            R.write_cell(rec)
        summary = R.summarize()
        assert (tmp_path / "s13_round.json").exists()
        assert (tmp_path / "s13_round.png").exists()
        blob = json.loads((tmp_path / "s13_round.json").read_text())
        assert blob["verdicts"]["spd1"]["verdict"] == "CONFIRMED"
        assert blob["verdicts"]["spd2"]["verdict"] == "CONFIRMED"
        assert blob["verdicts"]["hsafe"]["verdict"] == "HOLDS"
        assert blob["verdicts"]["complete"] is True
        assert summary["direction"] == R.direction(summary["verdicts"])
        assert len(blob["cells"]) == 64
        first = next(iter(blob["cells"].values()))
        assert "deadlock_ind" not in first and "capture_ind" not in first

    def test_a_partial_round_still_draws(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        recs = _records(_all_confirm)
        for name in list(recs)[:6]:
            R.write_cell(recs[name])
        blob = R.summarize()
        assert (tmp_path / "s13_round.png").exists()
        assert blob["verdicts"]["complete"] is False
