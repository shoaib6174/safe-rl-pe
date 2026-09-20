"""scripts/derisk_s15_mechanism.py — the S15 loop-mechanism round's own decisions.

Spec: docs/pathway/loop_mechanism_prereg_2026-09-20.md.
Nothing here touches a checkpoint or runs an episode: every test drives the
enumeration, the entry/escape definitions, the return-to-go arithmetic, the four
verdict rules and the cell bookkeeping on hand-built traces and fabricated
records.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s15_mechanism.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s15_mechanism", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()

V_MAX = 1.0
STOP = 0.0  # immobile: v_exec <= 0.05 * v_max
GO = 0.5  # moving: v_exec > 0.1 * v_max


# --------------------------------------------------------------- record fakes
def _entry(
    episode=0,
    step=100,
    escaped=False,
    steps_to_escape=None,
    outcome="timeout",
    steps_remaining=300,
    return_to_go=None,
    action_deltas_median=0.0,
    obs_deltas_median=0.02,
    visible_fraction=0.0,
):
    return {
        "episode": episode,
        "step": step,
        "escaped": escaped,
        "steps_to_escape": steps_to_escape,
        "outcome": outcome,
        "steps_remaining": steps_remaining,
        "return_to_go": (
            R.return_to_go(outcome, steps_remaining)
            if return_to_go is None
            else return_to_go
        ),
        "action_deltas_median": action_deltas_median,
        "obs_deltas_median": obs_deltas_median,
        "visible_fraction": visible_fraction,
    }


def _record(
    name,
    entries=(),
    immobile_step_rate=0.4,
    violations=0,
    sigma_stalled=0.35,
    sigma_free=0.35,
    sigma_state_dependent=False,
    capture_rate=0.1,
):
    spec = R.CELLS[name]
    return {
        "name": name,
        "policy": spec["policy"],
        "opponent": spec["opponent"],
        "hold": spec["hold"],
        "n": R.N_EPISODES,
        "seed0": R.SEED0,
        "entries": [dict(e) for e in entries],
        "per_episode": [
            {
                "seed": R.SEED0 + i,
                "outcome": "timeout",
                "steps": 1200,
                "n_immobile": 480,
                "n_stalled": 400,
                "longest_immobile_run": 300,
                "longest_stall_run": 200,
                "n_entries": 0,
                "violations": 0,
            }
            for i in range(2)
        ],
        "immobile_step_rate": immobile_step_rate,
        "deadlock_episode_rate": 0.9,
        "capture_rate": capture_rate,
        "violations": violations,
        "sigma_stalled": sigma_stalled,
        "sigma_free": sigma_free,
        "sigma_state_dependent": sigma_state_dependent,
        "provenance": {"git_commit": "deadbee"},
    }


def _entries_for(delta_rtg, capture_rate_escaped, action_delta, n_each=20):
    """Entries whose escaped group beats the non-escaped group by `delta_rtg` in
    discounted return-to-go and captures at `capture_rate_escaped`."""
    out = []
    for i in range(n_each):
        out.append(
            _entry(
                episode=i,
                escaped=True,
                steps_to_escape=12,
                outcome="capture"
                if i < round(capture_rate_escaped * n_each)
                else "timeout",
                return_to_go=delta_rtg,
                action_deltas_median=action_delta,
            )
        )
        out.append(
            _entry(
                episode=i,
                escaped=False,
                return_to_go=0.0,
                action_deltas_median=action_delta,
            )
        )
    return out


def _records(
    delta_rtg=0.0,
    capture_rate_escaped=0.0,
    action_delta=0.0,
    immobile_h10=0.40,
    immobile_h1=0.40,
    sigma_stalled=0.35,
    sigma_free=0.35,
    sigma_state_dependent=False,
    violations_fn=None,
):
    """A full 8-cell record set under the named behaviour."""
    recs = {}
    for name, spec in R.CELLS.items():
        main = spec["hold"] == R.MAIN_HOLD
        recs[name] = _record(
            name,
            entries=_entries_for(delta_rtg, capture_rate_escaped, action_delta)
            if main
            else [],
            immobile_step_rate=immobile_h10 if main else immobile_h1,
            violations=0 if violations_fn is None else violations_fn(spec),
            sigma_stalled=sigma_stalled,
            sigma_free=sigma_free,
            sigma_state_dependent=sigma_state_dependent,
        )
    return recs


# ---------------------------------------------------------------- enumeration
class TestCells:
    def test_eight_cells_two_policies_two_opponents_two_holds(self):
        assert len(R.POLICIES) == 2
        assert len(R.OPPONENTS) == 2
        assert R.HOLDS == (10, 1)
        assert len(R.CELLS) == 8
        for hold in R.HOLDS:
            assert len([s for s in R.CELLS.values() if s["hold"] == hold]) == 4

    def test_cell_naming(self):
        assert R.cell_name("RC19", "ATK4-45", 10) == "RC19__ATK4-45__h10"
        assert R.cell_name("FT-B-41", "GREEDY", 1) == "FT-B-41__GREEDY__h1"
        for name, spec in R.CELLS.items():
            assert name == R.cell_name(spec["policy"], spec["opponent"], spec["hold"])

    def test_policies_and_opponents_are_the_pre_registered_ones(self):
        assert set(R.POLICIES) == {"FT-B-41", "RC19"}
        assert set(R.OPPONENTS) == {"ATK4-45", "GREEDY"}
        assert R.POLICIES["FT-B-41"]["ckpt"].name == "FT-B-41_5000000.zip"
        assert R.POLICIES["RC19"]["ckpt"].name == "RC19_10000000.zip"
        assert all(p["shield_obs"] for p in R.POLICIES.values())

    def test_constants_are_the_pre_registered_ones(self):
        assert R.SEED0 == 1_710_000
        assert R.N_EPISODES == 200
        assert R.GAMMA == 0.99
        assert R.ENTRY_IMMOBILE_STEPS == 10
        assert R.ESCAPE_WINDOW == 40
        assert R.ESCAPE_MOVING_STEPS == 5
        assert R.ESCAPE_V_FRAC == 0.1
        assert R.P1_RTG_BAR == 0.05
        assert R.P1_CAPTURE_BAR == 0.15
        assert R.P3_SIGMA_RATIO_BAR == 0.8
        assert R.P4_IMMOBILE_BAR == 0.05
        assert R.MAIN_HOLD == 10


# ------------------------------------------------------------- stall entries
class TestStallEntries:
    def test_nine_immobile_steps_are_not_an_entry(self):
        assert R.stall_entries([True] * 9) == []

    def test_ten_immobile_steps_are_an_entry(self):
        assert R.stall_entries([True] * 10) == [(0, 10)]

    def test_entry_is_the_first_step_of_the_run(self):
        assert R.stall_entries([False] * 7 + [True] * 12) == [(7, 12)]

    def test_two_runs_separated_by_motion_are_two_entries(self):
        trace = [True] * 10 + [False] * 3 + [True] * 11
        assert R.stall_entries(trace) == [(0, 10), (13, 11)]

    def test_a_short_run_between_two_long_ones_is_not_an_entry(self):
        trace = [True] * 10 + [False] + [True] * 9 + [False] + [True] * 10
        assert R.stall_entries(trace) == [(0, 10), (21, 10)]

    def test_an_empty_trace_has_no_entries(self):
        assert R.stall_entries([]) == []


# ------------------------------------------------------------------- escapes
def _trace(entry_len, gap, move_len, tail=5):
    """v_exec for one stall run of `entry_len` immobile steps, `gap` further
    stopped steps, then `move_len` moving steps. The moving run therefore begins
    `entry_len + gap` steps after the entry (which is step 0)."""
    return [STOP] * entry_len + [STOP] * gap + [GO] * move_len + [STOP] * tail


class TestEscape:
    def test_five_moving_steps_are_an_escape(self):
        v = _trace(10, 0, 5)
        assert R.escape_after(0, v, V_MAX) == (True, 10)

    def test_four_moving_steps_are_not_an_escape(self):
        v = _trace(10, 0, 4)
        assert R.escape_after(0, v, V_MAX) == (False, None)

    def test_an_escape_beginning_at_exactly_forty_steps_counts(self):
        v = _trace(10, 30, 5)  # moving run starts at index 40
        assert R.escape_after(0, v, V_MAX) == (True, 40)

    def test_an_escape_beginning_at_forty_one_steps_does_not_count(self):
        v = _trace(10, 31, 5)  # moving run starts at index 41
        assert R.escape_after(0, v, V_MAX) == (False, None)

    def test_the_window_is_measured_from_the_entry_not_from_step_zero(self):
        v = [GO] * 20 + _trace(10, 30, 5)
        assert R.escape_after(20, v, V_MAX) == (True, 40)

    def test_speed_must_exceed_a_tenth_of_v_max(self):
        assert R.escape_after(0, [STOP] * 10 + [0.10] * 5, V_MAX) == (False, None)
        assert R.escape_after(0, [STOP] * 10 + [0.11] * 5, V_MAX) == (True, 10)

    def test_the_first_qualifying_run_is_the_escape(self):
        v = [STOP] * 10 + [GO] * 4 + [STOP] * 2 + [GO] * 6
        assert R.escape_after(0, v, V_MAX) == (True, 16)

    def test_a_moving_run_cut_short_by_the_episode_end_is_not_an_escape(self):
        assert R.escape_after(0, [STOP] * 10 + [GO] * 4, V_MAX) == (False, None)


# -------------------------------------------------------------- return-to-go
class TestReturnToGo:
    def test_capture_three_hundred_steps_after_the_entry(self):
        # entry at step 600, capture at step 900 -> 300 steps remaining
        assert R.return_to_go("capture", 900 - 600) == pytest.approx(0.99**300)

    def test_timeout_is_the_negative_of_the_same_discount(self):
        assert R.return_to_go("timeout", 300) == pytest.approx(-(0.99**300))
        assert R.return_to_go("timeout", 7) == pytest.approx(-(0.99**7))

    def test_zero_steps_remaining_is_the_undiscounted_terminal(self):
        assert R.return_to_go("capture", 0) == pytest.approx(1.0)
        assert R.return_to_go("timeout", 0) == pytest.approx(-1.0)

    def test_an_unknown_outcome_is_refused(self):
        with pytest.raises(ValueError):
            R.return_to_go("crash", 10)


# ------------------------------------------------------------------------ M1
class TestM1:
    def test_holds_when_the_escape_advantage_is_negligible(self):
        v = R.m1_verdict(_records(delta_rtg=0.01, capture_rate_escaped=0.05))
        assert v["verdict"] == "HOLDS"
        assert v["delta_return_to_go"] == pytest.approx(0.01)
        assert v["capture_rate_escaped"] == pytest.approx(0.05)

    def test_fails_when_the_discounted_advantage_clears_the_bar(self):
        v = R.m1_verdict(_records(delta_rtg=0.2, capture_rate_escaped=0.05))
        assert v["verdict"] == "FAILS"

    def test_fails_when_escapes_are_followed_by_capture(self):
        v = R.m1_verdict(_records(delta_rtg=0.0, capture_rate_escaped=0.5))
        assert v["verdict"] == "FAILS"

    def test_a_negative_advantage_of_the_same_size_also_fails(self):
        v = R.m1_verdict(_records(delta_rtg=-0.2, capture_rate_escaped=0.0))
        assert v["verdict"] == "FAILS"

    def test_reads_only_the_main_hold_cells(self):
        recs = _records(delta_rtg=0.0, capture_rate_escaped=0.0)
        for name, spec in R.CELLS.items():
            if spec["hold"] != R.MAIN_HOLD:
                recs[name]["entries"] = _entries_for(0.9, 0.9, 0.0)
        assert R.m1_verdict(recs)["verdict"] == "HOLDS"
        assert set(R.m1_verdict(recs)["per_cell"]) == {
            n for n, s in R.CELLS.items() if s["hold"] == R.MAIN_HOLD
        }

    def test_incomplete_when_a_main_cell_is_missing(self):
        recs = _records()
        del recs[R.cell_name("RC19", "ATK4-45", 10)]
        assert R.m1_verdict(recs)["verdict"] == "INCOMPLETE"

    def test_incomplete_when_a_group_is_empty(self):
        recs = _records()
        for name, spec in R.CELLS.items():
            if spec["hold"] == R.MAIN_HOLD:
                recs[name]["entries"] = [
                    e for e in recs[name]["entries"] if not e["escaped"]
                ]
        assert R.m1_verdict(recs)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------------ M2
class TestM2:
    def test_holds_when_the_emitted_action_is_near_constant(self):
        v = R.m2_verdict(_records(action_delta=0.001))
        assert v["verdict"] == "HOLDS"
        assert v["action_delta_median"] == pytest.approx(0.001)

    def test_fails_when_the_action_varies(self):
        v = R.m2_verdict(_records(action_delta=0.2))
        assert v["verdict"] == "FAILS"

    def test_the_bar_is_strict(self):
        assert R.m2_verdict(_records(action_delta=0.01))["verdict"] == "FAILS"
        assert R.m2_verdict(_records(action_delta=0.0099))["verdict"] == "HOLDS"

    def test_the_observation_change_is_reported_and_not_gated(self):
        recs = _records(action_delta=0.001)
        for name, spec in R.CELLS.items():
            if spec["hold"] == R.MAIN_HOLD:
                for e in recs[name]["entries"]:
                    e["obs_deltas_median"] = 0.5
        v = R.m2_verdict(recs)
        assert v["verdict"] == "HOLDS"
        assert v["obs_delta_median"] == pytest.approx(0.5)
        assert "visible_fraction" in v

    def test_incomplete_without_entries(self):
        recs = _records(action_delta=0.001)
        for name in recs:
            recs[name]["entries"] = []
        assert R.m2_verdict(recs)["verdict"] == "INCOMPLETE"

    def test_incomplete_when_a_main_cell_is_missing(self):
        recs = _records(action_delta=0.001)
        del recs[R.cell_name("FT-B-41", "GREEDY", 10)]
        assert R.m2_verdict(recs)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------------ M3
class TestM3:
    def test_state_independent_sigma_is_reported_as_not_applicable(self):
        v = R.m3_verdict(_records(sigma_state_dependent=False))
        assert v["verdict"] == "N/A (state-independent sigma)"
        assert v["state_dependent"] is False
        assert v["ratio"] is None

    def test_holds_when_sigma_at_stalls_is_not_materially_lower(self):
        v = R.m3_verdict(
            _records(sigma_stalled=0.9, sigma_free=1.0, sigma_state_dependent=True)
        )
        assert v["verdict"] == "HOLDS"
        assert v["ratio"] == pytest.approx(0.9)

    def test_fails_when_sigma_collapses_at_stalls(self):
        v = R.m3_verdict(
            _records(sigma_stalled=0.3, sigma_free=1.0, sigma_state_dependent=True)
        )
        assert v["verdict"] == "FAILS"
        assert v["ratio"] == pytest.approx(0.3)

    def test_the_bar_is_strict(self):
        at = _records(sigma_stalled=0.8, sigma_free=1.0, sigma_state_dependent=True)
        assert R.m3_verdict(at)["verdict"] == "FAILS"
        above = _records(
            sigma_stalled=0.801, sigma_free=1.0, sigma_state_dependent=True
        )
        assert R.m3_verdict(above)["verdict"] == "HOLDS"

    def test_incomplete_when_a_main_cell_is_missing(self):
        recs = _records(sigma_state_dependent=True)
        del recs[R.cell_name("RC19", "GREEDY", 10)]
        assert R.m3_verdict(recs)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------------ M4
class TestM4:
    def test_holds_when_the_hold_does_not_change_immobility(self):
        v = R.m4_verdict(_records(immobile_h10=0.40, immobile_h1=0.42))
        assert v["verdict"] == "HOLDS"
        assert len(v["pairs"]) == 4
        assert all(p["delta"] == pytest.approx(0.02) for p in v["pairs"])

    def test_fails_when_hold_one_removes_immobility(self):
        v = R.m4_verdict(_records(immobile_h10=0.40, immobile_h1=0.10))
        assert v["verdict"] == "FAILS"

    def test_fails_when_hold_one_adds_immobility(self):
        assert (
            R.m4_verdict(_records(immobile_h10=0.10, immobile_h1=0.40))["verdict"]
            == "FAILS"
        )

    def test_the_bar_is_strict(self):
        assert (
            R.m4_verdict(_records(immobile_h10=0.40, immobile_h1=0.35))["verdict"]
            == "FAILS"
        )
        assert (
            R.m4_verdict(_records(immobile_h10=0.40, immobile_h1=0.3501))["verdict"]
            == "HOLDS"
        )

    def test_one_offending_pair_fails_the_whole_measurement(self):
        recs = _records(immobile_h10=0.40, immobile_h1=0.40)
        recs[R.cell_name("RC19", "GREEDY", 1)]["immobile_step_rate"] = 0.0
        v = R.m4_verdict(recs)
        assert v["verdict"] == "FAILS"
        assert [p["pass"] for p in v["pairs"]].count(False) == 1

    def test_incomplete_when_a_hold_one_cell_is_missing(self):
        recs = _records()
        del recs[R.cell_name("FT-B-41", "ATK4-45", 1)]
        assert R.m4_verdict(recs)["verdict"] == "INCOMPLETE"


# ---------------------------------------------------------------------- H-SAFE
class TestHSafe:
    def test_holds_on_a_clean_full_round(self):
        h = R.hsafe(_records())
        assert h["verdict"] == "HOLDS"
        assert h["violation_steps"] == 0
        assert h["offenders"] == []

    def test_fails_and_names_the_offending_cell(self):
        bad = R.cell_name("RC19", "GREEDY", 1)
        h = R.hsafe(_records(violations_fn=lambda s: 4 if s["name"] == bad else 0))
        assert h["verdict"] == "FAILS"
        assert h["violation_steps"] == 4
        assert h["offenders"] == [bad]

    def test_incomplete_when_cells_are_missing_and_none_offend(self):
        recs = _records()
        del recs[R.cell_name("RC19", "ATK4-45", 10)]
        assert R.hsafe(recs)["verdict"] == "INCOMPLETE"


# -------------------------------------------------------------------- direction
class TestDirection:
    def test_the_predicted_pattern_returns_the_first_row_only(self):
        rows = R.direction(
            {
                "m1": {"verdict": "HOLDS"},
                "m2": {"verdict": "HOLDS"},
                "m3": {"verdict": "FAILS"},
                "m4": {"verdict": "FAILS"},
            }
        )
        assert [r["key"] for r in rows] == ["predicted"]
        assert "Phase 2's arms are then well motivated" in rows[0]["direction"]

    def test_p1_failing_selects_the_optimisation_row(self):
        rows = R.direction(
            {
                "m1": {"verdict": "FAILS"},
                "m2": {"verdict": "HOLDS"},
                "m3": {"verdict": "FAILS"},
                "m4": {"verdict": "FAILS"},
            }
        )
        keys = [r["key"] for r in rows]
        assert "predicted" not in keys
        assert "p1_fails" in keys
        assert (
            "optimisation, not credit assignment"
            in rows[keys.index("p1_fails")]["direction"]
        )

    def test_every_failing_prediction_contributes_its_row(self):
        rows = R.direction(
            {
                "m1": {"verdict": "FAILS"},
                "m2": {"verdict": "FAILS"},
                "m3": {"verdict": "FAILS"},
                "m4": {"verdict": "FAILS"},
            }
        )
        assert [r["key"] for r in rows] == [
            "p1_fails",
            "p2_fails",
            "p3_fails",
            "p4_fails",
        ]

    def test_a_not_applicable_m3_cannot_complete_the_predicted_pattern(self):
        rows = R.direction(
            {
                "m1": {"verdict": "HOLDS"},
                "m2": {"verdict": "HOLDS"},
                "m3": {"verdict": "N/A (state-independent sigma)"},
                "m4": {"verdict": "FAILS"},
            }
        )
        assert [r["key"] for r in rows] == ["p4_fails"]

    def test_the_rows_are_the_five_of_spec_section_four(self):
        assert [r["key"] for r in R.DIRECTIONS] == [
            "predicted",
            "p1_fails",
            "p2_fails",
            "p3_fails",
            "p4_fails",
        ]


# ------------------------------------------------------------------- verdicts
class TestVerdicts:
    def test_a_full_round_carries_all_four_measurements_and_hsafe(self):
        v = R.verdicts(_records())
        assert set(v) >= {"m1", "m2", "m3", "m4", "hsafe", "direction", "complete"}
        assert v["complete"] is True
        assert v["missing"] == []

    def test_a_state_independent_sigma_is_noted(self):
        v = R.verdicts(_records(sigma_state_dependent=False))
        assert any("state-independent" in n for n in v["notes"])

    def test_missing_cells_make_the_round_incomplete(self):
        recs = _records()
        gone = R.cell_name("FT-B-41", "GREEDY", 1)
        del recs[gone]
        v = R.verdicts(recs)
        assert v["complete"] is False
        assert gone in v["missing"]


# ------------------------------------------------------------- cell bookkeeping
class TestRunCells:
    def test_skips_cells_that_already_have_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        done = R.cell_name("RC19", "ATK4-45", 10)
        todo = R.cell_name("RC19", "ATK4-45", 1)
        R.cells_dir().mkdir(parents=True)
        (R.cells_dir() / f"{done}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells([done, todo])
        assert called == [todo]
        assert json.loads((R.cells_dir() / f"{todo}.json").read_text())["name"] == todo
        assert (R.cells_dir() / f"{done}.json").read_text() == "{}"

    def test_no_names_runs_every_missing_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        done = list(R.CELLS)[:3]
        for name in done:
            (R.cells_dir() / f"{name}.json").write_text("{}")
        called = []
        monkeypatch.setattr(
            R,
            "run_cell",
            lambda name, device="cpu": called.append(name) or _record(name),
        )
        R.run_cells(None)
        assert len(called) == len(R.CELLS) - 3
        assert set(called).isdisjoint(done)

    def test_unknown_cell_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit):
            R.run_cells(["NOPE"])

    def test_records_are_written_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        name = R.cell_name("FT-B-41", "ATK4-45", 10)
        R.write_cell(_record(name))
        assert json.loads((R.cells_dir() / f"{name}.json").read_text())["name"] == name
        assert list(R.cells_dir().glob("*.tmp")) == []


# -------------------------------------------------------------------- summarize
class TestSummarize:
    def test_writes_round_json_and_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        for rec in _records().values():
            R.write_cell(rec)
        blob = R.summarize()
        assert (tmp_path / "s15_round.json").exists()
        assert (tmp_path / "s15_round.png").exists()
        on_disk = json.loads((tmp_path / "s15_round.json").read_text())
        assert on_disk["verdicts"]["hsafe"]["verdict"] == "HOLDS"
        assert blob["verdicts"]["complete"] is True
        assert len(on_disk["cells"]) == 8
        # the per-episode table stays in the cell records
        assert "per_episode" not in next(iter(on_disk["cells"].values()))
