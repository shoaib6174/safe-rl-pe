"""scripts/derisk_s18_lambda.py and scripts/derisk_s18_construct.py — the S18
round's own decisions (spec: docs/pathway/boundary_condition_prereg_2026-09-23.md
sections 3, 6 and amendment A1).

Nothing here runs the pre-registered rollouts: the weight arithmetic, its
refusals, the artefact schema, CG1's variance computation, CG1/CG2's verdict
rules and the admissible-set enumeration are all driven on synthetic states and
fabricated records. The one place an environment appears is the enumeration
test, which must bind `admissible_actions` to `CBFShield.filter` itself — a
single filter call on a hand-placed pose, no episode.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.rewards import SearchStalenessTracker
from pe2.shield import CBFShield

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAM = _module("derisk_s18_lambda")
CON = _module("derisk_s18_construct")

CFG = EnvConfig()


def _episode(intervention=10.0, vis=4.0, search=2.0, n_decisions=5, outcome="timeout"):
    return {
        "seed": 1_860_000,
        "outcome": outcome,
        "steps": 1200,
        "n_decisions": n_decisions,
        "violations": 0,
        "intervention": intervention,
        "vis": vis,
        "search": search,
    }


# ------------------------------------------------------------------ section 3
class TestWeightArithmetic:
    """Section 3: lambda makes the expected episode penalty 1.0, the magnitude
    of the terminal reward. A1.2: both dense weights make the expected episode
    dense return 0.25, the SAME 0.25 in each column."""

    def test_lambda_makes_the_expected_episode_penalty_one(self):
        w = LAM.weights({"intervention": 40.0, "vis": 1.0, "search": 1.0})
        assert w["lambda"] * 40.0 == pytest.approx(LAM.TARGET_PENALTY)

    def test_both_dense_weights_make_the_same_expected_return(self):
        means = {"intervention": 40.0, "vis": 8.0, "search": 0.002}
        w = LAM.weights(means)
        assert w["w_vis"] * means["vis"] == pytest.approx(LAM.TARGET_DENSE)
        assert w["w_search"] * means["search"] == pytest.approx(LAM.TARGET_DENSE)
        # A1.2's whole point: the columns are equal in scale after calibration,
        # whatever their raw means were.
        assert w["w_vis"] * means["vis"] == pytest.approx(
            w["w_search"] * means["search"]
        )

    def test_targets_are_the_pre_registered_ones(self):
        assert LAM.TARGET_PENALTY == 1.0
        assert LAM.TARGET_DENSE == 0.25
        assert LAM.SEED0 == 1_860_000  # A1.1
        assert LAM.N_EPISODES == 200

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    @pytest.mark.parametrize("key", ["intervention", "vis", "search"])
    def test_refuses_a_non_positive_or_non_finite_mean(self, key, bad):
        means = {"intervention": 40.0, "vis": 8.0, "search": 0.002}
        means[key] = bad
        with pytest.raises(ValueError, match=key):
            LAM.weights(means)

    def test_a_zero_mean_is_refused_by_build_record_too(self):
        """The refusal must reach the artefact: a calibration whose term never
        fired must produce NO file rather than a file full of infinities."""
        rows = [_episode(vis=0.0) for _ in range(3)]
        with pytest.raises(ValueError, match="vis"):
            LAM.build_record(rows)


class TestCalibrationRecord:
    def test_schema(self):
        rows = [_episode(), _episode(intervention=30.0, vis=6.0, search=1.0)]
        rec = LAM.build_record(rows, seed0=LAM.SEED0)
        for key in ("lambda", "w_vis", "w_search", "means", "n_episodes", "seed0"):
            assert key in rec
        assert set(rec["means"]) == set(LAM.KEYS)
        assert rec["n_episodes"] == 2
        assert rec["seed0"] == LAM.SEED0
        assert rec["provenance"]["numpy_version"] == np.__version__
        json.dumps(rec)  # the artefact must be serialisable as written

    def test_means_are_over_every_episode_including_the_undecided_ones(self):
        """Section 3 and A1.2 say "200 episodes ... the mean per-episode sum".
        An episode the fixed law finished before the learner was consulted
        contributes 0 and stays in that mean; the conditional mean is reported
        beside it so the effect of that reading is visible."""
        rows = [
            _episode(intervention=10.0, vis=4.0, search=2.0),
            _episode(intervention=0.0, vis=0.0, search=0.0, n_decisions=0),
        ]
        rec = LAM.build_record(rows)
        assert rec["means"]["intervention"] == pytest.approx(5.0)
        assert rec["diagnostics"]["n_zero_decision_episodes"] == 1
        assert rec["diagnostics"]["means_given_a_decision"]["intervention"] == 10.0

    def test_the_weights_invert_the_reported_means(self):
        rows = [_episode(), _episode(intervention=30.0, vis=6.0, search=1.0)]
        rec = LAM.build_record(rows)
        assert rec["lambda"] * rec["means"]["intervention"] == pytest.approx(1.0)
        assert rec["w_vis"] * rec["means"]["vis"] == pytest.approx(0.25)
        assert rec["w_search"] * rec["means"]["search"] == pytest.approx(0.25)


class TestSpanTermRecorder:
    """The recorder is an OBSERVER: it must accumulate the dense terms and
    change nothing about the step it observes."""

    class _Stub:
        def __init__(self, terms):
            self.terms = terms

        def step(self, action):
            return "obs", 0.5, False, True, {"dense_terms": self.terms, "k": 1}

    def test_accumulates_until_cleared_and_passes_the_step_through(self):
        rec = LAM.SpanTermRecorder.__new__(LAM.SpanTermRecorder)
        rec.env = self._Stub({"vis": 0.1, "search": 0.02})
        rec.clear()
        out = rec.step(None)
        rec.step(None)
        assert out == (
            "obs",
            0.5,
            False,
            True,
            {"dense_terms": {"vis": 0.1, "search": 0.02}, "k": 1},
        )
        assert rec.vis == pytest.approx(0.2)
        assert rec.search == pytest.approx(0.04)
        rec.clear()
        assert (rec.vis, rec.search) == (0.0, 0.0)

    def test_a_step_with_no_dense_terms_contributes_zero(self):
        rec = LAM.SpanTermRecorder.__new__(LAM.SpanTermRecorder)
        rec.env = self._Stub({})
        rec.clear()
        rec.step(None)
        assert (rec.vis, rec.search) == (0.0, 0.0)


# ------------------------------------------------------------------ section 6
class TestLoadWeights:
    def test_refuses_to_run_without_the_calibration(self, tmp_path):
        with pytest.raises(SystemExit, match="derisk_s18_lambda"):
            CON.load_weights(tmp_path)

    def test_refuses_a_file_that_is_not_a_calibration(self, tmp_path):
        (tmp_path / CON.LAMBDA_NAME).write_text(json.dumps({"lambda": 1.0}))
        with pytest.raises(SystemExit, match="w_vis"):
            CON.load_weights(tmp_path)

    def test_reads_the_three_weights(self, tmp_path):
        blob = {"lambda": 0.01, "w_vis": 0.002, "w_search": 0.005}
        (tmp_path / CON.LAMBDA_NAME).write_text(json.dumps(blob))
        assert CON.load_weights(tmp_path)["w_vis"] == 0.002

    def test_the_stream_is_the_amended_one(self):
        assert CON.SEED0 == 1_880_000  # A1.1
        assert CON.CG1_N_STATES == 150
        assert CON.FLAT_ATOL == 1e-9


class TestAdmissibleActions:
    """CG1's action set is the SHIELD'S, not a new discretisation (D126)."""

    @staticmethod
    def _state(x=0.0, y=0.0, theta=0.0):
        return np.array([x, y, theta], dtype=float)

    # Driving a wall at full speed from here: the filter intervenes and still
    # reaches its exact tier, which is the state CG1 measures at.
    PINNED = (4.7, 0.0, 0.0)

    def test_every_returned_action_passes_the_shields_own_exact_test(self):
        shield = CBFShield(CFG)
        state = self._state(*self.PINNED)
        U = CON.admissible_actions(shield, (CFG.v_max, 0.0), state, [])
        assert len(U)
        for u in U:
            assert (shield.progress(state, [], u) >= 0.0).all()

    def test_it_contains_exactly_what_the_filter_returns_on_the_exact_tier(self):
        shield = CBFShield(CFG)
        state = self._state(*self.PINNED)
        nominal = (CFG.v_max, 0.0)
        u, info = shield.filter(nominal, state, [])
        assert info["method"] == "exact" and info["intervention"]
        U = CON.admissible_actions(shield, nominal, state, [])
        assert any(np.allclose(u, row) for row in U)

    def test_zero_is_always_admissible_in_an_empty_arena(self):
        """`u = 0` keeps every hard row bit-identical (shield spec B4), so the
        set can never be empty at a state the filter itself reached."""
        U = CON.admissible_actions(CBFShield(CFG), (0.5, 0.0), self._state(), [])
        assert any(np.allclose(row, [0.0, 0.0]) for row in U)

    def test_the_set_is_deduplicated(self):
        U = CON.admissible_actions(CBFShield(CFG), (0.5, 0.3), self._state(), [])
        assert len(np.unique(U, axis=0)) == len(U)


class TestColumnValues:
    """CG1's per-step task reward, on a synthetic state with no environment."""

    @staticmethod
    def _state(evader_xy=(3.0, 0.0), obstacles=()):
        half = CFG.arena_size / 2.0 - CFG.agent_radius
        return {
            "id": 0,
            "episode": 0,
            "method": "exact",
            "pursuer_pose": np.array([0.0, 0.0, 0.0]),
            "evader_pose": np.array([evader_xy[0], evader_xy[1], 0.0]),
            "obstacles": [dict(o) for o in obstacles],
            "current_step": 100,
            # Varied, and inside the staleness window: an all-zero
            # `last_observed` saturates every cell at staleness 1.0 and the
            # term would be constant for a reason that has nothing to do with
            # the action.
            "staleness": np.linspace(52, 100, 100).astype(np.int64),
            "x_lim": (-half, half),
            "y_lim": (-half, half),
            "nominal": (CFG.v_max, 0.0),
            "v_max": CFG.v_max,
        }

    def test_sparse_is_identically_zero_across_the_action_set(self):
        U = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        vals = CON.column_values(
            self._state(), U, CFG, 0.002, 0.005, SearchStalenessTracker(CFG)
        )
        assert vals["SPARSE"] == [0.0, 0.0, 0.0]
        assert np.var(vals["SPARSE"]) == 0.0

    def test_the_visibility_column_is_flat_when_the_evader_is_out_of_sight(self):
        """The DENSE-RARE claim, in miniature: with the evader behind the
        pursuer, no one-step action brings it into the cone, so the term pays
        the same (nothing) for every admissible action."""
        st = self._state(evader_xy=(-3.0, 0.0))
        U = np.array([[0.0, 0.0], [CFG.v_max, 0.0], [0.0, CFG.omega_max]])
        vals = CON.column_values(st, U, CFG, 0.002, 0.005, SearchStalenessTracker(CFG))
        assert np.var(vals["DRARE"]) == 0.0

    def test_the_visibility_column_varies_when_a_turn_acquires_the_evader(self):
        """...and it is a real measurement, not a constant: put the evader just
        outside the cone and one large turn does change the term."""
        # 50 degrees off the heading, just outside the 45-degree half-angle:
        # one full-rate turn (omega_max * dt = 8.1 degrees) brings it in.
        bearing, rng = np.radians(50.0), 6.0
        st = self._state(evader_xy=(rng * np.cos(bearing), rng * np.sin(bearing)))
        stay = np.array([[0.0, 0.0]])
        turn = np.array([[0.0, 0.0], [0.0, CFG.omega_max]])
        flat = CON.column_values(st, stay, CFG, 1.0, 0.0, SearchStalenessTracker(CFG))
        both = CON.column_values(st, turn, CFG, 1.0, 0.0, SearchStalenessTracker(CFG))
        assert flat["DRARE"] == [0.0]
        assert np.var(both["DRARE"]) > 0.0

    def test_the_search_column_varies_with_the_heading(self):
        st = self._state()
        U = np.array([[0.0, 0.0], [0.0, CFG.omega_max], [0.0, -CFG.omega_max]])
        vals = CON.column_values(st, U, CFG, 0.002, 0.005, SearchStalenessTracker(CFG))
        assert np.var(vals["DVARY"]) > 0.0

    def test_the_scratch_tracker_is_reset_per_candidate(self):
        """Each candidate must be scored from the SAME staleness state; if the
        tracker carried a candidate's own update into the next one the values
        would depend on the enumeration order."""
        st = self._state()
        U = np.array([[0.0, 0.0], [0.0, 0.0]])
        vals = CON.column_values(st, U, CFG, 0.0, 1.0, SearchStalenessTracker(CFG))
        assert vals["DVARY"][0] == pytest.approx(vals["DVARY"][1])
        assert vals["DVARY"][0] > 0.0


def _scored(sparse=0.0, dflat=0.0, dvary=1e-6, n=10, episode_of=lambda i: i):
    return [
        {
            "state": i,
            "episode": episode_of(i),
            "step": 10 * i,
            "method": "exact",
            "n_admissible": 6,
            "variance": {"SPARSE": sparse, "DRARE": dflat, "DVARY": dvary},
            "mean": {"SPARSE": 0.0, "DRARE": 0.0, "DVARY": 0.0},
        }
        for i in range(n)
    ]


class TestCG1Verdict:
    def test_the_pre_registered_pattern_passes(self):
        v = CON.cg1_verdict(_scored())
        assert v["verdict"] == "PASS"
        assert v["per_column"]["DRARE"]["label"] == "FLAT"
        assert v["n_scored"] == 10 and v["n_episodes_spanned"] == 10

    def test_it_counts_the_states_each_column_varies_at(self):
        """A mean variance can be lifted off zero by one state in a hundred;
        the count is what says which of those happened."""
        rows = _scored(dflat=0.0, n=4) + _scored(dflat=1e-6, n=1)
        v = CON.cg1_verdict(rows)
        assert v["per_column"]["DRARE"]["n_states_varying"] == 1
        assert v["per_column"]["DVARY"]["n_states_varying"] == 5
        assert v["per_column"]["SPARSE"]["n_states_varying"] == 0

    def test_a_varying_dense_flat_column_is_flagged_not_flat(self):
        v = CON.cg1_verdict(_scored(dflat=1e-6))
        assert v["verdict"] == "FAIL"
        assert v["per_column"]["DRARE"]["label"] == "NOT-FLAT"
        assert v["per_column"]["DVARY"]["pass"] is True

    def test_the_flat_tolerance_is_the_brief_s_and_nothing_larger(self):
        assert CON.cg1_verdict(_scored(dflat=CON.FLAT_ATOL / 2))["verdict"] == "PASS"
        assert CON.cg1_verdict(_scored(dflat=CON.FLAT_ATOL * 2))["verdict"] == "FAIL"

    def test_a_flat_dense_vary_column_fails(self):
        v = CON.cg1_verdict(_scored(dvary=0.0))
        assert v["verdict"] == "FAIL"
        assert v["per_column"]["DVARY"]["pass"] is False

    def test_states_with_an_empty_admissible_set_are_counted_not_dropped(self):
        rows = _scored(n=3) + [
            {
                "state": 9,
                "episode": 9,
                "step": 1,
                "method": "backup",
                "n_admissible": 0,
                "variance": None,
                "mean": None,
            }
        ]
        v = CON.cg1_verdict(rows)
        assert v["n_states"] == 4 and v["n_scored"] == 3

    def test_no_scorable_state_is_incomplete_not_a_pass(self):
        rows = [
            {
                "state": 0,
                "episode": 0,
                "step": 1,
                "method": "backup",
                "n_admissible": 0,
                "variance": None,
                "mean": None,
            }
        ]
        assert CON.cg1_verdict(rows)["verdict"] == "INCOMPLETE"


def _log(outcome="timeout", deadlock=False):
    return {
        "seed": 1_880_000,
        "role": "pursuer",
        "outcome": outcome,
        "steps": 1200,
        "final_distance": 5.0,
        "los_breaks": 0,
        "shielded_role": "pursuer",
        "violations": 0,
        "shield": {
            "deadlock": deadlock,
            "intervention_rate": 0.1,
            "immobile_step_rate": 1.0,
        },
    }


class TestCG2:
    def test_the_zero_motion_policy_commands_exactly_zero(self):
        from pe2.exploitability import SingleAgentWrapper

        action, _ = CON.ZeroMotion(CFG).predict(None)
        wrapper = SingleAgentWrapper.__new__(SingleAgentWrapper)
        wrapper.cfg, wrapper.role = CFG, "pursuer"
        assert wrapper._scale_action(np.asarray(action)) == (0.0, 0.0)

    def test_no_capture_passes(self):
        v = CON.cg2_verdict([_log() for _ in range(50)])
        assert v["verdict"] == "PASS" and v["capture_rate"] == 0.0

    def test_a_single_capture_fails_the_gate(self):
        logs = [_log() for _ in range(49)] + [_log(outcome="capture")]
        v = CON.cg2_verdict(logs)
        assert v["verdict"] == "FAIL"
        assert v["capture_rate"] == pytest.approx(0.02)
        assert v["n_captures"] == 1

    def test_the_mechanism_metrics_are_reported_and_never_gate(self):
        logs = [_log(deadlock=True) for _ in range(10)]
        v = CON.cg2_verdict(logs)
        assert v["verdict"] == "PASS"
        assert v["mechanism_metrics_report_only"]["deadlock_rate"] == 1.0


class TestGateRecord:
    @staticmethod
    def _record(cg1_rows=None, cg2_logs=None):
        cg1 = CON.cg1_verdict(cg1_rows if cg1_rows is not None else _scored())
        cg2 = CON.cg2_verdict(cg2_logs if cg2_logs is not None else [_log()] * 5)
        cg3 = CON.cg3_report(
            {c: [0.0, 1.0] for c in CON.COLUMNS},
            {"lambda": 0.01, "w_vis": 0.002, "w_search": 0.005},
        )
        return CON.build_record(
            cg1, [], cg2, cg3, {"lambda": 0.01, "w_vis": 0.002, "w_search": 0.005}
        )

    def test_a_clean_gate_passes_and_serialises(self):
        rec = self._record()
        assert rec["verdict"] == "PASS"
        assert rec["blocking"] == ["CG1", "CG2"]
        json.dumps(rec)

    def test_either_blocking_check_fails_the_gate(self):
        assert self._record(cg1_rows=_scored(dvary=0.0))["verdict"] == "FAIL"
        assert self._record(cg2_logs=[_log(outcome="capture")])["verdict"] == "FAIL"

    def test_cg3_is_reported_and_does_not_gate(self):
        rec = self._record()
        assert rec["CG3"]["verdict"].startswith("REPORTED")
        assert set(rec["CG3"]["columns"]) == set(CON.COLUMNS)
        assert "CG3" not in rec["blocking"]

    def test_the_verdict_table_prints(self, capsys):
        CON.print_record(self._record())
        out = capsys.readouterr().out
        assert "CG1" in out and "DENSE-VARY" in out and "GATE: PASS" in out
