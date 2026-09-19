"""scripts/derisk_s14_matrix.py — the S14 empirical game matrix's own decisions.

Spec: docs/pathway/game_matrix_prereg_2026-09-19.md.

No test here loads a checkpoint. The enumeration, the verdict rules, the
meta-game and the cell bookkeeping are driven on fabricated records; `run_cell`'s
bindings to the environment — that each defender carries ITS OWN `shield_obs`
and hold, and that each attacker is built as ITS OWN opponent object — are
checked with the evaluation stubbed out.

The two things this round could silently get wrong, and the tests that pin them:
  * the TEACHER row is a SCRIPTED pursuer (`OrbitSearch` at hold 1, which is
    `SearchingPursuer` episode-for-episode, validity check V1) with no
    `shield_obs` at all — `TestDefenders` and `TestRunCell`;
  * `meta_game` is a zero-sum LP, not an argmax — `TestMetaGame` checks it
    against two hand-solved games (one mixed, one with a pure saddle point).
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s14_matrix.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s14_matrix", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


# ----------------------------------------------------------------- fabricators
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
        "defender": spec["defender"],
        "attacker": spec["attacker"],
        "delta": R.DELTA,
        "n": R.N_OBSTACLES,
        "gamma": R.GAMMA,
        "shield_obs": spec["shield_obs"],
        "hold": spec["hold"],
        "transfer_only": spec["transfer_only"],
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
    """A full 42-cell record set whose deadlock rate is rate_fn(spec)."""
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


def _set_ind(recs, name, ind):
    """Overwrite one cell's deadlock indicator (and everything derived)."""
    rec = recs[name]
    rec["deadlock_ind"] = list(ind)
    rec["n_episodes"] = len(ind)
    rec["deadlock_rate"] = sum(ind) / len(ind)
    rec["wilson_deadlock"] = R._ci(sum(ind), len(ind))


# The reference world, in which both hypotheses CONFIRM.
#
# Every learned attacker is far strongest against the defender it was trained
# against (H-MAT-1), and the worst-case column reproduces the pre-registered
# ordering (H-MAT-2). Note what that forces and what the real round may well
# show: TEACHER and SEL-A-50 have no attacker of their own, so the only way
# their worst case can sit at the top of the ordering is for a SCRIPTED evader
# to be their worst opponent.
_BASE = {
    "TEACHER": 0.35,
    "STR14": 0.30,
    "RC19": 0.28,
    "RCM28": 0.05,
    "FT-B-41": 0.20,
    "SEL-A-50": 0.22,
}
_OWN_BONUS = 0.45
_SCRIPTED_BASE = {
    "TEACHER": 0.80,
    "STR14": 0.10,
    "RC19": 0.10,
    "RCM28": 0.10,
    "FT-B-41": 0.10,
    "SEL-A-50": 0.68,
}
_SCRIPTED_STEP = {"SPLITHUG": 0.0, "HUG7": -0.02, "GREEDY": -0.04}
# The worst case each of those produces, by hand (asserted in TestMat2).
_W = {
    "RCM28": 0.50,
    "FT-B-41": 0.65,
    "SEL-A-50": 0.68,
    "RC19": 0.73,
    "STR14": 0.75,
    "TEACHER": 0.80,
}


def _all_confirm(spec):
    d, a = spec["defender"], spec["attacker"]
    if a in R.SCRIPTED:
        return _SCRIPTED_BASE[d] + _SCRIPTED_STEP[a]
    return _BASE[d] + (_OWN_BONUS if R.TRAINED_AGAINST.get(a) == d else 0.0)


# ---------------------------------------------------------------- enumeration
class TestCells:
    def test_six_defenders_seven_attackers_is_forty_two_cells(self):
        assert len(R.DEFENDERS) == 6
        assert len(R.ATTACKERS) == 7
        assert len(R.CELLS) == 42

    def test_every_defender_meets_every_attacker(self):
        for d in R.DEFENDERS:
            attackers = sorted(
                s["attacker"] for s in R.CELLS.values() if s["defender"] == d
            )
            assert attackers == sorted(R.ATTACKERS)

    def test_cell_names_are_defender_double_underscore_attacker(self):
        assert R.cell_name("TEACHER", "SPLITHUG") == "TEACHER__SPLITHUG"
        assert R.cell_name("SEL-A-50", "ATK4-45") == "SEL-A-50__ATK4-45"
        for name, spec in R.CELLS.items():
            assert name == R.cell_name(spec["defender"], spec["attacker"])

    def test_every_cell_is_the_base_shield_setting(self):
        assert (R.DELTA, R.N_OBSTACLES, R.GAMMA) == (0.0, 2, 0.2)

    def test_pre_registered_constants(self):
        assert R.SEED0 == 1_370_000
        assert R.N_EPISODES == 300
        assert R.P_GATE == 0.05
        assert R.MAX_WORKERS == 6
        assert R.CONFIRM_MIN == 3
        assert R.SUPPORT_TOL == 1e-6
        assert R.EVADER_HOLD == 10
        assert R.ARTEFACT_PREFIX == "s14"


class TestDefenders:
    def test_the_six_rows_in_the_spec_order(self):
        assert list(R.DEFENDERS) == [
            "TEACHER",
            "STR14",
            "RC19",
            "RCM28",
            "FT-B-41",
            "SEL-A-50",
        ]

    def test_the_teacher_row_is_scripted_at_hold_one_with_no_shield_obs(self):
        spec = R.DEFENDERS["TEACHER"]
        assert spec["hold"] == 1
        assert "ckpt" not in spec
        # the spec's "shield_obs as trained": TEACHER was never trained, so it
        # has no flag at all — not a False one that could drift to True.
        assert "shield_obs" not in spec

    def test_the_teacher_policy_is_orbitsearch_on_the_cell_config(self):
        from pe2.structured import OrbitSearch

        cfg = R.cell_config()
        policy = R.defender_policy("TEACHER", cfg, "cpu")
        assert isinstance(policy, OrbitSearch)
        assert policy.cfg is cfg

    def test_each_learned_defender_carries_its_trained_shield_obs(self):
        assert R.DEFENDERS["STR14"]["shield_obs"] is False  # trained unshielded
        for d in ("RC19", "RCM28", "FT-B-41", "SEL-A-50"):
            assert R.DEFENDERS[d]["shield_obs"] is True, d

    def test_each_learned_defender_holds_for_the_structured_hold(self):
        from pe2.structured import HOLD

        for d in ("STR14", "RC19", "RCM28", "FT-B-41", "SEL-A-50"):
            assert R.DEFENDERS[d]["hold"] == HOLD, d

    def test_the_checkpoints(self):
        assert R.DEFENDERS["STR14"]["ckpt"].name == "STR14_10000000.zip"
        assert R.DEFENDERS["RC19"]["ckpt"].name == "RC19_10000000.zip"
        assert R.DEFENDERS["RCM28"]["ckpt"].name == "RCM28_10000000.zip"
        assert R.DEFENDERS["FT-B-41"]["ckpt"].name == "FT-B-41_5000000.zip"
        assert R.DEFENDERS["SEL-A-50"]["ckpt"].name == "SEL-A-50_5000000.zip"

    def test_teacher_and_sel_a_50_are_labelled_transfer_only(self):
        """Spec section 2: no attacker was trained against either, so no
        worst-case number of theirs is a best response."""
        assert set(R.TRANSFER_ONLY) == {"TEACHER", "SEL-A-50"}
        for d in R.DEFENDERS:
            assert (d in R.TRANSFER_ONLY) == (d not in R.TRAINED_AGAINST.values())
        for name, spec in R.CELLS.items():
            assert spec["transfer_only"] == (spec["defender"] in R.TRANSFER_ONLY), name


class TestAttackers:
    def test_the_seven_columns_in_the_spec_order(self):
        assert list(R.ATTACKERS) == [
            "CTL-D125",
            "CTL-D225",
            "ATK2-31",
            "ATK4-45",
            "SPLITHUG",
            "HUG7",
            "GREEDY",
        ]
        assert R.LEARNED == ("CTL-D125", "CTL-D225", "ATK2-31", "ATK4-45")
        assert R.SCRIPTED == ("SPLITHUG", "HUG7", "GREEDY")

    def test_each_learned_attacker_names_the_defender_it_was_trained_against(self):
        assert R.TRAINED_AGAINST == {
            "CTL-D125": "STR14",
            "CTL-D225": "RC19",
            "ATK2-31": "RCM28",
            "ATK4-45": "FT-B-41",
        }

    def test_the_learned_checkpoints(self):
        assert R.ATTACKERS["CTL-D125"]["ckpt"].name == "CTL-D125_10000000.zip"
        assert R.ATTACKERS["CTL-D225"]["ckpt"].name == "CTL-D225_10000000.zip"
        assert R.ATTACKERS["ATK2-31"]["ckpt"].name == "ATK2-31_10000000.zip"
        assert R.ATTACKERS["ATK4-45"]["ckpt"].name == "ATK4-45_10000000.zip"

    def test_a_learned_attacker_is_a_structured_evader_policy(self, monkeypatch):
        from pe2.evade import D_POS, StructuredEvaderPolicy

        model = object()
        monkeypatch.setattr(R, "_model", lambda ckpt, device: model)
        cfg = R.cell_config()
        policy = R.opponent_factory("ATK2-31", cfg, "cpu")()
        assert isinstance(policy, StructuredEvaderPolicy)
        assert policy.model is model
        assert policy.cfg is cfg
        assert (policy.hold, policy.d_pos) == (R.EVADER_HOLD, D_POS)

    def test_splithug_is_the_scripted_law_inside_the_same_wrapper(self):
        """S7's `splithug_factory`: the SplitHugEvader shim in the seat a
        learned attacker occupies, so the two differ only in what decides."""
        from pe2.evade import D_POS, SplitHugEvader, StructuredEvaderPolicy

        cfg = R.cell_config()
        policy = R.opponent_factory("SPLITHUG", cfg, "cpu")()
        assert isinstance(policy, StructuredEvaderPolicy)
        assert isinstance(policy.model, SplitHugEvader)
        assert policy.model.cfg is cfg
        assert policy.cfg is cfg
        assert (policy.hold, policy.d_pos) == (R.EVADER_HOLD, D_POS)

    def test_hug7_and_greedy_are_the_plain_scripted_classes(self):
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        cfg = R.cell_config()
        assert R.opponent_factory("HUG7", cfg, "cpu") is ObstacleHuggingEvader
        assert R.opponent_factory("GREEDY", cfg, "cpu") is GreedyEvader


# --------------------------------------------------------------------- run_cell
class _Stop(Exception):
    """Raised by the evaluation stub once it has captured its arguments."""


class TestRunCell:
    def _call(self, monkeypatch, name):
        seen = {}

        def fake_eval(model, cfg, opponent_factory, n_episodes, seed0, **kw):
            seen.update(
                model=model,
                cfg=cfg,
                opponent=opponent_factory(),
                n_episodes=n_episodes,
                seed0=seed0,
                kw=kw,
            )
            raise _Stop

        monkeypatch.setattr("pe2.structured.evaluate_structured", fake_eval)
        monkeypatch.setattr(R, "_model", lambda ckpt, device: ("model", ckpt))
        with pytest.raises(_Stop):
            R.run_cell(name)
        return seen

    def test_the_stream_and_the_arena_are_the_pre_registered_ones(self, monkeypatch):
        from pe2.config import EnvConfig

        seen = self._call(monkeypatch, "RC19__CTL-D225")
        assert seen["seed0"] == R.SEED0
        assert seen["n_episodes"] == R.N_EPISODES
        cfg = seen["cfg"]
        assert cfg.n_physical_obstacles == R.N_OBSTACLES
        assert cfg.evader_speed_ratio == EnvConfig().evader_speed_ratio
        shield = seen["kw"]["shield_factory"]()
        assert (shield.gamma, shield.margin) == (R.GAMMA, R.DELTA)

    @pytest.mark.parametrize(
        ("defender", "shield_obs"),
        [
            ("STR14", False),
            ("RC19", True),
            ("RCM28", True),
            ("FT-B-41", True),
            ("SEL-A-50", True),
            ("TEACHER", False),
        ],
    )
    def test_each_defender_is_evaluated_with_its_own_shield_obs(
        self, monkeypatch, defender, shield_obs
    ):
        seen = self._call(monkeypatch, R.cell_name(defender, "GREEDY"))
        assert seen["kw"]["shield_obs"] is shield_obs

    def test_a_learned_defender_is_its_checkpoint_at_the_structured_hold(
        self, monkeypatch
    ):
        from pe2.structured import HOLD

        seen = self._call(monkeypatch, "STR14__GREEDY")
        assert seen["model"] == ("model", R.DEFENDERS["STR14"]["ckpt"])
        assert seen["kw"]["hold"] == HOLD

    def test_the_teacher_row_is_the_scripted_pursuer_at_hold_one(self, monkeypatch):
        """The spec's non-learned baseline. S4 drove it the same way: OrbitSearch
        inside SearchMDP at hold 1 IS SearchingPursuer (validity check V1), so no
        checkpoint is loaded for this row."""
        from pe2.structured import OrbitSearch

        seen = self._call(monkeypatch, "TEACHER__HUG7")
        assert isinstance(seen["model"], OrbitSearch)
        assert seen["model"].cfg is seen["cfg"]
        assert seen["kw"]["hold"] == 1
        assert seen["kw"]["shield_obs"] is False

    @pytest.mark.parametrize("attacker", ["CTL-D125", "CTL-D225", "ATK2-31", "ATK4-45"])
    def test_a_learned_attacker_reaches_the_evaluation_as_its_policy(
        self, monkeypatch, attacker
    ):
        from pe2.evade import D_POS, StructuredEvaderPolicy

        seen = self._call(monkeypatch, R.cell_name("RC19", attacker))
        opp = seen["opponent"]
        assert isinstance(opp, StructuredEvaderPolicy)
        assert opp.model == ("model", R.ATTACKERS[attacker]["ckpt"])
        assert opp.cfg is seen["cfg"]
        assert (opp.hold, opp.d_pos) == (R.EVADER_HOLD, D_POS)

    def test_splithug_reaches_the_evaluation_wrapped(self, monkeypatch):
        from pe2.evade import SplitHugEvader, StructuredEvaderPolicy

        opp = self._call(monkeypatch, "RC19__SPLITHUG")["opponent"]
        assert isinstance(opp, StructuredEvaderPolicy)
        assert isinstance(opp.model, SplitHugEvader)

    def test_the_scripted_attackers_reach_the_evaluation_bare(self, monkeypatch):
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        assert isinstance(
            self._call(monkeypatch, "RC19__HUG7")["opponent"], ObstacleHuggingEvader
        )
        assert isinstance(
            self._call(monkeypatch, "RC19__GREEDY")["opponent"], GreedyEvader
        )

    def test_an_unknown_cell_is_refused(self):
        with pytest.raises(KeyError):
            R.run_cell("NOPE__GREEDY")


# --------------------------------------------------------------------- matrices
class TestMatrices:
    def test_the_deadlock_matrix_is_defenders_by_attackers(self):
        rows, cols, m = R.deadlock_matrix(_records(_all_confirm))
        assert rows == list(R.DEFENDERS)
        assert cols == list(R.ATTACKERS)
        assert len(m) == 6
        assert all(len(r) == 7 for r in m)
        i, j = rows.index("RCM28"), cols.index("ATK2-31")
        assert m[i][j] == pytest.approx(0.50)
        assert m[rows.index("TEACHER")][cols.index("SPLITHUG")] == pytest.approx(0.80)

    def test_the_capture_matrix_reads_the_capture_rate(self):
        recs = _records(
            _all_confirm,
            capture_fn=lambda s: 0.9 if s["defender"] == "TEACHER" else 0.1,
        )
        rows, _cols, m = R.capture_matrix(recs)
        assert m[rows.index("TEACHER")][0] == pytest.approx(0.9)
        assert m[rows.index("RC19")][0] == pytest.approx(0.1)

    def test_a_missing_cell_is_none_not_zero(self):
        recs = _records(_all_confirm)
        del recs["STR14__GREEDY"]
        rows, cols, m = R.deadlock_matrix(recs)
        assert m[rows.index("STR14")][cols.index("GREEDY")] is None


# -------------------------------------------------------------------- H-MAT-1
class TestMat1:
    def test_every_attacker_peaking_on_its_own_defender_confirms(self):
        s = R.mat1_verdicts(_records(_all_confirm))
        assert s["hypothesis"] == "H-MAT-1"
        assert s["verdict"] == "CONFIRMED"
        assert s["n_specialised"] == 4
        for a in R.LEARNED:
            block = s["attackers"][a]
            assert block["verdict"] == "SPECIALISED", a
            assert block["own_defender"] == R.TRAINED_AGAINST[a]
            assert block["argmax_defender"] == R.TRAINED_AGAINST[a]

    def test_the_comparator_is_the_next_highest_defender(self):
        s = R.mat1_verdicts(_records(_all_confirm))
        block = s["attackers"]["CTL-D125"]
        # own STR14 0.75; the next highest in that column is TEACHER at 0.35
        assert block["comparator_defender"] == "TEACHER"
        assert block["test"]["rate_a"] == pytest.approx(0.75)
        assert block["test"]["rate_b"] == pytest.approx(0.35)
        assert block["test"]["pass"] is True

    def test_an_attacker_stronger_elsewhere_is_general_and_reports_the_p(self):
        def rate(spec):
            if spec["attacker"] == "ATK2-31" and spec["defender"] == "TEACHER":
                return 0.90
            return _all_confirm(spec)

        s = R.mat1_verdicts(_records(rate))
        block = s["attackers"]["ATK2-31"]
        assert block["verdict"] == "GENERAL"
        assert block["argmax_defender"] == "TEACHER"
        assert block["comparator_defender"] == "TEACHER"
        assert 0.0 <= block["test"]["p"] <= 1.0
        assert block["test"]["pass"] is False  # own does not beat the comparator
        # one GENERAL still leaves 3 of 4 SPECIALISED, so the round confirms
        assert s["n_specialised"] == 3
        assert s["verdict"] == "CONFIRMED"

    def test_two_general_attackers_fail_the_round(self):
        def rate(spec):
            if (
                spec["attacker"] in ("ATK2-31", "ATK4-45")
                and spec["defender"] == "TEACHER"
            ):
                return 0.90
            return _all_confirm(spec)

        s = R.mat1_verdicts(_records(rate))
        assert s["n_specialised"] == 2
        assert s["verdict"] == "NOT CONFIRMED"
        assert sorted(s["general"]) == ["ATK2-31", "ATK4-45"]

    def test_a_missing_cell_makes_the_attacker_incomplete(self):
        recs = _records(_all_confirm)
        del recs["TEACHER__ATK2-31"]
        s = R.mat1_verdicts(recs)
        assert s["attackers"]["ATK2-31"]["verdict"] == "INCOMPLETE"
        assert s["verdict"] == "INCOMPLETE"

    def test_only_the_four_learned_attackers_are_scored(self):
        s = R.mat1_verdicts(_records(_all_confirm))
        assert sorted(s["attackers"]) == sorted(R.LEARNED)


# -------------------------------------------------------------------- H-MAT-2
class TestMat2:
    def test_the_pre_registered_ordering_is_the_specs_chain(self):
        """RCM28 < FT-B-41 ~ SEL-A-50 < RC19 <= STR14 <= TEACHER, expanded to
        pairs: only the strict `<` are tested."""
        assert R.ORDERING == (
            ("RCM28", "<", "FT-B-41"),
            ("RCM28", "<", "SEL-A-50"),
            ("FT-B-41", "~", "SEL-A-50"),
            ("FT-B-41", "<", "RC19"),
            ("SEL-A-50", "<", "RC19"),
            ("RC19", "<=", "STR14"),
            ("STR14", "<=", "TEACHER"),
        )
        tested = [p for p in R.ORDERING if p[1] == "<"]
        assert len(tested) == 4

    def test_the_worst_case_is_the_max_over_the_seven_attackers(self):
        s = R.mat2_verdicts(_records(_all_confirm))
        for d, w in _W.items():
            assert s["worst_case"][d]["deadlock_rate"] == pytest.approx(w), d
        assert s["worst_case"]["RCM28"]["attacker"] == "ATK2-31"
        assert s["worst_case"]["SEL-A-50"]["attacker"] == "SPLITHUG"
        assert s["worst_case"]["TEACHER"]["attacker"] == "SPLITHUG"

    def test_the_ordering_confirms(self):
        s = R.mat2_verdicts(_records(_all_confirm))
        assert s["hypothesis"] == "H-MAT-2"
        assert s["verdict"] == "CONFIRMED"
        assert s["inversions"] == []

    def test_each_comparison_runs_on_the_larger_sides_worst_attacker(self):
        s = R.mat2_verdicts(_records(_all_confirm))
        rel = s["relations"]["RCM28 < FT-B-41"]
        assert rel["comparator_attacker"] == "ATK4-45"  # FT-B-41's worst
        assert rel["test"]["rate_a"] == pytest.approx(0.65)  # FT-B-41 there
        assert rel["test"]["rate_b"] == pytest.approx(0.05)  # RCM28 there
        assert rel["holds"] is True
        assert (
            R.mat2_verdicts(_records(_all_confirm))["relations"]["RCM28 < SEL-A-50"][
                "comparator_attacker"
            ]
            == "SPLITHUG"
        )

    def test_the_weak_and_approximate_relations_are_reported_not_tested(self):
        s = R.mat2_verdicts(_records(_all_confirm))
        for key in ("FT-B-41 ~ SEL-A-50", "RC19 <= STR14", "STR14 <= TEACHER"):
            assert s["relations"][key]["tested"] is False
        for key in ("RCM28 < FT-B-41", "SEL-A-50 < RC19"):
            assert s["relations"][key]["tested"] is True

    def test_an_inverted_worst_case_fails_and_names_the_pair(self):
        # RCM28's own attacker pins it at 0.66, above FT-B-41's 0.65 but below
        # SEL-A-50's 0.68 — so exactly one pre-registered `<` inverts.
        recs = _records(_all_confirm)
        _set_ind(recs, "RCM28__ATK2-31", _ind(0.66, 300))
        s = R.mat2_verdicts(recs)
        assert s["verdict"] == "NOT CONFIRMED"
        assert s["inversions"] == ["RCM28 < FT-B-41"]
        rel = s["relations"]["RCM28 < FT-B-41"]
        assert rel["holds"] is False
        assert rel["worst_case_ordered"] is False

    def test_an_ordered_but_insignificant_gap_fails(self):
        """The `<` needs p < 0.05 as well as the ordering: b = 5, c = 0 gives
        a two-sided exact p of 0.0625."""
        recs = _records(_all_confirm)
        _set_ind(recs, "RCM28__ATK4-45", _ind(190 / 300, 300))
        s = R.mat2_verdicts(recs)
        rel = s["relations"]["RCM28 < FT-B-41"]
        assert rel["worst_case_ordered"] is True
        assert rel["test"]["b"] == 5
        assert rel["test"]["c"] == 0
        assert rel["test"]["p"] == pytest.approx(0.0625)
        assert rel["test"]["pass"] is False
        assert rel["holds"] is False
        assert s["verdict"] == "NOT CONFIRMED"

    def test_a_missing_cell_is_incomplete(self):
        recs = _records(_all_confirm)
        del recs["RC19__HUG7"]
        assert R.mat2_verdicts(recs)["verdict"] == "INCOMPLETE"


# -------------------------------------------------------------------- H-MAT-3
class TestMat3:
    def test_a_defender_best_on_both_axes_is_reported_as_dominating(self):
        recs = _records(
            _all_confirm,
            capture_fn=lambda s: 0.5 if s["defender"] == "RCM28" else 0.2,
        )
        r = R.mat3_report(recs)
        assert r["hypothesis"] == "H-MAT-3"
        assert r["dominating"] == "RCM28"
        assert r["free_lunch"] is True
        assert r["worst_case_deadlock"]["RCM28"] == pytest.approx(0.50)
        assert r["worst_case_capture"]["RCM28"] == pytest.approx(0.5)

    def test_otherwise_the_pareto_set_is_reported(self):
        recs = _records(
            _all_confirm,
            capture_fn=lambda s: 0.5 if s["defender"] == "TEACHER" else 0.2,
        )
        r = R.mat3_report(recs)
        assert r["dominating"] is None
        assert r["free_lunch"] is False
        assert sorted(r["pareto"]) == ["RCM28", "TEACHER"]

    def test_the_capture_axis_is_the_worst_case_too(self):
        recs = _records(
            _all_confirm,
            capture_fn=lambda s: 0.1 if s["attacker"] == "GREEDY" else 0.9,
        )
        r = R.mat3_report(recs)
        for d in R.DEFENDERS:
            assert r["worst_case_capture"][d] == pytest.approx(0.1), d
            assert r["worst_case_capture_attacker"][d] == "GREEDY"

    def test_an_incomplete_round_reports_no_frontier(self):
        recs = _records(_all_confirm)
        del recs["RC19__HUG7"]
        r = R.mat3_report(recs)
        assert r["complete"] is False
        assert r["pareto"] is None
        assert r["dominating"] is None


# ------------------------------------------------------------------ meta-game
# Two hand-solved zero-sum games. MIXED: rows minimise, columns maximise.
#   A = [[3, 1],
#        [0, 2]]
#   x = (p, 1-p) -> column payoffs 3p and 2-p; they cross at p = 1/2, so the
#   minimiser's unique optimum is (0.5, 0.5) and the value is 1.5.
#   y = (q, 1-q) -> row payoffs 1+2q and 2-2q; they cross at q = 1/4, so the
#   maximiser's unique optimum is (0.25, 0.75), value 1.5 again.
_MIXED = [[3.0, 1.0], [0.0, 2.0]]
# SADDLE: B = [[2, 3], [1, 4]]. With rows MINIMISING, max(1+p, 4-p) is smallest
# at p = 1 -> row 0 pure, and min(3-q, 4-3q) is largest at q = 0 -> column 1
# pure; the saddle is B[0][1] = 3. With rows MAXIMISING the saddle moves to
# B[0][0] = 2 (row 0 still, column 0 now) — the flag is not cosmetic.
_SADDLE = [[2.0, 3.0], [1.0, 4.0]]
_ROWS2 = ["d0", "d1"]
_COLS2 = ["a0", "a1"]
_TOL = 1e-6


class TestMetaGame:
    def test_a_hand_solved_mixed_equilibrium(self):
        g = R.meta_game(_MIXED, _ROWS2, _COLS2, minimise_rows=True)
        assert g["value"] == pytest.approx(1.5, abs=_TOL)
        assert g["row_strategy"]["d0"] == pytest.approx(0.5, abs=_TOL)
        assert g["row_strategy"]["d1"] == pytest.approx(0.5, abs=_TOL)
        assert g["col_strategy"]["a0"] == pytest.approx(0.25, abs=_TOL)
        assert g["col_strategy"]["a1"] == pytest.approx(0.75, abs=_TOL)
        assert g["row_support"] == ["d0", "d1"]
        assert g["col_support"] == ["a0", "a1"]
        assert g["duality_gap"] == pytest.approx(0.0, abs=_TOL)
        assert g["row_player"] == "minimiser"

    def test_a_pure_saddle_point_gives_pure_strategies(self):
        g = R.meta_game(_SADDLE, _ROWS2, _COLS2, minimise_rows=True)
        assert g["value"] == pytest.approx(3.0, abs=_TOL)
        assert g["row_strategy"]["d0"] == pytest.approx(1.0, abs=_TOL)
        assert g["row_strategy"]["d1"] == pytest.approx(0.0, abs=_TOL)
        assert g["col_strategy"]["a1"] == pytest.approx(1.0, abs=_TOL)
        assert g["row_support"] == ["d0"]
        assert g["col_support"] == ["a1"]

    def test_the_direction_flag_changes_the_game_that_is_solved(self):
        g = R.meta_game(_SADDLE, _ROWS2, _COLS2, minimise_rows=False)
        assert g["value"] == pytest.approx(2.0, abs=_TOL)
        assert g["row_strategy"]["d0"] == pytest.approx(1.0, abs=_TOL)
        assert g["col_strategy"]["a0"] == pytest.approx(1.0, abs=_TOL)
        assert g["row_support"] == ["d0"]
        assert g["col_support"] == ["a0"]
        assert g["row_player"] == "maximiser"

    def test_the_value_is_between_the_pure_bounds(self):
        g = R.meta_game(_MIXED, _ROWS2, _COLS2)
        maximin = max(min(row) for row in zip(*_MIXED, strict=True))  # column side
        minimax = min(max(row) for row in _MIXED)
        assert maximin <= g["value"] + _TOL
        assert g["value"] <= minimax + _TOL

    def test_the_support_tolerance_drops_a_zero_weight_strategy(self):
        # d1 is strictly dominated for a minimiser (every entry larger), so it
        # carries weight 0 and must not appear in the support.
        m = [[1.0, 1.0], [2.0, 2.0]]
        g = R.meta_game(m, _ROWS2, _COLS2, minimise_rows=True)
        assert g["row_support"] == ["d0"]
        assert g["row_strategy"]["d1"] < R.SUPPORT_TOL
        assert g["support_tol"] == R.SUPPORT_TOL

    def test_the_strategies_are_distributions(self):
        g = R.meta_game(_MIXED, _ROWS2, _COLS2)
        assert sum(g["row_strategy"].values()) == pytest.approx(1.0, abs=_TOL)
        assert sum(g["col_strategy"].values()) == pytest.approx(1.0, abs=_TOL)
        assert all(v >= 0.0 for v in g["row_strategy"].values())
        assert all(v >= 0.0 for v in g["col_strategy"].values())

    def test_a_missing_entry_is_refused(self):
        with pytest.raises(ValueError, match="measured"):
            R.meta_game([[1.0, None], [0.0, 2.0]], _ROWS2, _COLS2)

    def test_a_mislabelled_matrix_is_refused(self):
        with pytest.raises(ValueError, match="shape"):
            R.meta_game(_MIXED, _ROWS2, ["a0", "a1", "a2"])

    def test_the_round_solves_the_full_deadlock_matrix(self):
        recs = _records(_all_confirm)
        g = R.verdicts(recs)["meta_deadlock"]
        assert g["row_player"] == "minimiser"
        assert set(g["row_strategy"]) == set(R.DEFENDERS)
        assert set(g["col_strategy"]) == set(R.ATTACKERS)
        # the value cannot be below the best row's worst case
        assert g["value"] <= min(_W.values()) + _TOL

    def test_the_round_solves_the_capture_matrix_with_the_defender_maximising(self):
        recs = _records(_all_confirm)
        g = R.verdicts(recs)["meta_capture"]
        assert g["row_player"] == "maximiser"
        assert g["value"] == pytest.approx(0.2, abs=_TOL)  # flat capture fixture

    def test_an_incomplete_round_reports_no_meta_game(self):
        recs = _records(_all_confirm)
        del recs["RC19__HUG7"]
        v = R.verdicts(recs)
        assert v["meta_deadlock"]["status"] == "INCOMPLETE"
        assert v["meta_deadlock"]["value"] is None


# --------------------------------------------------------------------- H-SAFE
class TestHSafe:
    def test_holds_on_a_clean_full_round(self):
        h = R.hsafe(_records(_all_confirm))
        assert h["verdict"] == "HOLDS"
        assert h["violation_steps"] == 0
        assert h["offenders"] == []

    def test_fails_and_names_the_offending_cell(self):
        bad = R.cell_name("TEACHER", "SPLITHUG")
        h = R.hsafe(
            _records(_all_confirm, violations_fn=lambda s: 3 if s["name"] == bad else 0)
        )
        assert h["verdict"] == "FAILS"
        assert h["violation_steps"] == 3
        assert h["offenders"] == [bad]

    def test_incomplete_when_cells_are_missing_and_none_offend(self):
        recs = _records(_all_confirm)
        del recs["RC19__HUG7"]
        assert R.hsafe(recs)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------ direction
class TestDirection:
    def test_both_confirmed(self):
        d = R.direction(R.verdicts(_records(_all_confirm)))
        assert "H-MAT-1 CONFIRMED" in d
        assert "iterated best-response" in d
        assert "H-MAT-2 CONFIRMED" in d
        assert "worst-case test" in d

    def test_general_attackers_become_a_generality_claim(self):
        def rate(spec):
            if spec["attacker"] in ("ATK2-31", "ATK4-45", "CTL-D125") and (
                spec["defender"] == "TEACHER"
            ):
                return 0.95
            return _all_confirm(spec)

        v = R.verdicts(_records(rate))
        assert v["mat1"]["verdict"] == "NOT CONFIRMED"
        d = R.direction(v)
        assert "property of the shield" in d
        assert "generality claim" in d

    def test_a_failed_ordering_corrects_the_narrative(self):
        recs = _records(_all_confirm)
        _set_ind(recs, "RCM28__ATK2-31", _ind(0.66, 300))
        v = R.verdicts(recs)
        assert v["mat2"]["verdict"] == "NOT CONFIRMED"
        d = R.direction(v)
        assert "RCM28 < FT-B-41" in d
        assert "narrative is corrected" in d

    def test_a_concentrated_meta_game_names_the_recommendation(self):
        # one defender dominates every column, so the equilibrium puts all of
        # the defender's weight on it
        def rate(spec):
            return 0.05 if spec["defender"] == "RCM28" else 0.80

        v = R.verdicts(_records(rate))
        assert v["meta_deadlock"]["row_support"] == ["RCM28"]
        d = R.direction(v)
        assert "RCM28" in d
        assert "recommendation" in d

    def test_a_violation_stops_the_round_before_anything_else_is_read(self):
        bad = R.cell_name("RC19", "CTL-D225")
        recs = _records(
            _all_confirm, violations_fn=lambda s: 1 if s["name"] == bad else 0
        )
        d = R.direction(R.verdicts(recs))
        assert d.startswith("H-SAFE FAILS")
        assert "Stop" in d
        assert bad in d

    def test_an_incomplete_round_reads_nothing(self):
        recs = _records(_all_confirm)
        del recs["RC19__HUG7"]
        assert R.direction(R.verdicts(recs)).startswith("incomplete")


# ------------------------------------------------------------- cell bookkeeping
class TestRunCells:
    def test_skips_cells_that_already_have_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        done, todo = "RC19__CTL-D225", "RC19__HUG7"
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

    def test_records_are_written_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        name = "STR14__CTL-D125"
        R.cells_dir().mkdir(parents=True)
        R.write_cell(_record(name))
        assert json.loads((R.cells_dir() / f"{name}.json").read_text())["name"] == name
        assert list(R.cells_dir().glob("*.tmp")) == []

    def test_the_cells_directory_carries_the_s14_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.cells_dir() == tmp_path / "s14_cells"


# -------------------------------------------------------------------- summarize
class TestSummarize:
    def test_writes_round_json_and_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        for rec in _records(_all_confirm).values():
            R.write_cell(rec)
        summary = R.summarize()
        assert (tmp_path / "s14_round.json").exists()
        assert (tmp_path / "s14_round.png").exists()
        blob = json.loads((tmp_path / "s14_round.json").read_text())
        v = blob["verdicts"]
        assert v["mat1"]["verdict"] == "CONFIRMED"
        assert v["mat2"]["verdict"] == "CONFIRMED"
        assert v["hsafe"]["verdict"] == "HOLDS"
        assert v["complete"] is True
        assert v["meta_deadlock"]["value"] is not None
        assert v["meta_capture"]["value"] is not None
        assert blob["matrices"]["deadlock"]["rows"] == list(R.DEFENDERS)
        assert blob["matrices"]["capture"]["cols"] == list(R.ATTACKERS)
        assert summary["direction"] == R.direction(summary["verdicts"])
        assert len(blob["cells"]) == 42
        first = next(iter(blob["cells"].values()))
        assert "deadlock_ind" not in first and "capture_ind" not in first

    def test_a_partial_round_still_draws(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        recs = _records(_all_confirm)
        for name in list(recs)[:6]:
            R.write_cell(recs[name])
        blob = R.summarize()
        assert (tmp_path / "s14_round.png").exists()
        assert blob["verdicts"]["complete"] is False
