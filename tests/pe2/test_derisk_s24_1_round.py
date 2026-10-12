"""scripts/derisk_s24_1_gate.py — the S24.1 gate's cell table and verdicts.

Pre-registration: `docs/pathway/tactics_prereg_s24_1_2026-10-13.md` sections 4
to 6 with Amendments A1 and A2. Everything here is either a constant the prereg
fixes, a verdict rule it states, or one of the properties the round's only claim
rests on:

- the cell table really is 8 x 3, and the ONLY difference between the rows is
  the pursuer's `select` — asserted on the BUILT env, because "a capture
  difference is attributable to choosing" is false the moment anything else
  differs;
- a learned row REFUSES to run without its frozen final checkpoint;
- H-TAC-1's direction is reported even when it fails, and an inverted direction
  is a FAIL and not a pass;
- H-TAC-2 reads the per-DECISION `picks` histogram (A2.2) and FAILS on a
  selector whose profile is the same against every opponent — and on a
  single-tactic default, which section 4 names as a failure and not a vacuum;
- CG-T1 and A2.1 REFUSE the summary before any capture rate is read;
- CG-T2 labels a row that does not switch `LAW-LIKE`, and the label travels with
  the number;
- record validity (S19 A4.4/A4.5/A5.2): a record at the wrong N or stream is
  REFUSED, a crashed cell is UNDETERMINED and never re-run, duplicates reject
  BOTH;
- CG-T3: the watched seeds are the first three GATE seeds (a rule, not a pick),
  the pass refuses without a measurable record, no caption carries a measured
  number, and `--gifs` dispatches before `--summarize`;
- the launch wrapper refuses on the Step 1.1 preconditions, checked on fixture
  probe JSONs through the wrapper's OWN checker.

No real 500-episode cell runs here: one N = 3 smoke per row type into a tmp
dir, the learned one driven by a FRESHLY INITIALISED MaskablePPO (the arms are
still training; no checkpoint exists yet).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

from pe2.config import EnvConfig
from pe2.selector_mdp import ACTIONS, OBS_LEN, SelectorMDP
from pe2.structured import HOLD

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s24_1_gate.py"
_LAUNCH = _PATH.parent / "launch_s24_1.sh"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s24_1_gate", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic records below
ALL = ("GREEDY", "HUG", "TACTIC-V0")
SEL, NOMEM = "SEL-80", "NOMEM-80"
EVEN = dict.fromkeys(ACTIONS, 20)  # a flat pick histogram


# ----------------------------------------------------------------------------
# synthetic records, in the shape `run_cell` writes
# ----------------------------------------------------------------------------
def _caps(rate: float, n: int = N) -> list[bool]:
    k = round(rate * n)
    return [i < k for i in range(n)]


def _cell(
    name: str,
    caps: list[bool],
    picks: dict | None = None,
    *,
    switches: float = 5.0,
):
    """One cell record with only the fields the verdicts read. The per-episode
    arrays and the aggregate agree, so a verdict that recomputed one from the
    other would still say the same thing."""
    spec = R.CELLS[name]
    n = len(caps)
    picks = dict(EVEN) if picks is None else picks
    total = max(1, sum(picks.values()))
    return {
        "name": name,
        "row": spec["row"],
        "opponent": spec["opponent"],
        "in_distribution": spec["in_distribution"],
        "kind": spec["kind"],
        "memory": spec.get("memory"),
        "n_episodes": n,
        "seed0": R.GATE_SEED0,
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "violations": [0] * n,
        "violation_steps": 0,
        "n_switches": [switches] * n,
        "n_zero_length_spans": [0] * n,
        "n_decisions": [sum(picks.values())] * n,
        "inapplicable_picks": 0,
        "picks": {a: int(picks.get(a, 0)) for a in ACTIONS},
        "usage_steps": {a: int(picks.get(a, 0)) for a in ACTIONS},
        "search_policy_steps": 0,
        "aggregate": {
            "capture_rate": sum(caps) / n,
            "mean_switches": float(switches),
            "picks_share": {a: picks.get(a, 0) / total for a in ACTIONS},
        },
        "provenance": {"spec_sha256": "x"},
    }


def _round(per_row: dict, **kw) -> dict:
    """`per_row` maps a row to `{opponent: (caps, picks)}`."""
    out = {}
    for row, by_opp in per_row.items():
        for o, (caps, picks) in by_opp.items():
            name = R.cell_name(row, o)
            out[name] = _cell(name, caps, picks, **kw)
    return out


def _flat(caps, picks=None):
    return dict.fromkeys(ALL, (caps, picks))


# Profiles: `LEAN` differs by opponent, `SAME` does not, `ONLY_LEAD` is the
# single-tactic default section 4 names.
LEAN = {
    "GREEDY": {"INTERCEPT": 60, "LEAD": 20, "COUNTER-ORBIT": 5, "AMBUSH": 5, "SWEEP": 10},
    "HUG": {"INTERCEPT": 10, "LEAD": 10, "COUNTER-ORBIT": 55, "AMBUSH": 5, "SWEEP": 20},
    "TACTIC-V0": {"INTERCEPT": 10, "LEAD": 10, "COUNTER-ORBIT": 5, "AMBUSH": 55, "SWEEP": 20},
}
RULES_PROFILE = {
    o: {"INTERCEPT": 20, "LEAD": 55, "COUNTER-ORBIT": 5, "AMBUSH": 0, "SWEEP": 20}
    for o in ALL
}
ONLY_LEAD = dict.fromkeys(ACTIONS, 0) | {"LEAD": 100}


def _pass_round() -> dict:
    """Every SEL seed beats `LEAD-1.0` on all three opponents, chooses
    differently per opponent and differently from `RULES-V0`, and beats its own
    `NOMEM` on `TACTIC-V0`."""
    recs = {}
    for s in R.SEEDS:
        recs |= _round({f"SEL-{s}": {o: (_caps(0.60), LEAN[o]) for o in ALL}})
        recs |= _round({f"NOMEM-{s}": _flat(_caps(0.30), dict(EVEN))})
    recs |= _round({R.BASELINE_ROW: _flat(_caps(0.20), ONLY_LEAD)}, switches=0.0)
    recs |= _round({R.RULES_ROW: {o: (_caps(0.40), RULES_PROFILE[o]) for o in ALL}})
    return recs


def _probes(train_viol: int = 0, bad: int = 0) -> dict:
    return {
        row: {
            "path": f"p_{row}.json",
            "missing": False,
            "rows": [
                {
                    "mark": R.TRAIN_TOTAL,
                    "train_violations": train_viol,
                    "train_inapplicable_picks": bad,
                    "violations": 0,
                    "inapplicable_picks": 0,
                }
            ],
        }
        for row in R.LEARNED_ROWS
    }


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_twentyfour_cells_eight_rows_three_opponents(self):
        assert len(R.CELLS) == 24
        assert list(R.ROWS) == [
            "SEL-80", "SEL-81", "SEL-82", "NOMEM-80", "NOMEM-81", "NOMEM-82",
            "LEAD-1.0", "RULES-V0",
        ]
        assert list(R.OPPONENTS) == ["GREEDY", "HUG", "TACTIC-V0"]
        assert R.SEL_ROWS == ("SEL-80", "SEL-81", "SEL-82")
        assert len(R.LEARNED_ROWS) == 6
        assert set(R.CELLS) == {f"{r}_{o}" for r in R.ROWS for o in R.OPPONENTS}

    def test_the_training_opponent_is_labelled_in_distribution(self):
        assert R.OPPONENTS["TACTIC-V0"]["in_distribution"] is True
        assert [R.CELLS[R.cell_name(SEL, o)]["in_distribution"] for o in ALL] == [
            False,
            False,
            True,
        ]

    def test_pre_registered_constants(self):
        assert R.N_EPISODES == 500
        assert R.GATE_SEED0 == 2_400_000
        assert R.P_GATE == 0.05
        assert R.P_CHI2 == 0.01
        assert R.PASS_MIN == 2
        assert R.SEEDS == (80, 81, 82)
        assert R.TRAIN_TOTAL == 3_000_000
        assert R.CG_T2_MIN_SWITCHES == 2.0
        assert R.OMP_THREADS == "4"
        assert R.BASELINE_ROW == "LEAD-1.0" and R.RULES_ROW == "RULES-V0"
        assert R.cell_config() == EnvConfig()
        assert R.cell_config().evader_speed_ratio == 1.2

    def test_the_checkpoint_path_is_step_1_1s_layout(self):
        assert R.ckpt_path("SEL-80") == pathlib.Path(
            "results/derisk/ckpts_S24-1-SEL-80/S24-1-SEL-80_3000000.zip"
        )
        assert R.probe_file("NOMEM-82") == pathlib.Path(
            "results/derisk/s24_1_probe_NOMEM-82.json"
        )

    @pytest.mark.parametrize("opponent", ALL)
    def test_built_envs_differ_only_in_the_pursuers_select(self, opponent):
        """The round's whole attribution rests on this: build one cell of every
        row and assert the selector is the only difference."""
        cfg = R.cell_config()
        envs = {row: R.build_cell_env(R.cell_name(row, opponent), cfg) for row in R.ROWS}
        for row, env in envs.items():
            game = env.game if R.ROWS[row]["kind"] == "learned" else env
            assert type(game).__name__ == "TacticGameMDP"
            assert game.cfg is cfg
            assert game.hold == HOLD == 10
            assert (game.shield_obs, game.escape_obs) == (True, True)
            assert game.stall_penalty == 0.0
            assert game.intervention_penalty == 0.0
            assert game._driver is game.inner  # reward_wrapper=None
            # D131's recovery shield, fresh per episode, at the CBF defaults
            assert type(game.shield).__name__ == "RecoveryShieldV2"
            assert (game.shield.margin, game.shield.gamma) == (0.0, 0.2)
            # the same five-tactic library in every row
            assert len(game.law.tactics) == len(R.ROWS and game.law.tactics)
            assert {t.name.split("(")[0] for t in game.law.tactics} == set(ACTIONS)
        # the learned rows are the SAME env class and observation width
        learned = [envs[r] for r in R.LEARNED_ROWS]
        assert all(isinstance(e, SelectorMDP) for e in learned)
        assert {e.observation_space.shape for e in learned} == {(OBS_LEN,)}
        assert [e.memory for e in learned] == [True] * 3 + [False] * 3

    def test_the_lead_row_is_the_lead_tactic_and_rules_v0_is_the_rule_table(self):
        from pe2.planner import LeadPlannerInterceptLaw
        from pe2.selectors import PursuerRulesV0

        cfg = R.cell_config()
        lead = R.build_cell_env(R.cell_name("LEAD-1.0", "GREEDY"), cfg)
        obs = lead._begin(R.GATE_SEED0)
        picked = lead.law.select(obs, lead.inner.env, lead.law.history)
        inner = picked.tactic if hasattr(picked, "tactic") else picked
        assert type(inner.law) is LeadPlannerInterceptLaw
        assert inner.law.tau == 1.0  # S23's LEAD-1.0
        rules = R.build_cell_env(R.cell_name("RULES-V0", "GREEDY"), cfg)
        assert isinstance(rules.law.select, PursuerRulesV0)

    @pytest.mark.parametrize(("opponent", "cls"), [("GREEDY", "GreedyEvader"), ("HUG", "ObstacleHuggingEvader")])
    def test_the_scripted_opponents_occupy_the_evader_seat(self, opponent, cls):
        env = R.build_cell_env(R.cell_name(SEL, opponent))
        assert env.game.inner.opponent_factory.__name__ == cls
        # ... and TACTIC-V0 keeps the assembly's own tactic adapter
        tv0 = R.build_cell_env(R.cell_name(SEL, "TACTIC-V0"))
        assert type(tv0.game.inner.opponent_factory()).__name__ == "TacticOpponent"

    def test_unknown_cell_name_refuses(self):
        with pytest.raises(SystemExit, match="unknown cell"):
            R.run_cells(["SEL-80_NOPE"])

    def test_a_learned_row_refuses_without_its_final_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.require_ckpt("SEL-81")
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("an episode ran before refusal")
        )
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.run_cells(["SEL-81_HUG"])
        # ... and a chain is refused whole, before its FIRST cell, when any
        # learned row in it has no checkpoint
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.run_cells(["LEAD-1.0_HUG", "SEL-81_HUG"])


# ----------------------------------------------------------------------------
# record validity (S19 A4.4), the crashed cell (A4.5) and duplicates (A5.2)
# ----------------------------------------------------------------------------
def _on_disk(out: pathlib.Path, rec: dict, name: str | None = None) -> pathlib.Path:
    d = out / "s24_1_cells"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name or rec['name']}.json"
    path.write_text(json.dumps(rec))
    return path


def _measurable(name: str) -> dict:
    return _cell(name, _caps(0.5, R.N_EPISODES))


class TestRecordValidity:
    def test_a_measurable_record_is_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("SEL-80_GREEDY"))
        assert R.record_status(_measurable("SEL-80_GREEDY")) == ("MEASURABLE", "")
        assert list(R.load_records()) == ["SEL-80_GREEDY"]

    @pytest.mark.parametrize(("field", "value"), [("n_episodes", 20), ("seed0", 2_350_000)])
    def test_the_wrong_n_or_stream_is_not_measurable(self, tmp_path, monkeypatch, field, value):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable("SEL-80_GREEDY") | {field: value}
        assert R.record_status(rec)[0] == "INVALID"
        _on_disk(tmp_path, rec)
        assert R.load_records(quiet=True) == {}

    def test_a_foreign_name_is_not_measurable(self):
        assert R.record_status(_measurable("SEL-80_GREEDY") | {"name": "S23-THING"})[0] == "INVALID"

    def test_an_invalid_record_is_refused_not_overwritten(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("LEAD-1.0_GREEDY") | {"n_episodes": 3})
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("the smoke was overwritten")
        )
        with pytest.raises(SystemExit, match="REFUSED before any episode ran"):
            R.run_cells(["LEAD-1.0_GREEDY"])

    def test_a_crashed_cell_is_undetermined_and_never_re_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)

        def _crash(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(R, "run_cell", _crash)
        R.run_cells(["LEAD-1.0_GREEDY"])
        rec = json.loads((tmp_path / "s24_1_cells" / "LEAD-1.0_GREEDY.json").read_text())
        assert rec["verdict"] == R.UNDETERMINED
        assert rec["n_episodes"] == 0
        assert R.record_status(rec)[0] == R.UNDETERMINED
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("an UNDETERMINED cell re-ran")
        )
        with pytest.raises(SystemExit, match="REFUSED before any episode ran"):
            R.run_cells(["LEAD-1.0_GREEDY"])

    def test_two_files_claiming_one_cell_reject_both(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable("LEAD-1.0_GREEDY")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a-copy-by-hand")
        measurable, rejected, by_cell = R.scan_records()
        assert measurable == {}
        assert len(rejected) == 2
        assert set(by_cell) == {"LEAD-1.0_GREEDY"}
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("a poisoned cell re-ran")
        )
        with pytest.raises(SystemExit, match="duplicate"):
            R.run_cells(["LEAD-1.0_GREEDY"])

    def test_a_measurable_record_is_skipped_as_done(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("LEAD-1.0_GREEDY"))
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("a recorded cell re-ran")
        )
        assert R.run_cells(["LEAD-1.0_GREEDY"]) == []

    def test_no_rate_is_printed_while_the_cells_run(self):
        line = R._cell_line(_measurable("SEL-80_GREEDY"))
        assert line == "[SEL-80_GREEDY] recorded, n_episodes 500"
        for forbidden in ("capture", "switch", "viol", "pick"):
            assert forbidden not in line


# ----------------------------------------------------------------------------
# H-TAC-1 / H-TAC-1b
# ----------------------------------------------------------------------------
class TestHTac1:
    def test_passes_when_the_selector_captures_more_on_every_opponent(self):
        v = R.h_tac_1(_pass_round(), 80)
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 3
        assert v["comparator"] == "LEAD-1.0"
        assert all(b["direction"] == "SEL-80 > LEAD-1.0" for b in v["opponents"].values())

    def test_two_of_three_opponents_is_enough(self):
        recs = _pass_round()
        name = R.cell_name(SEL, "HUG")
        recs[name] = _cell(name, _caps(0.20), LEAN["HUG"])
        v = R.h_tac_1(recs, 80)
        assert v["verdict"] == "PASS"
        assert (v["n_pass"], v["pass_min"]) == (2, 2)
        assert v["opponents"]["HUG"]["verdict"] == "FAIL"

    def test_fails_when_the_direction_inverts_and_still_reports_it(self):
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.05), LEAN[o])
        v = R.h_tac_1(recs, 80)
        assert v["verdict"] == "FAIL"
        assert v["n_pass"] == 0
        for b in v["opponents"].values():
            assert b["direction"] == "SEL-80 < LEAD-1.0"
            assert b["delta"] < 0

    def test_a_tie_is_not_a_pass(self):
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.20), LEAN[o])
        v = R.h_tac_1(recs, 80)
        assert v["verdict"] == "FAIL"
        assert all(b["direction"] == "SEL-80 == LEAD-1.0" for b in v["opponents"].values())

    def test_incomplete_when_a_comparator_cell_is_missing(self):
        recs = _pass_round()
        del recs[R.cell_name("LEAD-1.0", "HUG")]
        v = R.h_tac_1(recs, 80)
        assert v["verdict"] == "INCOMPLETE"
        assert v["opponents"]["HUG"]["missing"] == ["LEAD-1.0_HUG"]

    def test_the_round_needs_two_of_three_seeds(self):
        recs = _pass_round()
        for o in ALL:  # seed 82 loses everywhere
            name = R.cell_name("SEL-82", o)
            recs[name] = _cell(name, _caps(0.05), LEAN[o])
        v = R.h_tac_1_round(recs)
        assert v["verdict"] == "PASS"
        assert v["passing_seeds"] == ["80", "81"]
        assert v["gated"] is True
        # one more losing seed and the round fails
        for o in ALL:
            name = R.cell_name("SEL-81", o)
            recs[name] = _cell(name, _caps(0.05), LEAN[o])
        assert R.h_tac_1_round(recs)["verdict"] == "FAIL"

    def test_h_tac_1b_is_reported_not_gated(self):
        v = R.h_tac_1b_round(_pass_round())
        assert v["gated"] is False
        assert v["seeds"]["80"]["comparator"] == "RULES-V0"
        assert v["seeds"]["80"]["verdict"] == "PASS"  # 0.60 vs 0.40


# ----------------------------------------------------------------------------
# H-TAC-2 (A2.2: the per-DECISION picks histogram)
# ----------------------------------------------------------------------------
class TestChi2:
    def test_it_matches_a_hand_computed_table(self):
        """2x3, row sums 60/60 and column sums 40/40/40, so every expected cell
        is 20 and chi^2 = 2 * (100 + 0 + 100) / 20 = 20 on 2 dof."""
        got = R.chi2_table([[10, 20, 30, 0, 0], [30, 20, 10, 0, 0]])
        assert got["verdict"] == "OK"
        assert got["chi2"] == pytest.approx(20.0)
        assert got["dof"] == 2  # the all-zero columns are dropped
        assert got["p"] < 0.01 and got["differs"] is True

    def test_all_zero_rows_and_columns_are_dropped_and_degenerate_is_named(self):
        one_tactic = R.chi2_table([[0, 5, 0, 0, 0], [0, 7, 0, 0, 0]])
        assert one_tactic["verdict"] == "DEGENERATE"
        assert (one_tactic["n_nonzero_cols"], one_tactic["n"]) == (1, 12)
        empty = R.chi2_table([[0] * 5, [0] * 5])
        assert empty["verdict"] == "DEGENERATE" and empty["n"] == 0


class TestHTac2:
    def test_passes_when_the_profile_differs_by_opponent_and_from_rules_v0(self):
        v = R.h_tac_2(_pass_round(), 80)
        assert v["verdict"] == "PASS"
        assert v["across_opponents"]["verdict"] == "PASS"
        assert v["n_pass_vs_rules"] == 3
        assert v["across_opponents"]["tactics"] == list(ACTIONS)

    def test_fails_when_the_usage_is_identical_across_opponents(self):
        """A selector with ONE profile has not chosen per opponent, however much
        it captures."""
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), LEAN["GREEDY"])
        v = R.h_tac_2(recs, 80)
        assert v["across_opponents"]["verdict"] == "FAIL"
        assert v["across_opponents"]["p"] == pytest.approx(1.0)
        assert v["verdict"] == "FAIL"
        # ... while H-TAC-1 passes on the very same records
        assert R.h_tac_1(recs, 80)["verdict"] == "PASS"

    def test_a_single_tactic_default_fails_and_is_not_vacuous(self):
        """Section 4: 'a selector that passes H-TAC-1 by always picking LEAD has
        not chosen'. The table is degenerate but the verdict is FAIL."""
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), ONLY_LEAD)
        v = R.h_tac_2(recs, 80)
        assert v["across_opponents"]["verdict"] == "FAIL"
        assert "single-tactic default" in v["across_opponents"]["note"]
        assert v["verdict"] == "FAIL"

    def test_an_empty_histogram_is_vacuous_never_pass(self):
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), dict.fromkeys(ACTIONS, 0))
        v = R.h_tac_2(recs, 80)
        assert v["across_opponents"]["verdict"] == "VACUOUS"
        assert v["verdict"] == "FAIL"  # vacuous is never a PASS

    def test_a_profile_equal_to_rules_v0_fails_part_b(self):
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), RULES_PROFILE[o])
        v = R.h_tac_2(recs, 80)
        assert [b["verdict"] for b in v["vs_rules_v0"].values()] == ["FAIL"] * 3
        assert v["verdict"] == "FAIL"

    def test_two_of_three_opponents_is_enough_for_part_b(self):
        recs = _pass_round()
        name = R.cell_name(SEL, "HUG")
        recs[name] = _cell(name, _caps(0.60), RULES_PROFILE["HUG"])
        v = R.h_tac_2(recs, 80)
        assert v["vs_rules_v0"]["HUG"]["verdict"] == "FAIL"
        assert (v["n_pass_vs_rules"], v["verdict"]) == (2, "PASS")

    def test_it_reads_picks_and_never_usage_steps(self):
        """A2.2, asserted on the source as well as on the behaviour: a
        per-env-step table would weigh a span by its length."""
        recs = _pass_round()
        for o in ALL:
            recs[R.cell_name(SEL, o)]["usage_steps"] = {"LEAD": 10_000}
        assert R.h_tac_2(recs, 80)["verdict"] == "PASS"
        assert "_picks_row" in _PATH.read_text()
        assert 'rec["usage_steps"]' not in _PATH.read_text()

    def test_incomplete_without_the_rules_v0_comparator(self):
        recs = _pass_round()
        del recs[R.cell_name("RULES-V0", "GREEDY")]
        assert R.h_tac_2(recs, 80)["verdict"] == "INCOMPLETE"

    def test_the_round_needs_two_of_three_seeds(self):
        recs = _pass_round()
        for o in ALL:
            recs[R.cell_name("SEL-82", o)] = _cell(
                R.cell_name("SEL-82", o), _caps(0.60), LEAN["GREEDY"]
            )
        v = R.h_tac_2_round(recs)
        assert v["passing_seeds"] == ["80", "81"]
        assert v["verdict"] == "PASS"


# ----------------------------------------------------------------------------
# H-TAC-3
# ----------------------------------------------------------------------------
class TestHTac3:
    def test_passes_when_sel_beats_its_own_nomem_on_the_training_opponent(self):
        v = R.h_tac_3(_pass_round(), 80)
        assert v["verdict"] == "PASS"
        assert v["comparator"] == NOMEM
        assert list(v["opponents"]) == ["TACTIC-V0"]
        assert v["opponents"]["TACTIC-V0"]["in_distribution"] is True

    def test_fails_when_nomem_matches_or_beats_it(self):
        recs = _pass_round()
        name = R.cell_name(NOMEM, "TACTIC-V0")
        recs[name] = _cell(name, _caps(0.60), dict(EVEN))
        assert R.h_tac_3(recs, 80)["verdict"] == "FAIL"
        recs[name] = _cell(name, _caps(0.80), dict(EVEN))
        v = R.h_tac_3(recs, 80)
        assert v["verdict"] == "FAIL"
        assert v["opponents"]["TACTIC-V0"]["direction"] == "SEL-80 < NOMEM-80"

    def test_only_the_training_opponent_is_tested(self):
        recs = _pass_round()
        for o in ("GREEDY", "HUG"):  # NOMEM wins off-distribution: irrelevant
            name = R.cell_name(NOMEM, o)
            recs[name] = _cell(name, _caps(0.99), dict(EVEN))
        assert R.h_tac_3(recs, 80)["verdict"] == "PASS"

    def test_the_round_needs_two_of_three_seeds(self):
        recs = _pass_round()
        name = R.cell_name("NOMEM-82", "TACTIC-V0")
        recs[name] = _cell(name, _caps(0.90), dict(EVEN))
        v = R.h_tac_3_round(recs)
        assert (v["verdict"], v["passing_seeds"]) == ("PASS", ["80", "81"])

    def test_the_ablation_is_named_as_inputs_only(self):
        assert "INPUTS only" in R.h_tac_3(_pass_round(), 80)["ablation"]


# ----------------------------------------------------------------------------
# H-SAFE, CG-T1, A2.1, CG-T2
# ----------------------------------------------------------------------------
class TestHSafe:
    def test_holds_at_zero_contact_steps_in_all_cells_and_in_training(self):
        v = R.h_safe(_pass_round(), _probes())
        assert v["verdict"] == "HOLDS"
        assert v["violation_steps"] == 0
        assert len(v["gated_cells"]) == 24

    def test_one_gate_contact_step_anywhere_fails_it(self):
        recs = _pass_round()
        recs["SEL-81_HUG"]["violation_steps"] = 1
        v = R.h_safe(recs, _probes())
        assert v["verdict"] == "FAILS"
        assert v["offenders"] == ["SEL-81_HUG"]

    def test_training_contact_fails_it_too(self):
        v = R.h_safe(_pass_round(), _probes(train_viol=4))
        assert v["verdict"] == "FAILS"
        assert v["training_offenders"] == sorted(R.LEARNED_ROWS)
        assert v["training"]["SEL-80"]["train_violations_max"] == 4

    def test_incomplete_while_a_cell_is_missing(self):
        recs = _pass_round()
        del recs["RULES-V0_HUG"]
        assert R.h_safe(recs, _probes())["verdict"] == "INCOMPLETE"


class TestConstructGates:
    def test_cg_t1_passes_at_zero_inapplicable_picks(self):
        v = R.cg_t1(_pass_round(), _probes())
        assert v["verdict"] == "PASS"
        assert set(v["cells"]) == {R.cell_name(r, o) for r in R.LEARNED_ROWS for o in ALL}

    def test_cg_t1_refuses_on_a_single_gate_pick(self):
        recs = _pass_round()
        recs["SEL-80_GREEDY"]["inapplicable_picks"] = 1
        v = R.cg_t1(recs, _probes())
        assert v["verdict"] == "REFUSE"
        assert v["offenders"] == ["SEL-80_GREEDY"]

    def test_cg_t1_refuses_on_a_training_pick(self):
        v = R.cg_t1(_pass_round(), _probes(bad=1))
        assert v["verdict"] == "REFUSE"
        assert v["offenders"] == sorted(R.LEARNED_ROWS)

    def test_cg_t1_refuses_the_summary_before_any_rate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "read_probes", lambda: _probes())
        rec = _measurable("SEL-80_GREEDY")
        rec["inapplicable_picks"] = 2
        _on_disk(tmp_path, rec)
        monkeypatch.setattr(
            R, "write_figure", lambda *a, **k: pytest.fail("a table was written")
        )
        with pytest.raises(SystemExit, match="CG-T1"):
            R.summarize()
        assert not (tmp_path / "s24_1_round.json").exists()

    def test_a2_1_refuses_on_a_search_policy_step_in_a_learned_cell(self, tmp_path, monkeypatch):
        recs = _pass_round()
        recs["NOMEM-82_HUG"]["search_policy_steps"] = 3
        v = R.cg_a2_1(recs)
        assert v["verdict"] == "REFUSE"
        assert v["offenders"] == ["NOMEM-82_HUG"]
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "read_probes", lambda: _probes())
        _on_disk(tmp_path, _measurable("NOMEM-82_HUG") | {"search_policy_steps": 3})
        with pytest.raises(SystemExit, match="A2.1"):
            R.summarize()

    def test_a2_1_ignores_the_comparator_rows(self):
        """Reading 5: LEAD-1.0's law is DONE on entry whenever the evader is
        unseen, exactly as S23 measured it, and those steps are the search
        policy's by design."""
        recs = _pass_round()
        recs["LEAD-1.0_GREEDY"]["search_policy_steps"] = 2054
        assert R.cg_a2_1(recs)["verdict"] == "PASS"

    def test_cg_t2_labels_a_row_that_does_not_switch(self):
        v = R.cg_t2(_pass_round())
        assert v["verdict"] == "REPORTED"  # reported, not gated
        assert v["law_like_rows"] == ["LEAD-1.0"]
        assert v["rows"]["LEAD-1.0"]["mean_switches"] == 0.0
        assert v["rows"][SEL]["label"] == "SWITCHES"
        assert R.row_label(v, "LEAD-1.0") == "LEAD-1.0 [LAW-LIKE]"
        assert R.row_label(v, SEL) == SEL

    def test_the_label_travels_with_the_number_in_every_table(self):
        recs = _pass_round()
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), LEAN[o], switches=1.0)
        v = R.verdicts(recs, _probes())
        assert v["cg_t2"]["rows"][SEL]["label"] == "LAW-LIKE"
        assert v["labels"][SEL] == "SEL-80 [LAW-LIKE]"
        # and a mean of exactly 2 is NOT law-like (">= 2")
        for o in ALL:
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), LEAN[o], switches=2.0)
        assert R.cg_t2(recs)["rows"][SEL]["label"] == "SWITCHES"

    def test_the_mean_pools_the_rows_three_opponents_and_reports_each(self):
        recs = _pass_round()
        for o, s in zip(ALL, (0.0, 3.0, 3.0), strict=True):
            name = R.cell_name(SEL, o)
            recs[name] = _cell(name, _caps(0.60), LEAN[o], switches=s)
        block = R.cg_t2(recs)["rows"][SEL]
        assert block["mean_switches"] == pytest.approx(2.0)
        assert block["per_opponent"] == {"GREEDY": 0.0, "HUG": 3.0, "TACTIC-V0": 3.0}


# ----------------------------------------------------------------------------
# direction() — section 4's refutation paragraph
# ----------------------------------------------------------------------------
class TestDirection:
    def test_hsafe_failure_precedes_everything(self):
        recs = _pass_round()
        recs["SEL-80_GREEDY"]["violation_steps"] = 3
        text = R.direction(R.verdicts(recs, _probes()))
        assert text.startswith("H-SAFE FAILS")
        assert "nothing else is read" in text

    def test_cg_t1_precedes_the_hypotheses(self):
        recs = _pass_round()
        recs["SEL-80_GREEDY"]["inapplicable_picks"] = 1
        text = R.direction(R.verdicts(recs, _probes()))
        assert text.startswith("CG-T1 REFUSES")
        assert "H-TAC-1 PASSES" not in text

    def test_a2_1_precedes_the_hypotheses(self):
        recs = _pass_round()
        recs["SEL-80_GREEDY"]["search_policy_steps"] = 7
        text = R.direction(R.verdicts(recs, _probes()))
        assert text.startswith("A2.1 REFUSES")

    def test_incompleteness_precedes_the_hypotheses(self):
        recs = _pass_round()
        del recs["RULES-V0_TACTIC-V0"]
        assert "incomplete" in R.direction(R.verdicts(recs, _probes()))

    def test_the_pass_narrative_names_the_passing_seeds(self):
        text = R.direction(R.verdicts(_pass_round(), _probes()))
        assert "H-TAC-1 PASSES on seeds 80, 81, 82" in text
        assert "SEL-80: H-TAC-2 PASS" in text
        assert "SEL-80: H-TAC-3 PASS" in text
        assert "S24.2 IS NOT LICENSED" not in text
        assert "in-distribution" in text  # TACTIC-V0 is labelled in every table
        assert "AMBUSH" in text  # section 2's expectation rides along

    def test_h_tac_1_failing_everywhere_blocks_s24_2(self):
        recs = _pass_round()
        for s in R.SEEDS:
            for o in ALL:
                name = R.cell_name(f"SEL-{s}", o)
                recs[name] = _cell(name, _caps(0.05), LEAN[o])
        text = R.direction(R.verdicts(recs, _probes()))
        assert "H-TAC-1 FAILS" in text
        assert "does not beat the best fixed law" in text
        assert "S24.2 IS NOT LICENSED" in text

    def test_a_pass_without_a_mechanism_is_a_single_tactic_default(self):
        recs = _pass_round()
        for s in R.SEEDS:
            for o in ALL:
                name = R.cell_name(f"SEL-{s}", o)
                recs[name] = _cell(name, _caps(0.60), ONLY_LEAD)
        text = R.direction(R.verdicts(recs, _probes()))
        assert "H-TAC-1 passes but H-TAC-2 FAILS" in text
        assert "SINGLE-TACTIC DEFAULT" in text
        assert "the selector is not the contribution" in text

    def test_h_tac_3_failing_says_memory_is_inert_and_the_sweep_still_runs(self):
        recs = _pass_round()
        for s in R.SEEDS:
            name = R.cell_name(f"NOMEM-{s}", "TACTIC-V0")
            recs[name] = _cell(name, _caps(0.90), dict(EVEN))
        text = R.direction(R.verdicts(recs, _probes()))
        assert "H-TAC-3 FAILS" in text
        assert "memory is INERT" in text
        assert "speed sweep is still" in text
        assert "A2.3" in text

    def test_the_law_like_label_is_named_in_the_narrative(self):
        text = R.direction(R.verdicts(_pass_round(), _probes()))
        assert "CG-T2" in text and "LAW-LIKE" in text and "LEAD-1.0" in text


# ----------------------------------------------------------------------------
# CG-T3 — the watched episodes
# ----------------------------------------------------------------------------
class TestGifs:
    def test_the_seeds_are_the_first_three_gate_seeds_and_a_rule(self):
        assert R.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)
        assert R.GIF_N == 3
        # a rule, not a pick: arithmetic on the gate stream, with no scan and
        # no frozen-seeds artefact to choose from (S23 needed one; this does not)
        assert R.gif_seeds() == tuple(
            R.GATE_SEED0 + i for i in range(R.GIF_N)
        )

    def test_thirty_gifs_five_rows_two_opponents_three_seeds(self):
        assert R.GIF_ROWS == ("SEL-80", "SEL-81", "SEL-82", "NOMEM-80", "RULES-V0")
        assert R.GIF_OPPONENTS == ("TACTIC-V0", "GREEDY")
        assert len(R.GIF_ROWS) * len(R.GIF_OPPONENTS) * R.GIF_N == 30

    def test_gif_paths_are_row_opponent_and_seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.gif_path("NOMEM-80", "GREEDY", 2_400_001) == (
            tmp_path / "s24_1_gifs" / "NOMEM-80_GREEDY_2400001.gif"
        )

    def test_it_refuses_without_a_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered")
        )
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs(["RULES-V0"])

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "not measurable"),
            ({"seed0": 2_350_000}, "not measurable"),
            ({"verdict": R.UNDETERMINED, "n_episodes": 0}, "UNDETERMINED"),
        ],
    )
    def test_a_non_measurable_record_refuses_rather_than_captioning(
        self, tmp_path, monkeypatch, overrides, match
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("RULES-V0_TACTIC-V0") | overrides)
        monkeypatch.setattr(
            R, "render_cell", lambda *a, **k: pytest.fail("rendered from a bad record")
        )
        with pytest.raises(SystemExit, match=match):
            R.write_gifs(["RULES-V0"])

    def test_a_learned_row_refuses_without_its_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        for o in R.GIF_OPPONENTS:
            _on_disk(tmp_path, _measurable(R.cell_name(SEL, o)))
        monkeypatch.setattr(
            R, "render_cell", lambda *a, **k: pytest.fail("rendered without a policy")
        )
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            R.write_gifs([SEL])

    def test_unknown_row_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown CG-T3 row"):
            R.write_gifs(["NOMEM-81"])

    def test_existing_gifs_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s24_1_gifs").mkdir(parents=True)
        for o in R.GIF_OPPONENTS:
            for s in R.gif_seeds():
                R.gif_path("RULES-V0", o, s).write_bytes(b"")
        monkeypatch.setattr(R, "render_cell", lambda *a, **k: pytest.fail("re-rendered"))
        assert R.write_gifs(["RULES-V0"]) == []

    def test_the_caption_carries_no_measured_number(self):
        title = R.gif_title("SEL-80", "GREEDY", 2_400_000)
        assert title == "SEL-80 vs GREEDY · episode 2400000"
        rec = _measurable("SEL-80_GREEDY")
        for forbidden in (
            "capture",
            "switch",
            "pick",
            "measured",
            "N=",
            f"{rec['capture_rate']:.3f}",
            "PASS",
            "FAIL",
        ):
            assert forbidden not in title
        assert "rec" not in R.gif_title.__code__.co_varnames

    def test_the_frame_label_is_the_two_roles_and_the_belief_marker(self):
        """The format is the v0b game's own `frame_label`, and the PURSUER half
        is read off the law's CURRENT tactic inside the step the frame shows —
        A2 forbids indexing frames by a span's law-call boundaries."""
        env = R.build_cell_env(R.cell_name("RULES-V0", "TACTIC-V0"), recorder=True, label="x")
        assert env.caption("law") == "EVADER: UNKNOWN | PURSUER: UNKNOWN"
        assert re.fullmatch(r"EVADER: .+ \| PURSUER: .+", env.caption("law"))
        assert [type(o).__name__ for o in env._ov] == [
            "FovOverlay",
            "ShieldOverlay",
            "BeliefOverlay",
            "AimOverlay",
        ]
        # a scripted-evader cell names the evader instead of printing UNKNOWN
        greedy = R.build_cell_env(R.cell_name("RULES-V0", "GREEDY"), recorder=True, label="x")
        assert greedy.caption("law") == "EVADER: GREEDY | PURSUER: UNKNOWN"
        for forbidden in ("capture", "rate", "PASS"):
            assert forbidden not in greedy.caption("law")


# ----------------------------------------------------------------------------
# CLI dispatch order, provenance and an empty round
# ----------------------------------------------------------------------------
class TestCli:
    def test_gifs_are_written_before_the_table_is_summarised(self, monkeypatch):
        """CG-T3 is a BLOCKING construct gate: the episodes are watched before
        any capture rate is read."""
        order = []
        monkeypatch.setattr(R, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(R, "write_gifs", lambda *a, **k: order.append("gifs"))
        monkeypatch.setattr(
            R,
            "summarize",
            lambda *a, **k: (
                order.append("summarize"),
                {"verdicts": R.verdicts({}, None), "direction": "x"},
            )[1],
        )
        monkeypatch.setattr(
            "sys.argv", ["derisk_s24_1_gate.py", "--summarize", "--gifs", "--cells"]
        )
        R.main()
        assert order == ["cells", "gifs", "summarize"]

    def test_summarize_survives_a_round_with_no_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "read_probes", lambda: _probes())
        blob = R.summarize()
        assert blob["verdicts"]["n_records"] == 0
        assert "incomplete" in blob["direction"]
        assert (tmp_path / "s24_1_round.json").exists()
        assert (tmp_path / "s24_1_round.png").exists()

    def test_provenance_carries_the_preregs_sha256_and_the_shield(self):
        p = R.provenance("cuda")
        assert p["spec"] == "docs/pathway/tactics_prereg_s24_1_2026-10-13.md"
        assert p["spec_sha256"] == hashlib.sha256(R.spec_path().read_bytes()).hexdigest()
        assert R.spec_path().exists()
        assert p["device"] == "cuda"
        assert p["omp_num_threads"] is not None
        assert "RecoveryShieldV2" in p["shield"]
        assert p["label_branch"] == "labels: none (D147/D148)"
        assert p["gate_stream"] == 2_400_000

    def test_the_s23_helpers_are_imported_not_reimplemented(self):
        assert R._write_json is R.S23._write_json
        assert R._ci is R.S23._ci
        assert R._cell_line is R.S23._cell_line
        assert R.scripted_opponent is R.S19.opponent_factory
        assert R.UNDETERMINED == "UNDETERMINED"
        # and the v0b assembly is the SAME module object Step 1.1 loads
        assert R.GAME is sys.modules["s24_tactic_game"]


# ----------------------------------------------------------------------------
# the launch wrapper
# ----------------------------------------------------------------------------
def _launch_checker(tmp_path: pathlib.Path) -> pathlib.Path:
    """The wrapper's OWN probe checker, extracted from the heredoc so the
    refusal conditions are tested rather than described."""
    body = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", _LAUNCH.read_text(), re.DOTALL).group(1)
    path = tmp_path / "checker.py"
    path.write_text(body)
    return path


def _probe_fixture(root: pathlib.Path, rows: list[dict], arm: str = "SEL-80") -> None:
    d = root / "results" / "derisk"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"s24_1_probe_{arm}.json").write_text(json.dumps({"checkpoints": rows}))


CLEAN_ROW = {
    "mark": 3_000_000,
    "inapplicable_picks": 0,
    "train_inapplicable_picks": 0,
    "violations": 0,
    "train_violations": 0,
}


class TestLauncher:
    def _run(self, tmp_path, rows):
        _probe_fixture(tmp_path, rows)
        return subprocess.run(
            [sys.executable, str(_launch_checker(tmp_path)), str(tmp_path), "SEL-80"],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_it_accepts_clean_probe_rows(self, tmp_path):
        got = self._run(tmp_path, [dict(CLEAN_ROW)])
        assert got.returncode == 0
        assert "clean" in got.stdout

    @pytest.mark.parametrize(
        "key",
        ["train_violations", "violations", "inapplicable_picks", "train_inapplicable_picks"],
    )
    def test_it_refuses_on_any_non_zero_counter(self, tmp_path, key):
        got = self._run(tmp_path, [dict(CLEAN_ROW) | {key: 1}])
        assert got.returncode == 1
        assert "REFUSED" in got.stdout and key in got.stdout

    def test_it_refuses_an_arm_with_no_probe_rows(self, tmp_path):
        got = self._run(tmp_path, [])
        assert got.returncode == 1
        assert "no probe rows" in got.stdout

    def test_the_wrapper_gates_the_worktree_the_symlink_and_the_checkpoints(self):
        src = _LAUNCH.read_text()
        assert "REFUSED: another round is using the host" in src
        assert "no frozen worktree" in src
        assert "is not a symlink to" in src
        assert "without the gate runner" in src
        assert "S24-1-${a}_3000000.zip" in src
        assert "OMP_NUM_THREADS=4" in src
        assert "git rev-parse --short HEAD" in src
        assert "confirm by results/derisk/s24_1_cells/*.json, not by PIDs" in src

    def test_the_four_chains_cover_every_cell_exactly_once(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH.read_text().splitlines()
            if "--cells" in line
        ]
        assert len(chains) == 4
        assert all(len(c) == 6 for c in chains)
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(R.CELL_ORDER)
        assert len(set(flat)) == 24

    def test_the_wrapper_never_summarises_or_renders(self):
        """A rate may not be read by the launch: `--gifs` (CG-T3) comes first and
        both are run by hand."""
        for line in _LAUNCH.read_text().splitlines():
            if line.startswith("nohup"):
                assert "--summarize" not in line and "--gifs" not in line


# ----------------------------------------------------------------------------
# the smoke: one N = 3 cell per ROW TYPE, into a tmp dir
# ----------------------------------------------------------------------------
_RECORD_KEYS = (
    "capture_ind",
    "capture_rate",
    "wilson_capture",
    "steps",
    "violations",
    "violation_steps",
    "n_switches",
    "n_zero_length_spans",
    "n_decisions",
    "inapplicable_picks",
    "picks",
    "usage_steps",
    "search_policy_steps",
    "tactics",
    "branch",
    "device",
    "provenance",
)


class TestSmoke:
    def test_a_scripted_row_cell_carries_every_pre_registered_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = R.run_cell("RULES-V0_TACTIC-V0", n_episodes=3)
        for key in _RECORD_KEYS:
            assert key in rec, key
        assert len(rec["capture_ind"]) == len(rec["branch"]) == 3
        assert rec["seed0"] == R.GATE_SEED0
        assert rec["kind"] == "scripted"
        assert sum(rec["picks"].values()) == sum(rec["n_decisions"])
        assert rec["violation_steps"] == 0  # H-SAFE's quantity, on 3 episodes
        # the evader's tactic history is recorded against the tactic evader
        assert rec["tactics"][0]["evader"]
        assert rec["aggregate"]["evader_tactics_recorded"] is True
        path = R.write_cell(rec)
        assert json.loads(path.read_text())["name"] == "RULES-V0_TACTIC-V0"
        # a smoke is NOT a measurable record and may never become a number
        assert R.record_status(rec)[0] == "INVALID"
        assert R.load_records(quiet=True) == {}

    def test_a_learned_row_cell_runs_on_a_fresh_policy_and_keeps_the_invariants(
        self, tmp_path, monkeypatch
    ):
        """The arms are still training, so the smoke drives a FRESHLY
        INITIALISED MaskablePPO — enough to assert A2.1's and CG-T1's
        invariants, the mask being the env's and not the policy's."""
        from sb3_contrib import MaskablePPO

        monkeypatch.setattr(R, "OUT", tmp_path)
        made = MaskablePPO(
            "MlpPolicy",
            R.build_cell_env("SEL-80_GREEDY"),
            seed=0,
            gamma=0.99,
            device="cpu",
        )
        monkeypatch.setattr(R, "load_policy", lambda row, device="cpu": made)
        rec = R.run_cell("SEL-80_GREEDY", n_episodes=3)
        for key in _RECORD_KEYS:
            assert key in rec, key
        assert rec["kind"] == "learned" and rec["memory"] is True
        assert rec["inapplicable_picks"] == 0  # CG-T1
        assert rec["search_policy_steps"] == 0  # A2.1
        assert rec["violation_steps"] == 0  # H-SAFE
        assert sum(rec["picks"].values()) == sum(rec["n_decisions"]) > 0
        assert rec["aggregate"]["evader_tactics_recorded"] is False
        assert all(s >= 0 for s in rec["n_switches"])
