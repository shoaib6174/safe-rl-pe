"""scripts/derisk_s23_strategy.py — the S23 round's cell table and verdicts.

Spec: `docs/pathway/pursuit_strategy_prereg_2026-10-12.md`, sections 3 to 7.
Everything here is either a constant the prereg fixes, a verdict rule it states,
or one of the properties the round's only claim rests on:

- the cell table really is 4 x 1 x 3, and the ONLY difference between the rows
  is the LAW OBJECT — the aim point. That is asserted on the BUILT env, not on
  the table's own bookkeeping, because "a capture difference is attributable to
  the aim point" is false the moment any other ingredient differs;
- the ATK opponent REFUSES to run without its frozen checkpoint;
- CG-B (prereg section 5) refuses inside `run_cells`, BEFORE the first episode,
  when a replayed S18 episode's `intervention_total` moved — the additive
  fields must be additive;
- H-STRAT-1's direction is reported even when it fails, and an inverted
  direction is a FAIL and not a pass;
- H-STRAT-2's ORBIT set is a property of the `PLAN-SHIELD` BASELINE alone, and
  an opponent with fewer than 20 orbit episodes is VACUOUS, never PASS;
- the NULL FAILS against a law that is not `PLAN` on an obstacle-free arena
  against a stationary evader, and `direction()` then says no verdict is about
  the aim point;
- record validity (S19 A4.4/A4.5/A5.2): a record at the wrong N or stream is
  REFUSED (not skipped as done, not overwritten, never summarised), a crashed
  cell is UNDETERMINED and never re-run, and duplicates reject BOTH;
- CG-S3 (A5.1/A4.6): `--gifs` uses EXACTLY the three frozen seeds and refuses
  without them, no caption carries a measured number, no rate is printed while
  the cells run, and `--gifs` dispatches before `--summarize`.

No real 500-episode cell runs here: one N = 3 smoke cell into a tmp dir and a
one-episode MPC cell, both for the record's schema.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.planner import (
    LeadPlannerInterceptLaw,
    MpcInterceptLaw,
    PlannerInterceptLaw,
)
from pe2.structured import HOLD

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s23_strategy.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s23_strategy", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic records below
ALL = ("GREEDY", "HUG", "ATK")


# ----------------------------------------------------------------------------
# synthetic records, in the shape `run_cell` writes
# ----------------------------------------------------------------------------
def _caps(rate: float, n: int = N) -> list[bool]:
    k = round(rate * n)
    return [i < k for i in range(n)]


def _cell(name: str, caps: list[bool], runs: list[int] | None = None, viol: int = 0):
    """One cell record with only the fields the verdicts read. The per-episode
    arrays and the aggregate agree, so a verdict that recomputed one from the
    other would still say the same thing."""
    spec = R.CELLS[name]
    n = len(caps)
    runs = [0] * n if runs is None else runs
    orbit = [r >= R.ORBIT_RUN_STEPS and not c for r, c in zip(runs, caps)]
    return {
        "name": name,
        "row": spec["row"],
        "col": spec["col"],
        "opponent": spec["opponent"],
        "law": spec["law"],
        "tau": spec["tau"],
        "shielded": True,
        "n_episodes": n,
        "seed0": R.GATE_SEED0,
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "longest_visible_law_run": runs,
        "orbit_ind": orbit,
        "violations": [0] * n,
        "violation_steps": viol,
        "aggregate": {
            "capture_rate": sum(caps) / n,
            "n_orbit": sum(orbit),
            "orbit_run_steps": R.ORBIT_RUN_STEPS,
        },
        "provenance": {"spec_sha256": "x"},
    }


def _round(per_row: dict, viol: dict | None = None) -> dict:
    """`per_row` maps a row to `{opponent: (caps, runs)}`."""
    viol = viol or {}
    out = {}
    for row, by_opp in per_row.items():
        for o, (caps, runs) in by_opp.items():
            name = R.cell_name(row, R.COL, o)
            out[name] = _cell(name, caps, runs, viol.get(name, 0))
    return out


def _flat(caps, runs=None):
    return dict.fromkeys(ALL, (caps, runs))


# The baseline: 40 orbit episodes (long visible law runs, no capture), 30
# non-orbit losses, 30 captures.
BASE_RUNS = [150] * 40 + [0] * 60
BASE_CAPS = [False] * 70 + [True] * 30


def _law_caps(orbit_k: int, non_orbit_k: int, base_k: int = 30) -> list[bool]:
    """A treated law that captured `orbit_k` of the baseline's 40 orbit
    episodes, `non_orbit_k` of its 30 non-orbit losses, and `base_k` of the
    episodes the baseline already won."""
    caps = [i < orbit_k for i in range(40)]
    caps += [i < non_orbit_k for i in range(30)]
    caps += [i < base_k for i in range(30)]
    return caps


def _pass_round() -> dict:
    """`LEAD-1.0` captures more than `PLAN` on all three opponents AND the gain
    is concentrated in the baseline's orbit episodes; the other two laws match
    the baseline exactly."""
    base = _flat(BASE_CAPS, BASE_RUNS)
    law = _flat(_law_caps(30, 3), BASE_RUNS)
    same = _flat(list(BASE_CAPS), BASE_RUNS)
    return _round({"PLAN": base, "LEAD-1.0": law, "LEAD-0.5": same, "MPC-2s": same})


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------
class _StillEvader:
    """A do-nothing opponent, so the table assertions never load a checkpoint."""

    uses_truth = True

    def reset(self):
        return None

    def act(self, obs, env):
        return (0.0, 0.0)


def _still_evader():
    return _StillEvader()


class TestCellTable:
    def test_twelve_cells_four_rows_one_column_three_opponents(self):
        assert len(R.CELLS) == 12
        assert list(R.ROWS) == ["PLAN", "LEAD-0.5", "LEAD-1.0", "MPC-2s"]
        assert R.COL == "SHIELD"
        assert list(R.OPPONENTS) == ["GREEDY", "HUG", "ATK"]
        assert R.TREATED == ("LEAD-0.5", "LEAD-1.0", "MPC-2s")
        assert set(R.CELLS) == {f"{r}-SHIELD_{o}" for r in R.ROWS for o in R.OPPONENTS}

    def test_pre_registered_constants(self):
        assert R.N_EPISODES == 500
        assert R.GATE_SEED0 == 2_100_000
        assert R.CONSTRUCT_SEED0 == 2_090_000  # NOT the unit tests' 1_990_000
        assert R.P_GATE == 0.05
        assert R.PASS_MIN == 2
        assert R.NULL_N == 50
        assert R.ORBIT_RUN_STEPS == 100
        assert R.ORBIT_MIN_EPISODES == 20
        assert R.OMP_THREADS == "4"
        assert R.MPC_BINDING == {"h_steps": 40, "n_seg": 4, "hold": 10, "k": 64}
        assert R.ATK_CKPT == pathlib.Path(
            "results/derisk/ckpts_ATK4-45/ATK4-45_10000000.zip"
        )

    def test_every_cell_runs_the_same_hold_and_observation_flags(self):
        assert {c["hold"] for c in R.CELLS.values()} == {HOLD} == {10}
        assert (R.SHIELD_OBS, R.ESCAPE_OBS) == (True, True)
        assert all(c["shielded"] for c in R.CELLS.values())

    def test_row_law_is_the_rows_only_variable(self):
        cfg = EnvConfig()
        assert type(R.row_law("PLAN", cfg)) is PlannerInterceptLaw
        for row, tau in (("LEAD-0.5", 0.5), ("LEAD-1.0", 1.0)):
            law = R.row_law(row, cfg)
            assert type(law) is LeadPlannerInterceptLaw
            assert law.tau == tau
        assert type(R.row_law("MPC-2s", cfg)) is MpcInterceptLaw

    def test_the_mpc_row_refuses_if_its_binding_defaults_move(self, monkeypatch):
        monkeypatch.setitem(R.MPC_BINDING, "k", 65)
        with pytest.raises(SystemExit, match="defaults moved"):
            R.row_law("MPC-2s", EnvConfig())

    @pytest.mark.parametrize("opponent", ALL)
    def test_built_envs_differ_only_in_the_law(self, opponent, monkeypatch):
        """The round's whole attribution rests on this: build all four cells of
        one opponent and assert the law object is the only difference."""
        monkeypatch.setattr(R, "opponent_factory", lambda *a, **k: _still_evader)
        cfg = EnvConfig()
        envs = {
            row: R.build_cell_env(R.cell_name(row, "SHIELD", opponent), cfg)
            for row in R.ROWS
        }
        want = {
            "PLAN": PlannerInterceptLaw,
            "LEAD-0.5": LeadPlannerInterceptLaw,
            "LEAD-1.0": LeadPlannerInterceptLaw,
            "MPC-2s": MpcInterceptLaw,
        }
        for row, env in envs.items():
            assert type(env).__name__ == "BranchTaggedSearchMDP"
            assert env.cfg == cfg
            assert env.hold == HOLD
            assert env.shield_obs is True
            assert env.escape_obs is True
            assert env.stall_penalty == 0.0
            assert env.intervention_penalty == 0.0
            assert env._driver is env.inner  # reward_wrapper=None
            assert type(env.law) is want[row]
            # the shield: the round's own, at the asserted defaults
            assert type(env.shield).__name__ == "CBFShield"
            assert (env.shield.margin, env.shield.gamma) == (0.0, 0.2)
        # and the observation width is the same everywhere, so the search
        # policy is literally the same function of the same input
        assert len({e.observation_space.shape for e in envs.values()}) == 1

    def test_unknown_cell_name_refuses(self):
        with pytest.raises(SystemExit, match="unknown cell"):
            R.run_cells(["PLAN-SHIELD_NOPE"])

    def test_default_config_is_the_locked_envconfig(self):
        assert R.cell_config() == EnvConfig()
        assert R.null_config().n_physical_obstacles == 0


# ----------------------------------------------------------------------------
# CG-B — the additive-field check (prereg section 5)
# ----------------------------------------------------------------------------
def _cgb(monkeypatch, replayed: dict, stored: dict | None = None):
    stored = replayed if stored is None else stored
    monkeypatch.setattr(R, "cgb_replay", lambda device="cpu": dict(replayed))
    monkeypatch.setattr(R, "cgb_stored", lambda: dict(stored))


def _twenty(offset: float = 0.0) -> dict:
    return {R.CGB_SEED0 + i: 1.5 + i + offset for i in range(R.CGB_N)}


def _pass_cgb(monkeypatch):
    """Make CG-B pass, so a test about something else is not shadowed by it."""
    _cgb(monkeypatch, _twenty())
    return R.cgb_status(_twenty(), _twenty())


class TestCgBAdditive:
    def test_twenty_exact_per_episode_reproductions_are_required(self):
        assert R.CGB_N == 20
        assert (R.CGB_ARM, R.CGB_OPPONENT) == ("SPARSE-PEN-70", "GREEDY")
        assert R.CGB_SEED0 == 1_900_000

    def test_the_stored_artefact_is_the_s18_replay_record(self, monkeypatch, tmp_path):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.cgb_artefact().name == "s18_branch_split_SPARSE-PEN-70_GREEDY.json"

    def test_passes_when_every_episode_is_exact(self, monkeypatch):
        _cgb(monkeypatch, _twenty())
        status = R.require_cgb_additive()
        assert status["verdict"] == "PASS"
        assert (status["n_exact"], status["n_episodes"]) == (20, 20)

    def test_refuses_on_a_single_mismatching_episode(self, monkeypatch):
        replayed = _twenty()
        replayed[R.CGB_SEED0 + 7] += 1e-9  # not a tolerance: this is a failure
        _cgb(monkeypatch, replayed, _twenty())
        with pytest.raises(SystemExit, match="19/20"):
            R.require_cgb_additive()
        status = R.cgb_status(replayed, _twenty())
        assert status["mismatching_seeds"] == [R.CGB_SEED0 + 7]

    def test_refuses_when_fewer_than_twenty_episodes_were_replayed(self, monkeypatch):
        short = {s: v for s, v in list(_twenty().items())[:19]}
        _cgb(monkeypatch, short, _twenty())
        with pytest.raises(SystemExit, match="CG-B REFUSES"):
            R.require_cgb_additive()

    def test_refuses_when_a_seed_is_absent_from_the_record(self, monkeypatch):
        stored = _twenty()
        del stored[R.CGB_SEED0 + 3]
        _cgb(monkeypatch, _twenty(), stored)
        with pytest.raises(SystemExit, match="CG-B REFUSES"):
            R.require_cgb_additive()

    def test_a_missing_artefact_refuses(self, monkeypatch, tmp_path):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no S18 replay artefact"):
            R.cgb_stored()

    def test_run_cells_refuses_before_the_first_episode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _cgb(monkeypatch, _twenty(1.0), _twenty())
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("an episode ran before CG-B")
        )
        with pytest.raises(SystemExit, match="CG-B REFUSES"):
            R.run_cells(["PLAN-SHIELD_GREEDY"])

    def test_the_tally_lands_in_provenance(self, monkeypatch):
        status = R.cgb_status(_twenty(), _twenty())
        p = R.provenance("cpu", status)
        assert p["cg_b"]["verdict"] == "PASS"
        assert p["cg_b"]["n_exact"] == 20
        assert "ADDITIVE" in p["cg_b_scope"]


# ----------------------------------------------------------------------------
# record validity (S19 A4.4), the crashed cell (A4.5) and duplicates (A5.2)
# ----------------------------------------------------------------------------
def _on_disk(out: pathlib.Path, rec: dict, name: str | None = None) -> pathlib.Path:
    d = out / "s23_cells"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name or rec['name']}.json"
    path.write_text(json.dumps(rec))
    return path


def _measurable_record(name: str) -> dict:
    return _cell(name, _caps(0.5, R.N_EPISODES), [0] * R.N_EPISODES)


class TestRecordValidity:
    def test_a_measurable_record_is_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable_record("PLAN-SHIELD_GREEDY"))
        assert R.record_status(_measurable_record("PLAN-SHIELD_GREEDY")) == (
            "MEASURABLE",
            "",
        )
        assert list(R.load_records()) == ["PLAN-SHIELD_GREEDY"]

    @pytest.mark.parametrize(
        ("field", "value"), [("n_episodes", 20), ("seed0", 2_090_000)]
    )
    def test_the_wrong_n_or_stream_is_not_measurable(
        self, tmp_path, monkeypatch, field, value
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable_record("PLAN-SHIELD_GREEDY") | {field: value}
        assert R.record_status(rec)[0] == "INVALID"
        _on_disk(tmp_path, rec)
        assert R.load_records(quiet=True) == {}

    def test_a_foreign_name_is_not_measurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable_record("PLAN-SHIELD_GREEDY") | {"name": "S19-THING"}
        assert R.record_status(rec)[0] == "INVALID"

    def test_an_invalid_record_is_refused_not_skipped(self, tmp_path, monkeypatch):
        """A smoke sitting in the right directory may not become a gate number,
        and may not be silently overwritten either."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_cgb(monkeypatch)
        _on_disk(tmp_path, _measurable_record("PLAN-SHIELD_GREEDY") | {"n_episodes": 3})
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("the smoke was overwritten")
        )
        with pytest.raises(SystemExit, match="not a\n?.*measurable record|REFUSED"):
            R.run_cells(["PLAN-SHIELD_GREEDY"])

    def test_an_invalid_record_is_never_summarised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "null_identity", lambda **k: dict(HOLDS))
        _on_disk(tmp_path, _measurable_record("PLAN-SHIELD_GREEDY") | {"n_episodes": 3})
        blob = R.summarize()
        assert blob["verdicts"]["n_records"] == 0
        assert list(blob["rejected_records"]) == ["PLAN-SHIELD_GREEDY.json"]

    def test_a_crashed_cell_is_undetermined_and_never_re_run(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_cgb(monkeypatch)

        def _crash(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(R, "run_cell", _crash)
        R.run_cells(["PLAN-SHIELD_GREEDY"])
        rec = json.loads(
            (tmp_path / "s23_cells" / "PLAN-SHIELD_GREEDY.json").read_text()
        )
        assert rec["verdict"] == R.UNDETERMINED
        assert rec["n_episodes"] == 0
        assert R.record_status(rec)[0] == R.UNDETERMINED
        # and it is never re-run
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("an UNDETERMINED cell re-ran")
        )
        with pytest.raises(SystemExit, match="REFUSED before any episode ran"):
            R.run_cells(["PLAN-SHIELD_GREEDY"])

    def test_two_files_claiming_one_cell_reject_both(self, tmp_path, monkeypatch):
        """A5.2: a duplicate pair may be named anything, so the block is keyed by
        CELL and not by `{cell}.json`."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_cgb(monkeypatch)
        rec = _measurable_record("PLAN-SHIELD_GREEDY")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a-copy-by-hand")
        measurable, rejected, by_cell = R.scan_records()
        assert measurable == {}
        assert len(rejected) == 2
        assert set(by_cell) == {"PLAN-SHIELD_GREEDY"}
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("a poisoned cell re-ran")
        )
        with pytest.raises(SystemExit, match="duplicate"):
            R.run_cells(["PLAN-SHIELD_GREEDY"])

    def test_a_third_copy_does_not_resurrect_the_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = _measurable_record("PLAN-SHIELD_GREEDY")
        for name in ("PLAN-SHIELD_GREEDY", "copy-1", "copy-2"):
            _on_disk(tmp_path, rec, name=name)
        assert R.scan_records()[0] == {}

    def test_a_measurable_record_is_skipped_as_done(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_cgb(monkeypatch)
        _on_disk(tmp_path, _measurable_record("PLAN-SHIELD_GREEDY"))
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("a recorded cell re-ran")
        )
        assert R.run_cells(["PLAN-SHIELD_GREEDY"]) == []

    def test_no_rate_is_printed_while_the_cells_run(self):
        line = R._cell_line(_measurable_record("PLAN-SHIELD_GREEDY"))
        assert line == "[PLAN-SHIELD_GREEDY] recorded, n_episodes 500"
        for forbidden in ("capture", "orbit", "viol", "conversion"):
            assert forbidden not in line


# ----------------------------------------------------------------------------
# the ATK refusal
# ----------------------------------------------------------------------------
class TestAtkRefusal:
    def test_require_atk_ckpt_refuses_when_missing(self, tmp_path, monkeypatch):
        # `opponent_factory` / `require_atk_ckpt` are S19's own functions,
        # imported and called, so the checkpoint path lives in S19's namespace.
        monkeypatch.setattr(R.S19, "ATK_CKPT", tmp_path / "nope.zip")
        with pytest.raises(SystemExit, match="ATK4-45"):
            R.require_atk_ckpt()

    def test_opponent_factory_refuses_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S19, "ATK_CKPT", tmp_path / "nope.zip")
        with pytest.raises(SystemExit):
            R.opponent_factory("ATK", EnvConfig(), "cpu")

    def test_run_cells_refuses_before_running_anything(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S19, "ATK_CKPT", tmp_path / "nope.zip")
        monkeypatch.setattr(R, "OUT", tmp_path)
        _pass_cgb(monkeypatch)  # isolate the ATK refusal from CG-B's
        monkeypatch.setattr(
            R, "run_cell", lambda *a, **k: pytest.fail("an episode ran before refusal")
        )
        with pytest.raises(SystemExit, match="ATK4-45"):
            R.run_cells(["PLAN-SHIELD_GREEDY", "MPC-2s-SHIELD_ATK"])

    def test_scripted_opponents_need_no_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S19, "ATK_CKPT", tmp_path / "nope.zip")
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        assert R.opponent_factory("GREEDY", EnvConfig(), "cpu") is GreedyEvader
        assert R.opponent_factory("HUG", EnvConfig(), "cpu") is ObstacleHuggingEvader


# ----------------------------------------------------------------------------
# H-STRAT-1
# ----------------------------------------------------------------------------
class TestHStrat1:
    def test_passes_when_the_law_captures_more_on_every_opponent(self):
        v = R.h_strat_1(_pass_round(), "LEAD-1.0")
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 3
        assert all(b["direction"] == "LEAD-1.0 > PLAN" for b in v["opponents"].values())

    def test_two_of_three_is_enough(self):
        recs = _pass_round()
        name = R.cell_name("LEAD-1.0", "SHIELD", "ATK")
        recs[name] = _cell(name, list(BASE_CAPS), BASE_RUNS)
        v = R.h_strat_1(recs, "LEAD-1.0")
        assert v["verdict"] == "PASS"
        assert (v["n_pass"], v["pass_min"]) == (2, 2)
        assert v["opponents"]["ATK"]["verdict"] == "FAIL"

    def test_fails_when_the_direction_inverts_and_still_reports_it(self):
        """A refutation that does not say which way it went is not usable."""
        recs = _round(
            {
                "PLAN": _flat(_law_caps(30, 3), BASE_RUNS),
                "LEAD-1.0": _flat(BASE_CAPS, BASE_RUNS),
                "LEAD-0.5": _flat(BASE_CAPS, BASE_RUNS),
                "MPC-2s": _flat(BASE_CAPS, BASE_RUNS),
            }
        )
        v = R.h_strat_1(recs, "LEAD-1.0")
        assert v["verdict"] == "FAIL"
        assert v["n_pass"] == 0
        for b in v["opponents"].values():
            assert b["direction"] == "LEAD-1.0 < PLAN"
            assert b["delta"] < 0

    def test_a_tie_is_not_a_pass(self):
        recs = _round(
            {
                "PLAN": _flat(BASE_CAPS, BASE_RUNS),
                "LEAD-1.0": _flat(list(BASE_CAPS), BASE_RUNS),
                "LEAD-0.5": _flat(list(BASE_CAPS), BASE_RUNS),
                "MPC-2s": _flat(list(BASE_CAPS), BASE_RUNS),
            }
        )
        v = R.h_strat_1(recs, "LEAD-1.0")
        assert v["verdict"] == "FAIL"
        assert all(
            b["direction"] == "LEAD-1.0 == PLAN" for b in v["opponents"].values()
        )

    def test_incomplete_when_a_cell_is_missing(self):
        recs = _pass_round()
        del recs[R.cell_name("PLAN", "SHIELD", "HUG")]
        v = R.h_strat_1(recs, "LEAD-1.0")
        assert v["verdict"] == "INCOMPLETE"
        assert v["opponents"]["HUG"]["missing"] == ["PLAN-SHIELD_HUG"]

    def test_the_round_passes_if_any_law_passes(self):
        v = R.h_strat_1_round(_pass_round())
        assert v["verdict"] == "PASS"
        assert v["passing_laws"] == ["LEAD-1.0"]
        assert v["laws"]["LEAD-0.5"]["verdict"] == "FAIL"
        assert set(v["laws"]) == set(R.TREATED)

    def test_the_round_fails_only_when_every_law_fails(self):
        same = _flat(list(BASE_CAPS), BASE_RUNS)
        recs = _round(
            {"PLAN": _flat(BASE_CAPS, BASE_RUNS), **dict.fromkeys(R.TREATED, same)}
        )
        v = R.h_strat_1_round(recs)
        assert v["verdict"] == "FAIL"
        assert v["passing_laws"] == []


# ----------------------------------------------------------------------------
# the ORBIT mask and H-STRAT-2
# ----------------------------------------------------------------------------
class TestOrbitMask:
    def test_the_mask_is_the_baselines_long_visible_law_runs_without_capture(self):
        om = R.orbit_mask(_pass_round(), "GREEDY")
        assert om["baseline"] == "PLAN-SHIELD_GREEDY"
        assert om["n_episodes"] == N
        assert om["n_orbit"] == 40
        assert om["n_baseline_lost"] == 70
        assert om["n_non_orbit_lost"] == 30
        assert om["mask"][:40] == [True] * 40
        assert om["mask"][40:] == [False] * 60

    def test_a_captured_episode_is_never_an_orbit_episode(self):
        """An orbit episode is a long visible law run AND no capture: a long run
        that ended in a capture is the law working, not the law orbiting."""
        name = R.cell_name("PLAN", "SHIELD", "GREEDY")
        recs = {name: _cell(name, [True] * 40 + [False] * 60, [150] * 100)}
        om = R.orbit_mask(recs, "GREEDY")
        assert om["n_orbit"] == 60
        assert om["mask"][:40] == [False] * 40

    def test_a_short_run_is_not_an_orbit_episode(self):
        name = R.cell_name("PLAN", "SHIELD", "GREEDY")
        runs = [R.ORBIT_RUN_STEPS - 1] * 50 + [R.ORBIT_RUN_STEPS] * 50
        recs = {name: _cell(name, [False] * 100, runs)}
        assert R.orbit_mask(recs, "GREEDY")["n_orbit"] == 50

    def test_incomplete_without_the_baseline(self):
        assert R.orbit_mask({}, "GREEDY")["verdict"] == "INCOMPLETE"


class TestHStrat2:
    def test_passes_when_the_gain_is_concentrated_in_the_orbit_episodes(self):
        v = R.h_strat_2(_pass_round(), "LEAD-1.0")
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 3
        g = v["opponents"]["GREEDY"]
        assert (g["n_orbit"], g["n_non_orbit_lost"]) == (40, 30)
        assert g["conversion_orbit"] == pytest.approx(30 / 40)
        assert g["conversion_non_orbit"] == pytest.approx(3 / 30)

    def test_fails_when_the_gain_lands_outside_the_orbit_episodes(self):
        """The prereg's own alternative: a real gain that is NOT the orbit
        mechanism, which must FAIL H-STRAT-2 and be reported as a gain from
        something else."""
        recs = _round(
            {
                "PLAN": _flat(BASE_CAPS, BASE_RUNS),
                "LEAD-1.0": _flat(_law_caps(2, 25), BASE_RUNS),
                "LEAD-0.5": _flat(list(BASE_CAPS), BASE_RUNS),
                "MPC-2s": _flat(list(BASE_CAPS), BASE_RUNS),
            }
        )
        v = R.h_strat_2(recs, "LEAD-1.0")
        assert v["verdict"] == "FAIL"
        g = v["opponents"]["GREEDY"]
        assert g["conversion_orbit"] < g["conversion_non_orbit"]
        assert g["margin"] < 0
        # ... while H-STRAT-1 passes on the very same records
        assert R.h_strat_1(recs, "LEAD-1.0")["verdict"] == "PASS"

    def test_an_equal_conversion_is_not_a_pass(self):
        recs = _round(
            {
                "PLAN": _flat(BASE_CAPS, BASE_RUNS),
                "LEAD-1.0": _flat(_law_caps(20, 15), BASE_RUNS),
                "LEAD-0.5": _flat(list(BASE_CAPS), BASE_RUNS),
                "MPC-2s": _flat(list(BASE_CAPS), BASE_RUNS),
            }
        )
        v = R.h_strat_2(recs, "LEAD-1.0")
        assert v["opponents"]["GREEDY"]["conversion_orbit"] == pytest.approx(0.5)
        assert v["opponents"]["GREEDY"]["conversion_non_orbit"] == pytest.approx(0.5)
        assert v["verdict"] == "FAIL"

    def test_fewer_than_twenty_orbit_episodes_is_vacuous_never_pass(self):
        """The prereg's guard: with no orbit phenomenon there is nothing to
        concentrate in, so the opponent can never PASS — not even when the
        conversion ordering is perfect."""
        runs = [150] * 10 + [0] * 90
        caps_base = [False] * 70 + [True] * 30
        law = [i < 10 for i in range(10)] + [False] * 60 + [True] * 30
        recs = _round(
            {
                "PLAN": _flat(caps_base, runs),
                "LEAD-1.0": _flat(law, runs),
                "LEAD-0.5": _flat(list(caps_base), runs),
                "MPC-2s": _flat(list(caps_base), runs),
            }
        )
        v = R.h_strat_2(recs, "LEAD-1.0")
        for b in v["opponents"].values():
            assert b["verdict"] == "VACUOUS"
            assert b["n_orbit"] == 10 < R.ORBIT_MIN_EPISODES
        assert v["verdict"] == "VACUOUS"  # not PASS, and not FAIL either
        assert v["n_pass"] == 0

    def test_one_vacuous_opponent_does_not_block_a_pass_on_the_other_two(self):
        recs = _pass_round()
        for row in ("PLAN", "LEAD-1.0"):
            name = R.cell_name(row, "SHIELD", "ATK")
            caps = BASE_CAPS if row == "PLAN" else _law_caps(30, 3)
            recs[name] = _cell(name, list(caps), [150] * 10 + [0] * 90)
        v = R.h_strat_2(recs, "LEAD-1.0")
        assert v["opponents"]["ATK"]["verdict"] == "VACUOUS"
        assert (v["n_pass"], v["n_vacuous"]) == (2, 1)
        assert v["verdict"] == "PASS"

    def test_an_empty_non_orbit_comparator_is_vacuous(self):
        """Stated reading: with every baseline loss an orbit loss there is no
        comparator, so "exceeds" is undefined rather than satisfied."""
        runs = [150] * 70 + [0] * 30
        recs = _round(
            {
                "PLAN": _flat(BASE_CAPS, runs),
                "LEAD-1.0": _flat(_law_caps(40, 30), runs),
                "LEAD-0.5": _flat(list(BASE_CAPS), runs),
                "MPC-2s": _flat(list(BASE_CAPS), runs),
            }
        )
        v = R.h_strat_2(recs, "LEAD-1.0")
        assert v["opponents"]["GREEDY"]["n_non_orbit_lost"] == 0
        assert v["verdict"] == "VACUOUS"

    def test_incomplete_when_a_cell_is_missing(self):
        recs = _pass_round()
        del recs[R.cell_name("LEAD-1.0", "SHIELD", "HUG")]
        assert R.h_strat_2(recs, "LEAD-1.0")["verdict"] == "INCOMPLETE"

    def test_it_is_reported_for_every_law(self):
        v = R.verdicts(_pass_round(), dict(HOLDS))
        assert set(v["h_strat_2"]["laws"]) == set(R.TREATED)


# ----------------------------------------------------------------------------
# H-SAFE
# ----------------------------------------------------------------------------
class TestHSafe:
    def test_holds_at_zero_contact_steps_in_all_twelve_cells(self):
        v = R.h_safe(_pass_round())
        assert v["verdict"] == "HOLDS"
        assert v["violation_steps"] == 0
        assert len(v["gated_cells"]) == 12

    def test_one_contact_step_anywhere_fails_it(self):
        recs = _pass_round()
        recs["MPC-2s-SHIELD_HUG"]["violation_steps"] = 1
        v = R.h_safe(recs)
        assert v["verdict"] == "FAILS"
        assert v["offenders"] == ["MPC-2s-SHIELD_HUG"]

    def test_incomplete_while_a_cell_is_missing(self):
        recs = _pass_round()
        del recs["PLAN-SHIELD_ATK"]
        assert R.h_safe(recs)["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------
# the NULL (CG-S1)
# ----------------------------------------------------------------------------
class _DivergentLaw(PlannerInterceptLaw):
    """`PLAN` with the turn reversed — a law that is NOT `PLAN` even with the
    evader stationary, which is exactly what the NULL must catch."""

    def act(self, obs):
        a = super().act(obs)
        return None if a is None else np.array([a[0], -a[1]], dtype=np.float32)


HOLDS = {
    "verdict": "HOLDS",
    "n_differing": 0,
    "n_episodes": 50,
    "differing_seeds": [],
    "failing_rows": [],
}


class TestNull:
    def test_it_runs_on_the_rounds_construct_stream(self):
        v = R.null_identity(n=1)
        assert v["seed0"] == 2_090_000  # NOT the unit tests' 1_990_000
        assert v["n_episodes"] == 1
        assert v["pre_registered"] is False  # n != 50
        assert set(v["rows"]) == set(R.TREATED)

    def test_the_treated_laws_are_plan_against_a_stationary_evader(self):
        v = R.null_identity(n=2)
        assert v["verdict"] == "HOLDS"
        assert v["n_differing"] == 0
        assert all(b["n_aimed"] == 0 for b in v["rows"].values())

    def test_it_fails_against_a_divergent_law(self, monkeypatch):
        real = R.row_law
        monkeypatch.setattr(
            R,
            "row_law",
            lambda row, cfg: (
                _DivergentLaw(cfg) if row == "LEAD-0.5" else real(row, cfg)
            ),
        )
        v = R.null_identity(n=1)
        assert v["verdict"] == "FAILS"
        assert v["failing_rows"] == ["LEAD-0.5"]
        assert v["rows"]["LEAD-1.0"]["verdict"] == "HOLDS"
        assert v["differing_seeds"] == [2_090_000]

    def test_the_exposure_counters_are_excluded_and_named(self):
        """`MpcInterceptLaw` always exposes `last_feasible_count`, so the
        recorder's `feasible_count` / `n_planned_calls` differ from `PLAN`'s on
        IDENTICAL episodes. Those fields are a property of which law object is
        installed, so the identity excludes them — and says so in the record."""
        assert R.NULL_EXPOSURE_KEYS == (
            "aim_point",
            "v_est",
            "feasible_count",
            "chose_direct",
        )
        assert R.NULL_EXEMPT_SUMMARY == (
            "n_planned_calls",
            "n_zero_feasible",
            "n_chose_direct",
        )
        text = R.null_identity(n=1)["comparison"]
        assert "per-step record" in text and "aim_point is None" in text

    def test_direction_stops_the_round_when_the_null_fails(self):
        v = R.verdicts(
            _pass_round(),
            {
                "verdict": "FAILS",
                "n_differing": 7,
                "n_episodes": 50,
                "differing_seeds": [1],
                "failing_rows": ["MPC-2s"],
            },
        )
        text = R.direction(v)
        assert "NULL FAILS" in text
        assert "NO VERDICT in this round is about the aim point" in text
        assert "H-STRAT-1 PASS" not in text

    def test_direction_refuses_to_interpret_an_unevaluated_null(self):
        assert "NULL has not been evaluated" in R.direction(
            R.verdicts(_pass_round(), None)
        )


# ----------------------------------------------------------------------------
# direction() — prereg section 4's refutation paragraph
# ----------------------------------------------------------------------------
class TestDirection:
    def test_hsafe_failure_precedes_everything(self):
        recs = _pass_round()
        recs["PLAN-SHIELD_GREEDY"]["violation_steps"] = 3
        text = R.direction(R.verdicts(recs, HOLDS))
        assert text.startswith("H-SAFE FAILS")
        assert "nothing else is read" in text

    def test_incompleteness_precedes_the_hypotheses(self):
        recs = _pass_round()
        del recs["MPC-2s-SHIELD_ATK"]
        assert "incomplete" in R.direction(R.verdicts(recs, HOLDS))

    def test_the_pass_narrative_names_the_passing_law(self):
        text = R.direction(R.verdicts(_pass_round(), HOLDS))
        assert "The round PASSES on LEAD-1.0" in text
        assert "LEAD-1.0: H-STRAT-1 PASS" in text
        assert "LEAD-1.0: H-STRAT-2 PASS" in text
        assert "ALL THREE LAWS FAIL" not in text
        assert "ALL-STEP" in text  # the scope note

    def test_every_law_failing_is_the_bound_not_a_null_result(self):
        same = _flat(list(BASE_CAPS), BASE_RUNS)
        recs = _round(
            {"PLAN": _flat(BASE_CAPS, BASE_RUNS), **dict.fromkeys(R.TREATED, same)}
        )
        text = R.direction(R.verdicts(recs, HOLDS))
        assert "ALL THREE LAWS FAIL H-STRAT-1" in text
        assert "IRREDUCIBLE BY THE NOMINAL CONTROLLER" in text
        assert "the agent-intelligence line stops" in text

    def test_a_gain_that_is_not_the_orbit_mechanism_is_said_so(self):
        recs = _round(
            {
                "PLAN": _flat(BASE_CAPS, BASE_RUNS),
                "LEAD-1.0": _flat(_law_caps(2, 25), BASE_RUNS),
                "LEAD-0.5": _flat(list(BASE_CAPS), BASE_RUNS),
                "MPC-2s": _flat(list(BASE_CAPS), BASE_RUNS),
            }
        )
        text = R.direction(R.verdicts(recs, HOLDS))
        assert "H-STRAT-1 passes but H-STRAT-2 fails on LEAD-1.0" in text
        assert "REAL BUT NOT THE ORBIT MECHANISM" in text
        assert "the mechanism question stays open" in text

    def test_the_cg_s2_expectation_rides_with_every_mpc_number(self):
        text = R.direction(R.verdicts(_pass_round(), HOLDS))
        assert "CG-S2" in text
        assert "DESPITE the model" in text


# ----------------------------------------------------------------------------
# CG-S3 — the watched episodes
# ----------------------------------------------------------------------------
def _seeds_file(out: pathlib.Path, seeds) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "s23_cg_s3_seeds.json").write_text(json.dumps({"seeds": list(seeds)}))


class TestGifs:
    def test_the_frozen_seeds_are_read_not_chosen(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [2_090_000, 2_090_005, 2_090_006])
        assert R.require_cg_s3_seeds() == (2_090_000, 2_090_005, 2_090_006)

    def test_the_real_artefact_carries_exactly_three_seeds(self):
        seeds = R.read_cg_s3_seeds()
        assert seeds is not None and len(seeds) == R.GIF_N == 3

    def test_it_refuses_without_the_seeds_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R, "render_row", lambda *a, **k: pytest.fail("rendered without seeds")
        )
        assert R.read_cg_s3_seeds() is None
        with pytest.raises(SystemExit, match="CG-S3 REFUSES"):
            R.write_gifs(["PLAN"])

    def test_it_refuses_on_the_wrong_number_of_seeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [2_090_000, 2_090_005])
        with pytest.raises(SystemExit, match="exactly"):
            R.require_cg_s3_seeds()

    def test_twelve_gifs_four_rows_times_three_seeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [1, 2, 3])
        assert len(R.ROWS) * len(R.require_cg_s3_seeds()) == 12

    def test_gif_paths_are_row_and_seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.gif_path("LEAD-0.5", 2_090_005) == (
            tmp_path / "s23_gifs" / "LEAD-0.5_2090005.gif"
        )

    def test_each_row_is_gated_on_its_greedy_cell(self):
        assert R.gif_cell("MPC-2s") == "MPC-2s-SHIELD_GREEDY"
        assert R.GIF_OPPONENT == "GREEDY"

    def test_a_row_without_a_record_refuses_before_rendering(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [2_090_000, 2_090_005, 2_090_006])
        monkeypatch.setattr(
            R, "render_row", lambda *a, **k: pytest.fail("a frame was rendered")
        )
        with pytest.raises(SystemExit, match="no cell record"):
            R.write_gifs(["PLAN"])

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "not measurable"),
            ({"seed0": 2_090_000}, "not measurable"),
            ({"verdict": R.UNDETERMINED, "n_episodes": 0}, "UNDETERMINED"),
        ],
    )
    def test_a_non_measurable_record_refuses_rather_than_captioning(
        self, tmp_path, monkeypatch, overrides, match
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [2_090_000, 2_090_005, 2_090_006])
        _on_disk(tmp_path, _measurable_record("PLAN-SHIELD_GREEDY") | overrides)
        monkeypatch.setattr(
            R, "render_row", lambda *a, **k: pytest.fail("rendered from a bad record")
        )
        with pytest.raises(SystemExit, match=match):
            R.write_gifs(["PLAN"])
        with pytest.raises(SystemExit, match=match):
            R.read_cell("PLAN-SHIELD_GREEDY")

    def test_unknown_row_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown row"):
            R.write_gifs(["PLAN-SHIELD_GREEDY"])

    def test_existing_gifs_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _seeds_file(tmp_path, [2_090_000, 2_090_005, 2_090_006])
        (tmp_path / "s23_gifs").mkdir(parents=True)
        for s in R.require_cg_s3_seeds():
            R.gif_path("PLAN", s).write_bytes(b"")
        monkeypatch.setattr(R, "render_row", lambda *a, **k: pytest.fail("re-rendered"))
        assert R.write_gifs(["PLAN"]) == []

    def test_the_caption_is_the_row_and_the_seed_and_no_number(self):
        title = R.gif_title("LEAD-0.5", 2_090_000)
        assert title == "LEAD-0.5 · episode 2090000"
        rec = _measurable_record("PLAN-SHIELD_GREEDY")
        for forbidden in (
            "capture",
            "orbit",
            "conversion",
            "measured",
            "N=",
            f"{rec['capture_rate']:.3f}",
            "PASS",
            "FAIL",
        ):
            assert forbidden not in title
        # and nothing in the signature can reach a record
        assert "rec" not in R.gif_title.__code__.co_varnames

    def test_the_frame_caption_adds_only_per_episode_quantities(self, monkeypatch):
        monkeypatch.setattr(R, "opponent_factory", lambda *a, **k: _still_evader)
        cfg = R.cell_config()
        rec = R.CellRecorder(
            cfg,
            None,
            label=R.gif_title("MPC-2s", 2_090_000),
            hold=HOLD,
            shield=R.build_shield(cfg),
            shield_obs=R.SHIELD_OBS,
            escape_obs=R.ESCAPE_OBS,
            reward_wrapper=None,
            law=R.row_law("MPC-2s", cfg),
        )
        assert rec.caption("law") == ("[law] MPC-2s · episode 2090000 · unreachable 0")
        assert "capture" not in rec.caption("CAPTURE @ 12")

    def test_the_recorder_is_s19s_plus_the_aim_marker(self, monkeypatch):
        """The branch tag, the FOV cone and the nominal-vs-executed arcs are
        S19's and unchanged; S23 adds the aim point and nothing else."""
        from pe2.recorder import BranchTaggedSearchMDP

        assert issubclass(R.CellRecorder, R.S19.CellRecorder)
        assert issubclass(R.CellRecorder, BranchTaggedSearchMDP)
        monkeypatch.setattr(R, "opponent_factory", lambda *a, **k: _still_evader)
        cfg = R.cell_config()
        made = {
            row: R.CellRecorder(
                cfg,
                None,
                label="x",
                hold=HOLD,
                shield=R.build_shield(cfg),
                reward_wrapper=None,
                law=R.row_law(row, cfg),
            )
            for row in ("PLAN", "LEAD-1.0")
        }
        for row, mdp in made.items():
            aim = [o for o in mdp._ov if isinstance(o, R.AimOverlay)]
            assert len(aim) == 1
            assert aim[0].law is mdp.law
            # `PLAN` exposes no aim point, so its frames carry no marker.
            assert (getattr(mdp.law, "last_aim", None) is None) is True
            assert hasattr(mdp.law, "last_aim") is (row != "PLAN")


# ----------------------------------------------------------------------------
# CLI dispatch order and an empty round
# ----------------------------------------------------------------------------
class TestCli:
    def test_gifs_are_written_before_the_table_is_summarised(self, monkeypatch):
        """CG-S3 is a BLOCKING construct gate: the episodes are watched before
        any capture rate is read."""
        order = []
        monkeypatch.setattr(R, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(R, "write_gifs", lambda *a, **k: order.append("gifs"))
        monkeypatch.setattr(
            R,
            "summarize",
            lambda *a, **k: (
                order.append("summarize"),
                {"verdicts": R.verdicts({}, dict(HOLDS)), "direction": "x"},
            )[1],
        )
        monkeypatch.setattr(
            "sys.argv", ["derisk_s23_strategy.py", "--summarize", "--gifs", "--cells"]
        )
        R.main()
        assert order == ["cells", "gifs", "summarize"]

    def test_summarize_survives_a_round_with_no_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "null_identity", lambda **k: dict(HOLDS))
        blob = R.summarize()
        assert blob["verdicts"]["n_records"] == 0
        assert "incomplete" in blob["direction"]
        assert (tmp_path / "s23_round.json").exists()
        assert (tmp_path / "s23_round.png").exists()
        assert (tmp_path / "s23_null.json").exists()


# ----------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------
class TestProvenance:
    def test_it_carries_the_preregs_sha256(self):
        p = R.provenance()
        assert p["spec"] == "docs/pathway/pursuit_strategy_prereg_2026-10-12.md"
        assert (
            p["spec_sha256"] == hashlib.sha256(R.spec_path().read_bytes()).hexdigest()
        )
        assert len(p["spec_sha256"]) == 64
        assert R.spec_path().exists()

    def test_it_carries_the_package_device_threads_and_cg_s3_seeds(self):
        p = R.provenance("cuda")
        assert p["pe2_file"].endswith("pe2/__init__.py")
        assert p["omp_num_threads"] == "4"
        assert p["device"] == "cuda"
        assert "git_commit" in p
        assert p["cg_s3_seeds"] == list(R.read_cg_s3_seeds())
        assert R.provenance()["cg_b"] is None


# ----------------------------------------------------------------------------
# the orbit statistic, and the smoke
# ----------------------------------------------------------------------------
class TestVisibleLawRun:
    def test_only_consecutive_visible_law_steps_count(self):
        steps = [
            {"branch": "law", "visible": True},
            {"branch": "law", "visible": True},
            {"branch": "law", "visible": False},  # a fresh last-seen step
            {"branch": "law", "visible": True},
            {"branch": "search", "visible": True},  # the learner, not the law
            {"branch": "law", "visible": True},
            {"branch": "law", "visible": True},
            {"branch": "law", "visible": True},
        ]
        assert R.longest_visible_law_run(steps) == 3
        assert R.longest_visible_law_run([]) == 0

    def test_the_recorder_exposes_the_field_additively(self):
        from pe2.recorder import BranchTaggedSearchMDP

        assert "visible" in BranchTaggedSearchMDP.__doc__


class TestSmoke:
    def test_a_three_episode_cell_carries_every_pre_registered_field(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = R.run_cell("LEAD-1.0-SHIELD_GREEDY", n_episodes=3)
        for key in (
            "capture_ind",
            "capture_rate",
            "wilson_capture",
            "steps",
            "violations",
            "violation_steps",
            "intervention_total",
            "longest_visible_law_run",
            "n_planned_calls",
            "n_zero_feasible",
            "n_chose_direct",
            "branch",
            "device",
            "provenance",
        ):
            assert key in rec, key
        assert rec["law"] == "LeadPlannerInterceptLaw"
        assert rec["tau"] == 1.0
        assert len(rec["capture_ind"]) == len(rec["branch"]) == 3
        assert rec["seed0"] == R.GATE_SEED0
        assert rec["device"] == "cpu" == rec["provenance"]["device"]
        assert rec["aggregate"]["intervention_scope"] == "all steps (S19 A1.2)"
        assert rec["aggregate"]["orbit_run_steps"] == 100
        assert all(r >= 0 for r in rec["longest_visible_law_run"])
        path = R.write_cell(rec)
        assert json.loads(path.read_text())["name"] == "LEAD-1.0-SHIELD_GREEDY"
        # the smoke is NOT a measurable record and may never become a number
        assert R.record_status(rec)[0] == "INVALID"
        assert R.load_records(quiet=True) == {}

    def test_the_mpc_row_records_its_planned_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = R.run_cell("MPC-2s-SHIELD_GREEDY", n_episodes=1)
        assert rec["n_planned_calls"][0] > 0  # every step of an MPC episode
        assert rec["n_zero_feasible"][0] >= 0
        assert rec["aggregate"]["n_chose_direct_total"] >= 0
