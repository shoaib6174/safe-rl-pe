"""scripts/derisk_s9_sweep.py — the S9 attack-surface sweep's own decisions.

Spec: docs/pathway/attack_surface_sweep_prereg_2026-09-17.md (+ amendments A1, A2).
Nothing here touches a checkpoint: every test drives the enumeration, the
verdict rules and the cell bookkeeping on fabricated records.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s9_sweep.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s9_sweep", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n):
    """n paired indicators whose first round(rate * n) entries are True, so a
    higher-rate list strictly contains a lower-rate one (discordant pairs all
    point one way — the paired test then sees the direction under test)."""
    k = round(rate * n)
    return [i < k for i in range(n)]


def _record(pursuer, opponent, delta, n, gamma, deadlock, violations=0, n_ep=300):
    ind = _ind(deadlock, n_ep)
    return {
        "name": R.cell_name(pursuer, opponent, delta, n, gamma),
        "pursuer": pursuer,
        "opponent": opponent,
        "delta": delta,
        "n": n,
        "gamma": gamma,
        "n_episodes": n_ep,
        "seed0": R.SEED0,
        "deadlock_ind": ind,
        "capture_ind": _ind(0.2, n_ep),
        "deadlock_rate": sum(ind) / n_ep,
        "capture_rate": 0.2,
        "escape_rate": 0.8,
        "outcomes": {"capture": round(0.2 * n_ep), "timeout": n_ep - round(0.2 * n_ep)},
        "stall_step_rate": 0.3,
        "intervention_rate": 0.4,
        "violations": violations,
        "min_clearance_floor": 0.01,
        "wilson_deadlock": R._ci(sum(ind), n_ep),
    }


def _stub_record(name, deadlock=0.5, violations=0):
    spec = R.CELLS[name]
    return _record(
        spec["pursuer"],
        spec["opponent"],
        spec["delta"],
        spec["n"],
        spec["gamma"],
        deadlock,
        violations=violations,
    )


def _records(rate_fn, violations_fn=None):
    """A full 132-cell record set whose deadlock rate is rate_fn(spec)."""
    return {
        name: _stub_record(
            name,
            rate_fn(spec),
            0 if violations_fn is None else violations_fn(spec),
        )
        for name, spec in R.CELLS.items()
    }


def _monotone(spec):
    """Deadlock grows with delta, grows with n, shrinks with gamma: every
    hypothesis of the round CONFIRMED. The n slope is large enough that the
    n = 4 and n = 2 Wilson intervals separate (A2's unpaired test)."""
    return 0.10 + 2.0 * spec["delta"] + 0.12 * spec["n"] + (0.2 - spec["gamma"]) / 2.0


def _flat(spec):
    return 0.30


# ---------------------------------------------------------------- enumeration
class TestCells:
    def test_count_is_three_pursuers_times_four_opponents_times_eleven(self):
        assert len(R.PURSUERS) == 3
        assert len(R.OPPONENTS) == 4
        assert len(R.CELLS) == 132
        for p in R.PURSUERS:
            for o in R.OPPONENTS:
                cells = [s for s in R.CELLS.values() if s["pursuer"] == p]
                assert len([s for s in cells if s["opponent"] == o]) == 11

    def test_levels_are_the_pre_registered_ones(self):
        assert R.DELTAS == (0.0, 0.05, 0.10, 0.20, 0.30)
        assert R.NS == (0, 1, 2, 3, 4)
        assert R.NS_CONFIRM == (2, 3, 4)
        assert R.NS_EXPLORATORY == (0, 1)
        assert R.GAMMAS == (0.1, 0.2, 0.4)
        assert (R.BASE_DELTA, R.BASE_N, R.BASE_GAMMA) == (0.0, 2, 0.2)
        assert R.SEED0 == 1_240_000
        assert R.N_EPISODES == 300

    def test_base_cell_is_shared_by_all_three_sweeps(self):
        name = R.cell_name("RC19", "ATT25", R.BASE_DELTA, R.BASE_N, R.BASE_GAMMA)
        assert name == "RC19__ATT25__d0.00_n2_g0.2"
        assert R.CELLS[name]["sweep"] == {"delta", "n", "gamma"}

    def test_each_cell_carries_exactly_the_sweeps_it_belongs_to(self):
        for spec in R.CELLS.values():
            expect = set()
            if (spec["n"], spec["gamma"]) == (R.BASE_N, R.BASE_GAMMA):
                expect.add("delta")
            if (spec["delta"], spec["gamma"]) == (R.BASE_DELTA, R.BASE_GAMMA):
                expect.add("n")
            if (spec["delta"], spec["n"]) == (R.BASE_DELTA, R.BASE_N):
                expect.add("gamma")
            assert spec["sweep"] == expect and expect

    def test_structured_and_scripted_opponents_are_declared(self):
        assert R.STRUCTURED == ("ATT25", "ATK2-31")
        assert R.SCRIPTED == ("HUG7", "GREEDY")
        assert set(R.STRUCTURED) | set(R.SCRIPTED) == set(R.OPPONENTS)


# ------------------------------------------------------------------- spearman
class TestSpearman:
    def test_monotone_increasing_is_plus_one(self):
        assert R.spearman([1, 2, 3, 4, 5]) == pytest.approx(1.0)

    def test_monotone_decreasing_is_minus_one(self):
        assert R.spearman([5, 4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_constant_is_zero_not_nan(self):
        assert R.spearman([0.3, 0.3, 0.3]) == 0.0

    def test_ties_use_average_ranks(self):
        # y = [1, 2, 2, 3] against index ranks [1, 2, 3, 4]: average ranks
        # [1, 2.5, 2.5, 4] -> rho = 0.9486832...
        assert R.spearman([1, 2, 2, 3]) == pytest.approx(0.9486832980505138)


# -------------------------------------------------------------------- verdicts
class TestVerdicts:
    def test_monotone_records_confirm_every_hypothesis(self):
        v = R.verdicts(_records(_monotone))
        for sweep in ("delta", "n", "gamma"):
            s = v["sweeps"][sweep]
            assert s["verdict"] == "CONFIRMED", sweep
            for p in R.PURSUERS:
                assert s["pursuers"][p]["verdict"] == "CONFIRMED", (sweep, p)
                for o in R.STRUCTURED:
                    assert s["pursuers"][p]["opponents"][o]["verdict"] == "CONFIRMED"

    def test_flat_records_confirm_nothing_except_the_weak_gamma_rule(self):
        v = R.verdicts(_records(_flat))
        assert v["sweeps"]["delta"]["verdict"] == "NOT CONFIRMED"
        assert v["sweeps"]["n"]["verdict"] == "NOT CONFIRMED"
        # H-GEO-gamma is a non-strict rule ("does not decrease"), so a flat
        # sweep confirms it by construction; that is the pre-registered rule.
        assert v["sweeps"]["gamma"]["verdict"] == "CONFIRMED"

    def test_delta_needs_both_structured_attackers(self):
        def rate(spec):
            if spec["opponent"] == "ATT25":
                return _flat(spec)
            return _monotone(spec)

        v = R.verdicts(_records(rate))
        d = v["sweeps"]["delta"]
        assert d["verdict"] == "NOT CONFIRMED"
        for p in R.PURSUERS:
            assert d["pursuers"][p]["verdict"] == "NOT CONFIRMED"
            assert d["pursuers"][p]["opponents"]["ATT25"]["verdict"] == "NOT CONFIRMED"
            assert d["pursuers"][p]["opponents"]["ATK2-31"]["verdict"] == "CONFIRMED"

    def test_overall_needs_two_of_three_pursuers(self):
        def rate(spec):
            if spec["pursuer"] == "STR14":
                return _flat(spec)
            return _monotone(spec)

        v = R.verdicts(_records(rate))
        assert v["sweeps"]["delta"]["verdict"] == "CONFIRMED"
        assert v["sweeps"]["delta"]["n_confirmed"] == 2
        assert v["sweeps"]["delta"]["pursuers"]["STR14"]["verdict"] == "NOT CONFIRMED"

    def test_delta_top_vs_bottom_tests_that_the_top_deadlocks_more(self):
        v = R.verdicts(_records(_monotone))
        t = v["sweeps"]["delta"]["pursuers"]["RC19"]["opponents"]["ATT25"][
            "top_vs_bottom"
        ]
        assert t["test"] == "paired_indicator"  # A2: delta stays seed-paired
        assert t["pass"] is True
        assert t["rate_a"] > t["rate_b"]  # a = top level, b = bottom level

    def test_gamma_keeps_the_paired_test(self):
        v = R.verdicts(_records(_monotone))
        t = v["sweeps"]["gamma"]["pursuers"]["RC19"]["opponents"]["ATT25"][
            "top_vs_bottom"
        ]
        assert t["test"] == "paired_indicator"

    def test_delta_fails_when_only_the_monotone_half_holds(self):
        # rho > 0 but the top vs bottom difference is a single episode: the
        # paired test must veto.
        def rate(spec):
            return 0.30 + 0.003 * R.DELTAS.index(spec["delta"])

        v = R.verdicts(_records(rate))
        d = v["sweeps"]["delta"]["pursuers"]["RC19"]["opponents"]["ATT25"]
        assert d["rho"] > 0
        assert d["top_vs_bottom"]["pass"] is False
        assert d["verdict"] == "NOT CONFIRMED"

    def test_n_verdict_uses_only_the_confirmatory_levels(self):
        # n = 0 and n = 1 are the highest rates and would flip both clauses if
        # they counted (A1: they are exploratory, no verdict is derived).
        def rate(spec):
            return {0: 0.9, 1: 0.9, 2: 0.1, 3: 0.2, 4: 0.3}[spec["n"]]

        v = R.verdicts(_records(rate))
        s = v["sweeps"]["n"]
        assert s["verdict"] == "CONFIRMED"
        cell = s["pursuers"]["RC19"]["opponents"]["ATT25"]
        assert cell["levels"] == list(R.NS_CONFIRM)
        assert cell["rates"] == [0.1, 0.2, 0.3]
        assert cell["rho"] > 0

    def test_n_exploratory_levels_are_reported_without_a_verdict(self):
        def rate(spec):
            return {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.2, 4: 0.3}[spec["n"]]

        v = R.verdicts(_records(rate))
        ex = v["sweeps"]["n"]["exploratory"]
        assert ex["levels"] == list(R.NS_EXPLORATORY)
        assert ex["rates"]["RC19"]["ATT25"] == [0.9, 0.8]
        assert "verdict" not in ex

    def test_n_uses_wilson_separation_not_a_paired_test(self):
        # A2: the n levels are different games (obstacles are drawn before the
        # poses), so the top-vs-bottom test is 95 % Wilson separation.
        def rate(spec):
            return {0: 0.9, 1: 0.9, 2: 0.10, 3: 0.20, 4: 0.30}[spec["n"]]

        cell = R.verdicts(_records(rate))["sweeps"]["n"]["pursuers"]["RC19"][
            "opponents"
        ]["ATT25"]
        assert cell["top_vs_bottom"]["test"] == "wilson_separation"
        assert (
            cell["top_vs_bottom"]["wilson_a"][0] > cell["top_vs_bottom"]["wilson_b"][1]
        )
        assert cell["top_vs_bottom"]["pass"] is True
        assert cell["verdict"] == "CONFIRMED"

    def test_n_not_confirmed_when_the_intervals_overlap(self):
        # rates rise and n = 2 is strictly lowest, but n = 4 and n = 2 overlap at
        # 95 %: the A2 test vetoes what rho alone would confirm.
        def rate(spec):
            return {0: 0.9, 1: 0.9, 2: 0.30, 3: 0.33, 4: 0.36}[spec["n"]]

        cell = R.verdicts(_records(rate))["sweeps"]["n"]["pursuers"]["RC19"][
            "opponents"
        ]["ATT25"]
        assert cell["rho"] > 0
        assert cell["rates"][0] < min(cell["rates"][1:])
        assert cell["top_vs_bottom"]["pass"] is False
        assert cell["verdict"] == "NOT CONFIRMED"

    def test_n_verdict_needs_the_lowest_rate_at_n_two(self):
        def rate(spec):
            return {0: 0.9, 1: 0.9, 2: 0.3, 3: 0.2, 4: 0.5}[spec["n"]]

        v = R.verdicts(_records(rate))
        assert v["sweeps"]["n"]["verdict"] == "NOT CONFIRMED"

    def test_gamma_verdict_compares_the_conservative_end_to_the_loose_end(self):
        def rate(spec):
            return {0.1: 0.2, 0.2: 0.5, 0.4: 0.5}[spec["gamma"]]

        v = R.verdicts(_records(rate))
        g = v["sweeps"]["gamma"]
        assert g["verdict"] == "NOT CONFIRMED"
        cell = g["pursuers"]["RC19"]["opponents"]["ATT25"]
        assert cell["rate_top"] == 0.2  # gamma = 0.1, the conservative filter
        assert cell["rate_bottom"] == 0.5  # gamma = 0.4

    def test_scripted_opponents_are_reported_with_no_verdict(self):
        v = R.verdicts(_records(_monotone))
        for sweep in ("delta", "n", "gamma"):
            for o in R.SCRIPTED:
                cell = v["sweeps"][sweep]["pursuers"]["RC19"]["opponents"][o]
                assert cell["verdict"] is None
                assert "rho" in cell and "rates" in cell

    def test_missing_records_make_the_round_incomplete(self):
        recs = _records(_monotone)
        gone = R.cell_name("RC19", "ATT25", 0.30, 2, 0.2)
        del recs[gone]
        v = R.verdicts(recs)
        assert v["complete"] is False
        assert gone in v["missing"]
        d = v["sweeps"]["delta"]
        assert d["pursuers"]["RC19"]["opponents"]["ATT25"]["verdict"] == "INCOMPLETE"
        assert d["pursuers"]["RC19"]["verdict"] == "INCOMPLETE"
        assert d["verdict"] == "INCOMPLETE"


# ---------------------------------------------------------------------- H-SAFE
class TestHSafe:
    def test_holds_on_a_clean_full_round(self):
        h = R.verdicts(_records(_monotone))["hsafe"]
        assert h["verdict"] == "HOLDS"
        assert h["violation_steps"] == 0
        assert h["offenders"] == []

    def test_fails_and_names_the_offending_cell(self):
        bad = R.cell_name("STR14", "GREEDY", 0.20, 2, 0.2)

        def viol(spec):
            return 3 if spec["name"] == bad else 0

        h = R.verdicts(_records(_monotone, violations_fn=viol))["hsafe"]
        assert h["verdict"] == "FAILS"
        assert h["violation_steps"] == 3
        assert h["offenders"] == [bad]

    def test_incomplete_when_cells_are_missing_and_none_offend(self):
        recs = _records(_monotone)
        del recs[R.cell_name("RC19", "ATT25", 0.30, 2, 0.2)]
        assert R.verdicts(recs)["hsafe"]["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------- cell bookkeeping
class TestRunCells:
    def test_skips_cells_that_already_have_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "check_obstacles_nested", lambda: {"pass": True})
        done = R.cell_name("RC19", "ATT25", 0.0, 2, 0.2)
        todo = R.cell_name("RC19", "ATT25", 0.05, 2, 0.2)
        R.cells_dir().mkdir(parents=True)
        (R.cells_dir() / f"{done}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _stub_record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells([done, todo], workers=1)
        assert called == [todo]
        assert json.loads((R.cells_dir() / f"{todo}.json").read_text())["name"] == todo
        assert (R.cells_dir() / f"{done}.json").read_text() == "{}"

    def test_no_names_runs_every_missing_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "check_obstacles_nested", lambda: {"pass": True})
        R.cells_dir().mkdir(parents=True)
        done = list(R.CELLS)[:5]
        for name in done:
            (R.cells_dir() / f"{name}.json").write_text("{}")
        called = []

        def stub(name, device="cpu"):
            called.append(name)
            return _stub_record(name)

        monkeypatch.setattr(R, "run_cell", stub)
        R.run_cells(None, workers=1)
        assert len(called) == len(R.CELLS) - 5
        assert set(called).isdisjoint(done)

    def test_unknown_cell_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit):
            R.run_cells(["NOPE"], workers=1)

    def test_a_failed_nested_check_stops_the_round(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(
            R, "check_obstacles_nested", lambda: {"pass": False, "reason": "drift"}
        )
        monkeypatch.setattr(R, "run_cell", lambda name, device="cpu": None)
        with pytest.raises(SystemExit):
            R.run_cells(None, workers=1)

    def test_records_are_written_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        name = R.cell_name("RC19", "ATT25", 0.0, 2, 0.2)
        R.cells_dir().mkdir(parents=True)
        R.write_cell(_record("RC19", "ATT25", 0.0, 2, 0.2, 0.5))
        assert json.loads((R.cells_dir() / f"{name}.json").read_text())["name"] == name
        assert list(R.cells_dir().glob("*.tmp")) == []


# ------------------------------------------------------------- nested obstacles
class TestNestedObstacles:
    def test_first_two_obstacles_coincide_between_n_two_and_n_four(self):
        out = R.check_obstacles_nested()
        assert out["pass"] is True
        assert out["seed"] == R.SEED0
        assert len(out["obstacles_n2"]) == 2
        assert len(out["obstacles_n4"]) == 4
        assert out["obstacles_n4"][:2] == out["obstacles_n2"]


# -------------------------------------------------------------------- summarize
class TestSummarize:
    def test_writes_round_json_and_png_with_the_nested_result(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.cells_dir().mkdir(parents=True)
        for rec in _records(_monotone).values():
            R.write_cell(rec)
        summary = R.summarize()
        assert (tmp_path / "s9_round.json").exists()
        assert (tmp_path / "s9_round.png").exists()
        blob = json.loads((tmp_path / "s9_round.json").read_text())
        assert blob["obstacles_nested"]["pass"] is True
        assert blob["verdicts"]["sweeps"]["delta"]["verdict"] == "CONFIRMED"
        assert summary["verdicts"]["hsafe"]["verdict"] == "HOLDS"
        assert len(blob["cells"]) == 132
        # the per-episode indicator lists stay in the cell records
        assert "deadlock_ind" not in next(iter(blob["cells"].values()))
