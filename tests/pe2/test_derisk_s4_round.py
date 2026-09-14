"""scripts/derisk_s4_shield.py — the shielded-learned-pursuer round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s4_shield.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s4_shield", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _logs(wins, n, seed0=0):
    return [
        {
            "seed": seed0 + i,
            "role": "pursuer",
            "outcome": "capture" if i < wins else "timeout",
        }
        for i in range(n)
    ]


def _cell(policy, opp, arm, wins, violations=0, n=500):
    return {
        "policy": policy,
        "opponent": opp,
        "arm": arm,
        "n": n,
        "logs": _logs(wins, n),
        "shield": {
            "violation_steps": violations,
            "episodes_with_violation": 1 if violations else 0,
            "violation_episode_rate": (1 if violations else 0) / n,
            "intervention_rate": 0.1,
            "infeasibility_rate": 0.0,
            "backup_rate": 0.0,
            "stall_step_rate": 0.2,
            "immobile_step_rate": 0.05,
            "deadlock_episode_rate": 0.1,
            "pinned_episode_rate": 0.02,
            "min_clearance_floor": 0.01,
            "min_clearance_hard_floor": 0.005,
            "tier_histogram": {"exact": n, "relaxed": 0, "backup": 0, "none": 0},
        },
        "rate": wins / n,
    }


def _arm_record(k=330, passes=True, violations=0):
    return {
        "train_violations": 0,
        "probe_violations": 0,
        "final": {
            "violations": violations,
            "gate": {
                "STR14_SHIELD": {"pass": passes, "rate": k / 500},
                "TEACHER_NONE": {"pass": True, "rate": k / 500},
            },
        },
    }


def _census(none_violations=30):
    """The complete round: all 21 cells (the 14 shielded ones clean, the NONE controls
    carrying their own contacts) and all three training arms with a final block."""
    cells = {}
    for c in R.CELLS:
        policy, opp, arm = c.split(":")
        cells[c] = _cell(
            policy,
            opp,
            arm,
            300,
            violations=none_violations if arm == "NONE" else 0,
        )
    return cells, {a: _arm_record() for a in R.ARMS}


class TestProtocol:
    def test_constants(self):
        from pe2.structured import HOLD

        assert (R.N_CELL, R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N) == (
            500,
            10_000_000,
            1_000_000,
            200,
        )
        assert (
            R.PROBE_SEED0 == 1_100_000 and R.SEEDS == (16, 17, 18) and R.PASS_MIN == 2
        )
        assert (
            R.P_GATE == 0.05
            and R.ROBUST_DELTA == 0.05
            and R.GAMMA_S == 0.99
            and R.HOLD == HOLD
        )
        assert (
            R.OPPONENTS["GREEDY12"]["seed0"] == 1_090_000
            and R.OPPONENTS["GREEDY14"]["speed"] == 1.4
        )
        assert set(R.OPPONENTS) == {
            "GREEDY12",
            "FOV",
            "HUG",
            "OPEN",
            "GREEDY10",
            "GREEDY14",
        }
        assert set(R.SHIELD_ARMS) == {"NONE", "SHIELD", "ROBUST"}

    def test_cells_are_21_and_teacher_only_on_greedy12(self):
        assert len(R.CELLS) == 21
        teacher = [c for c in R.CELLS if c.startswith("TEACHER:")]
        assert teacher == [
            "TEACHER:GREEDY12:NONE",
            "TEACHER:GREEDY12:SHIELD",
            "TEACHER:GREEDY12:ROBUST",
        ]
        assert "STR14:OPEN:ROBUST" in R.CELLS

    def test_training_arms(self):
        assert set(R.ARMS) == {"STRSH16", "STRSH17", "STRSH18"}
        assert R.ARMS["STRSH17"] == {"seed": 17}

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s4_cell_STR14_GREEDY12_NONE.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_cell_not_run("STR14:GREEDY12:NONE")
        R.require_cell_not_run("STR14:GREEDY12:SHIELD")
        (tmp_path / "ckpts_STRSH16").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("STRSH16")


class TestShift:
    def test_paired_shift_direction_and_ci(self):
        s = R.paired_shift(
            _logs(300, 500), _logs(340, 500)
        )  # shielded 0.60 vs unshielded 0.68: 40 losses, 0 wins
        assert (
            s["delta"] == pytest.approx(-0.08)
            and s["b"] == 0
            and s["c"] == 40
            and s["p"] < 0.05
        )
        assert s["ci"][0] <= s["delta"] <= s["ci"][1]

    def test_h1_confirmed(self):
        cells = {
            "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340),
            "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 300),
        }
        assert R.h1_verdict(cells)["verdict"] == "CONFIRMED"

    def test_h1_no_shift(self):
        cells = {
            "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340),
            "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 338),
        }
        assert R.h1_verdict(cells)["verdict"] == "NO SHIFT"

    def test_h1_reversed(self):
        cells = {
            "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 300),
            "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 340),
        }
        assert R.h1_verdict(cells)["verdict"] == "REVERSED"

    def test_h1_shift_not_separated(self):
        # amendment A3: 0.650 vs 0.680 — b = 0, c = 15, p < 0.05, but the shielded
        # Wilson upper bound (~0.691) does not clear the unshielded point 0.680
        cells = {
            "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340),
            "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 325),
        }
        r = R.h1_verdict(cells)
        assert r["verdict"] == "SHIFT NOT SEPARATED"
        assert r["b"] == 0 and r["c"] == 15 and r["p"] < 0.05

    def test_h1_incomplete(self):
        assert R.h1_verdict({})["verdict"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe_holds_only_on_a_complete_clean_census(self):
        cells, arms = _census()
        h = R.hsafe_verdict(cells, arms)
        assert h["verdict"] == "HOLDS" and h["shielded_violation_steps"] == 0
        assert h["missing"] == []

    def test_hsafe_incomplete_on_a_partial_census(self):
        # amendment A4: a clean but partial census is INCOMPLETE, never HOLDS
        cells, arms = _census()
        del cells["STR14:FOV:ROBUST"]
        del arms["STRSH18"]
        arms["STRSH17"] = {"verdict": "UNDETERMINED", "checkpoints": []}  # crashed
        h = R.hsafe_verdict(cells, arms)
        assert h["verdict"] == "INCOMPLETE"
        assert set(h["missing"]) == {"STR14:FOV:ROBUST", "STRSH17", "STRSH18"}
        assert "incomplete" in R.direction(
            {"hsafe": "INCOMPLETE", "h1": "CONFIRMED", "h3": "RECOVERED"}
        )

    def test_hsafe_counts_the_14_shielded_cells(self):
        cells, arms = _census()
        shielded = [c for c in R.CELLS if c.rsplit(":", 1)[1] in ("SHIELD", "ROBUST")]
        assert len(shielded) == 14
        h = R.hsafe_verdict({c: cells[c] for c in shielded}, arms)
        assert h["verdict"] == "HOLDS" and h["missing"] == []

    def test_hsafe_fails_on_any_shielded_violation(self):
        cells = {"STR14:FOV:SHIELD": _cell("STR14", "FOV", "SHIELD", 300, violations=1)}
        assert R.hsafe_verdict(cells, {})["verdict"] == "FAILS"
        cells = {"STR14:FOV:SHIELD": _cell("STR14", "FOV", "SHIELD", 300)}
        arms = {
            "STRSH16": {
                "train_violations": 2,
                "probe_violations": 0,
                "final": {"violations": 0},
            }
        }
        assert R.hsafe_verdict(cells, arms)["verdict"] == "FAILS"

    def test_hsafe_reads_a_crash_records_probe_rows(self):
        # no rolled-up probe_violations (a crashed arm): fall back to the checkpoint rows
        cells, arms = _census()
        arms["STRSH16"] = {
            "verdict": "UNDETERMINED",
            "train_violations": 0,
            "checkpoints": [{"violations": 0}, {"violations": 4}],
        }
        h = R.hsafe_verdict(cells, arms)
        assert h["verdict"] == "FAILS" and h["shielded_violation_steps"] == 4
        assert "STRSH16" in h["offenders"]

    def test_none_arm_weakness_is_reported(self):
        cells = {
            "STR14:GREEDY12:NONE": _cell("STR14", "GREEDY12", "NONE", 340, violations=0)
        }
        h = R.hsafe_verdict(cells, {})
        assert h["none_greedy12_violation_episode_rate"] is not None


class TestH3:
    def _arm(self, k, pass_vs_shield):
        return {
            "final": {
                "violations": 0,
                "gate": {
                    "STR14_SHIELD": {"pass": pass_vs_shield, "rate": k / 500},
                    "TEACHER_NONE": {"pass": True, "rate": k / 500},
                },
            }
        }

    def test_recovered_needs_two_of_three(self):
        arms = {
            "STRSH16": self._arm(330, True),
            "STRSH17": self._arm(320, True),
            "STRSH18": self._arm(280, False),
        }
        assert R.h3_verdict(arms)["verdict"] == "RECOVERED"
        arms["STRSH17"] = self._arm(280, False)
        assert R.h3_verdict(arms)["verdict"] == "NOT RECOVERED"

    def test_missing_seed_is_incomplete(self):
        assert (
            R.h3_verdict({"STRSH16": self._arm(330, True)})["verdict"] == "INCOMPLETE"
        )


class TestDirection:
    def test_rows(self):
        assert "central result" in R.direction(
            {"hsafe": "HOLDS", "h1": "CONFIRMED", "h3": "RECOVERED"}
        )
        assert "free for the learned pursuer" in R.direction(
            {"hsafe": "HOLDS", "h1": "NO SHIFT", "h3": "NOT RECOVERED"}
        )
        assert "refutation" in R.direction(
            {"hsafe": "HOLDS", "h1": "REVERSED", "h3": "INCOMPLETE"}
        )
        assert "Stop" in R.direction(
            {"hsafe": "FAILS", "h1": "CONFIRMED", "h3": "RECOVERED"}
        )
        assert "incomplete" in R.direction(
            {"hsafe": "HOLDS", "h1": "INCOMPLETE", "h3": "INCOMPLETE"}
        )
        d = R.direction(
            {"hsafe": "HOLDS", "h1": "SHIFT NOT SEPARATED", "h3": "RECOVERED"}
        )
        assert "not separated at N = 500" in d and "RECOVERED" in d

    def test_hsafe_fails_outranks_an_incomplete_census(self):
        assert "Stop" in R.direction(
            {"hsafe": "FAILS", "h1": "INCOMPLETE", "h3": "INCOMPLETE"}
        )


class TestRunnerRecords:
    def test_reference_cells_are_checked_before_any_training(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="reference cell"):
            R.run_train_arm("STRSH16", "cpu")
        assert not (tmp_path / "ckpts_STRSH16").exists()

    def test_crash_record_carries_the_violation_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)

        class _Env:
            violations_total = 3

        def _boom(arm, device, holder=None):
            holder["env"] = _Env()
            holder["rows"] = [{"violations": 1}, {"violations": 2}]
            raise RuntimeError("boom")

        monkeypatch.setattr(R, "run_train_arm", _boom)
        with pytest.raises(RuntimeError):
            R.run_arm("STRSH16", "cpu")
        rec = json.loads((tmp_path / "s4_arm_STRSH16.json").read_text())
        assert rec["verdict"] == "UNDETERMINED"
        assert rec["train_violations"] == 3 and rec["probe_violations"] == 3

    def test_systemexit_mid_run_still_leaves_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)

        def _refuse(arm, device, holder=None):
            raise SystemExit("reference cell STR14:GREEDY12:SHIELD has not run")

        monkeypatch.setattr(R, "run_train_arm", _refuse)
        with pytest.raises(SystemExit):
            R.run_arm("STRSH16", "cpu")
        rec = json.loads((tmp_path / "s4_arm_STRSH16.json").read_text())
        assert rec["verdict"] == "UNDETERMINED" and "reference cell" in rec["reason"]
        assert rec["train_violations"] == 0 and rec["probe_violations"] == 0

    def test_writes_are_atomic_and_leave_no_tmp(self, tmp_path):
        p = tmp_path / "x.json"
        R._write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}
        assert list(tmp_path.iterdir()) == [p]


class TestSummary:
    def test_complete_needs_a_final_block_in_every_arm(self):
        cells, arms = _census()
        assert R.summarize(cells, arms)["complete"]
        arms["STRSH17"] = {"verdict": "UNDETERMINED", "checkpoints": []}
        assert not R.summarize(cells, arms)["complete"]

    def test_cost_table_carries_the_full_h_cost_split(self):
        cells, arms = _census()
        c = R.summarize(cells, arms)["cost"]["STR14:GREEDY12:SHIELD"]
        for k in (
            "intervention_rate",
            "infeasibility_rate",
            "backup_rate",
            "stall_step_rate",
            "immobile_step_rate",
            "deadlock_episode_rate",
            "pinned_episode_rate",
            "tier_histogram",
            "violation_episode_rate",
            "min_clearance_floor",
            "min_clearance_hard_floor",
            "rate",
        ):
            assert c[k] is not None, k
