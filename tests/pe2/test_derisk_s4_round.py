"""scripts/derisk_s4_shield.py — the shielded-learned-pursuer round's own decisions."""

import importlib.util
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
            "intervention_rate": 0.1,
            "infeasibility_rate": 0.0,
            "backup_rate": 0.0,
            "stall_step_rate": 0.2,
            "deadlock_episode_rate": 0.1,
            "min_clearance_floor": 0.01,
            "tier_histogram": {"exact": n, "relaxed": 0, "backup": 0, "none": 0},
        },
        "rate": wins / n,
    }


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

    def test_h1_incomplete(self):
        assert R.h1_verdict({})["verdict"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe_holds_when_all_shielded_cells_and_arms_are_clean(self):
        cells = {
            "STR14:GREEDY12:SHIELD": _cell("STR14", "GREEDY12", "SHIELD", 300),
            "STR14:GREEDY12:ROBUST": _cell("STR14", "GREEDY12", "ROBUST", 290),
            "STR14:GREEDY12:NONE": _cell(
                "STR14", "GREEDY12", "NONE", 340, violations=30
            ),
        }
        arms = {
            "STRSH16": {
                "train_violations": 0,
                "probe_violations": 0,
                "final": {"violations": 0},
            }
        }
        h = R.hsafe_verdict(cells, arms)
        assert h["verdict"] == "HOLDS" and h["shielded_violation_steps"] == 0

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
