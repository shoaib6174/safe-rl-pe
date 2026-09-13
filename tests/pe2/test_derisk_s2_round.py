"""scripts/derisk_s2_cp_round.py — the S2 round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s2_cp_round.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s2_cp_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, sel_pass, fin_pass, kind="ppo"):
    return {
        "arm": name,
        "kind": kind,
        "selected": {
            "gate": {"PPO_CLONE": {"pass": sel_pass}, "TEACHER": {"pass": False}}
        },
        "final": {
            "gate": {"PPO_CLONE": {"pass": fin_pass}, "TEACHER": {"pass": False}}
        },
        "checkpoints": [],
    }


class TestProtocol:
    def test_pre_registered_constants(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.GATE_N) == (1_000_000, 100_000, 200, 500)
        assert (R.SEL_SEED0, R.GATE_SEED0, R.S1_EVAL_SEED0) == (
            1_040_000,
            1_050_000,
            1_010_000,
        )
        assert R.SEEDS == (7, 8, 9) and R.BETAS == {"ANC01": 0.1, "ANC001": 0.01}
        assert R.CP_LAMBDA == 1.0 and R.P_GATE == 0.05
        assert (R.TOTAL_LONG, R.EVERY_LONG) == (10_000_000, 1_000_000)

    def test_arm_table(self):
        names = set(R.ARMS)
        assert names == {
            "SEL7",
            "SEL8",
            "SEL9",
            "ANC01_7",
            "ANC01_8",
            "ANC01_9",
            "ANC001_7",
            "ANC001_8",
            "ANC001_9",
            "NULL7",
            "SBLONG",
        }
        assert R.ARMS["SEL8"] == {
            "kind": "ppo",
            "beta": 0.0,
            "seed": 8,
            "zero_reward": False,
        }
        assert R.ARMS["ANC01_9"] == {
            "kind": "ppo",
            "beta": 0.1,
            "seed": 9,
            "zero_reward": False,
        }
        assert R.ARMS["ANC001_7"] == {
            "kind": "ppo",
            "beta": 0.01,
            "seed": 7,
            "zero_reward": False,
        }
        assert R.ARMS["NULL7"] == {
            "kind": "ppo",
            "beta": 0.0,
            "seed": 7,
            "zero_reward": True,
        }
        assert R.ARMS["SBLONG"] == {"kind": "sblong", "seed": 7}

    def test_arms_refuse_without_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()

    def test_measure_start_refuses_to_re_measure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s2_start.json")
        (tmp_path / "s2_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()

    def test_an_arm_refuses_to_run_twice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s2_arm_SEL7.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("SEL7")
        (tmp_path / "ckpts_ANC01_8").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("ANC01_8")
        R.require_arm_not_run("SEL9")  # clean: no raise


class TestHypotheses:
    def test_h_sel_needs_two_of_three_selected_passes(self):
        res = {
            "SEL7": _arm("SEL7", True, False),
            "SEL8": _arm("SEL8", True, False),
            "SEL9": _arm("SEL9", False, False),
            "NULL7": _arm("NULL7", False, False),
        }
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"] == "PASS" and h["H_SEL_passes"] == 2

    def test_h_anc_uses_final_checkpoints_per_beta(self):
        res = {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res |= {f"ANC001_{s}": _arm(f"ANC001_{s}", True, False) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", False, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_ANC"]["ANC01"] == "PASS" and h["H_ANC"]["ANC001"] == "FAIL"

    def test_null_pass_makes_h_sel_undetermined_only(self):
        res = {f"SEL{s}": _arm(f"SEL{s}", True, False) for s in (7, 8, 9)}
        res |= {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", True, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"].startswith("UNDETERMINED") and h["H_ANC"]["ANC01"] == "PASS"

    def test_missing_arms_are_incomplete(self):
        h = R.hypothesis_verdicts({"SEL7": _arm("SEL7", True, False)})
        assert h["H_SEL"] == "INCOMPLETE"


class TestDirection:
    def test_anc_pass_is_the_artefact(self):
        d = R.direction(
            {
                "H_SEL": "FAIL",
                "H_ANC": {"ANC01": "PASS", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "no selection" in d and "Route C withdrawn" in d

    def test_sel_only(self):
        d = R.direction(
            {
                "H_SEL": "PASS",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "selection-based" in d and "needs the gate" in d

    def test_both_fail(self):
        d = R.direction(
            {
                "H_SEL": "FAIL",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "seed-7-specific" in d

    def test_null_pass(self):
        d = R.direction(
            {
                "H_SEL": "UNDETERMINED (selection manufactures improvement)",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "PASS",
            }
        )
        assert "manufactures" in d

    def test_incomplete(self):
        assert "incomplete" in R.direction(
            {
                "H_SEL": "INCOMPLETE",
                "H_ANC": {"ANC01": "INCOMPLETE", "ANC001": "INCOMPLETE"},
                "NULL": "INCOMPLETE",
            }
        )


class TestDeterminism:
    def test_reproduced_when_curves_match(self):
        rows = [
            {"steps": s, "win_rate": r}
            for s, r in zip(
                range(100_000, 1_000_001, 100_000),
                [0.64, 0.645, 0.585, 0.605, 0.51, 0.55, 0.56, 0.38, 0.285, 0.265],
                strict=True,
            )
        ]
        assert R.determinism_note(rows, rows) == "reproduced"
        other = [dict(r) for r in rows]
        other[3]["win_rate"] = 0.6
        assert R.determinism_note(rows, other).startswith("not reproduced")
