"""scripts/derisk_s3_confirm.py — the confirmatory round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s3_confirm.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s3_confirm", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, seed, teacher_pass, rate, kind="train"):
    return {
        "arm": name,
        "kind": kind,
        "seed": seed,
        "final": {
            "score": {"rate": rate, "k": int(rate * 500), "n": 500},
            "gate": {
                "TEACHER": {"pass": teacher_pass, "rate": rate},
                "PPO_CLONE": {"pass": teacher_pass, "rate": rate},
                "SAC_CLONE": {"pass": False, "rate": rate},
                "SEL7_100K": {"pass": False, "rate": rate},
                "GREEDY_FULLSTATE": {"pass": False, "rate": rate},
            },
        },
        "checkpoints": [],
    }


def _full(passes, sz_pass=False, rates=(0.66, 0.70, 0.62, 0.68, 0.64)):
    res = {
        f"STR{s}": _arm(f"STR{s}", s, p, r)
        for s, p, r in zip(R.SEEDS, passes, rates, strict=True)
    }
    res["SZLONG11"] = _arm("SZLONG11", 11, sz_pass, 0.30, kind="control")
    res["SBLONG"] = _arm("SBLONG", 7, True, 0.69, kind="existing")
    return res


class TestProtocol:
    def test_pre_registered_constants(self):
        from pe2.structured import HOLD

        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (
            10_000_000,
            1_000_000,
            200,
            500,
        )
        assert (R.PROBE_SEED0, R.GATE_SEED0, R.FAMILY_SEED0, R.FAMILY_N) == (
            1_060_000,
            1_070_000,
            1_080_000,
            200,
        )
        assert R.SEEDS == (11, 12, 13, 14, 15) and R.PASS_MIN == 4 and R.P_GATE == 0.05
        assert R.GAMMA_S == 0.99 and R.HOLD == HOLD == 10
        assert R.REFS == (
            "TEACHER",
            "PPO_CLONE",
            "SAC_CLONE",
            "SEL7_100K",
            "GREEDY_FULLSTATE",
        )

    def test_arm_table(self):
        assert set(R.ARMS) == {
            "STR11",
            "STR12",
            "STR13",
            "STR14",
            "STR15",
            "SZLONG11",
            "SBLONG",
        }
        assert R.ARMS["STR13"] == {"kind": "train", "seed": 13, "zero_reward": False}
        assert R.ARMS["SZLONG11"] == {
            "kind": "control",
            "seed": 11,
            "zero_reward": True,
        }
        assert R.ARMS["SBLONG"] == {
            "kind": "existing",
            "seed": 7,
            "ckpt": "results/derisk/ckpts_SBLONG/SBLONG_10000000.zip",
        }

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s3_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s3_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()
        (tmp_path / "s3_arm_STR12.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("STR12")
        R.require_arm_not_run("STR13")


class TestVerdicts:
    def test_pass_needs_four_of_five(self):
        h = R.hypothesis_verdict(_full([True, True, True, True, False]))
        assert h["H_STR"] == "PASS" and h["passes"] == 4
        h = R.hypothesis_verdict(_full([True, True, True, False, False]))
        assert h["H_STR"] == "FAIL" and h["passes"] == 3

    def test_control_passing_invalidates(self):
        h = R.hypothesis_verdict(_full([True] * 5, sz_pass=True))
        assert h["H_STR"].startswith("UNDETERMINED") and h["harness"] == "INVALID"

    def test_missing_seed_is_incomplete(self):
        res = _full([True] * 5)
        del res["STR14"]
        assert R.hypothesis_verdict(res)["H_STR"] == "INCOMPLETE"
        res = _full([True] * 5)
        del res["SZLONG11"]
        assert R.hypothesis_verdict(res)["H_STR"] == "INCOMPLETE"

    def test_artefact_is_highest_passing_fresh_seed_ties_to_lowest(self):
        res = _full(
            [True, True, True, True, False], rates=(0.66, 0.70, 0.70, 0.68, 0.99)
        )
        h = R.hypothesis_verdict(res)
        assert (
            h["artefact"] == "STR12"
        )  # 0.70 tie between 12 and 13 -> lowest seed; STR15 fails so 0.99 ignored
        res = _full([True, True, True, False, False])
        assert (
            R.hypothesis_verdict(res)["artefact"] is None
        )  # no artefact without a pass

    def test_sblong_is_never_the_artefact(self):
        res = _full([True] * 5, rates=(0.60, 0.61, 0.62, 0.63, 0.64))
        assert R.hypothesis_verdict(res)["artefact"] == "STR15"


class TestDirection:
    def test_pass(self):
        assert "paper's learned pursuer" in R.direction(
            {"H_STR": "PASS", "harness": "VALID"}
        )

    def test_fail(self):
        assert "seed-specific" in R.direction({"H_STR": "FAIL", "harness": "VALID"})

    def test_invalid(self):
        assert "harness" in R.direction(
            {"H_STR": "UNDETERMINED (harness)", "harness": "INVALID"}
        )

    def test_incomplete(self):
        assert "incomplete" in R.direction({"H_STR": "INCOMPLETE", "harness": "VALID"})


class TestFamilyTable:
    def test_rows_have_ci_and_mcnemar(self):
        logs_a = [
            {
                "seed": i,
                "role": "pursuer",
                "outcome": "capture" if i < 120 else "timeout",
            }
            for i in range(200)
        ]
        logs_t = [
            {
                "seed": i,
                "role": "pursuer",
                "outcome": "capture" if i < 100 else "timeout",
            }
            for i in range(200)
        ]
        row = R.family_row(logs_a, logs_t)
        assert (
            row["rate"] == 0.6
            and row["teacher_rate"] == 0.5
            and row["b"] == 20
            and row["c"] == 0
        )
        assert row["wilson"][0] < 0.6 < row["wilson"][1] and 0.0 <= row["p"] <= 1.0
