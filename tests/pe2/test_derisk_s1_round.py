"""scripts/derisk_s1_structured.py — the structured round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s1_structured.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s1_structured", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()

_STARTS = {
    "BAR_ORBIT": {"k": 245, "n": 500, "rate": 0.49, "wilson": [0.447, 0.533]},
    "FLOOR_RAND": {"k": 167, "n": 500, "rate": 0.334, "wilson": [0.294, 0.377]},
    "START_S": {"k": 100, "n": 500, "rate": 0.2, "wilson": [0.167, 0.237]},
    "START_CP": {
        "k": 230,
        "n": 500,
        "rate": 0.46,
        "wilson": [0.417, 0.504],
        "v2_pass": True,
    },
}


def _arm(name, verdict, rate=0.5):
    return {
        "arm": name,
        "verdict": verdict,
        "reason": "r",
        "checkpoints": [],
        "final": {
            "rate": rate,
            "k": int(rate * 500),
            "n": 500,
            "wilson": [rate - 0.04, rate + 0.04],
        },
    }


class TestProtocol:
    def test_pre_registered_constants(self):
        from pe2.structured import HOLD, R_ACQ

        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N) == (
            1_000_000,
            100_000,
            200,
            500,
        )
        assert R.EVAL_SEED0 == 1_010_000 and R.DEMO_SEED0 == 1_020_000
        assert (R.DEMO_EPISODES, R.BC_EPOCHS, R.SEED) == (300, 50, 7)
        assert R.GAMMA_S == 0.99 and R.CP_LAMBDA == 1.0 and R.V2_MIN_LO == 0.35
        assert (R.HOLD, R.R_ACQ) == (HOLD, R_ACQ) == (10, 0.2)
        assert set(R.ARMS) == {"SA", "SB", "SZ", "CP"}

    def test_arm_reward_settings(self):
        assert R.ARMS["SA"] == {"r_acq": 0.2, "zero_reward": False}
        assert R.ARMS["SB"] == {"r_acq": 0.0, "zero_reward": False}
        assert R.ARMS["SZ"] == {"r_acq": 0.0, "zero_reward": True}
        assert R.ARMS["CP"] == "control"

    def test_arms_refuse_to_run_without_measured_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()

    def test_measure_start_refuses_without_a_passed_preflight(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "missing.json").write_text(
            json.dumps({"pass": False, "mismatches": 3})
        )
        with pytest.raises(SystemExit, match="V1"):
            R.require_preflight()

    def test_measure_start_refuses_to_re_measure_after_an_arm_has_run(
        self, tmp_path, monkeypatch
    ):
        """Stop rule (spec section 5): re-measuring would move the bars the
        verdicts were computed against, and re-create the init SA/SB/SZ share."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s1_preflight.json")
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s1_start.json")
        # Also redirected so a regressed guard cannot write real artefacts.
        monkeypatch.setattr(R, "INIT_PATH", tmp_path / "s1_ppo_init.zip")
        monkeypatch.setattr(R, "CP_CLONE", tmp_path / "clone.zip")
        (tmp_path / "s1_preflight.json").write_text(json.dumps({"pass": True}))
        (tmp_path / "s1_arm_SA.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.measure_start("cpu")

    def test_the_stop_rule_guard_fires_on_bars_and_on_this_rounds_ckpts_only(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s1_start.json")
        (tmp_path / "ckpts_RQNEW").mkdir()  # another round's artefacts: not ours
        R.require_not_measured()
        (tmp_path / "ckpts_SB").mkdir()
        with pytest.raises(SystemExit, match="already run"):
            R.require_not_measured()
        (tmp_path / "s1_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already measured"):
            R.require_not_measured()


class TestDirection:
    def test_incomplete(self):
        assert "incomplete" in R.direction({"SA": "IMPROVED"})

    def test_sz_learned_invalidates_the_harness(self):
        d = R.direction(
            {"SA": "IMPROVED", "SB": "LEARNED", "SZ": "LEARNED", "CP": "COLLAPSED"}
        )
        assert "harness invalid" in d

    def test_sa_improved_cp_not(self):
        for cp in ("COLLAPSED", "PRESERVED", "UNDETERMINED"):
            d = R.direction(
                {"SA": "IMPROVED", "SB": "FAILED", "SZ": "FAILED", "CP": cp}
            )
            assert (
                "structure is what makes learning work" in d
                and "Route C withdrawn" in d
            )

    def test_both_improved(self):
        d = R.direction(
            {"SA": "IMPROVED", "SB": "FAILED", "SZ": "FAILED", "CP": "IMPROVED"}
        )
        assert "non-bootstrapping learner suffices" in d and "ablation" in d

    def test_sa_learned_not_improved(self):
        d = R.direction(
            {"SA": "LEARNED", "SB": "FAILED", "SZ": "FAILED", "CP": "COLLAPSED"}
        )
        assert "observation" in d and "not the learner" in d

    def test_sa_failed(self):
        d = R.direction(
            {"SA": "FAILED", "SB": "FAILED", "SZ": "FAILED", "CP": "COLLAPSED"}
        )
        assert "about the game" in d


class TestSummarize:
    def test_sz_learned_marks_structured_arms_undetermined(self):
        results = {
            "SA": _arm("SA", "IMPROVED"),
            "SB": _arm("SB", "LEARNED"),
            "SZ": _arm("SZ", "LEARNED"),
            "CP": _arm("CP", "COLLAPSED"),
        }
        s = R.summarize(results, _STARTS)
        assert s["arms"]["SA"]["verdict"] == "UNDETERMINED (harness)"
        assert s["arms"]["SB"]["verdict"] == "UNDETERMINED (harness)"
        assert s["arms"]["CP"]["verdict"] == "COLLAPSED"
        assert s["complete"] is True

    def test_sb_at_least_sa_is_reported(self):
        results = {
            "SA": _arm("SA", "LEARNED", 0.40),
            "SB": _arm("SB", "LEARNED", 0.42),
            "SZ": _arm("SZ", "FAILED", 0.2),
            "CP": _arm("CP", "COLLAPSED", 0.0),
        }
        s = R.summarize(results, _STARTS)
        assert "event reward added nothing" in s["direction"]

    def test_weak_prior_caveat_survives_into_the_summary(self):
        starts = dict(_STARTS, START_CP=dict(_STARTS["START_CP"], v2_pass=False))
        results = {
            "SA": _arm("SA", "FAILED"),
            "SB": _arm("SB", "FAILED"),
            "SZ": _arm("SZ", "FAILED"),
            "CP": _arm("CP", "COLLAPSED (weak prior)"),
        }
        s = R.summarize(results, starts)
        assert "weak prior" in s["arms"]["CP"]["verdict"]
        assert s["v2_pass"] is False
