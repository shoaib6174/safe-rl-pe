"""scripts/derisk_s5_remedy.py — the remedy round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s5_remedy.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s5_remedy", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, teacher_pass, rate, viol=0):
    return {
        "arm": name,
        "family": name[:2],
        "seed": int(name[2:]),
        "train_violations": viol,
        "probe_violations": 0,
        "final": {
            "violations": 0,
            "score": {"rate": rate},
            "gate": {
                "TEACHER_SHIELD": {"pass": teacher_pass, "rate": rate},
                "STR14_SHIELD": {"pass": True, "rate": rate},
                "TEACHER_NONE": {"pass": False, "rate": rate},
            },
            "shield": {
                "stall_step_rate": 0.3,
                "deadlock_episode_rate": 0.2,
                "intervention_rate": 0.5,
                "infeasibility_rate": 0.0,
            },
        },
        "checkpoints": [],
    }


def _full(
    ra=(True, True, False),
    rb=(False, False, False),
    rc=(True, True, True),
    rz=False,
    rates=None,
):
    res = {}
    for fam, passes in (("RA", ra), ("RB", rb), ("RC", rc)):
        for s, p in zip(R.SEEDS, passes, strict=True):
            res[f"{fam}{s}"] = _arm(
                f"{fam}{s}", p, (rates or {}).get(f"{fam}{s}", 0.5 if p else 0.4)
            )
    res["RZ19"] = _arm("RZ19", rz, 0.3)
    return res


class TestProtocol:
    def test_constants(self):
        from pe2.structured import HOLD

        assert (
            R.STALL_W == 1e-3
            and R.SEEDS == (19, 20, 21)
            and R.PASS_MIN == 2
            and R.P_GATE == 0.05
        )
        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (
            10_000_000,
            1_000_000,
            200,
            500,
        )
        assert (
            R.PROBE_SEED0 == 1_110_000
            and R.GATE_SEED0 == 1_090_000
            and R.GAMMA_S == 0.99
            and R.HOLD == HOLD
        )

    def test_arm_table(self):
        assert set(R.ARMS) == {
            "RA19",
            "RA20",
            "RA21",
            "RB19",
            "RB20",
            "RB21",
            "RC19",
            "RC20",
            "RC21",
            "RZ19",
        }
        assert R.ARMS["RA20"] == {
            "family": "RA",
            "seed": 20,
            "shield_obs": True,
            "stall_penalty": 0.0,
            "zero_reward": False,
        }
        assert R.ARMS["RB21"] == {
            "family": "RB",
            "seed": 21,
            "shield_obs": False,
            "stall_penalty": 1e-3,
            "zero_reward": False,
        }
        assert R.ARMS["RC19"] == {
            "family": "RC",
            "seed": 19,
            "shield_obs": True,
            "stall_penalty": 1e-3,
            "zero_reward": False,
        }
        assert R.ARMS["RZ19"] == {
            "family": "RZ",
            "seed": 19,
            "shield_obs": True,
            "stall_penalty": 0.0,
            "zero_reward": True,
        }

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s5_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s5_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()
        (tmp_path / "s5_arm_RA19.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("RA19")
        R.require_arm_not_run("RA20")


class TestVerdicts:
    def test_h3_per_arm_and_overall(self):
        h = R.hypothesis_verdicts(_full())
        assert (
            h["H3"] == {"RA": "PASS", "RB": "FAIL", "RC": "PASS"}
            and h["H3_overall"] == "PASS"
        )

    def test_rz_pass_invalidates(self):
        h = R.hypothesis_verdicts(_full(rz=True))
        assert all(v.startswith("UNDETERMINED") for v in h["H3"].values()) and h[
            "H3_overall"
        ].startswith("UNDETERMINED")

    def test_missing_seed_is_incomplete(self):
        res = _full()
        del res["RB20"]
        h = R.hypothesis_verdicts(res)
        assert h["H3"]["RB"] == "INCOMPLETE" and h["H3"]["RA"] == "PASS"
        del res["RZ19"]
        assert R.hypothesis_verdicts(res)["H3_overall"] == "INCOMPLETE"

    def test_artefact_rule_order_rc_ra_rb_then_lowest_seed(self):
        res = _full(rates={"RA19": 0.60, "RA20": 0.60, "RC19": 0.60, "RC21": 0.55})
        h = R.hypothesis_verdicts(res)
        assert (
            h["artefact"] == "RC19"
        )  # tie at 0.60 between RA19, RA20, RC19 -> RC first, then lowest seed
        res = _full(rc=(False, False, False), rates={"RA19": 0.55, "RA20": 0.60})
        assert R.hypothesis_verdicts(res)["artefact"] == "RA20"
        res = _full(ra=(False,) * 3, rc=(False,) * 3)
        assert R.hypothesis_verdicts(res)["artefact"] is None


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full())["verdict"] == "HOLDS"
        res = _full()
        res["RB19"]["train_violations"] = 1
        assert R.hsafe_verdict(res)["verdict"] == "FAILS"
        res = _full()
        del res["RC21"]
        assert R.hsafe_verdict(res)["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "safe learned pursuer" in R.direction(
            {"hsafe": "HOLDS", "H3_overall": "PASS", "RZ": "FAIL"}
        )
        assert "feasibility constraint" in R.direction(
            {"hsafe": "HOLDS", "H3_overall": "FAIL", "RZ": "FAIL"}
        )
        assert "harness" in R.direction(
            {"hsafe": "HOLDS", "H3_overall": "UNDETERMINED (harness)", "RZ": "PASS"}
        )
        assert "Stop" in R.direction(
            {"hsafe": "FAILS", "H3_overall": "PASS", "RZ": "FAIL"}
        )
        assert "incomplete" in R.direction(
            {"hsafe": "INCOMPLETE", "H3_overall": "INCOMPLETE", "RZ": "INCOMPLETE"}
        )
