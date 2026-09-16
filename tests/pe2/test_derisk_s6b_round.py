"""scripts/derisk_s6b_attacker.py — the second attacker round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s6b_attacker.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s6b_attacker", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _arm(name, deadlocks):
    fam, rest = name.split("-")
    return {
        "arm": name,
        "family": fam,
        "defender": rest[:2],
        "seed": int(rest[2:]),
        "train_pursuer_violations": 0,
        "probe_pursuer_violations": 0,
        "final": {
            "pursuer_violations": 0,
            "deadlock_ind": deadlocks,
            "escape_ind": _ind(0.8),
            "pursuer": {
                "deadlock_episode_rate": sum(deadlocks) / len(deadlocks),
                "escape_rate": 0.8,
                "stall_step_rate": 0.4,
                "intervention_rate": 0.5,
            },
        },
        "checkpoints": [],
    }


def _starts(hug7=(0.53, 0.34), greedy=(0.49, 0.32)):
    return {
        d: {
            "HUG7": {
                "deadlock_ind": _ind(h),
                "escape_ind": _ind(0.6),
                "pursuer_violations": 0,
                "pursuer": {"deadlock_episode_rate": h, "escape_rate": 0.6},
            },
            "GREEDY": {
                "deadlock_ind": _ind(g),
                "escape_ind": _ind(0.5),
                "pursuer_violations": 0,
                "pursuer": {"deadlock_episode_rate": g, "escape_rate": 0.5},
            },
        }
        for d, h, g in zip(("D1", "D2"), hug7, greedy, strict=True)
    }


def _full(
    ctl=((0.66, 0.68, 0.64), (0.62, 0.65, 0.61)),
    ez=((0.48, 0.47, 0.50), (0.34, 0.35, 0.33)),
):
    res = {}
    for d, c3, e3 in zip(("D1", "D2"), ctl, ez, strict=True):
        for s, c, e in zip(R.SEEDS, c3, e3, strict=True):
            res[f"CTL-{d}{s}"] = _arm(f"CTL-{d}{s}", _ind(c))
            res[f"EZ-{d}{s}"] = _arm(f"EZ-{d}{s}", _ind(e))
    return res


class TestProtocol:
    def test_constants_and_overrides(self):
        assert R.SEEDS == (25, 26, 27) and (R.PROBE_SEED0, R.GATE_SEED0) == (
            1_140_000,
            1_150_000,
        )
        assert (R.GATE_N, R.EVAL_N, R.TOTAL_LONG, R.EVERY_LONG) == (
            500,
            200,
            10_000_000,
            1_000_000,
        )
        assert R.PASS_MIN == 2 and R.P_GATE == 0.05
        assert (
            R.S6.PROBE_SEED0 == 1_140_000 and R.S6.GATE_SEED0 == 1_150_000
        )  # overrides applied to the S6 module
        assert (
            R.S6.OUT == R.OUT
            and R.S6.START_PATH == R.START_PATH == R.OUT / "s6b_start.json"
        )
        assert R.S6.ARMS is R.ARMS

    def test_arm_table(self):
        assert len(R.ARMS) == 12 and {a.split("-")[0] for a in R.ARMS} == {
            "CTL",
            "EZ",
        }
        assert R.ARMS["CTL-D125"] == {
            "family": "CTL",
            "defender": "D1",
            "seed": 25,
            "beta": 0.0,
            "zero_reward": False,
        }
        assert R.ARMS["EZ-D227"] == {
            "family": "EZ",
            "defender": "D2",
            "seed": 27,
            "beta": 0.0,
            "zero_reward": True,
        }
        assert all(v["beta"] == 0.0 for v in R.ARMS.values())

    def test_artefact_names_are_s6b(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R.S6, "OUT", tmp_path)
        assert R.S6._arm_file("CTL-D125").name == "s6b_arm_CTL-D125.json"
        assert R.S6._probe_file("CTL-D125").name == "s6b_probe_CTL-D125.json"

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R.S6, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s6b_start.json")
        monkeypatch.setattr(R.S6, "START_PATH", tmp_path / "s6b_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s6b_arm_CTL-D125.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.S6.require_arm_not_run("CTL-D125")


class TestVerdicts:
    def test_pass_both_clauses(self):
        h = R.h2_verdicts(_full(), _starts())
        assert h["H2"] == {"D1": "PASS", "D2": "PASS"} and h["H2_overall"] == "PASS"
        assert h["clauses"]["D1"]["vs_ez"] == 3 and h["clauses"]["D1"]["vs_hug7"] == 3
        assert h["harness"] == {"D1": "VALID", "D2": "VALID"}

    def test_fail_when_ctl_equals_ez(self):
        h = R.h2_verdicts(
            _full(ctl=((0.48, 0.47, 0.50), (0.34, 0.35, 0.33))), _starts()
        )
        assert h["H2"]["D1"] == "FAIL" and h["H2"]["D2"] == "FAIL"

    def test_pass_vs_ez_only_when_structure_beats_hug7(self):
        h = R.h2_verdicts(_full(ez=((0.60, 0.61, 0.62), (0.34, 0.35, 0.33))), _starts())
        assert (
            h["harness"]["D1"] == "STRUCTURE-DRIVEN"
            and h["H2"]["D1"] == "PASS (vs EZ only)"
        )
        assert h["harness"]["D2"] == "VALID" and h["H2"]["D2"] == "PASS"

    def test_missing_seed_incomplete(self):
        res = _full()
        del res["EZ-D226"]
        assert R.h2_verdicts(res, _starts())["H2"]["D2"] == "INCOMPLETE"

    def test_r3b(self):
        h = R.h2_verdicts(_full(), _starts())
        assert set(h["R3b"]) >= {
            "gains_D1",
            "gains_D2",
            "mean_gain_D1",
            "mean_gain_D2",
            "ctl_mean_D1",
            "ctl_mean_D2",
            "remedy_reduces_gain",
        }


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        st = _starts()
        st["D1"]["HUG7"]["pursuer_violations"] = 1
        assert R.hsafe_verdict(_full(), st)["verdict"] == "FAILS"
        res = _full()
        del res["CTL-D127"]
        assert R.hsafe_verdict(res, _starts())["verdict"] == "INCOMPLETE"


class TestDirection:
    def test_rows(self):
        assert "weaponises" in R.direction({"hsafe": "HOLDS", "H2_overall": "PASS"})
        assert "no R2" in R.direction({"hsafe": "HOLDS", "H2_overall": "FAIL"})
        assert "Stop" in R.direction({"hsafe": "FAILS", "H2_overall": "PASS"})
        assert "incomplete" in R.direction(
            {"hsafe": "INCOMPLETE", "H2_overall": "INCOMPLETE"}
        )
