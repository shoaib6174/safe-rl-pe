"""scripts/derisk_s6_attacker.py — the attacker round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s6_attacker.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s6_attacker", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500, seed0=0):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _arm(name, deadlocks, viol=0):
    fam, d = name.split("-")[0], name.split("-")[1][:2]
    seed = int(name.split("-")[1][2:]) if len(name.split("-")[1]) > 2 else 22
    return {
        "arm": name,
        "family": fam,
        "defender": d,
        "seed": seed,
        "train_pursuer_violations": viol,
        "probe_pursuer_violations": 0,
        "final": {
            "pursuer_violations": 0,
            "deadlock_ind": deadlocks,
            "escape_ind": _ind(0.5),
            "pursuer": {
                "deadlock_episode_rate": sum(deadlocks) / len(deadlocks),
                "stall_step_rate": 0.4,
                "intervention_rate": 0.6,
                "backup_rate": 0.0,
                "infeasibility_rate": 0.0,
            },
        },
        "checkpoints": [],
    }


def _comp(deadlock, escape):
    return {
        "deadlock_ind": _ind(deadlock),
        "escape_ind": _ind(escape),
        "pursuer_violations": 0,
        "pursuer": {"deadlock_episode_rate": deadlock, "escape_rate": escape},
    }


def _starts():
    out = {}
    for d in ("D1", "D2"):
        out[d] = {
            "SPLITHUG": _comp(0.50, 0.4),
            "HUG7": _comp(0.61, 0.5),
            "GREEDY": _comp(0.54, 0.6),
        }
    return out


def _full(atk=(0.70, 0.68, 0.66), ctl=(0.55, 0.56, 0.54), ez=0.50, d2_shift=-0.10):
    res = {}
    for d, shift in (("D1", 0.0), ("D2", d2_shift)):
        for s, ra, rc in zip(R.SEEDS, atk, ctl, strict=True):
            res[f"ATK-{d}{s}"] = _arm(f"ATK-{d}{s}", _ind(ra + shift))
            res[f"CTL-{d}{s}"] = _arm(f"CTL-{d}{s}", _ind(rc + shift))
        res[f"EZ-{d}22"] = _arm(f"EZ-{d}22", _ind(ez + shift))
    return res


class TestProtocol:
    def test_constants(self):
        assert (R.BETA, R.D_POS, R.HOLD) == (1e-3, 4.0, 10)
        assert (R.TOTAL_LONG, R.EVERY_LONG, R.EVAL_N, R.GATE_N) == (
            10_000_000,
            1_000_000,
            200,
            500,
        )
        assert (R.PROBE_SEED0, R.GATE_SEED0) == (1_120_000, 1_130_000)
        assert (
            R.SEEDS == (22, 23, 24)
            and R.PASS_MIN == 2
            and R.P_GATE == 0.05
            and R.GAMMA == 0.99
        )
        assert (
            set(R.DEFENDERS) == {"D1", "D2"}
            and R.DEFENDERS["D2"]["shield_obs"] is True
            and R.DEFENDERS["D1"]["shield_obs"] is False
        )

    def test_arm_table(self):
        assert len(R.ARMS) == 14
        assert R.ARMS["ATK-D122"] == {
            "family": "ATK",
            "defender": "D1",
            "seed": 22,
            "beta": 1e-3,
            "zero_reward": False,
        }
        assert R.ARMS["CTL-D224"] == {
            "family": "CTL",
            "defender": "D2",
            "seed": 24,
            "beta": 0.0,
            "zero_reward": False,
        }
        assert R.ARMS["EZ-D122"] == {
            "family": "EZ",
            "defender": "D1",
            "seed": 22,
            "beta": 0.0,
            "zero_reward": True,
        }

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s6_start.json")
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s6_preflight.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "s6_preflight.json").write_text('{"pass": false}')
        with pytest.raises(SystemExit, match="V1-P"):
            R.require_preflight()
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s6_arm_ATK-D122.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("ATK-D122")


class TestVerdicts:
    def test_h2_pass_on_both_defenders(self):
        h = R.h2_verdicts(_full(), _starts())
        assert h["H2"] == {"D1": "PASS", "D2": "PASS"} and h["H2_overall"] == "PASS"
        assert h["H2_implicit"]["D1"] in ("PASS", "FAIL") and h["harness"] == {
            "D1": "VALID",
            "D2": "VALID",
        }

    def test_h2_fail_when_atk_equals_ctl(self):
        h = R.h2_verdicts(_full(atk=(0.55, 0.56, 0.54)), _starts())
        assert h["H2"]["D1"] == "FAIL"

    def test_harness_invalid_when_ez_beats_splithug(self):
        h = R.h2_verdicts(_full(ez=0.70), _starts())
        assert h["harness"]["D1"] == "INVALID" and h["H2"]["D1"].startswith(
            "UNDETERMINED"
        )

    def test_missing_seed_incomplete(self):
        res = _full()
        del res["CTL-D223"]
        assert R.h2_verdicts(res, _starts())["H2"]["D2"] == "INCOMPLETE"

    def test_r3b_is_reported(self):
        h = R.h2_verdicts(_full(d2_shift=-0.10), _starts())
        assert (
            "R3b" in h
            and abs(h["R3b"]["mean_gain_D1"] - h["R3b"]["mean_gain_D2"]) < 0.02 + 1e-9
        )  # same gain shape under a pure shift


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full())["verdict"] == "HOLDS"
        res = _full()
        res["ATK-D122"]["train_pursuer_violations"] = 1
        assert R.hsafe_verdict(res)["verdict"] == "FAILS"
        res = _full()
        del res["EZ-D222"]
        assert R.hsafe_verdict(res)["verdict"] == "INCOMPLETE"

    def test_comparator_violations_fail_hsafe(self):
        starts = _starts()
        starts["D1"]["HUG7"]["pursuer_violations"] = 1
        h = R.hsafe_verdict(_full(), starts)
        assert h["verdict"] == "FAILS"
        assert h["pursuer_violation_steps"] == 1
        assert h["offenders"] == ["comparator:D1/HUG7"]

    def test_comparator_without_the_key_is_unverified(self):
        starts = _starts()
        del starts["D2"]["GREEDY"]["pursuer_violations"]
        h = R.hsafe_verdict(_full(), starts)
        assert h["verdict"] == "HOLDS"
        assert h["comparators_unverified"] == ["comparator:D2/GREEDY"]


class TestSummary:
    def test_summarize_runs_on_the_fixture(self):
        s = R.summarize(_full(), _starts())
        assert s["complete"] is True
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert s["comparators"]["D1"]["HUG7"]["pursuer_violations"] == 0


class TestDirection:
    def test_rows(self):
        assert "weaponises" in R.direction(
            {"hsafe": "HOLDS", "H2_overall": "PASS", "H2_implicit_any": True}
        )
        assert "implicit" in R.direction(
            {"hsafe": "HOLDS", "H2_overall": "FAIL", "H2_implicit_any": True}
        )
        assert "no exploitation" in R.direction(
            {"hsafe": "HOLDS", "H2_overall": "FAIL", "H2_implicit_any": False}
        )
        assert "Stop" in R.direction(
            {"hsafe": "FAILS", "H2_overall": "PASS", "H2_implicit_any": True}
        )
        assert "incomplete" in R.direction(
            {
                "hsafe": "INCOMPLETE",
                "H2_overall": "INCOMPLETE",
                "H2_implicit_any": False,
            }
        )
