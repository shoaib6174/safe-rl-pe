"""scripts/derisk_s7_remedy2.py — the second remedy round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s7_remedy2.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s7_remedy2", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _ind(rate, n=500):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _gate(deadlock, capture):
    return {
        "deadlock_ind": _ind(deadlock),
        "capture_ind": _ind(capture),
        "pursuer_violations": 0,
        "pursuer": {
            "deadlock_episode_rate": deadlock,
            "stall_step_rate": 0.3,
            "intervention_rate": 0.4,
        },
        "capture_rate": capture,
    }


def _starts(att=(0.60, 0.58, 0.61), greedy=0.56, hug=0.40):
    return {
        "RC19": {
            "ATT25": _gate(att[0], 0.15),
            "ATT26": _gate(att[1], 0.15),
            "ATT27": _gate(att[2], 0.15),
            "GREEDY": _gate(0.30, greedy),
            "HUG7": _gate(0.34, hug),
        }
    }


def _arm(name, att, greedy_capture=0.56, viol=0):
    return {
        "arm": name,
        "seed": int(name[3:]),
        "train_violations": viol,
        "probe_violations": 0,
        "final": {
            "pursuer_violations": 0,
            "gate": {
                "ATT25": _gate(att[0], 0.2),
                "ATT26": _gate(att[1], 0.2),
                "ATT27": _gate(att[2], 0.2),
                "GREEDY": _gate(0.28, greedy_capture),
                "HUG7": _gate(0.30, 0.42),
            },
        },
        "checkpoints": [],
    }


def _full(
    seeds=((0.40, 0.42, 0.45), (0.38, 0.41, 0.44), (0.62, 0.60, 0.59)),
    greedy=(0.56, 0.55, 0.57),
):
    return {
        f"RCM{s}": _arm(f"RCM{s}", a, g)
        for s, a, g in zip(R.SEEDS, seeds, greedy, strict=True)
    }


class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (28, 29, 30) and (R.PROBE_SEED0, R.GATE_SEED0) == (
            1_160_000,
            1_170_000,
        )
        assert (
            R.GATE_N,
            R.EVAL_N,
            R.TOTAL_LONG,
            R.EVERY_LONG,
            R.PASS_MIN,
            R.P_GATE,
        ) == (500, 200, 10_000_000, 1_000_000, 2, 0.05)
        assert R.RC_SETTINGS == {
            "shield_obs": True,
            "stall_penalty": 1e-3,
            "r_acq": 0.0,
        }
        assert R.SEEDS2 == (31, 32, 33) and (
            R.PROBE2_SEED0,
            R.GATE2_SEED0,
            R.RACE_BAR,
        ) == (1_180_000, 1_190_000, 0.58)
        assert set(R.ARMS) == {"RCM28", "RCM29", "RCM30"}
        assert R.HELD_OUT == ("ATT25", "ATT26", "ATT27") and R.GATE_OPPONENTS == (
            "ATT25",
            "ATT26",
            "ATT27",
            "GREEDY",
            "HUG7",
        )
        assert [m for m in R.MIXTURE] == [
            "GREEDY",
            "HUG7",
            "SPLITHUG",
            "CTL-D125",
            "CTL-D126",
            "CTL-D127",
        ]

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s7_start.json")
        monkeypatch.setattr(R, "PREFLIGHT_PATH", tmp_path / "s7_preflight.json")
        with pytest.raises(SystemExit, match="preflight"):
            R.require_preflight()
        (tmp_path / "s7_preflight.json").write_text('{"pass": true}')
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s7_arm_RCM28.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("RCM28")

    def test_stage2_requires_the_stage1_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="run --summarize first"):
            R._artefact_ckpt()


class TestVerdicts:
    def test_h32_pass_two_of_three_seeds(self):
        h = R.h32_verdicts(_full(), _starts())
        assert (
            h["per_seed"][28]["verdict"] == "PASS"
            and h["per_seed"][29]["verdict"] == "PASS"
            and h["per_seed"][30]["verdict"] == "FAIL"
        )
        assert h["H32"] == "PASS" and h["artefact"] in ("RCM28", "RCM29")

    def test_artefact_is_lowest_mean_deadlock_among_passing(self):
        h = R.h32_verdicts(_full(), _starts())
        assert h["artefact"] == "RCM29"  # mean 0.41 < 0.423

    def test_capture_guard_label(self):
        h = R.h32_verdicts(
            _full(greedy=(0.40, 0.55, 0.57)), _starts()
        )  # seed 28 captures far less than RC19's 0.56
        assert h["per_seed"][28]["verdict"] == "PASS (robust but weaker)"

    def test_fail_and_incomplete(self):
        h = R.h32_verdicts(_full(seeds=((0.60, 0.58, 0.61),) * 3), _starts())
        assert h["H32"] == "FAIL" and h["artefact"] is None
        res = _full()
        del res["RCM29"]
        assert R.h32_verdicts(res, _starts())["H32"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe(self):
        assert R.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        res = _full()
        res["RCM28"]["train_violations"] = 1
        assert R.hsafe_verdict(res, _starts())["verdict"] == "FAILS"
        st = _starts()
        st["RC19"]["HUG7"]["pursuer_violations"] = 2
        assert R.hsafe_verdict(_full(), st)["verdict"] == "FAILS"
        res = _full()
        del res["RCM30"]
        assert R.hsafe_verdict(res, _starts())["verdict"] == "INCOMPLETE"

    def test_hsafe_folds_in_stage_2(self):
        assert R.hsafe_verdict(_full(), _starts(), None)["verdict"] == "HOLDS"
        race = {"pursuer_violation_steps": 1, "offenders": ["ATK2-31"]}
        h = R.hsafe_verdict(_full(), _starts(), race)
        assert h["verdict"] == "FAILS" and "race:ATK2-31" in h["offenders"]
        clean = {"pursuer_violation_steps": 0, "offenders": []}
        assert R.hsafe_verdict(_full(), _starts(), clean)["verdict"] == "HOLDS"

    def test_hsafe_counts_a_crash_records_partial_gate(self):
        res = _full()
        del res["RCM30"]["final"]  # crashed inside the gate
        res["RCM30"]["gate_violations"] = 1
        assert R.hsafe_verdict(res, _starts())["verdict"] == "FAILS"


def _race_arm(viol_train=0, viol_probe=0, viol_gate=0, rate=0.45):
    return {
        "train_pursuer_violations": viol_train,
        "probe_pursuer_violations": viol_probe,
        "final": {
            "deadlock_ind": _ind(rate),
            "pursuer": {"deadlock_episode_rate": rate, "escape_rate": 0.7},
            "pursuer_violations": viol_gate,
        },
    }


class TestRace:
    def test_race_summary(self):
        arms = {
            f"ATK2-{s}": {
                "final": {
                    "deadlock_ind": _ind(r),
                    "pursuer": {"deadlock_episode_rate": r, "escape_rate": 0.7},
                    "pursuer_violations": 0,
                }
            }
            for s, r in zip(R.SEEDS2, (0.45, 0.50, 0.48), strict=True)
        }
        r = R.race_summary(arms)
        assert r["pooled_rate"] == pytest.approx(
            (0.45 + 0.50 + 0.48) / 3, abs=0.01
        ) and r["H_RACE"] in ("HOLDS", "FAILS")
        assert r["pooled_wilson"][1] < R.RACE_BAR and r["H_RACE"] == "HOLDS"
        arms2 = {
            f"ATK2-{s}": {
                "final": {
                    "deadlock_ind": _ind(0.62),
                    "pursuer": {"deadlock_episode_rate": 0.62, "escape_rate": 0.9},
                    "pursuer_violations": 0,
                }
            }
            for s in R.SEEDS2
        }
        assert R.race_summary(arms2)["H_RACE"] == "FAILS"

    def test_race_summary_counts_stage2_violations(self):
        clean = {f"ATK2-{s}": _race_arm() for s in R.SEEDS2}
        r = R.race_summary(clean)
        assert r["pursuer_violation_steps"] == 0 and r["offenders"] == []
        dirty = dict(clean, **{"ATK2-32": _race_arm(viol_train=1, viol_gate=2)})
        r = R.race_summary(dirty)
        assert r["pursuer_violation_steps"] == 3 and r["offenders"] == ["ATK2-32"]

    def test_race_summary_counts_the_stage2_comparators(self):
        clean = {f"ATK2-{s}": _race_arm() for s in R.SEEDS2}
        assert R.race_summary(clean, None)["pursuer_violation_steps"] == 0
        rs = {
            "D3": {
                "HUG7": {"pursuer_violations": 2},
                "GREEDY": {"pursuer_violations": 0},
            }
        }
        r = R.race_summary(clean, rs)
        assert r["pursuer_violation_steps"] == 2
        assert r["offenders"] == ["race_start:D3/HUG7"]


class TestGateHolder:
    """A crash part-way through the gate must not lose what the gate already saw."""

    def _fakes(self, monkeypatch, die_on):
        import pe2.structured

        calls = []

        def fake_eval(model, cfg, factory, n, seed0, **kw):
            calls.append(factory)
            if len(calls) == die_on:
                raise RuntimeError("gate died")
            return [
                {"outcome": "capture", "shield": {"deadlock": False}},
                {"outcome": "timeout", "shield": {"deadlock": True}},
            ]

        monkeypatch.setattr(pe2.structured, "evaluate_structured", fake_eval)
        monkeypatch.setattr(R, "win_count", lambda logs, role: (1, 2))
        monkeypatch.setattr(
            R,
            "summarize_shield",
            lambda logs: (
                dict.fromkeys(
                    (
                        "deadlock_episode_rate",
                        "stall_step_rate",
                        "intervention_rate",
                        "infeasibility_rate",
                        "backup_rate",
                        "min_clearance_floor",
                    ),
                    0.5,
                )
                | {"violation_steps": 3}
            ),
        )

    def test_holder_keeps_the_opponents_finished_before_the_crash(self, monkeypatch):
        self._fakes(monkeypatch, die_on=2)
        holder: dict = {}
        with pytest.raises(RuntimeError):
            R.gate_pursuer(None, {"ATT25": object(), "ATT26": object()}, "cpu", holder)
        assert list(holder["gate"]) == ["ATT25"]
        # exactly the expression run_arm's crash record uses
        assert sum(v["pursuer_violations"] for v in holder["gate"].values()) == 3

    def test_holder_is_optional(self, monkeypatch):
        self._fakes(monkeypatch, die_on=0)
        out = R.gate_pursuer(None, {"ATT25": object()}, "cpu")
        assert out["ATT25"]["pursuer_violations"] == 3


class TestDirection:
    def test_rows(self):
        assert "converges" in R.direction(
            {"hsafe": "HOLDS", "H32": "PASS", "H_RACE": "HOLDS"}
        )
        assert "open arms race" in R.direction(
            {"hsafe": "HOLDS", "H32": "PASS", "H_RACE": "FAILS"}
        )
        assert "does not buy" in R.direction(
            {"hsafe": "HOLDS", "H32": "FAIL", "H_RACE": None}
        )
        assert "Stop" in R.direction({"hsafe": "FAILS", "H32": "PASS", "H_RACE": None})
        assert "incomplete" in R.direction(
            {"hsafe": "INCOMPLETE", "H32": "INCOMPLETE", "H_RACE": None}
        )
