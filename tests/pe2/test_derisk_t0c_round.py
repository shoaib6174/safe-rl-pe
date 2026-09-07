"""scripts/derisk_t0c_round.py — the final round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t0c_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t0c_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, verdict, final=(120, 500), rows=None):
    return {
        "arm": name,
        "verdict": verdict,
        "reason": "test",
        "timesteps": 1_000_000,
        "checkpoints": rows
        or [
            {"steps": s, "win_rate": 0.2, "k": 40, "n": 200}
            for s in range(100_000, 1_000_001, 100_000)
        ],
        "final": {
            "k": final[0],
            "n": final[1],
            "win_rate": final[0] / final[1],
            "wilson": list(R.wilson_ci(final[0], final[1])),
        },
        "start": {"k": 37, "n": 500, "rate": 0.185},
    }


class TestPreRegisteredConstants:
    def test_protocol(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.FINAL_EVAL_N, R.PASS_FROM) == (
            1_000_000,
            100_000,
            200,
            500,
            500_000,
        )
        assert (R.NEG_TOTAL, R.BETA_RE, R.CRITIC_WARMUP) == (500_000, 1.0, 50_000)

    def test_fresh_seed_streams(self):
        assert (R.EVAL_SEED0, R.RF_SEED0, R.RFNEG_SEED0) == (840_000, 850_000, 860_000)
        assert R.EVAL_SEED0 not in (700_000, 780_000, 800_000, 810_000)

    def test_seeds_and_learners(self):
        assert R.SEEDS == {"RE": 7, "RF": 13, "RFNEG": 13}
        assert R.LEARNER_ARMS == ("RE", "RF")

    def test_pass_window_is_bound_to_the_library_constant(self):
        from pe2.derisk import ROUND_PASS_FROM

        assert R.PASS_FROM == ROUND_PASS_FROM


class TestStartBar:
    def test_arms_refuse_to_run_without_a_measured_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        with pytest.raises(SystemExit, match="start bar"):
            R.load_start()

    def test_load_start_reads_what_measure_start_wrote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        payload = {
            "k": 96,
            "n": 500,
            "rate": 0.192,
            "wilson": [0.16, 0.23],
            "eval_seed0": 840_000,
            "clone": "x.zip",
        }
        (tmp_path / "t0c_start.json").write_text(json.dumps(payload))
        assert R.load_start() == payload


class TestAnchorPattern:
    """Spec §3.1 fixes the reading of each outcome pattern in advance."""

    def test_improved(self):
        assert R.anchor_pattern("IMPROVED", True)[0] == "improved"
        assert R.anchor_pattern("IMPROVED", False)[0] == "improved"

    def test_pinned(self):
        pattern, reading = R.anchor_pattern("PRESERVED", False)
        assert pattern == "pinned"
        assert "cannot learn" in reading

    def test_late_collapse(self):
        pattern, reading = R.anchor_pattern("COLLAPSED", True)
        assert pattern == "late_collapse"
        assert "continuum" in reading

    def test_anything_else_is_named_not_forced(self):
        pattern, reading = R.anchor_pattern("COLLAPSED", False)
        assert pattern == "other" and reading


class TestSpecCheck:
    """Spec §3.2: the temperature's predicted diagnostics are falsifiable."""

    def _its(self, ratio, lam=0.15, n=5):
        return [
            {
                "lam_used": lam,
                "weight_ratio_1sigma": ratio,
                "weight_ratio_2sigma": ratio**2,
            }
            for _ in range(n)
        ]

    def test_matching_iterations_pass(self):
        import math

        assert R.spec_check(self._its(math.exp(2.0)))["pass"] is True

    def test_a_deviating_iteration_fails(self):
        out = R.spec_check(self._its(2.10))  # T0.4's inert value
        assert out["pass"] is False and "2.1" in out["reason"]

    def test_floored_iterations_are_excluded_not_failed(self):
        import math

        its = self._its(math.exp(2.0)) + [
            {
                "lam_used": R.LAM_FLOOR,
                "weight_ratio_1sigma": 1.02,
                "weight_ratio_2sigma": 1.04,
            }
        ]
        out = R.spec_check(its)
        assert out["pass"] is True and out["n_floored"] == 1


class TestSummarize:
    def test_survivors_are_improved_only(self):
        s = R.summarize(
            {
                "RE": _arm("RE", "IMPROVED"),
                "RF": _arm("RF", "PRESERVED"),
                "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["survivors"] == ["RE"] and s["complete"] is True

    def test_rf_withheld_when_its_control_did_not_collapse(self):
        s = R.summarize(
            {
                "RE": _arm("RE", "PRESERVED"),
                "RF": _arm("RF", "IMPROVED"),
                "RFNEG": {"arm": "RFNEG", "verdict": None, "reason": "not separated"},
            }
        )
        assert s["arms"]["RF"]["verdict"] == "UNDETERMINED"
        assert s["survivors"] == []
        assert "insensitive" in s["control_finding"].lower()

    def test_route_c_is_declared_final(self):
        s = R.summarize(
            {
                "RE": _arm("RE", "COLLAPSED"),
                "RF": _arm("RF", "COLLAPSED"),
                "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["survivors"] == []
        assert "final" in s["direction"].lower()
        assert "no further learner pre-registration" in s["direction"].lower()

    def test_a_crashed_learner_does_not_reach_route_c(self):
        s = R.summarize(
            {
                "RE": _arm("RE", "UNDETERMINED"),
                "RF": _arm("RF", "COLLAPSED"),
                "RFNEG": {"arm": "RFNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["complete"] is False and "incomplete" in s["direction"].lower()

    def test_re_does_not_depend_on_the_awr_control(self):
        s = R.summarize({"RE": _arm("RE", "IMPROVED")})
        assert s["arms"]["RE"]["verdict"] == "IMPROVED" and s["survivors"] == ["RE"]


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"RE": _arm("RE", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestRunArmWiring:
    """Pins the USE of the constants, not merely their values."""

    def test_per_arm_call_arguments(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        calls = {}
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t0c_start.json")
        (tmp_path / "t0c_start.json").write_text(
            json.dumps(
                {
                    "k": 96,
                    "n": 500,
                    "rate": 0.192,
                    "wilson": [0.16, 0.23],
                    "eval_seed0": 840_000,
                    "clone": "x.zip",
                }
            )
        )

        def fake_anchored(cfg, role, opp, path, total, **kw):
            calls["RE"] = {"path": str(path), "total": total, **kw}

        def fake_awr(cfg, role, opp, path, total, **kw):
            arm = "RFNEG" if kw["awr"].advantage_sign < 0 else "RF"
            calls[arm] = {
                "path": str(path),
                "total": total,
                "seed": kw["seed"],
                "rollout_seed0": kw["rollout_seed0"],
                "adaptive_lam": kw["awr"].adaptive_lam,
                "sign": kw["awr"].advantage_sign,
            }
            return None, {
                "total_steps": total,
                "n_episodes": 1,
                "positives": 0,
                "positives_per_1M": 0.0,
                "iterations": [],
                "first_action": [0.0, 0.0],
                "probe_marks": [],
                "config": {},
            }

        import pe2.awr as awr_mod
        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br_anchored", fake_anchored)
        monkeypatch.setattr(awr_mod, "train_awr", fake_awr)
        monkeypatch.setattr(
            R,
            "final_eval",
            lambda *a, **k: {
                "k": 100,
                "n": 500,
                "win_rate": 0.2,
                "wilson": [0.17, 0.24],
                "ckpt": "x",
            },
        )

        from pe2.config import EnvConfig

        for arm in ("RE", "RF", "RFNEG"):
            R.run_arm(arm, EnvConfig(), "cpu")

        assert calls["RE"]["beta"] == 1.0 and calls["RE"]["seed"] == 7
        assert calls["RE"]["total"] == 1_000_000
        assert calls["RE"]["critic_warmup"] == 50_000
        assert calls["RF"]["adaptive_lam"] is True and calls["RF"]["sign"] == 1.0
        assert calls["RF"]["rollout_seed0"] == 850_000
        assert calls["RF"]["total"] == 1_000_000
        assert calls["RFNEG"]["sign"] == -1.0
        assert calls["RFNEG"]["rollout_seed0"] == 860_000
        assert calls["RFNEG"]["total"] == 500_000
        assert calls["RF"]["seed"] == calls["RFNEG"]["seed"] == 13
