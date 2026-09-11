"""scripts/derisk_t1_round.py — the horizon round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t1_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t1_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()

_BARS = {
    "START_400": {
        "k": 90,
        "n": 500,
        "rate": 0.18,
        "wilson": [0.15, 0.22],
        "eval_seed0": 870_000,
        "max_steps": 400,
    },
    "START_1200": {
        "k": 97,
        "n": 500,
        "rate": 0.194,
        "wilson": [0.16, 0.23],
        "eval_seed0": 880_000,
        "max_steps": 1200,
    },
}


def _arm(name, verdict, final=(120, 500), bar=0.179, steps=400, bar_name="START_400"):
    return {
        "arm": name,
        "verdict": verdict,
        "reason": "test",
        "timesteps": 1_000_000,
        "max_steps": steps,
        "gamma": 0.999,
        "checkpoints": [
            {"steps": s, "win_rate": 0.2, "k": 40, "n": 200}
            for s in range(100_000, 1_000_001, 100_000)
        ],
        "final": {
            "k": final[0],
            "n": final[1],
            "win_rate": final[0] / final[1],
            "wilson": list(R.wilson_ci(final[0], final[1])),
        },
        "start": {"k": 90, "n": 500, "rate": bar, "bar": bar_name},
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
        assert R.CRITIC_WARMUP == 50_000 and R.SEED == 7

    def test_horizons_and_discounts(self):
        assert (R.SHORT_STEPS, R.LONG_STEPS) == (400, 1200)
        assert (R.GAMMA_ORIG, R.GAMMA_MATCHED) == (0.999, 0.9975)
        # The matched discount sets the effective horizon to the episode length.
        assert 1.0 / (1.0 - R.GAMMA_MATCHED) == pytest.approx(R.SHORT_STEPS, rel=1e-6)

    def test_fresh_seed_streams(self):
        assert (R.SHORT_SEED0, R.LONG_SEED0) == (870_000, 880_000)
        assert R.SHORT_SEED0 > 860_999 and R.LONG_SEED0 > 860_999

    def test_pass_window_is_bound_to_the_library_constant(self):
        from pe2.derisk import ROUND_PASS_FROM

        assert R.PASS_FROM == ROUND_PASS_FROM


class TestArmConfig:
    """Spec §3: every arm shares the learner and differs only in the task."""

    def test_ha_is_the_short_horizon_at_the_original_discount(self):
        cfg, gamma, seed0, bar = R.arm_config("HA")
        assert (cfg.max_steps, gamma, seed0, bar) == (400, 0.999, 870_000, "START_400")

    def test_hc_is_the_matched_control_at_the_original_length(self):
        cfg, gamma, seed0, bar = R.arm_config("HC")
        assert (cfg.max_steps, gamma, seed0, bar) == (
            1200,
            0.999,
            880_000,
            "START_1200",
        )

    def test_hb_adds_the_matched_discount(self):
        cfg, gamma, seed0, bar = R.arm_config("HB")
        assert (cfg.max_steps, gamma, seed0, bar) == (400, 0.9975, 870_000, "START_400")

    def test_the_control_matches_ha_exactly(self):
        ha = R.arm_config("HA")
        neg = R.arm_config("HNEG")
        assert (neg[0].max_steps, neg[1], neg[2], neg[3]) == (
            ha[0].max_steps,
            ha[1],
            ha[2],
            ha[3],
        )

    def test_only_the_horizon_differs_between_ha_and_hc(self):
        import dataclasses

        a = dataclasses.asdict(R.arm_config("HA")[0])
        c = dataclasses.asdict(R.arm_config("HC")[0])
        assert {k for k in a if a[k] != c[k]} == {"max_steps"}


class TestStartBars:
    def test_arms_refuse_to_run_without_measured_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        with pytest.raises(SystemExit, match="start bar"):
            R.load_starts()

    def test_load_starts_reads_both_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(json.dumps(_BARS))
        assert R.load_starts() == _BARS

    def test_a_missing_single_bar_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(
            json.dumps({"START_400": _BARS["START_400"]})
        )
        with pytest.raises(SystemExit, match="START_1200"):
            R.load_starts()


class TestSummarize:
    def _ctrl(self, verdict):
        return {"arm": "HNEG", "verdict": verdict, "reason": "test"}

    def _full(self, ha, hb, hc, ctrl=True):
        return {
            "HA": _arm("HA", ha),
            "HB": _arm("HB", hb),
            "HC": _arm("HC", hc, bar=0.194, steps=1200, bar_name="START_1200"),
            "HNEG": self._ctrl(ctrl),
        }

    def test_the_horizon_finding_names_both_arms(self):
        s = R.summarize(self._full("IMPROVED", "PRESERVED", "COLLAPSED"))
        assert s["survivors"] == ["HA"]
        assert "horizon was the binding constraint" in s["direction"]

    def test_both_horizons_improving_is_reported_as_a_coverage_gap(self):
        s = R.summarize(self._full("IMPROVED", "PRESERVED", "IMPROVED"))
        assert "not what changed" in s["direction"] and "coverage" in s["direction"]

    def test_the_discount_case_is_named(self):
        s = R.summarize(self._full("COLLAPSED", "IMPROVED", "COLLAPSED"))
        assert "matched discount is doing the work" in s["direction"]

    def test_zero_improved_leaves_route_c_standing(self):
        s = R.summarize(self._full("COLLAPSED", "PRESERVED", "COLLAPSED"))
        assert s["survivors"] == []
        assert "Route C stands" in s["direction"]

    def test_every_arm_is_withheld_when_the_control_does_not_collapse(self):
        s = R.summarize(self._full("IMPROVED", "IMPROVED", "COLLAPSED", ctrl=False))
        assert s["survivors"] == []
        assert all(
            s["arms"][a]["verdict"] == "UNDETERMINED" for a in ("HA", "HB", "HC")
        )
        assert "unvalidated" in s["arms"]["HA"]["reason"]
        assert s["control_withheld_everything"] is True

    def test_a_missing_control_withholds_everything_too(self):
        s = R.summarize({"HA": _arm("HA", "IMPROVED")})
        assert s["arms"]["HA"]["verdict"] == "UNDETERMINED"
        assert s["survivors"] == [] and s["complete"] is False

    def test_a_crashed_learner_does_not_reach_a_direction(self):
        s = R.summarize(self._full("UNDETERMINED", "COLLAPSED", "COLLAPSED"))
        assert s["complete"] is False and "incomplete" in s["direction"].lower()


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"HA": _arm("HA", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestRunArmWiring:
    """Pins that every arm uses the SAME learner and differs only in the task."""

    def _prepare(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "t1_start.json")
        (tmp_path / "t1_start.json").write_text(json.dumps(_BARS))
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

    def test_per_arm_call_arguments(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        self._prepare(tmp_path, monkeypatch)
        calls = {}
        current = {}

        def fake_finetune(cfg, role, opp, path, total, **kw):
            calls[current["arm"]] = {
                "max_steps": cfg.max_steps,
                "gamma": kw["gamma"],
                "seed": kw["seed"],
                "critic_warmup": kw["critic_warmup"],
                "total": total,
                "env_wrap": kw.get("env_wrap"),
            }

        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br", fake_finetune)
        for arm in ("HA", "HB", "HC", "HNEG"):
            current["arm"] = arm
            R.run_arm(arm, "cpu")

        assert all(c["seed"] == 7 for c in calls.values())
        assert all(c["critic_warmup"] == 50_000 for c in calls.values())
        assert all(c["total"] == 1_000_000 for c in calls.values())
        assert (calls["HA"]["max_steps"], calls["HA"]["gamma"]) == (400, 0.999)
        assert (calls["HC"]["max_steps"], calls["HC"]["gamma"]) == (1200, 0.999)
        assert (calls["HB"]["max_steps"], calls["HB"]["gamma"]) == (400, 0.9975)
        assert (calls["HNEG"]["max_steps"], calls["HNEG"]["gamma"]) == (400, 0.999)
        # Only the control wraps the training env.
        assert calls["HNEG"]["env_wrap"] is not None
        assert all(calls[a]["env_wrap"] is None for a in ("HA", "HB", "HC"))

    def test_each_arm_is_judged_against_its_own_horizon_bar(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("stable_baselines3")
        self._prepare(tmp_path, monkeypatch)
        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br", lambda *a, **k: None)
        short = R.run_arm("HA", "cpu")
        long_ = R.run_arm("HC", "cpu")
        assert short["start"]["bar"] == "START_400" and short["start"]["rate"] == 0.18
        assert long_["start"]["bar"] == "START_1200" and long_["start"]["rate"] == 0.194

    def test_the_control_wrap_negates_the_reward(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        self._prepare(tmp_path, monkeypatch)
        captured = {}

        def fake_finetune(cfg, role, opp, path, total, **kw):
            captured["wrap"] = kw.get("env_wrap")

        import pe2.exploitability as expl

        monkeypatch.setattr(expl, "finetune_br", fake_finetune)
        R.run_arm("HNEG", "cpu")

        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.rewards import NegatedRewardWrapper
        from pe2.scripted import GreedyEvader

        wrapped = captured["wrap"](
            SingleAgentWrapper(EnvConfig(max_steps=40), "pursuer", GreedyEvader)
        )
        assert isinstance(wrapped, NegatedRewardWrapper)
