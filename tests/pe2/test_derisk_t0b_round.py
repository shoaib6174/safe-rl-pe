"""scripts/derisk_t0b_round.py — the retry round's own decisions."""

import importlib.util
import json
import pathlib

import pytest

from pe2.eval import wilson_ci

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t0b_round.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t0b_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, verdict, rows=None, final=(120, 500)):
    return {
        "arm": name,
        "verdict": verdict,
        "reason": "test",
        "checkpoints": rows
        or [
            {"steps": s, "win_rate": 0.2, "k": 40, "n": 200}
            for s in range(100_000, 1_000_001, 100_000)
        ],
        "final": {
            "k": final[0],
            "n": final[1],
            "win_rate": final[0] / final[1],
            "wilson": list(wilson_ci(final[0], final[1])),
        },
        "start": {"k": 37, "n": 200, "rate": 0.185},
        # The fixture is wrong, not the code: write_figure reads both
        # r["timesteps"] and r["final"]["wilson"] for any arm carrying a
        # "final" block (see write_figure's errorbar), and the brief's
        # helper omitted both, which raises KeyError in TestFigure. Real
        # "final" blocks (final_eval's return shape) always carry both.
        "timesteps": 1_000_000,
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
        assert (R.BETA_RB, R.CRITIC_WARMUP, R.NEG_TOTAL) == (0.1, 50_000, 200_000)
        assert (R.EVAL_SEED0, R.AWR_SEED0, R.AWRNEG_SEED0) == (
            780_000,
            820_000,
            830_000,
        )

    def test_seeds_and_starts(self):
        assert R.SEEDS == {"RB": 7, "RC": 9, "RD": 11, "RDNEG": 11}
        assert R.STARTS == {
            "RB": (37, 200),
            "RC": (41, 200),
            "RD": (37, 200),
            "RDNEG": (37, 200),
        }

    def test_pass_from_is_bound_to_the_pre_registered_constant(self):
        # R.PASS_FROM = 500_000 is written a second time here, next to
        # ROUND_PASS_FROM in pe2.derisk. An unbound duplicate of a
        # pre-registered constant is exactly what pre-registration
        # discipline exists to prevent, so pin that they cannot drift apart.
        import pe2.derisk

        assert R.PASS_FROM == pe2.derisk.ROUND_PASS_FROM


class TestSummarize:
    def test_survivors_are_improved_only(self):
        s = R.summarize(
            {
                "RB": _arm("RB", "PRESERVED"),
                "RC": _arm("RC", "IMPROVED"),
                "RD": _arm("RD", "IMPROVED"),
                "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["survivors"] == ["RC", "RD"] and s["complete"] is True

    def test_rd_is_withheld_when_its_control_did_not_collapse(self):
        s = R.summarize(
            {
                "RD": _arm("RD", "IMPROVED"),
                "RDNEG": {"arm": "RDNEG", "verdict": None, "reason": "not separated"},
            }
        )
        assert s["arms"]["RD"]["verdict"] == "UNDETERMINED"
        assert "control" in s["arms"]["RD"]["reason"]
        assert s["survivors"] == []

    def test_rd_is_withheld_when_its_control_is_missing(self):
        s = R.summarize({"RD": _arm("RD", "IMPROVED")})
        assert s["arms"]["RD"]["verdict"] == "UNDETERMINED" and s["complete"] is False

    def test_rb_rc_do_not_depend_on_rdneg(self):
        s = R.summarize({"RB": _arm("RB", "IMPROVED")})
        assert s["arms"]["RB"]["verdict"] == "IMPROVED" and s["survivors"] == ["RB"]

    def test_direction_text_follows_c4(self):
        assert (
            "Route C"
            in R.summarize(
                {
                    "RB": _arm("RB", "PRESERVED"),
                    "RC": _arm("RC", "COLLAPSED"),
                    "RD": _arm("RD", "PRESERVED"),
                    "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": ""},
                }
            )["direction"]
        )
        assert "Track 2" in R.summarize({"RB": _arm("RB", "IMPROVED")})["direction"]


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"RB": _arm("RB", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0


class TestArmJsonRoundTrip:
    def test_load_results_reads_only_arm_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "t0b_arm_RB.json").write_text(json.dumps(_arm("RB", "PRESERVED")))
        (tmp_path / "t0_pilots.json").write_text("{}")
        assert set(R.load_results()) == {"RB"}


class TestCrashedArmKeepsItsProbeHistory:
    """A crashed arm must not get a fabricated final probe (ruling 3). If the
    saved final checkpoint isn't there, run_arm must not raise, must not call
    final_eval, must record UNDETERMINED with the missing path named in the
    reason, and must still write t0b_arm_{ARM}.json with the probe rows and
    training stats that WERE collected before the crash."""

    def test_missing_final_checkpoint_is_undetermined_not_a_crash(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("stable_baselines3")
        import pe2.exploitability as expl
        from pe2.config import EnvConfig

        mod = _runner()
        monkeypatch.setattr(mod, "OUT", tmp_path)

        def _dead_finetune_ppo(
            cfg,
            role,
            opponent_factory,
            model_path,
            total_timesteps,
            seed,
            device="auto",
            callback=None,
        ):
            # Simulates a run that died before writing anything: no probe
            # rows appended (the callback never fires), no checkpoint saved.
            return None

        monkeypatch.setattr(expl, "finetune_ppo", _dead_finetune_ppo)

        result = mod.run_arm("RC", EnvConfig(), "cpu")

        assert result["verdict"] == "UNDETERMINED"
        expected_path = tmp_path / "ckpts_RC" / "RC_1000000.zip"
        assert str(expected_path) in result["reason"]
        assert "final" not in result
        # The probe history and training stats collected before the crash
        # (empty here, but present as keys) must survive onto disk.
        assert "checkpoints" in result and "training" in result

        saved = json.loads((tmp_path / "t0b_arm_RC.json").read_text())
        assert saved["verdict"] == "UNDETERMINED"
        assert saved["checkpoints"] == result["checkpoints"]
        assert "training" in saved
