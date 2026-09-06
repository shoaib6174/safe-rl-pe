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
        # NEG_TOTAL = 500_000: amendment C3-a. train_awr's offline phase
        # collects initial_episodes=300 unconditionally, and the clone's
        # measured mean episode length is 1020.6 steps
        # (results/derisk/t0_signal_audit_pursuer.json: 1,000,161 / 980), so
        # the offline phase alone costs ~306k steps -- at the old 200_000
        # the online loop would never run and the control would score two
        # probes of the same post-offline model.
        assert (R.BETA_RB, R.CRITIC_WARMUP, R.NEG_TOTAL) == (0.1, 50_000, 500_000)
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

    def test_rd_is_withheld_when_its_control_held_the_bound(self):
        # `control_collapsed`'s False branch: the control PASSED (held at or
        # above threshold at every probe) rather than being ambiguous
        # (None). Distinct code path, same `ctrl["verdict"] is not True`
        # withhold condition -- only None was covered before this test.
        s = R.summarize(
            {
                "RD": _arm("RD", "IMPROVED"),
                "RDNEG": {
                    "arm": "RDNEG",
                    "verdict": False,
                    "reason": "held at or above threshold at every probe: did NOT collapse",
                },
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


class TestSurvivorTieBreak:
    def test_best_is_the_higher_wilson_lower_bound_not_list_order(self):
        # `test_survivors_are_improved_only` gives both survivors identical
        # final counts, so `max` would return the first candidate regardless
        # of the key -- a sign flip on the comparison would still pass. Give
        # RD (which comes AFTER RC in LEARNER_ARMS traversal order) the
        # clearly higher final win rate, so picking "best" by order instead
        # of by Wilson lower bound is caught rather than accidentally right.
        s = R.summarize(
            {
                "RC": _arm("RC", "IMPROVED", final=(120, 500)),  # 0.24
                "RD": _arm("RD", "IMPROVED", final=(300, 500)),  # 0.60
                "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert "Track-1 mechanism = RD" in s["direction"]


class TestCompleteRequiresJudgedLearners:
    """THE CRITICAL DEFECT the round-4 review pinned: `complete` used to test
    only file presence. A crashed RB (own UNDETERMINED verdict, no final
    checkpoint) with RC/RD PRESERVED and RDNEG collapsed used to read
    complete=True and emit the round-ending "-> Route C" direction off an
    arm that was never judged. RB is the 5-6 hour arm and the likeliest to
    die mid-run."""

    def test_a_crashed_learner_does_not_reach_route_c(self):
        s = R.summarize(
            {
                "RB": {
                    "arm": "RB",
                    "verdict": "UNDETERMINED",
                    "reason": "final checkpoint missing: results/derisk/ckpts_RB/RB_1000000.zip was never written",
                },
                "RC": _arm("RC", "PRESERVED"),
                "RD": _arm("RD", "PRESERVED"),
                "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["complete"] is False
        assert "Route C" not in s["direction"]

    def test_a_crashed_rd_with_a_collapsed_control_is_not_a_legitimate_withhold(self):
        # RD crashed on its OWN (own verdict UNDETERMINED) while RDNEG DID
        # collapse (ctrl["verdict"] is True), so the withhold branch never
        # fires (rd_withheld stays False) -- this must NOT be read as "the
        # round is decided on RB/RC alone", it is a genuine gap.
        s = R.summarize(
            {
                "RB": _arm("RB", "PRESERVED"),
                "RC": _arm("RC", "PRESERVED"),
                "RD": {
                    "arm": "RD",
                    "verdict": "UNDETERMINED",
                    "reason": "final checkpoint missing: results/derisk/ckpts_RD/RD_1000000.zip was never written",
                },
                "RDNEG": {"arm": "RDNEG", "verdict": True, "reason": "collapsed"},
            }
        )
        assert s["complete"] is False
        assert "Route C" not in s["direction"]


class TestFigure:
    def test_writes_with_partial_arms(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = {"RB": _arm("RB", "PRESERVED")}
        R.write_figure(results, R.summarize(results), tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0

    def test_the_controls_boolean_verdict_gets_a_readable_label(self, tmp_path):
        # RDNEG's verdict is True/False/None (control_collapsed's shape),
        # not one of the VERDICTS strings -- the legend must not read
        # literally "RDNEG (True)" on a signed visual gate. write_figure
        # closes its figure before returning, so introspect the label
        # source (_verdict_label) directly rather than pyplot's global
        # current-figure state, which is gone by then.
        assert R._verdict_label(True) == "collapsed"
        assert R._verdict_label(False) == "did not collapse"
        assert R._verdict_label(None) == "undetermined"
        assert R._verdict_label("PRESERVED") == "PRESERVED"

    def test_writes_with_a_boolean_verdict_arm_present(self, tmp_path):
        # RDNEG never carries a "final" block (run_arm never adds one for
        # it); write_figure must still render it without crashing.
        pytest.importorskip("matplotlib")
        rdneg = _arm("RDNEG", True)
        del rdneg["final"]
        results = {"RDNEG": rdneg}
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


class TestRunArmWiring:
    """Pins which learner entry point run_arm calls for each arm, with which
    model/clone path, seed, budget and distinguishing kwarg (beta for RB;
    rollout_seed0 and advantage_sign for the AWR arms), and that the final
    verdict is judged against THAT arm's own start rate. All the other
    tests in this file stay green under any of: advantage_sign=1.0 for both
    AWR arms (the control becomes a second treatment and validates
    nothing), CLONE passed where PPO_BC belongs (RC starts from the wrong
    clone but is judged against the PPO bar), or start_rate("RB") hardcoded
    where start_rate(arm) belongs (RC judged against 0.185, a bar it
    clears with 11 fewer captures) -- this test is the one that would
    catch them.
    """

    def test_per_arm_call_arguments_and_start_rate(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        import pe2.awr
        import pe2.exploitability
        from pe2.config import EnvConfig

        mod = _runner()
        monkeypatch.setattr(mod, "OUT", tmp_path)

        learner_calls = {}

        def _record_ppo(
            cfg,
            role,
            opponent_factory,
            model_path,
            total_timesteps,
            seed,
            device="auto",
            callback=None,
        ):
            learner_calls["RC"] = {
                "model_path": model_path,
                "total_timesteps": total_timesteps,
                "seed": seed,
            }

        def _record_br_anchored(
            cfg,
            role,
            opponent_factory,
            model_path,
            total_timesteps,
            seed,
            beta,
            device="auto",
            gamma=0.999,
            critic_warmup=200_000,
            callback=None,
        ):
            learner_calls["RB"] = {
                "model_path": model_path,
                "total_timesteps": total_timesteps,
                "seed": seed,
                "beta": beta,
                "critic_warmup": critic_warmup,
            }

        def _record_train_awr(
            cfg,
            role,
            opponent_factory,
            clone_path,
            total_timesteps,
            seed,
            awr,
            probe_every,
            probe,
            device="auto",
            rollout_seed0=820_000,
        ):
            key = "RD" if awr.advantage_sign == 1.0 else "RDNEG"
            learner_calls[key] = {
                "clone_path": clone_path,
                "total_timesteps": total_timesteps,
                "seed": seed,
                "advantage_sign": awr.advantage_sign,
                "rollout_seed0": rollout_seed0,
            }
            return object(), {
                "total_steps": 0,
                "n_episodes": 0,
                "positives": 0,
                "positives_per_1M": None,
            }

        monkeypatch.setattr(pe2.exploitability, "finetune_ppo", _record_ppo)
        monkeypatch.setattr(
            pe2.exploitability, "finetune_br_anchored", _record_br_anchored
        )
        monkeypatch.setattr(pe2.awr, "train_awr", _record_train_awr)

        def _fake_final_eval(cfg, arm, total, device):
            return {
                "ckpt": "x",
                "k": 10,
                "n": 50,
                "win_rate": 0.2,
                "wilson": [0.1, 0.3],
                "eval_seed0": mod.EVAL_SEED0,
            }

        verdict_calls = {}
        current = [None]

        def _record_round_verdict(
            rows,
            final_k,
            final_n,
            start_rate,
            threshold=mod.CLONE_CI_LO,
            from_step=mod.PASS_FROM,
        ):
            verdict_calls[current[0]] = start_rate
            return "PRESERVED", "test"

        monkeypatch.setattr(mod, "final_eval", _fake_final_eval)
        monkeypatch.setattr(mod, "round_verdict", _record_round_verdict)

        cfg = EnvConfig()
        for arm in ("RB", "RC", "RD", "RDNEG"):
            if arm != "RDNEG":
                # round_verdict is only reached past run_arm's own
                # missing-checkpoint gate (ruling 3): give it a checkpoint.
                ckpt_dir = tmp_path / f"ckpts_{arm}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                (ckpt_dir / f"{arm}_{mod.TOTAL}.zip").touch()
            current[0] = arm
            mod.run_arm(arm, cfg, "cpu")

        assert learner_calls["RB"] == {
            "model_path": mod.CLONE,
            "total_timesteps": mod.TOTAL,
            "seed": mod.SEEDS["RB"],
            "beta": mod.BETA_RB,
            "critic_warmup": mod.CRITIC_WARMUP,
        }
        assert learner_calls["RC"] == {
            "model_path": mod.PPO_BC,
            "total_timesteps": mod.TOTAL,
            "seed": mod.SEEDS["RC"],
        }
        assert learner_calls["RD"] == {
            "clone_path": mod.CLONE,
            "total_timesteps": mod.TOTAL,
            "seed": mod.SEEDS["RD"],
            "advantage_sign": 1.0,
            "rollout_seed0": mod.AWR_SEED0,
        }
        assert learner_calls["RDNEG"] == {
            "clone_path": mod.CLONE,
            "total_timesteps": mod.NEG_TOTAL,
            "seed": mod.SEEDS["RDNEG"],
            "advantage_sign": -1.0,
            "rollout_seed0": mod.AWRNEG_SEED0,
        }
        for arm in ("RB", "RC", "RD"):
            assert verdict_calls[arm] == mod.start_rate(arm)


class TestFinalEvalHonesty:
    """Nothing else in this file reaches final_eval. A regression to scoring
    the in-memory model the learner returns -- the exact thing the brief's
    "ALWAYS from the saved final checkpoint" docstring forbids -- or loading
    RC with the wrong model class, would pass every other test in this file
    green."""

    def test_loads_the_saved_checkpoint_with_the_right_model_class(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("stable_baselines3")
        import stable_baselines3

        import pe2.exploitability
        from pe2.config import EnvConfig

        mod = _runner()
        monkeypatch.setattr(mod, "OUT", tmp_path)

        def _fake_evaluate(
            model, cfg, role, opponent_factory, n_episodes, seed0, init_poses=None
        ):
            return [{"outcome": "capture"} for _ in range(n_episodes)]

        monkeypatch.setattr(pe2.exploitability, "evaluate_sb3_policy", _fake_evaluate)

        loads = {}

        def _make_loader(cls_name):
            def _load(path, device="auto"):
                loads[cls_name] = pathlib.Path(path)
                return object()

            return _load

        for arm, expected_cls in (("RB", "SAC"), ("RC", "PPO"), ("RD", "SAC")):
            ckpt_dir = tmp_path / f"ckpts_{arm}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / f"{arm}_{mod.TOTAL}.zip").touch()

            loads.clear()
            monkeypatch.setattr(stable_baselines3.SAC, "load", _make_loader("SAC"))
            monkeypatch.setattr(stable_baselines3.PPO, "load", _make_loader("PPO"))

            mod.final_eval(EnvConfig(), arm, mod.TOTAL, "cpu")

            assert set(loads) == {expected_cls}, (
                f"{arm} loaded {set(loads)}, expected {{{expected_cls!r}}}"
            )
            assert (
                loads[expected_cls]
                == tmp_path / f"ckpts_{arm}" / f"{arm}_{mod.TOTAL}.zip"
            )
