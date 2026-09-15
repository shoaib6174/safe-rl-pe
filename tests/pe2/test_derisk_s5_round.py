"""scripts/derisk_s5_remedy.py — the remedy round's own decisions."""

import importlib.util
import json
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


def _starts():
    return {k: {"rate": 0.4, "reproduced": True, "logs": []} for k in R.REFS}


class TestStopRule:
    def test_a_refusal_writes_nothing(self, tmp_path, monkeypatch):
        """C1: `run_arm` on an arm that has already run must refuse BEFORE it
        opens or writes anything — the crash recorder must never overwrite a
        finished arm's record with an `UNDETERMINED` stub."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s5_start.json")
        arm_path = tmp_path / "s5_arm_RA19.json"
        arm_path.write_text(
            json.dumps(
                {
                    "arm": "RA19",
                    "verdict": "final PASS vs TEACHER_SHIELD",
                    "final": {"score": {"k": 250, "n": 500, "rate": 0.5}},
                },
                indent=2,
            )
        )
        (tmp_path / "s5_start.json").write_text(json.dumps(_starts()))
        before = arm_path.read_bytes()

        with pytest.raises(SystemExit, match="stop rule"):
            R.run_arm("RA19", "cpu")

        assert arm_path.read_bytes() == before
        assert not (tmp_path / "ckpts_RA19").exists()
        assert sorted(q.name for q in tmp_path.iterdir()) == [
            "s5_arm_RA19.json",
            "s5_start.json",
        ]


class _VecEnv:
    def __init__(self, env_steps, violations_total):
        self._attrs = {"env_steps": env_steps, "violations_total": violations_total}

    def get_attr(self, name):
        return [self._attrs[name]]


class _Model:
    def __init__(self, env):
        self._env = env
        self.saved = []

    def get_env(self):
        return self._env

    def save(self, path):
        self.saved.append(pathlib.Path(path))
        pathlib.Path(path).write_text("ckpt")


class TestTrainingViolationCensus:
    def test_probe_rows_carry_the_running_train_violation_count(
        self, tmp_path, monkeypatch
    ):
        """I1: every probe row records the cumulative training violations at
        that mark, flushed with the report, so a crashed arm's contact steps
        survive in a file."""
        monkeypatch.setattr(
            "pe2.structured.evaluate_structured",
            lambda *a, **k: [
                {"outcome": "capture", "violations": 0},
                {"outcome": "timeout", "violations": 0},
            ],
        )
        monkeypatch.setattr(
            R,
            "summarize_shield",
            lambda logs: {
                "intervention_rate": 0.5,
                "stall_step_rate": 0.4,
                "deadlock_episode_rate": 0.1,
            },
        )
        report = tmp_path / "s5_probe_RA19.json"
        probe = R.ShieldedStructuredProbe(
            None,
            None,
            every=2,
            total=4,
            eval_n=2,
            seed0=0,
            ckpt_dir=tmp_path,
            report_path=report,
            label="RA19",
            shield_obs=True,
        )
        cb = probe.callback
        cb.model = _Model(_VecEnv(env_steps=7, violations_total=3))
        cb._on_step()

        assert [r["mark"] for r in probe.rows] == [2, 4]
        assert [r["train_violations"] for r in probe.rows] == [3, 3]
        assert [
            r["train_violations"] for r in json.loads(report.read_text())["checkpoints"]
        ] == [3, 3]

    def test_hsafe_falls_back_to_the_last_probe_row(self):
        assert R.train_violations_of({"train_violations": 0}) == 0
        assert (
            R.train_violations_of(
                {"train_violations": 0, "checkpoints": [{"train_violations": 9}]}
            )
            == 0
        )  # a record's own count wins over the rows
        res = _full()
        res["RB20"] = {
            "arm": "RB20",
            "family": "RB",
            "seed": 20,
            "verdict": "UNDETERMINED",
            "reason": "crashed: RuntimeError: boom",
            "checkpoints": [
                {"violations": 0, "train_violations": 0},
                {"violations": 0, "train_violations": 4},
            ],
        }
        hs = R.hsafe_verdict(res)
        assert hs["verdict"] == "FAILS"
        assert hs["offenders"] == ["RB20"] and hs["shielded_violation_steps"] == 4


class TestOrphanProbeFiles:
    def test_orphan_probe_enters_the_census_but_not_the_pass_counts(
        self, tmp_path, monkeypatch
    ):
        """I2: a probe file whose process died before a record was written is
        still evidence; H-SAFE must see its contact steps."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s5_probe_RC21.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "steps": 1_000_000,
                            "mark": 1_000_000,
                            "win_rate": 0.1,
                            "k": 20,
                            "n": 200,
                            "violations": 2,
                            "train_violations": 5,
                        }
                    ]
                }
            )
        )
        res = _full()
        del res["RC21"]
        summary = R.summarize(res, _starts())

        assert summary["hsafe"]["verdict"] == "FAILS"
        assert summary["hsafe"]["offenders"] == ["RC21"]
        assert summary["hsafe"]["shielded_violation_steps"] == 7  # 5 train + 2 probe
        orphan = summary["orphans"]["RC21"]
        assert orphan["probe"].endswith("s5_probe_RC21.json")
        assert orphan["checkpoints"] == 1
        assert (orphan["probe_violations"], orphan["train_violations"]) == (2, 5)
        assert "orphan probe file" in orphan["reason"]
        # the orphan is NOT a result: it cannot pass, and RC is short a seed
        assert (
            "RC21" not in summary["arms"] and summary["ingredient"]["RC"]["passes"] == 2
        )
        assert summary["hypotheses"]["H3"]["RC"] == "INCOMPLETE"
        assert summary["complete"] is False

    def test_no_orphans_when_every_probe_has_a_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s5_probe_RC21.json").write_text(
            json.dumps({"checkpoints": [{"violations": 0, "train_violations": 0}]})
        )
        summary = R.summarize(_full(), _starts())
        assert summary["orphans"] == {} and summary["hsafe"]["verdict"] == "HOLDS"


class TestIngredientReading:
    def test_mean_rate_is_none_when_no_seed_finished(self, tmp_path, monkeypatch):
        """M1: an unfinished family must not read as a 0.0 mean gate rate."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        res = _full()
        for s in R.SEEDS:
            del res[f"RB{s}"]
        ing = R.summarize(res, _starts())["ingredient"]
        assert ing["RB"] == {"passes": 0, "mean_rate": None, "n_finished": 0}
        assert ing["RA"]["n_finished"] == 3
        assert ing["RA"]["mean_rate"] == pytest.approx((0.5 + 0.5 + 0.4) / 3)

    def test_mean_rate_averages_only_finished_seeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        res = _full()
        del res["RA21"]
        ing = R.summarize(res, _starts())["ingredient"]
        assert ing["RA"]["n_finished"] == 2
        assert ing["RA"]["mean_rate"] == pytest.approx(0.5)
