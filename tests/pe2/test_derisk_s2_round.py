"""scripts/derisk_s2_cp_round.py — the S2 round's own decisions."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s2_cp_round.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s2_cp_round", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _arm(name, sel_pass, fin_pass, kind="ppo"):
    return {
        "arm": name,
        "kind": kind,
        "selected": {
            "gate": {"PPO_CLONE": {"pass": sel_pass}, "TEACHER": {"pass": False}}
        },
        "final": {
            "gate": {"PPO_CLONE": {"pass": fin_pass}, "TEACHER": {"pass": False}}
        },
        "checkpoints": [],
    }


class TestProtocol:
    def test_pre_registered_constants(self):
        assert (R.TOTAL, R.EVERY, R.EVAL_N, R.GATE_N) == (1_000_000, 100_000, 200, 500)
        assert (R.SEL_SEED0, R.GATE_SEED0, R.S1_EVAL_SEED0) == (
            1_040_000,
            1_050_000,
            1_010_000,
        )
        assert R.SEEDS == (7, 8, 9) and R.BETAS == {"ANC01": 0.1, "ANC001": 0.01}
        assert R.CP_LAMBDA == 1.0 and R.P_GATE == 0.05
        assert (R.TOTAL_LONG, R.EVERY_LONG) == (10_000_000, 1_000_000)

    def test_arm_table(self):
        names = set(R.ARMS)
        assert names == {
            "SEL7",
            "SEL8",
            "SEL9",
            "ANC01_7",
            "ANC01_8",
            "ANC01_9",
            "ANC001_7",
            "ANC001_8",
            "ANC001_9",
            "NULL7",
            "SBLONG",
        }
        assert R.ARMS["SEL8"] == {
            "kind": "ppo",
            "beta": 0.0,
            "seed": 8,
            "zero_reward": False,
        }
        assert R.ARMS["ANC01_9"] == {
            "kind": "ppo",
            "beta": 0.1,
            "seed": 9,
            "zero_reward": False,
        }
        assert R.ARMS["ANC001_7"] == {
            "kind": "ppo",
            "beta": 0.01,
            "seed": 7,
            "zero_reward": False,
        }
        assert R.ARMS["NULL7"] == {
            "kind": "ppo",
            "beta": 0.0,
            "seed": 7,
            "zero_reward": True,
        }
        assert R.ARMS["SBLONG"] == {"kind": "sblong", "seed": 7}

    def test_arms_refuse_without_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "START_PATH", tmp_path / "missing.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()

    def test_measure_start_refuses_to_re_measure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s2_start.json")
        (tmp_path / "s2_start.json").write_text("{}")
        with pytest.raises(SystemExit, match="already"):
            R.require_not_measured()

    def test_an_arm_refuses_to_run_twice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s2_arm_SEL7.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("SEL7")
        (tmp_path / "ckpts_ANC01_8").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("ANC01_8")
        R.require_arm_not_run("SEL9")  # clean: no raise


class TestHypotheses:
    def test_h_sel_needs_two_of_three_selected_passes(self):
        res = {
            "SEL7": _arm("SEL7", True, False),
            "SEL8": _arm("SEL8", True, False),
            "SEL9": _arm("SEL9", False, False),
            "NULL7": _arm("NULL7", False, False),
        }
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"] == "PASS" and h["H_SEL_passes"] == 2

    def test_h_anc_uses_final_checkpoints_per_beta(self):
        res = {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res |= {f"ANC001_{s}": _arm(f"ANC001_{s}", True, False) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", False, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_ANC"]["ANC01"] == "PASS" and h["H_ANC"]["ANC001"] == "FAIL"

    def test_null_pass_makes_h_sel_undetermined_only(self):
        res = {f"SEL{s}": _arm(f"SEL{s}", True, False) for s in (7, 8, 9)}
        res |= {f"ANC01_{s}": _arm(f"ANC01_{s}", False, True) for s in (7, 8, 9)}
        res["NULL7"] = _arm("NULL7", True, False)
        h = R.hypothesis_verdicts(res)
        assert h["H_SEL"].startswith("UNDETERMINED") and h["H_ANC"]["ANC01"] == "PASS"

    def test_missing_arms_are_incomplete(self):
        h = R.hypothesis_verdicts({"SEL7": _arm("SEL7", True, False)})
        assert h["H_SEL"] == "INCOMPLETE"

    def test_missing_null_makes_h_sel_incomplete(self):
        """Without the negative control there is no evidence that the selection
        rule did not manufacture the passes, so H-SEL may not be read."""
        res = {f"SEL{s}": _arm(f"SEL{s}", True, False) for s in (7, 8, 9)}
        res |= {f"ANC01_{s}": _arm(f"ANC01_{s}", False, False) for s in (7, 8, 9)}
        res |= {f"ANC001_{s}": _arm(f"ANC001_{s}", False, False) for s in (7, 8, 9)}
        h = R.hypothesis_verdicts(res)
        assert h["NULL"] == "INCOMPLETE" and h["H_SEL"] == "INCOMPLETE"
        assert "incomplete" in R.direction(h)


class TestDirection:
    def test_anc_pass_is_the_artefact(self):
        d = R.direction(
            {
                "H_SEL": "FAIL",
                "H_ANC": {"ANC01": "PASS", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "no selection" in d and "Route C withdrawn" in d

    def test_sel_only(self):
        d = R.direction(
            {
                "H_SEL": "PASS",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "selection-based" in d and "needs the gate" in d

    def test_both_fail(self):
        d = R.direction(
            {
                "H_SEL": "FAIL",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "FAIL",
            }
        )
        assert "seed-7-specific" in d

    def test_null_pass(self):
        d = R.direction(
            {
                "H_SEL": "UNDETERMINED (selection manufactures improvement)",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "PASS",
            }
        )
        assert "manufactures" in d

    def test_incomplete(self):
        assert "incomplete" in R.direction(
            {
                "H_SEL": "INCOMPLETE",
                "H_ANC": {"ANC01": "INCOMPLETE", "ANC001": "INCOMPLETE"},
                "NULL": "INCOMPLETE",
            }
        )

    def test_a_missing_null_control_is_incomplete(self):
        assert "incomplete" in R.direction(
            {
                "H_SEL": "PASS",
                "H_ANC": {"ANC01": "FAIL", "ANC001": "FAIL"},
                "NULL": "INCOMPLETE",
            }
        )


class TestProvenance:
    def test_records_what_the_run_depended_on(self):
        p = R.provenance()
        assert set(p) == {
            "git_commit",
            "sb3_version",
            "torch_threads",
            "omp_num_threads",
        }
        assert isinstance(p["torch_threads"], int)
        assert isinstance(p["sb3_version"], str)
        assert p["git_commit"] is None or isinstance(p["git_commit"], str)


class TestDeterminism:
    """SEL7 vs S1's CP is a WEIGHT comparison: the two rounds probe on
    different eval streams (1 040 000+ vs 1 010 000+), so equal win rates
    would prove nothing about determinism."""

    @staticmethod
    def _model(seed: int):
        from stable_baselines3 import PPO

        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper
        from pe2.scripted import GreedyEvader

        env = SingleAgentWrapper(EnvConfig(), "pursuer", GreedyEvader)
        return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)

    @staticmethod
    def _dirs(tmp_path):
        sel, cp = tmp_path / "ckpts_SEL7", tmp_path / "ckpts_CP"
        sel.mkdir()
        cp.mkdir()
        return sel, cp

    def test_identical_weights_are_reproduced(self, tmp_path):
        sel, cp = self._dirs(tmp_path)
        model = self._model(7)
        model.save(sel / "SEL7_100000.zip")
        model.save(cp / "CP_100000.zip")
        assert R.determinism_note(sel, cp, [100_000]) == "reproduced"

    def test_a_perturbed_weight_is_not_reproduced(self, tmp_path):
        import torch

        sel, cp = self._dirs(tmp_path)
        model = self._model(7)
        model.save(sel / "SEL7_100000.zip")
        with torch.no_grad():
            next(iter(model.policy.parameters())).add_(1.0)
        model.save(cp / "CP_100000.zip")
        assert R.determinism_note(sel, cp, [100_000]).startswith(
            "not reproduced (first difference at 100000"
        )

    def test_a_thread_count_caveats_the_negative(self, tmp_path):
        """A weight mismatch is only evidence about the RUN if both rounds used
        the same torch thread count, so the count travels with the answer."""
        import torch

        sel, cp = self._dirs(tmp_path)
        model = self._model(7)
        model.save(sel / "SEL7_100000.zip")
        with torch.no_grad():
            next(iter(model.policy.parameters())).add_(1.0)
        model.save(cp / "CP_100000.zip")
        note = R.determinism_note(sel, cp, [100_000], threads=8)
        assert note.startswith("not reproduced (first difference at 100000")
        assert "8 torch threads" in note and "same count" in note

    def test_a_missing_checkpoint_is_not_comparable(self, tmp_path):
        sel, cp = self._dirs(tmp_path)
        self._model(7).save(sel / "SEL7_100000.zip")
        assert (
            R.determinism_note(sel, cp, [100_000])
            == "not comparable (missing checkpoint at 100000)"
        )


class TestFigure:
    def test_a_crashed_sblong_still_draws(self, tmp_path):
        """A crashed SBLONG has probe rows but no `s1_bars`; the figure must
        still be written rather than dying on a KeyError."""
        starts = {r: {"rate": 0.4, "wilson": [0.3, 0.5], "logs": []} for r in R.REFS}
        results = {
            "SBLONG": {
                "arm": "SBLONG",
                "kind": "sblong",
                "verdict": "UNDETERMINED",
                "reason": "crashed: RuntimeError: boom",
                "checkpoints": [{"steps": 1_000_000, "win_rate": 0.4}],
            }
        }
        path = tmp_path / "s2_round.png"
        R.write_figure(results, starts, {"complete": False}, path)
        assert path.exists()

    def test_a_complete_sblong_draws_the_verdict_lines(self, tmp_path):
        """The SBLONG verdict is read against S1's Wilson UPPER bounds, so the
        panel must draw those lines and not only the point estimates."""
        starts = {r: {"rate": 0.4, "wilson": [0.3, 0.5], "logs": []} for r in R.REFS}
        results = {
            "SBLONG": {
                "arm": "SBLONG",
                "kind": "sblong",
                "verdict": "FAIL",
                "reason": "below START_S hi",
                "checkpoints": [{"steps": 1_000_000, "win_rate": 0.4}],
                "s1_bars": {"BAR_ORBIT": 0.9, "FLOOR_RAND": 0.1, "START_S": 0.3},
                "s1_bars_hi": {"FLOOR_RAND": 0.15, "START_S": 0.36},
            }
        }
        path = tmp_path / "s2_round_complete.png"
        labels = self._axhline_labels(R.write_figure, results, starts, path)
        assert path.exists()
        assert "FLOOR_RAND hi (verdict line)" in labels
        assert "START_S hi (verdict line)" in labels

    @staticmethod
    def _axhline_labels(write_figure, results, starts, path) -> list:
        """Every horizontal line the figure drew, by label."""
        import matplotlib.axes

        seen = []
        real = matplotlib.axes.Axes.axhline

        def spy(self, y=0, *a, **kw):
            seen.append(kw.get("label"))
            return real(self, y, *a, **kw)

        matplotlib.axes.Axes.axhline = spy
        try:
            write_figure(results, starts, {"complete": True}, path)
        finally:
            matplotlib.axes.Axes.axhline = real
        return seen
