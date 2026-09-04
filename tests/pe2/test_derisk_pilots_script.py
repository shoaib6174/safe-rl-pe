"""Tests for scripts/derisk_t0_pilots.py — the T0.3 runner's own decisions.

The training itself is a GPU job and is not exercised here; what is tested
is what the runner DECIDES with the numbers, which is where a wrongly
passed or wrongly failed arm would come from.
"""

import importlib.util
import pathlib

import pytest

# The runner imports matplotlib and stable-baselines3 at module scope, and
# both are optional dependencies (CI installs neither).
pytest.importorskip("stable_baselines3")
pytest.importorskip("matplotlib")

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_t0_pilots.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_t0_pilots", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(name, verdict, rows=None):
    return {
        "arm": name,
        "pass": verdict,
        "checkpoints": rows
        or [{"steps": 100_000, "k": 40, "n": 200, "win_rate": 0.200}],
    }


class TestSeedsArePaired:
    def test_the_three_off_policy_arms_share_one_seed(self):
        # The training seed drives critic init, exploration and the training
        # env's episode stream. One seed per arm confounds the mechanism with
        # between-seed variance (0.010-0.280 measured in this project), and
        # there is one seed per arm, so the confound is unmeasurable.
        s = _runner().SEEDS
        assert s["PA"] == s["PB"] == s["CTRL"]

    def test_the_on_policy_arm_keeps_its_own_seed(self):
        # PPO has no replay buffer, no critic warm-up and its own clone:
        # nothing about it is paired with the SAC arms, so sharing a seed
        # would suggest a pairing that does not exist.
        s = _runner().SEEDS
        assert s["PC"] != s["PA"]


class TestSurvivorsAndFigure:
    def test_an_undetermined_arm_is_not_a_survivor(self):
        # P-C's verdict is forced to None when its own clone was too weak to
        # be judged against the bound. `if r["pass"]` already excludes it,
        # but only by accident of None being falsy; the rule is explicit.
        mod = _runner()
        # CTRL collapsed: 4/200 = 0.020, Wilson hi 0.050 < 0.137.
        ctrl = _arm(
            "CTRL", False, [{"steps": 200_000, "k": 4, "n": 200, "win_rate": 0.020}]
        )
        results = {"PA": _arm("PA", True), "PC": _arm("PC", None), "CTRL": ctrl}
        summary = mod.summarize(results, every=50_000, eval_n=200, complete=True)
        assert summary["negative_control_collapsed"] is True
        assert summary["survivors"] == ["PA"]

    def test_the_figure_survives_all_three_verdicts(self, tmp_path, monkeypatch):
        # A None verdict must not crash the figure or be drawn as "fail":
        # the figure is the artifact most likely to be read on its own.
        mod = _runner()
        monkeypatch.setattr(mod, "OUT", tmp_path)
        results = {
            "PA": _arm("PA", True),
            "PB": _arm("PB", False),
            "PC": _arm("PC", None),
        }
        summary = {
            "negative_control_collapsed": True,
            "negative_control_reason": "collapse reproduced",
        }
        mod.write_figure(results, summary, eval_n=200)
        assert (tmp_path / "t0_pilots.png").exists()


class TestNondegeneracyGate:
    def test_the_declared_seed_governs_the_rollout(self, tmp_path, monkeypatch):
        # The gate seeded torch and THEN loaded the clone. `SAC.load`
        # re-applies the seed baked into the zip through `_setup_model`, so
        # the declared seed was overwritten before a single action was
        # sampled and `rollout.torch_seed` in the results file was a false
        # provenance claim — the exact accident `seed_torch`'s docstring
        # warns about. Three declared seeds gave byte-identical output.
        #
        # Changing the declared seed alone is not enough to catch it: the
        # seed ALSO drives `reset(seed=NONDEG_SEED0 + ep)`, so the rollout
        # would differ through the env even with the seeding dead. The
        # episode stream is therefore pinned here, leaving torch's global
        # generator as the ONLY thing the declared seed can still move.
        mod = _runner()
        if not mod.CLONE.exists():
            pytest.skip(f"gated clone not present: {mod.CLONE}")
        from pe2.config import EnvConfig

        class _PinnedStream(mod.SingleAgentWrapper):
            def reset(self, **kwargs):
                kwargs["seed"] = 4242
                return super().reset(**kwargs)

        monkeypatch.setattr(mod, "SingleAgentWrapper", _PinnedStream)
        monkeypatch.setattr(mod, "OUT", tmp_path)
        monkeypatch.setattr(mod, "NONDEG_EPISODES", 2)
        monkeypatch.setattr(mod, "NONDEG_BOOTSTRAP", 8)
        cfg = EnvConfig(max_steps=40)

        monkeypatch.setattr(mod, "NONDEG_SEED0", 11)
        a = mod.nondegeneracy_gate(cfg, "cpu")
        monkeypatch.setattr(mod, "NONDEG_SEED0", 22)
        b = mod.nondegeneracy_gate(cfg, "cpu")

        assert a["rollout"]["torch_seed"] == 11
        assert b["rollout"]["torch_seed"] == 22
        # `search` is continuous in the agent's pose, so two different
        # action streams over the same initial state cannot coincide.
        assert a["terms"]["search"]["mean"] != b["terms"]["search"]["mean"]


class TestGateDecisiveness:
    """The plateau gate is a hard stop. These pin that it fires on the
    reward and not on the sampling noise of a short rollout."""

    @staticmethod
    def _episodes(n_high, n_low, steps=100, positives=10):
        # "high" episodes carry `positives` steps of vis=+0.1 out of
        # `steps`; "low" episodes carry none.
        high = {"vis": [0.1] * positives + [0.0] * (steps - positives)}
        low = {"vis": [0.0] * steps}
        return [dict(high) for _ in range(n_high)] + [dict(low) for _ in range(n_low)]

    def test_a_live_term_is_decisively_passed(self):
        # 20 identical episodes, each 50 of 100 steps positive -> every
        # resample has frac_positive = 0.50, inside the 0.05-0.95 band with
        # std > 0. Nothing to be uncertain about.
        mod = _runner()
        eps = self._episodes(20, 0, steps=100, positives=50)
        assert mod.gate_decisiveness(eps, seed=1)["vis"] == 1.0

    def test_a_dead_term_is_decisively_failed(self):
        # Every step zero -> std = 0 and frac_positive = 0 in every
        # resample: the term fails the band in all of them.
        mod = _runner()
        eps = self._episodes(0, 20)
        assert mod.gate_decisiveness(eps, seed=1)["vis"] == 0.0

    def test_a_term_sitting_on_the_band_edge_is_not_decisive(self):
        # THE DEFECT THIS EXISTS FOR. 10 high + 10 low episodes of 100
        # steps, 10 positives in a high one: the POINT estimate is
        # 100/2000 = 0.0500, exactly the band edge. A bootstrap resample
        # drawing k high episodes reads 10k/2000 = 0.005k and passes iff
        # 0.005k > 0.05, i.e. k >= 11; k ~ Binomial(20, 0.5), so
        # P(pass) = 1 - P(k <= 10) = 1 - 0.5881 = 0.4119. A hard stop off a
        # coin flip is exactly what must not happen.
        mod = _runner()
        eps = self._episodes(10, 10)
        frac = mod.gate_decisiveness(eps, seed=1)["vis"]
        assert 0.30 < frac < 0.52  # 0.4119 +/- 4 bootstrap SE (0.025)
        assert not (
            frac >= mod.DECISIVE_FRACTION or frac <= 1.0 - mod.DECISIVE_FRACTION
        )


class TestGateBlocksTheRun:
    @staticmethod
    def _prepare(mod, monkeypatch, tmp_path, report):
        import sys

        clone = tmp_path / "clone.zip"
        clone.touch()
        monkeypatch.setattr(mod, "OUT", tmp_path)
        monkeypatch.setattr(mod, "CLONE", clone)
        monkeypatch.setattr(mod, "nondegeneracy_gate", lambda cfg, device: report)
        monkeypatch.setattr(sys, "argv", ["derisk_t0_pilots.py", "--arms", "PA"])

    def test_an_undecided_gate_stops_the_run_with_its_own_reason(
        self, tmp_path, monkeypatch
    ):
        # A point verdict of "passes" that the bootstrap only reproduces in
        # 41% of resamples must not let P-A train, and must not be reported
        # as a finding about the reward either.
        mod = _runner()
        report = {
            "terms": {"vis": {"passes": True, "decisive": False}},
            "all_pass": True,
            "all_decisive": False,
        }
        self._prepare(mod, monkeypatch, tmp_path, report)
        with pytest.raises(SystemExit, match="UNDECIDED"):
            mod.main()

    def test_a_decisive_failure_is_still_reported_as_a_finding(
        self, tmp_path, monkeypatch
    ):
        # The pre-registered response to a decisively flat term is
        # unchanged: fix the term or drop P-A, do not widen the band.
        mod = _runner()
        report = {
            "terms": {"vis": {"passes": False, "decisive": True}},
            "all_pass": False,
            "all_decisive": True,
        }
        self._prepare(mod, monkeypatch, tmp_path, report)
        with pytest.raises(SystemExit, match="FAILED"):
            mod.main()
