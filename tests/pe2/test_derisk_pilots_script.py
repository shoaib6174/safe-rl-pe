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
