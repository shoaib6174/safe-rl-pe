"""Tests for pe2.eval — Wilson CI, summarize, battery runner."""

from typing import ClassVar

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.eval import (
    binomial_pvalue,
    mcnemar_pvalue,
    run_battery,
    select_best_checkpoint,
    summarize,
    wilson_ci,
    win_count,
)
from pe2.scripted import (
    FovLimitedEvader,
    FovLimitedPursuer,
    GreedyEvader,
    GreedyPursuer,
)


class TestWilsonCi:
    def test_known_value(self):
        # k=50, n=100 -> approx (0.4038, 0.5962) at z=1.96
        lo, hi = wilson_ci(50, 100)
        assert lo == pytest.approx(0.4038, abs=1e-3)
        assert hi == pytest.approx(0.5962, abs=1e-3)

    def test_edge_zero(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < hi < 0.1

    def test_edge_full(self):
        lo, hi = wilson_ci(100, 100)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert 0.9 < lo < 1.0

    def test_empty(self):
        lo, hi = wilson_ci(0, 0)
        assert np.isnan(lo) and np.isnan(hi)

    def test_contains_point_estimate(self):
        lo, hi = wilson_ci(37, 500)
        assert lo < 37 / 500 < hi


class TestSummarize:
    def test_aggregation(self):
        logs = [
            {"outcome": "capture", "steps": 100, "los_breaks": 2},
            {"outcome": "timeout", "steps": 1200, "los_breaks": 0},
            {"outcome": "capture", "steps": 200, "los_breaks": 1},
            {"outcome": "timeout", "steps": 1200, "los_breaks": 3},
        ]
        s = summarize(logs)
        assert s["n"] == 4
        assert s["captures"] == 2
        assert s["cr"] == pytest.approx(0.5)
        assert s["cr_ci"][0] < 0.5 < s["cr_ci"][1]
        assert s["mean_steps"] == pytest.approx(675.0)
        assert s["mean_los_breaks"] == pytest.approx(1.5)
        assert s["frac_with_los_break"] == pytest.approx(0.75)


class TestRunBattery:
    def test_smoke(self):
        matchups = {
            "GxG": (GreedyPursuer, GreedyEvader),
            "FLpxGe": (FovLimitedPursuer, GreedyEvader),
            "GpxFLe": (GreedyPursuer, FovLimitedEvader),
        }
        results = run_battery(EnvConfig(), matchups, n_episodes=3, seed0=42)
        assert set(results) == set(matchups)
        for logs in results.values():
            assert len(logs) == 3
            assert [log["seed"] for log in logs] == [42, 43, 44]

    def test_paired_seeds_across_matchups(self):
        """D57: every matchup sees the same seed sequence (paired eval)."""
        matchups = {
            "A": (GreedyPursuer, GreedyEvader),
            "B": (FovLimitedPursuer, FovLimitedEvader),
        }
        results = run_battery(EnvConfig(), matchups, n_episodes=2, seed0=7)
        seeds_a = [log["seed"] for log in results["A"]]
        seeds_b = [log["seed"] for log in results["B"]]
        assert seeds_a == seeds_b == [7, 8]

    def test_deterministic(self):
        matchups = {"GxG": (GreedyPursuer, GreedyEvader)}
        r1 = run_battery(EnvConfig(), matchups, n_episodes=2, seed0=9)
        r2 = run_battery(EnvConfig(), matchups, n_episodes=2, seed0=9)
        assert r1 == r2

    def test_bit_identical_json(self):
        """Plan Step 5 verify: harness twice -> bit-identical serialized results."""
        matchups = {"GxG": (GreedyPursuer, GreedyEvader)}
        r1 = run_battery(EnvConfig(), matchups, n_episodes=3, seed0=11)
        r2 = run_battery(EnvConfig(), matchups, n_episodes=3, seed0=11)
        import json

        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
        s1 = {m: summarize(logs) for m, logs in r1.items()}
        s2 = {m: summarize(logs) for m, logs in r2.items()}
        assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


class TestBinomialPvalue:
    def test_hand_computed_zero_of_17(self):
        """D56's own number: 0/17 balanced -> one-sided p = 0.5^17 ~ 7.63e-6."""
        p = binomial_pvalue(0, 17, alternative="less")
        assert p == pytest.approx(0.5**17, rel=1e-12)
        assert p == pytest.approx(7.6294e-6, rel=1e-3)

    def test_hand_computed_two_sided(self):
        # k=3, n=10, p0=0.5: outcomes 0,1,2,3,7,8,9,10 -> 2*176/1024
        p = binomial_pvalue(3, 10, alternative="two-sided")
        assert p == pytest.approx(2 * 176 / 1024, rel=1e-9)

    def test_greater(self):
        # P[X >= 9] for n=10, p0=0.5 = (10 + 1)/1024
        p = binomial_pvalue(9, 10, alternative="greater")
        assert p == pytest.approx(11 / 1024, rel=1e-9)

    def test_bad_alternative(self):
        with pytest.raises(ValueError, match="unknown alternative"):
            binomial_pvalue(1, 5, alternative="sideways")


class TestWinCount:
    LOGS: ClassVar = [
        {"outcome": "capture"},
        {"outcome": "timeout"},
        {"outcome": "capture"},
    ]

    def test_pursuer_perspective(self):
        assert win_count(self.LOGS, "pursuer") == (2, 3)

    def test_evader_perspective(self):
        assert win_count(self.LOGS, "evader") == (1, 3)

    def test_bad_role(self):
        with pytest.raises(ValueError, match="unknown role"):
            win_count(self.LOGS, "referee")


class TestMcnemarPvalue:
    @staticmethod
    def _logs(outcomes):
        return [{"seed": 100 + i, "outcome": o} for i, o in enumerate(outcomes)]

    def test_discordant_counts_pursuer(self):
        """b = A wins only, c = B wins only; concordant pairs are ignored."""
        a = self._logs(["capture", "timeout", "capture", "timeout", "capture"])
        b = self._logs(["capture", "capture", "timeout", "timeout", "timeout"])
        p, b_, c_ = mcnemar_pvalue(a, b, "pursuer")
        assert (b_, c_) == (2, 1)
        # exact two-sided binomial of 2 out of 3 at p0=0.5 = 1.0
        assert p == pytest.approx(1.0)

    def test_strong_separation_is_significant(self):
        """A wins 12 pairs B loses, B wins 2: p < 0.05."""
        a = self._logs(["capture"] * 12 + ["timeout"] * 2 + ["capture"] * 86)
        b = self._logs(["timeout"] * 12 + ["capture"] * 2 + ["capture"] * 86)
        p, b_, c_ = mcnemar_pvalue(a, b, "pursuer")
        assert (b_, c_) == (12, 2)
        assert p < 0.05

    def test_evader_role_flips_win_outcome(self):
        """Same episodes, evader perspective: timeout is the win."""
        a = self._logs(["timeout"] * 10 + ["capture"] * 90)
        b = self._logs(["capture"] * 10 + ["capture"] * 90)
        p, b_, c_ = mcnemar_pvalue(a, b, "evader")
        assert (b_, c_) == (10, 0)
        assert p < 0.01

    def test_full_agreement_gives_p1(self):
        logs = self._logs(["capture", "timeout"])
        p, b_, c_ = mcnemar_pvalue(logs, logs, "pursuer")
        assert (p, b_, c_) == (1.0, 0, 0)

    def test_unpaired_seed_rejected(self):
        a = self._logs(["capture"])
        with pytest.raises(ValueError, match="unpaired episode seed"):
            mcnemar_pvalue(a, self._logs(["capture"])[:0], "pursuer")

    def test_bad_role(self):
        with pytest.raises(ValueError, match="unknown role"):
            mcnemar_pvalue([], [], "referee")


class TestSelectBestCheckpoint:
    def test_picks_best_not_last(self):
        """D14: selection is by paired-eval win rate, never recency."""

        # eval_fn receives the candidate VALUE (the checkpoint handle)
        candidates = {
            "ckpt_10": 30,
            "ckpt_20": 80,
            "ckpt_30": 50,
        }  # value = rigged wins

        def fake_eval(candidate):
            wins = candidate
            return [{"outcome": "capture", "steps": 100, "los_breaks": 0}] * wins + [
                {"outcome": "timeout", "steps": 1200, "los_breaks": 0}
            ] * (100 - wins)

        # insertion order = training order; the LAST checkpoint is not best
        best, table = select_best_checkpoint(candidates, fake_eval, role="pursuer")
        assert best == "ckpt_20"
        assert table["ckpt_20"]["win_rate"] == pytest.approx(0.8)
        assert "cr_ci" in table["ckpt_10"]

    def test_empty_candidates(self):
        with pytest.raises(ValueError, match="no candidates"):
            select_best_checkpoint({}, lambda c: [], role="pursuer")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
