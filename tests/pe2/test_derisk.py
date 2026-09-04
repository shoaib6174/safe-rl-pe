"""Tests for pe2.derisk — Tier 0 signal audit (T0.1)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.derisk import mc_returns, signal_audit
from pe2.scripted import GreedyEvader

CFG = EnvConfig()


class TestMcReturns:
    def test_terminal_only_returns_hand_computed(self):
        # Episode of length 3 ending in capture (R=+1), gamma=0.5:
        # G_0 = 0.5**3 = 0.125, G_1 = 0.25, G_2 = 0.5.
        g = mc_returns(length=3, terminal_reward=1.0, gamma=0.5)
        assert g == pytest.approx([0.125, 0.25, 0.5])

    def test_loss_flips_sign(self):
        g = mc_returns(length=2, terminal_reward=-1.0, gamma=1.0)
        assert g == pytest.approx([-1.0, -1.0])


class _Full:
    """model.predict shim: full-forward pursuer (a heuristic from D76)."""

    def predict(self, obs, deterministic=True):
        return np.array([1.0, 0.0], dtype=np.float32), None


class TestSignalAudit:
    def test_counts_are_consistent(self):
        rep = signal_audit(
            _Full(), CFG, "pursuer", GreedyEvader, min_steps=3000, seed0=790000
        )
        assert rep["total_steps"] >= 3000
        assert rep["n_episodes"] == len(rep["episode_lengths"])
        assert rep["captures"] == len(rep["capture_lengths"])
        assert rep["positives_per_1M"] == pytest.approx(
            rep["captures"] / rep["total_steps"] * 1e6
        )
        assert 0.0 <= rep["visible_fraction"] <= 1.0
        # bucket step counts must sum to the total
        assert (
            sum(b["n"] for b in rep["return_variance_by_bucket"]) == rep["total_steps"]
        )

    def test_deterministic_repeat(self):
        a = signal_audit(
            _Full(), CFG, "pursuer", GreedyEvader, min_steps=2000, seed0=790000
        )
        b = signal_audit(
            _Full(), CFG, "pursuer", GreedyEvader, min_steps=2000, seed0=790000
        )
        assert a == b

    def test_stand_still_never_captures(self):
        # v = (-1+1)/2 * v_max = 0, so the pursuer never moves and the greedy
        # evader (starting >= min_init_distance = 3 m away) is never caught:
        # every episode is a 1200-step timeout with terminal reward -1, so
        # every populated bucket's mean return is negative.
        class _Still:
            def predict(self, obs, deterministic=True):
                return np.array([-1.0, 0.0], dtype=np.float32), None

        rep = signal_audit(
            _Still(), CFG, "pursuer", GreedyEvader, min_steps=2400, seed0=790000
        )
        assert rep["captures"] == 0
        assert rep["positives_per_1M"] == 0.0
        assert all(b["mean"] < 0 for b in rep["return_variance_by_bucket"] if b["n"])
