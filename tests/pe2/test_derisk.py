"""Tests for pe2.derisk — Tier 0 signal audit (T0.1)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.derisk import CLONE_CI_LO, mc_returns, pilot_passes, signal_audit
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


class _Flee:
    """model.predict shim: full-speed-forward evader (only needs to not get
    caught by a stationary opponent)."""

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
        assert len(rep["positive_lengths"]) == rep["positives"]
        assert rep["positives_per_1M"] == pytest.approx(
            rep["positives"] / rep["total_steps"] * 1e6
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

    def test_evader_positives_are_wins_not_captures(self):
        # A stationary pursuer never catches the greedy evader, so every
        # episode is a timeout: for role="evader" that is a WIN on every
        # episode (positives == n_episodes) and zero captures. The pursuer's
        # audit of the same situation is the mirror image: zero positives.
        class _StillOpponent:
            def reset(self): ...
            def act(self, obs, env):
                return (0.0, 0.0)

        rep = signal_audit(
            _Flee(), CFG, "evader", _StillOpponent, min_steps=2400, seed0=790000
        )
        assert rep["captures"] == 0
        assert rep["positives"] == rep["n_episodes"]
        assert rep["positives_per_1M"] > 0.0


class TestPilotPassRule:
    def test_pass_requires_every_probe_from_100k(self):
        rows = [
            {"steps": 50_000, "win_rate": 0.05},  # before the window: ignored
            {"steps": 100_000, "win_rate": 0.150},
            {"steps": 150_000, "win_rate": 0.140},
            {"steps": 200_000, "win_rate": 0.137},  # equal to the bound passes
        ]
        assert pilot_passes(rows) is True
        rows[2]["win_rate"] = 0.136
        assert pilot_passes(rows) is False

    def test_no_probe_in_window_fails(self):
        assert pilot_passes([{"steps": 50_000, "win_rate": 0.9}]) is False

    def test_bound_is_the_gated_clone_ci_lo(self):
        assert CLONE_CI_LO == 0.137
