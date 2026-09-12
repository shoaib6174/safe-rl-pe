"""Tests for pe2.derisk — Tier 0 signal audit (T0.1) and pilot rules (T0.3)."""

import dataclasses

import gymnasium as gym
import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    PASS_FROM_STEP,
    TrainingCaptureCounter,
    control_collapsed,
    mc_returns,
    pilot_passes,
    signal_audit,
    start_is_adequate,
)
from pe2.exploitability import SingleAgentWrapper
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

    def test_probe_exactly_at_window_start_is_judged(self):
        # The window filter is `steps >= from_step`. Narrowing it to `>`
        # would drop the first probe from the judgement entirely — and that
        # is exactly where an arm that has not yet recovered sits, so a
        # below-bound arm would be wrongly passed. One row, on the edge.
        edge_low = [{"steps": PASS_FROM_STEP, "win_rate": CLONE_CI_LO - 0.001}]
        assert pilot_passes(edge_low) is False
        edge_at = [{"steps": PASS_FROM_STEP, "win_rate": CLONE_CI_LO}]
        assert pilot_passes(edge_at) is True


def _probe_rows(*pairs, n=200):
    """CheckpointEvalProbe-shaped rows from (steps, k) pairs at N=n."""
    return [{"steps": s, "k": k, "n": n, "win_rate": k / n} for s, k in pairs]


class TestControlCollapsed:
    """The negative control may only be CERTIFIED as collapsed when it
    actually reproduced the documented collapse — failing the pass rule is
    necessary but nowhere near sufficient."""

    def test_documented_d78_trajectory_is_a_collapse(self):
        # 0.185 -> 0.060 @100k -> 0.020 @200k, the measured naive fine-tune.
        rows = _probe_rows((50_000, 24), (100_000, 12), (150_000, 8), (200_000, 4))
        verdict, why = control_collapsed(rows)
        assert verdict is True, why

    def test_final_rate_near_the_clone_is_not_a_collapse(self):
        # 0.130 fails the pass rule (it is under 0.137) but is nowhere near
        # the documented collapse — it nearly PRESERVED the clone. Certifying
        # this as "the control collapsed" is the critical failure.
        rows = _probe_rows((100_000, 26), (200_000, 26))
        assert pilot_passes(rows) is False  # the pass rule alone says "fail"
        verdict, why = control_collapsed(rows)
        assert verdict is None
        assert "not separated" in why

    def test_empty_window_cannot_certify_itself(self):
        # A run shorter than the window gives EVERY arm an empty window, so
        # every arm "fails" — including the control, which would then certify
        # a harness that judged zero probes. This is what a smoke run does.
        verdict, why = control_collapsed(_probe_rows((200, 0), (400, 0)))
        assert verdict is None
        assert "window" in why

    def test_control_that_holds_the_bound_did_not_collapse(self):
        rows = _probe_rows((100_000, 40), (200_000, 40))  # 0.200, healthy
        verdict, why = control_collapsed(rows)
        assert verdict is False, why

    def test_reason_is_always_words(self):
        for rows in (
            _probe_rows((100_000, 4)),
            _probe_rows((100_000, 26)),
            _probe_rows((400, 0)),
            _probe_rows((100_000, 40)),
        ):
            _, why = control_collapsed(rows)
            assert isinstance(why, str) and len(why) > 20


class _Truth(gym.Wrapper):
    """Ground truth for the counter: sits INSIDE the vec env, so it sees
    every env.step directly rather than through SB3's rollout locals."""

    def __init__(self, env):
        super().__init__(env)
        self.steps = self.episodes = self.captures = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.steps += 1
        if term or trunc:
            self.episodes += 1
            self.captures += int(info["outcome"] == "capture")
        return obs, r, term, trunc, info


# capture_distance = 2 * agent_radius = 4.0 > min_init_distance 3.0, so
# captures actually happen in a few hundred steps and the outcome-read is
# exercised instead of trivially counting zero.
_CAPTURY = dataclasses.replace(EnvConfig(), max_steps=40, agent_radius=2.0)


class TestTrainingCaptureCounter:
    def test_matches_ground_truth_for_both_learner_families(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import PPO, SAC

        for ctor, kwargs, steps in (
            (SAC, {}, 300),
            (PPO, {"n_steps": 256, "batch_size": 64}, 256),
        ):
            truth = _Truth(
                SingleAgentWrapper(_CAPTURY, "pursuer", GreedyEvader, seed0=1)
            )
            counter = TrainingCaptureCounter(snapshot_every=100)
            model = ctor("MlpPolicy", truth, seed=0, verbose=0, device="cpu", **kwargs)
            model.learn(total_timesteps=steps, callback=counter.callback)
            assert counter.state["steps"] == truth.steps == model.num_timesteps
            assert counter.state["episodes"] == truth.episodes
            assert counter.state["captures"] == truth.captures
            # not vacuous: the capture branch really ran
            assert counter.state["captures"] > 0
            assert counter.positives_per_1M == pytest.approx(
                truth.captures / truth.steps * 1e6
            )

    def test_segments_decompose_the_aggregate(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC

        truth = _Truth(SingleAgentWrapper(_CAPTURY, "pursuer", GreedyEvader, seed0=1))
        counter = TrainingCaptureCounter(snapshot_every=100)
        model = SAC("MlpPolicy", truth, seed=0, verbose=0, device="cpu")
        model.learn(total_timesteps=250, callback=counter.callback)
        segs = counter.segments
        assert len(segs) > 1  # the run really was split
        assert sum(s["steps"] for s in segs) == counter.state["steps"]
        assert sum(s["captures"] for s in segs) == counter.state["captures"]
        assert sum(s["episodes"] for s in segs) == counter.state["episodes"]
        assert [s["from_step"] for s in segs] == [0, *[s["to_step"] for s in segs[:-1]]]

    def test_no_steps_means_no_rate_and_no_kill_verdict(self):
        pytest.importorskip("stable_baselines3")
        counter = TrainingCaptureCounter()
        # A rate off zero rollouts is UNDEFINED, not 0.0 — returning 0.0 would
        # fire the kill rule ("signal too sparse") having collected nothing.
        assert counter.positives_per_1M is None
        assert counter.segments == []


class TestStochasticRepeat:
    """The STOCHASTIC regime is the training-time regime the kill rule speaks
    about, so its number must be reproducible from the seed the report
    records. `TestSignalAudit.test_deterministic_repeat` cannot show this:
    its shim ignores the `deterministic` flag, so it never draws from the
    tensor library's global generator the way SB3's `predict` does.
    """

    def test_stochastic_repeat_is_identical(self):
        torch = pytest.importorskip("torch")

        class _TorchNoisy:
            """model.predict shim that is GENUINELY stochastic: off
            `deterministic=False` it draws from torch's global generator,
            exactly as SB3's squashed-Gaussian SAC actor does."""

            def predict(self, obs, deterministic=True):
                if deterministic:
                    return np.array([1.0, 0.0], dtype=np.float32), None
                return (2.0 * torch.rand(2) - 1.0).numpy().astype(np.float32), None

        # not vacuous: the shim really does move with the global generator.
        torch.manual_seed(1)
        first, _ = _TorchNoisy().predict(None, deterministic=False)
        torch.manual_seed(2)
        second, _ = _TorchNoisy().predict(None, deterministic=False)
        assert not np.array_equal(first, second)

        kw = {"min_steps": 2000, "seed0": 790000, "deterministic": False}
        a = signal_audit(_TorchNoisy(), CFG, "pursuer", GreedyEvader, **kw)
        b = signal_audit(_TorchNoisy(), CFG, "pursuer", GreedyEvader, **kw)
        assert a == b
        # and the report must SAY what was seeded: a number whose provenance
        # is not in the file is not a reproducible measurement.
        assert a["torch_seed"] == 790000


class TestStartIsAdequate:
    """P-C starts from its OWN clone of unknown quality yet is judged against
    a bound derived from a DIFFERENT clone. Measuring the start and then not
    using it is how a weak start gets reported as a refuted hypothesis."""

    def test_a_start_separated_below_the_bound_is_undetermined(self):
        # 4/200 = 0.020, Wilson [0.008, 0.050]: the upper bound 0.050 is
        # below 0.137, so this arm could not reach the bound even if the
        # mechanism were perfect and nothing about it can be concluded.
        verdict, why = start_is_adequate(4, 200)
        assert verdict is None
        assert "0.050" in why and "0.137" in why

    def test_a_start_whose_interval_reaches_the_bound_is_judgeable(self):
        # 20/200 = 0.100, Wilson [0.066, 0.149]: 0.149 >= 0.137, so the
        # start is consistent with clearing the bound and the arm's verdict
        # carries information about the mechanism.
        verdict, why = start_is_adequate(20, 200)
        assert verdict is True
        assert "0.149" in why

    def test_the_gated_clone_itself_is_judgeable(self):
        # 37/200 = 0.185 is the gated clone, whose CI-lo IS the bound.
        assert start_is_adequate(37, 200)[0] is True

    def test_reason_is_always_words(self):
        for k in (0, 4, 20, 37):
            _, why = start_is_adequate(k, 200)
            assert isinstance(why, str) and len(why) > 20


class TestRoundVerdict:
    """T0.4 retry-round rules (spec C2): pass = >= 0.137 at every probe from
    500k; improved = final N=500 Wilson lower bound above the start."""

    def _rows(self, rates, every=100_000):
        return [
            {"steps": (i + 1) * every, "win_rate": r, "k": round(r * 200), "n": 200}
            for i, r in enumerate(rates)
        ]

    def test_improvement_needs_the_lower_bound_not_the_point(self):
        from pe2.derisk import improved

        # 110/500 = 0.220 has Wilson lo 0.1859 > 0.185; 109/500 does not.
        assert improved(110, 500, 0.185)
        assert not improved(109, 500, 0.185)
        # PPO's start is 0.205: needs 121/500.
        assert improved(121, 500, 0.205)
        assert not improved(120, 500, 0.205)

    def test_round_verdict_improved_branch_at_the_same_boundaries(self):
        # M4: `round_verdict` must not re-implement the improvement rule.
        # It delegates to `improved`, so it has to agree with it AT THE EDGE,
        # not merely far inside (150/500). Same four points as the test above.
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 10)  # passes the window at every probe
        assert round_verdict(rows, 110, 500, 0.185)[0] == "IMPROVED"
        assert round_verdict(rows, 109, 500, 0.185)[0] == "PRESERVED"
        assert round_verdict(rows, 121, 500, 0.205)[0] == "IMPROVED"
        assert round_verdict(rows, 120, 500, 0.205)[0] == "PRESERVED"

    def test_round_verdict_delegates_to_improved(self):
        # The binding between the two: whatever `improved` says on the final
        # counts is what a window-passing arm's verdict says.
        from pe2.derisk import improved, round_verdict

        rows = self._rows([0.2] * 10)
        for k, start in ((109, 0.185), (110, 0.185), (120, 0.205), (121, 0.205)):
            expected = "IMPROVED" if improved(k, 500, start) else "PRESERVED"
            assert round_verdict(rows, k, 500, start)[0] == expected

    def test_collapse_inside_the_window(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 4 + [0.10] + [0.2] * 5)  # dip at 500k
        verdict, reason = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "COLLAPSED"
        assert "500000" in reason and "0.100" in reason

    def test_dip_before_the_window_is_forgiven(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2, 0.03, 0.09, 0.14] + [0.2] * 6)  # P-C's shape
        verdict, _ = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "IMPROVED"

    def test_preserved_when_final_is_inside_the_start_interval(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 10)
        verdict, reason = round_verdict(rows, 100, 500, 0.185)  # 0.200, lo 0.168
        assert verdict == "PRESERVED"
        assert "0.185" in reason

    def test_empty_window_is_undetermined_not_collapsed(self):
        from pe2.derisk import round_verdict

        rows = self._rows([0.2] * 4)  # run died before 500k
        verdict, reason = round_verdict(rows, 150, 500, 0.185)
        assert verdict == "UNDETERMINED"
        assert "500000" in reason

    def test_every_verdict_is_in_the_registry(self):
        from pe2.derisk import VERDICTS, round_verdict

        for rows, k in (
            (self._rows([0.2] * 10), 150),
            (self._rows([0.2] * 10), 100),
            (self._rows([0.2] * 4 + [0.05] * 6), 150),
            (self._rows([0.2]), 150),
        ):
            assert round_verdict(rows, k, 500, 0.185)[0] in VERDICTS


class TestStructuredVerdict:
    def test_improved_when_lower_bound_beats_the_orbit_bar(self):
        from pe2.derisk import structured_verdict

        v, reason = structured_verdict(
            300, 500, bar_rate=0.49, floor_hi=0.38, start_hi=0.30
        )
        assert v == "IMPROVED" and "0.490" in reason

    def test_learned_when_above_floor_and_start_but_not_the_bar(self):
        from pe2.derisk import structured_verdict

        v, reason = structured_verdict(
            220, 500, bar_rate=0.49, floor_hi=0.38, start_hi=0.30
        )
        assert v == "LEARNED" and "0.380" in reason

    def test_failed_when_not_separated_from_the_floor_or_the_start(self):
        from pe2.derisk import structured_verdict

        assert structured_verdict(180, 500, 0.49, 0.38, 0.30)[0] == "FAILED"
        # a high START_S upper bound is as binding as the floor
        assert structured_verdict(220, 500, 0.49, 0.30, 0.45)[0] == "FAILED"

    def test_verdict_vocabulary(self):
        from pe2.derisk import STRUCTURED_VERDICTS

        assert STRUCTURED_VERDICTS == ("IMPROVED", "LEARNED", "FAILED", "UNDETERMINED")
