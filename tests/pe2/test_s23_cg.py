"""S23 step 4 — the construct checks CG-S2, CG-S3 and CG-S4.

Spec: `docs/pathway/pursuit_strategy_plan_2026-10-12.md` section 4 and
amendment A2 notes (i) and (ii).

Three things are load-bearing here and the rest is bookkeeping:

- CG-S4's record must say what the law DECIDED, and must not say it on a step
  the law did not decide: a learner's hold span is not a law decision, and a
  stale `last_aim` logged against it would make the round's mechanism reading
  unfalsifiable.
- CG-S2's probe law must be `PLAN` bit for bit. The model error is a property of
  the opponent on the REFERENCE trajectory; measured on a trajectory the probe
  itself bent, it would be a property of the probe.
- CG-S3's rule must be a rule: the first three obstructed seeds in seed order,
  stopping at the third, refusing rather than falling back to a weaker set.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import scripts.derisk_s19_planner as S19
import scripts.s23_cg_s2_model_error as CGS2
import scripts.s23_cg_s3_seeds as CGS3
from pe2.config import EnvConfig
from pe2.planner import (
    LeadPlannerInterceptLaw,
    MpcInterceptLaw,
    PlannerInterceptLaw,
    _episode_entropy,
    _state_bytes,
)
from pe2.recorder import BranchTaggedSearchMDP, branch_summary
from pe2.scripted import GreedyEvader
from pe2.shield import CBFShield
from pe2.structured import HOLD, OrbitSearch, SearchMDP

CFG = EnvConfig()
OPEN_CFG = EnvConfig(n_physical_obstacles=0)  # observation layout unchanged
CONSTRUCT_SEED0 = 2_090_000  # the round's construct stream (plan section 5)


def _mdp(law, cfg: EnvConfig = CFG, shield: bool = True) -> BranchTaggedSearchMDP:
    """The cell's construction (hold, observation flags, no reward wrapper) with
    this law, so a per-step record here is the record the round would carry."""
    return BranchTaggedSearchMDP(
        cfg,
        GreedyEvader,
        hold=HOLD,
        shield=CBFShield(cfg) if shield else None,
        shield_obs=True,
        escape_obs=True,
        reward_wrapper=None,
        law=law,
    )


def _episode(law, seed: int, cfg: EnvConfig = CFG) -> list[dict]:
    mdp = _mdp(law, cfg)
    mdp.play_episode(OrbitSearch(cfg).predict, seed)
    return mdp.step_records


# ----------------------------------------------------------------------------
# CG-S4 — the recorder's view of the law's decision
# ----------------------------------------------------------------------------
class TestRecorderAimPoint:
    def test_lead_records_its_aim_point_on_law_steps_and_nothing_on_search_steps(
        self,
    ):
        steps = _episode(LeadPlannerInterceptLaw(CFG, 1.0), CONSTRUCT_SEED0)
        aimed = [s for s in steps if s["aim_point"] is not None]
        assert aimed, "no law step aimed ahead; this seed cannot test the field"
        assert any(s["branch"] == "search" for s in steps), "no search step either"
        for s in steps:
            moved = s["v_est"] != [0.0, 0.0]
            # LEAD aims ahead exactly when it has a nonzero estimate, and only
            # a law step can carry one (the law is not consulted in a span).
            assert (s["aim_point"] is not None) == (s["branch"] == "law" and moved)
            assert len(s["v_est"]) == 2
            if s["branch"] == "search":
                assert s["aim_point"] is None and s["v_est"] == [0.0, 0.0]

    def test_a_law_without_the_fields_records_none_throughout(self):
        steps = _episode(None, CONSTRUCT_SEED0)  # SearchMDP's own InterceptLaw
        assert steps
        for s in steps:
            assert s["aim_point"] is None
            assert s["v_est"] is None
            assert s["feasible_count"] is None
            assert s["chose_direct"] is None

    def test_plan_records_no_aim_point_but_the_episode_is_otherwise_recorded(self):
        steps = _episode(PlannerInterceptLaw(CFG), CONSTRUCT_SEED0)
        assert steps
        assert all(s["aim_point"] is None and s["v_est"] is None for s in steps)
        assert any(s["branch"] == "law" for s in steps)

    def test_mpc_records_the_feasible_count_and_the_direct_choice(self):
        mdp = _mdp(MpcInterceptLaw(OPEN_CFG), OPEN_CFG)
        log = mdp.play_episode(OrbitSearch(OPEN_CFG).predict, CONSTRUCT_SEED0)
        steps, b = mdp.step_records, log["branch"]
        planned = [s for s in steps if s["aim_point"] is not None]
        assert planned, "the sampling never ran; this seed cannot test the field"
        for s in steps:
            assert isinstance(s["feasible_count"], int)
            assert isinstance(s["chose_direct"], bool)
            if s["aim_point"] is None:  # the CG-S1 delegation: no plan ran
                assert s["feasible_count"] == 0 and s["chose_direct"] is False
        # A2 note (i): the record must be able to say "0 of K were feasible".
        assert all(
            0 <= s["feasible_count"] <= MpcInterceptLaw(OPEN_CFG).k for s in steps
        )
        assert b["n_planned_calls"] == len(steps)  # every step exposes the field
        assert b["n_zero_feasible"] == sum(s["feasible_count"] == 0 for s in steps)
        assert b["n_chose_direct"] == sum(s["chose_direct"] for s in steps)

    def test_the_new_fields_do_not_change_the_episode(self):
        """The recorder regression, on the law that writes the new fields."""
        cfg = CFG
        plain = SearchMDP(
            cfg,
            GreedyEvader,
            hold=HOLD,
            shield=CBFShield(cfg),
            shield_obs=True,
            escape_obs=True,
            law=LeadPlannerInterceptLaw(cfg, 1.0),
        )
        tagged = _mdp(LeadPlannerInterceptLaw(cfg, 1.0))
        policy = OrbitSearch(cfg)
        for seed in (CONSTRUCT_SEED0, CONSTRUCT_SEED0 + 1):
            assert plain.play_episode(policy.predict, seed) == {
                k: v
                for k, v in tagged.play_episode(policy.predict, seed).items()
                if k != "branch"
            }


class TestSummaryCounters:
    @staticmethod
    def _steps(rows: list[tuple]) -> list[dict]:
        return [
            {
                "branch": "law",
                "in_span": True,
                "immobile": False,
                "stalled": False,
                "intervention_magnitude": 0.0,
                "feasible_count": f,
                "chose_direct": d,
            }
            for f, d in rows
        ]

    def test_the_three_mpc_counters(self):
        b = branch_summary(self._steps([(64, True), (0, False), (3, False)]), 0.0)
        assert b["n_planned_calls"] == 3
        assert b["n_zero_feasible"] == 1
        assert b["n_chose_direct"] == 1

    def test_a_record_without_the_fields_summarises_unchanged(self):
        """Hand-written and pre-S23 step records must still summarise: the
        fields are additive, which is what makes S19's records readable."""
        rows = self._steps([(None, None)])
        b_missing = branch_summary(
            [
                {
                    k: v
                    for k, v in rows[0].items()
                    if k not in ("feasible_count", "chose_direct")
                }
            ],
            0.0,
        )
        b_none = branch_summary(rows, 0.0)
        assert b_missing == b_none
        assert b_none["n_planned_calls"] == 0
        assert b_none["n_zero_feasible"] == 0
        assert b_none["n_chose_direct"] == 0


# ----------------------------------------------------------------------------
# amendment A2 note (ii) — the episode digest hashes integers, not a repr
# ----------------------------------------------------------------------------
# One PCG64 state, as `bit_generator.state` hands it over.
_STATE = {
    "bit_generator": "PCG64",
    "state": {"state": 12345678901234567890, "inc": 987654321},
    "has_uint32": 0,
    "uinteger": 0,
}


class TestEpisodeDigest:
    class _Env:
        def __init__(self, state):
            self.np_random = type(
                "_R", (), {"bit_generator": type("_B", (), {"state": state})()}
            )()

    def test_the_digest_is_the_hand_computed_one_of_the_integers(self):
        want = int.from_bytes(
            hashlib.blake2b(
                b"inc:987654321|state:12345678901234567890", digest_size=8
            ).digest(),
            "big",
        )
        assert _episode_entropy(self._Env(_STATE)) == want

    def test_formatting_of_the_same_integers_does_not_change_the_digest(self):
        """The numbers, not their `repr`: numpy scalars, a different key order
        and a different `bit_generator` label are all the same state."""
        other = {
            "uinteger": 0,
            "state": {
                "inc": np.uint64(987654321),
                "state": 12345678901234567890,
            },
            "bit_generator": "PCG64DXSM-ish label",
            "has_uint32": 0,
        }
        assert _state_bytes(other) == _state_bytes(_STATE)
        assert _episode_entropy(self._Env(other)) == _episode_entropy(self._Env(_STATE))
        assert repr(other) != repr(_STATE)  # the old digest WOULD differ

    def test_the_real_generator_state_differs_per_episode_and_repeats(self):
        from pe2.env import PursuitEvasionEnv

        law, env = MpcInterceptLaw(OPEN_CFG), PursuitEvasionEnv(OPEN_CFG)
        env.reset(seed=CONSTRUCT_SEED0)
        law.reset(env=env)
        first = law._ep_seed
        env.reset(seed=CONSTRUCT_SEED0 + 1)
        law.reset(env=env)
        assert law._ep_seed != first
        env.reset(seed=CONSTRUCT_SEED0)
        law.reset(env=env)
        assert law._ep_seed == first


# ----------------------------------------------------------------------------
# CG-S2 — the model error
# ----------------------------------------------------------------------------
def _straight_line(p0, v, n_steps: int, cfg: EnvConfig = CFG) -> dict:
    """A true evader trajectory at constant velocity `v` from `p0`, keyed by env
    step index as `ModelErrorMDP` keys it."""
    p0, v = np.asarray(p0, dtype=float), np.asarray(v, dtype=float)
    return {t: list(p0 + t * cfg.dt * v) for t in range(n_steps + 1)}


class TestPredictionError:
    def test_a_constant_velocity_evader_is_predicted_exactly(self):
        v = [0.4, -0.3]
        true = _straight_line([0.0, 0.0], v, 100)
        samples = [{"step": t, "pos": true[t], "v_est": v} for t in (0, 10, 50)]
        errors, dropped = CGS2.prediction_errors(samples, true, CFG)
        assert dropped == 0
        assert errors == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)

    def test_an_accelerating_evader_misses_by_half_a_t_squared(self):
        a = np.array([0.6, -0.2])  # m/s^2
        v0 = np.array([0.5, 0.1])
        dt, h = CFG.dt, CGS2.H_STEPS
        true = {t: list(v0 * (t * dt) + 0.5 * a * (t * dt) ** 2) for t in range(h + 1)}
        samples = [{"step": 0, "pos": true[0], "v_est": list(v0)}]
        errors, dropped = CGS2.prediction_errors(samples, true, CFG)
        assert dropped == 0
        assert errors[0] == pytest.approx(
            float(np.hypot(*(0.5 * a * (h * dt) ** 2))), rel=1e-12
        )

    def test_the_prediction_is_clipped_into_the_inset_arena(self):
        lim = CFG.arena_size / 2.0 - CFG.agent_radius
        fast = [100.0, 0.0]  # 100 m/s for 2 s: far outside the arena
        pred = CGS2.predict_xy([0.0, 0.0], fast, CFG)
        assert pred[0] == pytest.approx(lim)
        # ... and the error is measured against that clipped point, not the ray.
        true = {CGS2.H_STEPS: [lim, 0.0]}
        errors, _ = CGS2.prediction_errors(
            [{"step": 0, "pos": [0.0, 0.0], "v_est": fast}], true, CFG
        )
        assert errors == pytest.approx([0.0])

    def test_a_horizon_past_the_end_of_the_episode_is_dropped_not_zeroed(self):
        true = _straight_line([0.0, 0.0], [0.2, 0.0], CGS2.H_STEPS - 1)
        samples = [{"step": 0, "pos": [0.0, 0.0], "v_est": [0.2, 0.0]}]
        errors, dropped = CGS2.prediction_errors(samples, true, CFG)
        assert (errors, dropped) == ([], 1)


class TestVelocityProbe:
    """The probe law must be `PLAN`, and the probe env must be the cell."""

    def test_the_probe_env_is_the_cell_with_only_the_recording_changed(self):
        cfg = S19.cell_config()
        ref = S19.build_cell_env(S19.cell_name("PLAN", "SHIELD", "GREEDY"), cfg)
        mdp = CGS2.build_probe_env("GREEDY", cfg)
        assert isinstance(mdp, CGS2.ModelErrorMDP)
        assert isinstance(mdp, BranchTaggedSearchMDP)
        # the swap is scoped: the runner's own name is back
        assert S19.BranchTaggedSearchMDP is BranchTaggedSearchMDP
        assert type(ref) is BranchTaggedSearchMDP
        assert (mdp.hold, mdp.shield_obs, mdp.escape_obs) == (
            ref.hold,
            ref.shield_obs,
            ref.escape_obs,
        )
        assert type(mdp.shield) is type(ref.shield)
        assert (mdp.shield.margin, mdp.shield.gamma) == (
            ref.shield.margin,
            ref.shield.gamma,
        )
        assert type(mdp.inner.opponent_factory) is type(ref.inner.opponent_factory)
        assert isinstance(mdp.law, PlannerInterceptLaw)

    def test_the_probe_plays_plans_episode_bit_for_bit(self):
        cfg = S19.cell_config()
        ref = S19.build_cell_env(S19.cell_name("PLAN", "SHIELD", "GREEDY"), cfg)
        mdp = CGS2.build_probe_env("GREEDY", cfg)
        policy = OrbitSearch(cfg)
        for seed in (CONSTRUCT_SEED0, CONSTRUCT_SEED0 + 1):
            ref.shield, mdp.shield = S19.build_shield(cfg), S19.build_shield(cfg)
            ref.trace, mdp.trace = [], []
            assert mdp.play_episode(policy.predict, seed) == ref.play_episode(
                policy.predict, seed
            )
            assert mdp.trace == ref.trace  # every executed command, in order
            assert mdp.law.n_planned_steps == ref.law.n_planned_steps

    def test_the_probe_estimates_the_velocity_it_does_not_use(self):
        cfg = S19.cell_config()
        mdp = CGS2.build_probe_env("GREEDY", cfg)
        mdp.shield = S19.build_shield(cfg)
        mdp.play_episode(OrbitSearch(cfg).predict, CONSTRUCT_SEED0)
        assert mdp.samples, "no law step carried a nonzero estimate"
        assert mdp.n_visible_law_steps >= len(mdp.samples)
        for s in mdp.samples:
            assert s["v_est"] != [0.0, 0.0]
            # the true position at the sample's own step index is the one the
            # estimate was decoded from, to float32 decoding precision
            true = np.asarray(mdp.evader_xy[s["step"]])
            assert np.hypot(*(true - np.asarray(s["pos"]))) < 1e-4
        errors, _ = CGS2.prediction_errors(mdp.samples, mdp.evader_xy, cfg)
        assert errors and all(e >= 0.0 for e in errors)


# ----------------------------------------------------------------------------
# CG-S3 — the watched-seed rule
# ----------------------------------------------------------------------------
class _StubLaw:
    def __init__(self):
        self.n_planned_steps = 0


class _StubMDP:
    """Just enough to be scanned: a law carrying the counter and a
    `play_episode` that advances it by `planned[seed]` steps."""

    def __init__(self, planned: dict[int, int]):
        self.law = _StubLaw()
        self.planned = planned
        self.seeds_played: list[int] = []
        self.shield = None

    def play_episode(self, predict, seed: int) -> dict:
        self.seeds_played.append(seed)
        self.law.n_planned_steps += self.planned.get(seed, 0)
        return {
            "outcome": "timeout",
            "steps": 1200,
            "branch": {"steps_by_branch": {"law": 7, "search": 3}},
        }


class TestSeedRule:
    S0 = 1000

    def _mdp(self, hits: set[int], n: int = 20) -> _StubMDP:
        return _StubMDP({self.S0 + i: (4 if i in hits else 0) for i in range(n)})

    def test_it_picks_the_first_three_obstructed_seeds_in_seed_order(self):
        mdp = self._mdp({1, 2, 5, 7, 9})
        rec = CGS3.scan(mdp, None, seed0=self.S0, max_scan=20)
        assert rec["seeds"] == [self.S0 + 1, self.S0 + 2, self.S0 + 5]

    def test_it_stops_at_the_third_hit(self):
        mdp = self._mdp({1, 2, 5, 7, 9})
        rec = CGS3.scan(mdp, None, seed0=self.S0, max_scan=20)
        assert rec["scanned"] == 6  # seeds +0 .. +5 inclusive
        assert mdp.seeds_played == [self.S0 + i for i in range(6)]
        assert len(rec["episodes"]) == 6
        assert [e["hit"] for e in rec["episodes"]] == [
            False,
            True,
            True,
            False,
            False,
            True,
        ]

    def test_fewer_than_three_hits_in_the_cap_is_a_refusal(self):
        mdp = self._mdp({3, 8})
        with pytest.raises(SystemExit, match="REFUSES"):
            CGS3.scan(mdp, None, seed0=self.S0, max_scan=20)
        assert mdp.seeds_played == [self.S0 + i for i in range(20)]

    def test_the_shield_is_rebuilt_once_per_episode_when_asked(self):
        mdp, built = self._mdp({0, 1, 2}), []
        CGS3.scan(
            mdp,
            None,
            seed0=self.S0,
            max_scan=20,
            shield_factory=lambda: built.append(1) or len(built),
        )
        assert len(built) == 3 and mdp.shield == 3

    def test_the_rule_text_is_the_record_and_names_the_counter(self):
        mdp = self._mdp({0, 1, 2})
        rec = CGS3.scan(mdp, None, seed0=self.S0, max_scan=20)
        assert rec["rule"] == CGS3.RULE
        assert "n_planned_steps" in rec["rule"]
        assert set(rec) >= {"seeds", "rule", "scanned"}


class TestPlannedStepCounter:
    """The counter CG-S3's rule is read off: the law's own clear-line test."""

    def test_it_counts_a_visible_branch_call_that_planned(self):
        cfg = EnvConfig()
        law = PlannerInterceptLaw(cfg)
        law.set_map([{"x": 1.0, "y": 0.0, "radius": 0.5}])
        from pe2.obs import full_state_observation

        own = np.array([0.0, 0.0, 0.0])
        blocked = full_state_observation(own, np.array([2.0, 0.0, 0.0]), [], cfg)
        clear = full_state_observation(own, np.array([0.0, 2.0, 0.0]), [], cfg)
        law.act(clear)
        assert law.n_planned_steps == 0  # the line was clear: no planning
        law.act(blocked)
        assert law.n_planned_steps == 1  # the obstacle sits on the line
        law.act(blocked)
        assert law.n_planned_steps == 2  # cumulative, as `n_unreachable` is

    def test_an_invisible_evader_is_not_counted(self):
        cfg = EnvConfig()
        law = PlannerInterceptLaw(cfg)
        law.set_map([{"x": 1.0, "y": 0.0, "radius": 0.5}])
        from pe2.obs import full_state_observation

        own = np.array([0.0, 0.0, 0.0])
        obs = full_state_observation(own, np.array([2.0, 0.0, 0.0]), [], cfg)
        law.act(obs)  # one visible, obstructed call
        obs = obs.copy()
        obs[3:7] = 0.0  # now invisible: the last-seen branch may still plan
        law.act(obs)
        assert law.n_planned_steps == 1
