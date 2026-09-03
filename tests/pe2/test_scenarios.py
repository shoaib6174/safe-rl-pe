"""Tests for pe2.scenarios — scenario-stratified eval inits (D70)."""

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.eval import summarize
from pe2.exploitability import SingleAgentWrapper
from pe2.scenarios import (
    LAYOUT_SEED,
    default_scenarios,
    evaluate_scenarios,
    scenario_inits,
)
from pe2.scripted import GreedyEvader

CFG = EnvConfig()
SCENARIOS = default_scenarios(CFG)


class TestScenarioDefinitions:
    def test_six_canonical_scenarios(self):
        assert [s.name for s in SCENARIOS] == [
            "close_open",
            "mid_open",
            "far_open",
            "occluded_start",
            "shadow_adjacent",
            "corner_trap",
        ]

    def test_nominal_poses_satisfy_predicates(self):
        """Each scenario's NOMINAL poses must pass its own predicate —
        checked through the real obs builder (visibility flag, band)."""
        env = PursuitEvasionEnv(CFG)
        env.reset(seed=LAYOUT_SEED)
        for scn in SCENARIOS:
            env.pursuer_pose = np.array(scn.pursuer_pose)
            env.evader_pose = np.array(scn.evader_pose)
            visible = env._get_obs()["pursuer"][6] > 0.5
            d = float(np.hypot(*(env.pursuer_pose[:2] - env.evader_pose[:2])))
            assert visible == scn.expect_visible, f"{scn.name}: visibility"
            assert scn.d_band[0] <= d <= scn.d_band[1], f"{scn.name}: distance {d}"


class TestScenarioInits:
    def test_deterministic(self):
        scn = SCENARIOS[0]
        assert scenario_inits(scn, 5, CFG) == scenario_inits(scn, 5, CFG)

    def test_variants_differ_but_stay_in_band(self):
        for scn in SCENARIOS:
            inits = scenario_inits(scn, 4, CFG)
            env = PursuitEvasionEnv(CFG)
            env.reset(seed=LAYOUT_SEED)
            for pp, ep in inits:
                env.pursuer_pose = np.array(pp)
                env.evader_pose = np.array(ep)
                visible = env._get_obs()["pursuer"][6] > 0.5
                d = float(np.hypot(*(env.pursuer_pose[:2] - env.evader_pose[:2])))
                assert visible == scn.expect_visible, scn.name
                assert scn.d_band[0] <= d <= scn.d_band[1], scn.name
            distinct = {tuple(np.round(np.asarray(pp), 6)) for pp, _ in inits}
            assert len(distinct) > 1, f"{scn.name}: jitter produces no variety"

    def test_occluded_start_has_variants_near_obstacle(self):
        """Sanity: occluded-start inits must line up through obstacle 0 —
        verified implicitly: visibility False asserted for all variants."""
        scn = next(s for s in SCENARIOS if s.name == "occluded_start")
        assert len(scenario_inits(scn, 3, CFG)) == 3


class TestScenarioEval:
    def test_init_poses_respected_by_wrapper(self):
        w = SingleAgentWrapper(CFG, "pursuer", GreedyEvader)
        pp, ep = SCENARIOS[0].pursuer_pose, SCENARIOS[0].evader_pose
        w.reset(seed=LAYOUT_SEED, init_poses=(pp, ep))
        np.testing.assert_allclose(w.env.pursuer_pose, pp, atol=1e-9)
        np.testing.assert_allclose(w.env.evader_pose, ep, atol=1e-9)

    def test_evaluate_scenarios_smoke(self):
        pytest.importorskip("stable_baselines3")

        class Shim:
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0]), None

        results = evaluate_scenarios(
            Shim(), CFG, "pursuer", GreedyEvader, SCENARIOS[:2], n_variants=2
        )
        assert set(results) == {"close_open", "mid_open"}
        for name, logs in results.items():
            assert len(logs) == 2
            assert all(log["scenario"] == name for log in logs)
            assert all(log["seed"] == LAYOUT_SEED for log in logs)  # fixed layout
            s = summarize(logs)  # same fields as all other eval logs
            assert s["n"] == 2

    def test_evaluate_scenarios_deterministic(self):
        pytest.importorskip("stable_baselines3")

        class Shim:
            def predict(self, obs, deterministic=True):
                return np.array([1.0, 0.0]), None

        r1 = evaluate_scenarios(Shim(), CFG, "pursuer", GreedyEvader, SCENARIOS[:1], 2)
        r2 = evaluate_scenarios(Shim(), CFG, "pursuer", GreedyEvader, SCENARIOS[:1], 2)
        assert r1 == r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
