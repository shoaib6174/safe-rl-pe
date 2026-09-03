"""Step 0 exit gate: same config + seed twice -> identical RNG streams.

This is the CI-gated determinism test required by
docs/pathway/from_scratch_build_plan.md (Step 0, hard rule 1 / L1.1).
It simulates what a run does — derive named sub-streams from a root seed and
draw from each — and requires bit-identical results across repetitions.
"""

import numpy as np

from pe2.config import EnvConfig
from pe2.rng import derive_seed, make_rng

STREAMS = ("env_reset", "eval_init", "trainer")


def simulate_run_streams(cfg: EnvConfig, seed: int) -> dict[str, np.ndarray]:
    """Draw a fixed sequence from each named sub-stream, as a real run would."""
    assert hash(cfg)  # config must be hashable to be loggable
    return {
        label: make_rng(derive_seed(seed, label)).standard_normal(256)
        for label in STREAMS
    }


class TestDeterminism:
    def test_same_config_seed_identical_streams(self):
        cfg = EnvConfig()
        run_a = simulate_run_streams(cfg, seed=0)
        run_b = simulate_run_streams(cfg, seed=0)
        assert run_a.keys() == run_b.keys()
        for label in run_a:
            np.testing.assert_array_equal(run_a[label], run_b[label])

    def test_different_seed_different_streams(self):
        cfg = EnvConfig()
        run_a = simulate_run_streams(cfg, seed=0)
        run_b = simulate_run_streams(cfg, seed=1)
        assert any(not np.array_equal(run_a[label], run_b[label]) for label in run_a)
