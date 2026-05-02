"""Verify that train_br_sac.py loads a SAC checkpoint as a frozen opponent
and that the frozen side's parameters do not change after a trainer update step.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

# Skip if no snapshot is available (CI/dev machines without niro-2 mount).
SNAP = Path("results/BR_frozen/s48/evader")
pytestmark = pytest.mark.skipif(
    not (SNAP / "ppo.zip").exists(),
    reason="snapshot results/BR_frozen/s48/evader/ppo.zip not available",
)


def _hash_params(model: SAC) -> str:
    """Return a stable hash of all policy + critic + target params."""
    h = hashlib.sha256()
    for tensor in model.policy.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_load_snapshot_returns_sac():
    """SAC.load on the snapshot returns a SAC instance with the expected obs space."""
    model = SAC.load(SNAP / "ppo.zip", device="cpu")
    assert isinstance(model, SAC)
    # The cohort uses Dict obs (obs_history + lidar + state) per partial_obs_wrapper.
    assert hasattr(model.observation_space, "spaces"), \
        "expected Dict observation space (partial-obs policy)"
    assert {"obs_history", "lidar", "state"}.issubset(model.observation_space.spaces.keys())


def test_frozen_opponent_params_unchanged_after_trainer_step(tmp_path):
    """Round-trip: build the BR trainer, run one update step, assert frozen
    opponent's state_dict hash is unchanged."""
    from scripts.train_br_sac import build_br_setup, run_one_update

    setup = build_br_setup(
        frozen_opponent_path=str(SNAP),
        frozen_role="evader",
        seed=148,
        output_dir=str(tmp_path / "br_smoke"),
        env_kwargs=None,  # use cohort defaults
    )
    h_before = _hash_params(setup.frozen_model)
    run_one_update(setup, n_train_steps=200)  # tiny update batch
    h_after = _hash_params(setup.frozen_model)
    assert h_before == h_after, "frozen opponent parameters were mutated"


if __name__ == "__main__":
    # Self-test runner (no pytest needed — avoids ROS plugin conflicts).
    import sys
    errors = []
    try:
        test_load_snapshot_returns_sac()
        print("PASS: test_load_snapshot_returns_sac")
    except Exception as e:
        errors.append(("test_load_snapshot_returns_sac", e))
        print(f"FAIL: test_load_snapshot_returns_sac — {e}")

    try:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            test_frozen_opponent_params_unchanged_after_trainer_step(Path(tmp))
        print("PASS: test_frozen_opponent_params_unchanged_after_trainer_step")
    except Exception as e:
        errors.append(("test_frozen_opponent_params_unchanged_after_trainer_step", e))
        print(f"FAIL: test_frozen_opponent_params_unchanged_after_trainer_step — {e}")

    if errors:
        sys.exit(1)
    print("All tests passed.")
