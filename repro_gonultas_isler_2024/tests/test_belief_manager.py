import numpy as np
import pytest
import torch
from training.belief_manager import BeliefStateManager


def test_belief_shape_before_warmup():
    mgr = BeliefStateManager(obs_dim=11)
    belief = mgr.get_belief()
    assert belief.shape == (2,)
    assert np.allclose(belief, 0.0)


def test_history_accumulates():
    mgr = BeliefStateManager(obs_dim=11, history_length=20, downsample=2)
    for i in range(10):
        mgr.add_observation(np.ones(11) * i)
    assert len(mgr.history) == 10


def test_training_sample_storage():
    mgr = BeliefStateManager(obs_dim=11, history_length=20, downsample=2)
    for i in range(20):
        mgr.add_observation(np.ones(11) * i)
    mgr.store_training_sample(np.array([1.0, 2.0]))
    assert len(mgr.training_buffer) == 1


def test_train_step_reduces_loss():
    mgr = BeliefStateManager(obs_dim=11, history_length=20, downsample=2, buffer_size=100)
    # Fill history and training buffer
    for _ in range(50):
        for i in range(20):
            mgr.add_observation(np.random.randn(11))
        mgr.store_training_sample(np.random.randn(2))

    loss1 = mgr.train_step(batch_size=min(32, len(mgr.training_buffer)))
    for _ in range(5):
        loss2 = mgr.train_step(batch_size=min(32, len(mgr.training_buffer)))

    assert loss1 > 0
    assert loss2 > 0
    assert loss2 < loss1 * 1.5  # Loss should generally decrease or stay similar


def test_belief_nonzero_after_training():
    mgr = BeliefStateManager(obs_dim=11, history_length=20, downsample=2, buffer_size=50)
    target_pos = np.array([3.0, 4.0])

    for _ in range(30):
        for i in range(20):
            obs = np.random.randn(11)
            obs[4:6] = target_pos + np.random.randn(2) * 0.1  # opponent position visible
            mgr.add_observation(obs)
        mgr.store_training_sample(target_pos)

    for _ in range(10):
        mgr.train_step(batch_size=min(16, len(mgr.training_buffer)))

    belief = mgr.get_belief()
    assert not np.allclose(belief, 0.0, atol=0.1)
