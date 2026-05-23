import sys
sys.path.insert(0, "/home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024")

import torch
import numpy as np
from training.population_manager import PopulationManager
from agents.maddpg import ActorNetwork


def test_add_and_sample():
    """Add 3 state dicts, sample 300 times, verify roughly uniform distribution."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    for i in range(3):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict())

    assert len(pm) == 3

    counts = [0, 0, 0]
    for _ in range(300):
        sampled = pm.sample()
        val = sampled.fc1.weight.data.mean().item()
        counts[int(round(val))] += 1

    assert all(c >= 50 for c in counts), f"Counts: {counts}"


def test_fifo_eviction():
    """Add K+1 state dicts; verify oldest is evicted."""
    pm = PopulationManager(
        max_size=3, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    for i in range(4):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict())

    assert len(pm) == 3

    for _ in range(300):
        sampled = pm.sample()
        val = sampled.fc1.weight.data.mean().item()
        assert val != 0.0, f"Evicted member 0 was sampled"


def test_sample_returns_fresh_module():
    """Sample twice; verify different module objects."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict())

    s1 = pm.sample()
    s2 = pm.sample()
    assert s1 is not s2


def test_sample_no_grad_and_eval_mode():
    """Verify sampled module has no grad and is in eval mode."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict())

    sampled = pm.sample()
    assert not sampled.training
    assert all(not p.requires_grad for p in sampled.parameters())


def test_empty_population_raises():
    """Sampling from empty population must raise RuntimeError."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    try:
        pm.sample()
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "empty" in str(e).lower()


def test_save_load(tmp_path):
    """Save population, reload, verify identical samples."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    actor.fc1.weight.data.fill_(3.14)
    pm.add(actor.state_dict())

    path = tmp_path / "pop.pt"
    pm.save(str(path))

    pm2 = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    pm2.load(str(path))

    assert len(pm2) == 1
    sampled = pm2.sample()
    assert abs(sampled.fc1.weight.data.mean().item() - 3.14) < 1e-6
