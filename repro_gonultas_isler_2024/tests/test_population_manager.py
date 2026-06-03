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


def test_add_tracks_episode():
    """Verify episode numbers are tracked with each addition."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict(), episode=100)
    pm.add(actor.state_dict(), episode=250)
    pm.add(actor.state_dict(), episode=500)

    assert pm.episodes_added == [100, 250, 500]


def test_fifo_eviction_tracks_episodes():
    """Verify episode tracking survives FIFO eviction."""
    pm = PopulationManager(
        max_size=2, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict(), episode=100)
    pm.add(actor.state_dict(), episode=200)
    pm.add(actor.state_dict(), episode=300)  # evicts episode 100

    assert pm.episodes_added == [200, 300]


def test_sample_recency_bias():
    """Recency-biased sampling should favor recently added members."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    for i in range(4):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict(), episode=i * 250)

    # Sample 400 times with recency bias at episode 1000
    counts = [0, 0, 0, 0]
    for _ in range(400):
        sampled = pm.sample_recency(current_episode=1000, lambda_decay=0.15)
        val = sampled.fc1.weight.data.mean().item()
        counts[int(round(val))] += 1

    # Most recent (index 3, added at ep 750) should be sampled most
    assert counts[3] > counts[0], f"Recent member not favored: {counts}"
    # Oldest (index 0, added at ep 0) should be sampled least
    assert counts[0] < counts[3], f"Old member not deprioritized: {counts}"


def test_sample_recency_zero_lambda_is_uniform():
    """lambda=0 should give roughly uniform sampling."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    for i in range(3):
        actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
        actor.fc1.weight.data.fill_(float(i))
        pm.add(actor.state_dict(), episode=i * 250)

    counts = [0, 0, 0]
    for _ in range(300):
        sampled = pm.sample_recency(current_episode=1000, lambda_decay=0.0)
        val = sampled.fc1.weight.data.mean().item()
        counts[int(round(val))] += 1

    # Should be roughly uniform (each >= 50 out of 300)
    assert all(c >= 50 for c in counts), f"Not uniform with lambda=0: {counts}"


def test_sample_recency_single_member():
    """With only 1 member, recency sampling should return that member."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    actor.fc1.weight.data.fill_(7.0)
    pm.add(actor.state_dict(), episode=100)

    sampled = pm.sample_recency(current_episode=500, lambda_decay=0.15)
    assert abs(sampled.fc1.weight.data.mean().item() - 7.0) < 1e-6


def test_save_load_preserves_episodes(tmp_path):
    """Save/load roundtrip must preserve episode metadata."""
    pm = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    actor = ActorNetwork(obs_dim=4, action_dim=2, hidden_dim=16)
    pm.add(actor.state_dict(), episode=42)
    pm.add(actor.state_dict(), episode=99)

    path = tmp_path / "pop_v2.pt"
    pm.save(str(path))

    pm2 = PopulationManager(
        max_size=8, obs_dim=4, action_dim=2, hidden_dim=16, device="cpu"
    )
    pm2.load(str(path))

    assert len(pm2) == 2
    assert pm2.episodes_added == [42, 99]
