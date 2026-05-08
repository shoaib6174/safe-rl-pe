import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train, load_config
from pathlib import Path


def test_load_config():
    config = load_config("config.yaml")
    assert "maddpg" in config
    assert config["maddpg"]["episodes"] == 20000
    assert config["maddpg"]["batch_size"] == 512


def test_training_runs_10_episodes():
    """Smoke test: verify training loop runs without errors for 10 episodes."""
    config = load_config("config.yaml")
    config["maddpg"]["episodes"] = 10
    config["maddpg"]["replay_buffer_size"] = 1000
    config["eval"]["checkpoint_freq"] = 5

    # Write temporary config
    import yaml
    temp_config_path = "/tmp/smoke_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    train(temp_config_path)

    # Verify checkpoints were created
    assert Path("checkpoints/episode_0").exists()
    assert Path("checkpoints/episode_5").exists()
    assert Path("checkpoints/episode_0/pursuer_actor.pt").exists()
    assert Path("checkpoints/episode_0/evader_actor.pt").exists()
