#!/usr/bin/env python3
"""Evaluation script for shared policy trained agents."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import yaml

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.shared_maddpg import SharedMADDPGAgent


def evaluate(env, agent, num_episodes, frame_stack, device, role_p=0, role_e=1):
    """Evaluate shared policy.

    role_p: role for pursuer (default 0)
    role_e: role for evader (default 1)
    """
    capture_count = 0
    total_len = 0
    for ep in range(num_episodes):
        obs_p, obs_e = env.reset()
        obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
        obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]
        done = False
        step = 0
        while not done and step < env.max_steps:
            stacked_p = np.concatenate(obs_p_buffer[-frame_stack:])
            stacked_e = np.concatenate(obs_e_buffer[-frame_stack:])
            with torch.no_grad():
                a_p = agent.select_action(stacked_p, role=role_p, noise_sigma=0)
                a_e = agent.select_action(stacked_e, role=role_e, noise_sigma=0)
            obs_p, obs_e, r_p, r_e, done = env.step(a_p, a_e)
            obs_p_buffer.append(obs_p.copy())
            obs_e_buffer.append(obs_e.copy())
            step += 1
        if step < env.max_steps:
            capture_count += 1
        total_len += step
    return capture_count / num_episodes, total_len / num_episodes


with open("config_shared_policy.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_EPISODES = 50
frame_stack = cfg["maddpg"]["frame_stack"]
raw_obs_dim = 11
obs_dim = raw_obs_dim * frame_stack
action_dim = 2
total_obs_dim = obs_dim * 2
total_action_dim = action_dim * 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

env = PursuitEvasionEnv(cfg)

# Find checkpoint
base_dir = Path("/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024/results_shared")
import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cp_dir = base_dir / f"seed_{seed}" / "checkpoints" / "episode_10000"

print(f"\n=== Seed {seed} ===")
print(f"Checkpoint: {cp_dir}")

if not (cp_dir / "shared_actor.pt").exists():
    print(f"ERROR: checkpoint not found")
    sys.exit(1)

agent = SharedMADDPGAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    total_obs_dim=total_obs_dim,
    total_action_dim=total_action_dim,
    use_amp=False,
    device=device,
)
agent.actor.load_state_dict(torch.load(cp_dir / "shared_actor.pt", map_location=device))
agent.actor.eval()

print("  shared_actor vs shared_actor (role-conditioned)...")
cr, avg_len = evaluate(env, agent, NUM_EPISODES, frame_stack, device)
print(f"    CR={cr:.3f}, len={avg_len:.1f}")

# EMA eval
ema_path = cp_dir / "ema_actor.pt"
if ema_path.exists():
    print("  EMA vs EMA...")
    agent.actor.load_state_dict(torch.load(ema_path, map_location=device))
    cr2, avg_len2 = evaluate(env, agent, NUM_EPISODES, frame_stack, device)
    print(f"    CR={cr2:.3f}, len={avg_len2:.1f}")
