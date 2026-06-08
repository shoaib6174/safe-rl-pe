#!/usr/bin/env python3
"""Evaluate symmetric seeds 3,4,5,6,7,8."""
import sys
sys.path.insert(0, "/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024")

import numpy as np
import torch
import yaml
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent


def evaluate_fast(env, pursuer, evader, num_episodes, frame_stack, device):
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
                a_p = pursuer.actor(torch.FloatTensor(stacked_p).unsqueeze(0).to(device)).cpu().numpy()[0]
                a_e = evader.actor(torch.FloatTensor(stacked_e).unsqueeze(0).to(device)).cpu().numpy()[0]
            obs_p, obs_e, r_p, r_e, done = env.step(a_p, a_e)
            obs_p_buffer.append(obs_p.copy())
            obs_e_buffer.append(obs_e.copy())
            step += 1
        if step < env.max_steps:
            capture_count += 1
        total_len += step
    return capture_count / num_episodes, total_len / num_episodes


with open("config_symmetric.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_EPISODES = 50
frame_stack = cfg["maddpg"]["frame_stack"]
raw_obs_dim = 11
obs_dim = raw_obs_dim * frame_stack
action_dim = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

env = PursuitEvasionEnv(cfg)

base_dir = Path("/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024/results_symmetric")

for seed in [3, 4, 5, 6, 7, 8]:
    cp_dir = base_dir / f"seed_{seed}" / "checkpoints" / "episode_5000"
    print(f"\n=== Seed {seed} ===")
    print(f"Checkpoint: {cp_dir}")

    if not (cp_dir / "pursuer_actor.pt").exists():
        print(f"  ERROR: checkpoint not found")
        continue

    pursuer = MADDPGAgent(obs_dim=obs_dim, action_dim=action_dim,
                          total_obs_dim=obs_dim*2, total_action_dim=action_dim*2,
                          use_amp=False, device=device)
    evader = MADDPGAgent(obs_dim=obs_dim, action_dim=action_dim,
                         total_obs_dim=obs_dim*2, total_action_dim=action_dim*2,
                         use_amp=False, device=device)
    pursuer.actor.load_state_dict(torch.load(cp_dir / "pursuer_actor.pt", map_location=device))
    evader.actor.load_state_dict(torch.load(cp_dir / "evader_actor.pt", map_location=device))
    pursuer.actor.eval()
    evader.actor.eval()

    print("  pursuer_vs_random...")
    cr, avg_len = evaluate_fast(env, pursuer, evader, NUM_EPISODES, frame_stack, device)
    print(f"    CR={cr:.3f}, len={avg_len:.1f}")

    print("  pursuer_vs_selfplay...")
    cr2, avg_len2 = evaluate_fast(env, pursuer, evader, NUM_EPISODES, frame_stack, device)
    print(f"    CR={cr2:.3f}, len={avg_len2:.1f}")

    ema_p_path = cp_dir / "ema_pursuer.pt"
    ema_e_path = cp_dir / "ema_evader.pt"
    if ema_p_path.exists() and ema_e_path.exists():
        print("  ema_pursuer_vs_ema_evader...")
        pursuer.actor.load_state_dict(torch.load(ema_p_path, map_location=device))
        evader.actor.load_state_dict(torch.load(ema_e_path, map_location=device))
        cr3, avg_len3 = evaluate_fast(env, pursuer, evader, NUM_EPISODES, frame_stack, device)
        print(f"    CR={cr3:.3f}, len={avg_len3:.1f}")
