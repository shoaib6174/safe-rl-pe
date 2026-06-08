#!/usr/bin/env python3
"""Debug evaluation for symmetric point-mass."""
import sys
sys.path.insert(0, "/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024")

import torch
import yaml
from pathlib import Path

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent

print("Loading config...")
with open("config_symmetric.yaml") as f:
    cfg = yaml.safe_load(f)

print("Creating env...")
env = PursuitEvasionEnv(cfg)

raw_obs_dim = 11
frame_stack = cfg["maddpg"]["frame_stack"]
obs_dim = raw_obs_dim * frame_stack
action_dim = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Creating agents...")
pursuer = MADDPGAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    total_obs_dim=obs_dim * 2,
    total_action_dim=action_dim * 2,
    use_amp=False,
    device=device,
)
evader = MADDPGAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    total_obs_dim=obs_dim * 2,
    total_action_dim=action_dim * 2,
    use_amp=False,
    device=device,
)

seed = 0
cp_dir = Path(f"./repro_gonultas_isler_2024/results_symmetric/seed_{seed}/seed_{seed}/checkpoints/episode_5000")
print(f"Loading checkpoint from {cp_dir}")

pursuer.actor.load_state_dict(torch.load(cp_dir / "pursuer_actor.pt", map_location=device))
evader.actor.load_state_dict(torch.load(cp_dir / "evader_actor.pt", map_location=device))
pursuer.actor.eval()
evader.actor.eval()

print("Running 1 episode...")
obs_p, obs_e = env.reset()
obs_p_buffer = [obs_p.copy() for _ in range(frame_stack)]
obs_e_buffer = [obs_e.copy() for _ in range(frame_stack)]
done = False
step = 0
while not done and step < env.max_steps:
    stacked_p = torch.FloatTensor(np.concatenate(obs_p_buffer[-frame_stack:])).unsqueeze(0).to(device)
    stacked_e = torch.FloatTensor(np.concatenate(obs_e_buffer[-frame_stack:])).unsqueeze(0).to(device)
    with torch.no_grad():
        action_p = pursuer.actor(stacked_p).cpu().numpy()[0]
        action_e = evader.actor(stacked_e).cpu().numpy()[0]
    obs_p, obs_e, r_p, r_e, done = env.step(action_p, action_e)
    obs_p_buffer.append(obs_p.copy())
    obs_e_buffer.append(obs_e.copy())
    step += 1
    if step % 100 == 0:
        print(f"  step {step}")

print(f"Episode done: step={step}, done={done}")
print("Debug complete.")
