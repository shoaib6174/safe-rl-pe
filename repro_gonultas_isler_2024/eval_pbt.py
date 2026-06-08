#!/usr/bin/env python3
"""Evaluation script for PBT-trained population."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import yaml

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.shared_maddpg import SharedMADDPGAgent


def evaluate_match(env, agent_p, agent_e, num_episodes, frame_stack, device):
    """Evaluate agent_p (pursuer) vs agent_e (evader)."""
    capture_count = 0
    total_len = 0
    for _ in range(num_episodes):
        obs_p, obs_e = env.reset()
        obs_p_buf = [obs_p.copy() for _ in range(frame_stack)]
        obs_e_buf = [obs_e.copy() for _ in range(frame_stack)]
        done = False
        step = 0
        while not done and step < env.max_steps:
            sp = np.concatenate(obs_p_buf[-frame_stack:])
            se = np.concatenate(obs_e_buf[-frame_stack:])
            with torch.no_grad():
                a_p = agent_p.select_action(sp, role=0, noise_sigma=0)
                a_e = agent_e.select_action(se, role=1, noise_sigma=0)
            obs_p, obs_e, _, _, done = env.step(a_p, a_e)
            obs_p_buf.append(obs_p.copy())
            obs_e_buf.append(obs_e.copy())
            step += 1
        if step < env.max_steps:
            capture_count += 1
        total_len += step
    return capture_count / num_episodes, total_len / num_episodes


def evaluate_population(env, population, num_episodes, frame_stack, device):
    """Run round-robin tournament within population."""
    pop_size = len(population)
    win_matrix = np.zeros((pop_size, pop_size))
    cr_matrix = np.zeros((pop_size, pop_size))

    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            cr, avg_len = evaluate_match(env, population[i], population[j],
                                          num_episodes, frame_stack, device)
            win_matrix[i, j] = cr
            cr_matrix[i, j] = cr

    # Population-level stats
    mean_cr = cr_matrix[cr_matrix > 0].mean() if np.any(cr_matrix > 0) else 0.0
    std_cr = cr_matrix[cr_matrix > 0].std() if np.any(cr_matrix > 0) else 0.0

    # Best agent (highest win rate as pursuer)
    win_rates = win_matrix.sum(axis=1) / (pop_size - 1)
    best_idx = int(np.argmax(win_rates))

    return {
        "win_matrix": win_matrix,
        "cr_matrix": cr_matrix,
        "mean_cr": mean_cr,
        "std_cr": std_cr,
        "win_rates": win_rates,
        "best_idx": best_idx,
    }


with open("config_pbt.yaml") as f:
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

base_dir = Path("/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024/results_pbt")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cp_dir = base_dir / f"seed_{seed}" / "checkpoints" / "episode_10000"

print(f"\n=== PBT Seed {seed} ===")
print(f"Checkpoint: {cp_dir}")

if not cp_dir.exists():
    print("ERROR: checkpoint directory not found")
    sys.exit(1)

# Load population
population_size = cfg["pbt"]["population_size"]
population = []
for i in range(population_size):
    cp_path = cp_dir / f"agent_{i}_actor.pt"
    if not cp_path.exists():
        print(f"WARNING: agent {i} checkpoint not found, skipping")
        continue

    agent = SharedMADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        use_amp=False,
        device=device,
    )
    agent.actor.load_state_dict(torch.load(cp_path, map_location=device))
    agent.actor.eval()
    population.append(agent)

if len(population) == 0:
    print("ERROR: no agents loaded")
    sys.exit(1)

print(f"Loaded {len(population)} agents")

# Population tournament
print(f"\n--- Population Round-Robin ({NUM_EPISODES} games per matchup) ---")
results = evaluate_population(env, population, NUM_EPISODES, frame_stack, device)

print(f"\nWin Rate Matrix (pursuer vs evader):")
print(f"{'':>8}", end="")
for j in range(len(population)):
    print(f"E{j:>7}", end="")
print()
for i in range(len(population)):
    print(f"P{i:>6}", end="")
    for j in range(len(population)):
        if i == j:
            print(f"{'---':>8}", end="")
        else:
            print(f"{results['cr_matrix'][i, j]:>7.3f}", end="")
    print(f"  | WR={results['win_rates'][i]:.3f}")

print(f"\nPopulation Mean CR: {results['mean_cr']:.3f} ± {results['std_cr']:.3f}")
print(f"Best Agent: {results['best_idx']} (WR={results['win_rates'][results['best_idx']]:.3f})")

# Balanced agents: those with win_rate closest to 0.5
balanced_scores = np.abs(results['win_rates'] - 0.5)
balanced_idx = int(np.argmin(balanced_scores))
print(f"Most Balanced Agent: {balanced_idx} (WR={results['win_rates'][balanced_idx]:.3f})")
