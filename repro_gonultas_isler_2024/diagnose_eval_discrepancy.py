"""Diagnose training-log CR vs frozen-opponent eval discrepancy (seeds 9, 16).

Training-log CR (TensorBoard episode/capture_rate) differs from frozen-opponent
tournament eval. Candidate mechanisms isolated here, one variable at a time:

  H1  Snapshot artifact: logged CR windows are ~2 episodes -> last value (0.500)
      is noise; proper estimate is the mean over many windows.
  H2  Exploration noise: training acts with Gaussian sigma=0.1 on BOTH agents;
      frozen eval is deterministic (noise_sigma=0.0).
  H3  Frame-skip mismatch: training holds each action for frame_skip=2 sim
      steps (obs stacked at decision cadence); evaluate_agent re-acts and
      re-stacks every sim step.
  H4  Live co-adapting opponent: training opponent drifts continuously.
      Proxied by evaluating final-vs-near-final (episode_4750) checkpoints.

Conditions per seed (50 episodes each, final = episode_5000 checkpoint):
  A  det,  skip=1  -> must reproduce S115 frozen eval (sanity check)
  B  det,  skip=2  -> isolates control-frequency/obs-cadence effect (H3)
  C  noisy, skip=1 -> isolates exploration noise (H2)
  D  noisy, skip=2 -> full training-like measurement, frozen nets (H2+H3)
  E  det,  skip=1, one role from episode_4750 -> opponent-drift proxy (H4)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent

REPRO = Path(__file__).parent
# 300 episodes: env resets use an UNSEEDED np_random stream, so 50-episode
# evals carry large Monte Carlo error (Wilson CI for 12/50 ~ [0.14, 0.38]).
N_EPISODES = 300
SEEDS = [9, 16]


def make_agent(obs_dim, action_dim=2):
    return MADDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_obs_dim=obs_dim * 2,
        total_action_dim=action_dim * 2,
        use_amp=False,
        device="cpu",
    )


def load_actor(agent, path):
    agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
    agent.actor.eval()


def run_matchup(env, pursuer, evader, n_episodes, frame_stack, frame_skip, noise_sigma):
    """Evaluate with training-fidelity controls.

    frame_skip: hold each action for this many sim steps (obs stacked at
        decision cadence, matching VectorizedEnvRunner when >1).
    noise_sigma: Gaussian exploration noise on BOTH agents (0.0 = deterministic).
    """
    captures = 0
    lengths = []
    for _ in range(n_episodes):
        obs_p, obs_e = env.reset()
        buf_p = [obs_p.copy() for _ in range(frame_stack)]
        buf_e = [obs_e.copy() for _ in range(frame_stack)]
        done = False
        step = 0
        while not done and step < env.max_steps:
            stacked_p = np.concatenate(buf_p[-frame_stack:])
            stacked_e = np.concatenate(buf_e[-frame_stack:])
            a_p = pursuer.select_action(stacked_p, noise_sigma=noise_sigma)
            a_e = evader.select_action(stacked_e, noise_sigma=noise_sigma)
            for _ in range(frame_skip):
                obs_p, obs_e, r_p, r_e, done = env.step(a_p, a_e)
                step += 1
                if done:
                    break
            buf_p.append(obs_p.copy())
            buf_e.append(obs_e.copy())
        if step < env.max_steps:
            captures += 1
        lengths.append(step)
    return captures / n_episodes, float(np.mean(lengths))


def parse_training_log_cr(log_path, last_n_episodes=500, total_episodes=5000):
    """Windowed CR values from console log; each window ~2 episodes."""
    vals = []
    ep_nums = []
    with open(log_path) as f:
        for line in f:
            if "CR=" in line:
                try:
                    ep = int(line.split("|")[0].replace("Ep", "").strip())
                    cr = float(line.split("CR=")[1].split("|")[0].strip())
                except (ValueError, IndexError):
                    continue
                ep_nums.append(ep)
                vals.append(cr)
    cutoff = total_episodes - last_n_episodes
    tail = [v for e, v in zip(ep_nums, vals) if e >= cutoff]
    return np.array(vals), np.array(tail)


def main():
    out = {}

    # --- H1: training-log window statistics ---
    for seed in SEEDS:
        log = REPRO / f"symmetric_seed_{seed}.log"
        all_w, tail_w = parse_training_log_cr(log)
        out[f"seed_{seed}"] = {
            "trainlog_windows_total": len(all_w),
            "trainlog_mean_all": float(all_w.mean()),
            "trainlog_mean_last500ep": float(tail_w.mean()),
            "trainlog_std_last500ep": float(tail_w.std()),
            "trainlog_frac_balanced_windows_last500ep": float(
                np.mean((tail_w >= 0.45) & (tail_w <= 0.55))
            ),
            "trainlog_final_value": float(all_w[-1]),
        }

    # --- H2/H3/H4: controlled matchups ---
    for seed in SEEDS:
        with open(REPRO / f"config_symmetric_seed_{seed}.yaml") as f:
            config = yaml.safe_load(f)
        frame_stack = config["maddpg"]["frame_stack"]
        obs_dim = 11 * frame_stack  # use_belief=false
        ckpt = REPRO / "results_symmetric" / f"seed_{seed}" / "checkpoints"

        env = PursuitEvasionEnv(config)

        p_final, e_final = make_agent(obs_dim), make_agent(obs_dim)
        load_actor(p_final, ckpt / "episode_5000" / "pursuer_actor.pt")
        load_actor(e_final, ckpt / "episode_5000" / "evader_actor.pt")

        p_4750, e_4750 = make_agent(obs_dim), make_agent(obs_dim)
        load_actor(p_4750, ckpt / "episode_4750" / "pursuer_actor.pt")
        load_actor(e_4750, ckpt / "episode_4750" / "evader_actor.pt")

        conditions = {
            "A_det_skip1": (p_final, e_final, 1, 0.0),
            "B_det_skip2": (p_final, e_final, 2, 0.0),
            "C_noisy_skip1": (p_final, e_final, 1, 0.1),
            "D_noisy_skip2": (p_final, e_final, 2, 0.1),
            "E1_pfinal_vs_e4750": (p_final, e_4750, 1, 0.0),
            "E2_p4750_vs_efinal": (p_4750, e_final, 1, 0.0),
        }

        for name, (p, e, skip, sigma) in conditions.items():
            np.random.seed(1000 + seed)  # same episode inits across conditions
            torch.manual_seed(1000 + seed)
            cr, ml = run_matchup(env, p, e, N_EPISODES, frame_stack, skip, sigma)
            out[f"seed_{seed}"][name] = {"capture_rate": cr, "mean_length": ml}
            print(f"seed {seed} {name}: CR={cr:.3f} len={ml:.1f}", flush=True)

    out_path = REPRO / "results_symmetric" / "eval_discrepancy_diagnosis_n300.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
