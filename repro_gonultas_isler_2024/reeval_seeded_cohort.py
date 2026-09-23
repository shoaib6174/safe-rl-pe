"""Seeded full-cohort re-evaluation of symmetric seeds 9-33 (frozen protocol).

Problem found in S117: env resets use np.random.default_rng() with OS entropy
(pursuit_evasion_env.py:74), so eval episode inits are never reproducible and
per-seed CR carries Monte Carlo error even at N=300 (seed 16: 0.503 vs 0.563
across two unseeded N=300 draws).

Fix: paired, seeded evaluation — every seed is evaluated on the SAME fixed set
of episode initial conditions (env.reset(seed=INIT0+i), i=0..N-1). Policies
deterministic (noise_sigma=0.0), frame_skip=1 (paper frozen protocol, §III-H).
With seeded inits and deterministic policies, results are exactly reproducible.

Primary balance criterion: pursuer_vs_selfplay CR in [0.45, 0.55].
Seeds 0-8 checkpoints are on niro-2 (offline) and cannot be re-evaluated;
their N=50 unseeded classifications stand with an explicit caveat.
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
N_EPISODES = 500
INIT0 = 700000  # fixed init-seed base, shared across all seeds (paired design)
SEEDS = list(range(9, 34))


def run_seeded_selfplay(env, pursuer, evader, n_episodes, frame_stack):
    captures = 0
    lengths = []
    for i in range(n_episodes):
        obs_p, obs_e = env.reset(seed=INIT0 + i)
        buf_p = [obs_p.copy() for _ in range(frame_stack)]
        buf_e = [obs_e.copy() for _ in range(frame_stack)]
        done = False
        step = 0
        while not done and step < env.max_steps:
            a_p = pursuer.select_action(np.concatenate(buf_p[-frame_stack:]), noise_sigma=0.0)
            a_e = evader.select_action(np.concatenate(buf_e[-frame_stack:]), noise_sigma=0.0)
            obs_p, obs_e, r_p, r_e, done = env.step(a_p, a_e)
            step += 1
            buf_p.append(obs_p.copy())
            buf_e.append(obs_e.copy())
        if step < env.max_steps:
            captures += 1
        lengths.append(step)
    return captures / n_episodes, float(np.mean(lengths))


def main():
    out = {}
    for seed in SEEDS:
        with open(REPRO / f"config_symmetric_seed_{seed}.yaml") as f:
            config = yaml.safe_load(f)
        frame_stack = config["maddpg"]["frame_stack"]
        obs_dim = 11 * frame_stack
        ckpt = REPRO / "results_symmetric" / f"seed_{seed}" / "checkpoints" / "episode_5000"

        env = PursuitEvasionEnv(config)
        mk = lambda: MADDPGAgent(obs_dim=obs_dim, action_dim=2, total_obs_dim=obs_dim * 2,
                                 total_action_dim=4, use_amp=False, device="cpu")
        pursuer, evader = mk(), mk()
        pursuer.actor.load_state_dict(torch.load(ckpt / "pursuer_actor.pt", map_location="cpu"))
        pursuer.actor.eval()
        evader.actor.load_state_dict(torch.load(ckpt / "evader_actor.pt", map_location="cpu"))
        evader.actor.eval()

        cr, ml = run_seeded_selfplay(env, pursuer, evader, N_EPISODES, frame_stack)
        band = "BALANCED" if 0.45 <= cr <= 0.55 else ""
        out[str(seed)] = {"capture_rate": cr, "mean_length": ml,
                          "n_episodes": N_EPISODES, "init_seed_base": INIT0}
        print(f"seed {seed:2d}: CR={cr:.3f} len={ml:.0f} {band}", flush=True)

    out_path = REPRO / "results_symmetric" / "eval_seeds_9_33_seeded_n500.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    balanced = [s for s, v in out.items() if 0.45 <= v["capture_rate"] <= 0.55]
    print(f"\nBalanced (seeded N={N_EPISODES}): {len(balanced)}/25 -> {balanced}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
