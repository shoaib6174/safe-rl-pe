"""Re-evaluate borderline symmetric seeds at N=300 (frozen protocol, det, skip=1).

N=50 unseeded evals (S115) leave seeds within ~2 SE of the balanced-band edges
[0.45, 0.55] at misclassification risk. Seeds flagged borderline: 12, 16, 20,
21, 30. This script re-runs pursuer_vs_{selfplay,random,greedy} at N=300 for
those seeds, using the same deterministic frozen protocol as run_evaluation.
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
from evaluation.evaluate import evaluate_agent, random_policy, greedy_evader_policy, load_policy

REPRO = Path(__file__).parent
N_EPISODES = 300
BORDERLINE = [12, 16, 20, 21, 30]


def main():
    out = {}
    for seed in BORDERLINE:
        with open(REPRO / f"config_symmetric_seed_{seed}.yaml") as f:
            config = yaml.safe_load(f)
        frame_stack = config["maddpg"]["frame_stack"]
        obs_dim = 11 * frame_stack  # use_belief=false
        ckpt = REPRO / "results_symmetric" / f"seed_{seed}" / "checkpoints" / "episode_5000"

        env = PursuitEvasionEnv(config)
        pursuer = MADDPGAgent(obs_dim=obs_dim, action_dim=2, total_obs_dim=obs_dim * 2,
                              total_action_dim=4, use_amp=False, device="cpu")
        evader = MADDPGAgent(obs_dim=obs_dim, action_dim=2, total_obs_dim=obs_dim * 2,
                             total_action_dim=4, use_amp=False, device="cpu")
        load_policy(pursuer, ckpt / "pursuer_actor.pt")
        load_policy(evader, ckpt / "evader_actor.pt")

        p_pol = lambda obs: pursuer.select_action(obs, noise_sigma=0.0)
        e_pol = lambda obs: evader.select_action(obs, noise_sigma=0.0)

        res = {}
        res["pursuer_vs_selfplay"] = evaluate_agent(env, p_pol, e_pol, N_EPISODES, frame_stack)
        res["pursuer_vs_random"] = evaluate_agent(env, p_pol, random_policy, N_EPISODES, frame_stack)
        res["pursuer_vs_greedy"] = evaluate_agent(env, p_pol, greedy_evader_policy, N_EPISODES, frame_stack)
        out[str(seed)] = res
        cr = res["pursuer_vs_selfplay"]["capture_rate"]
        band = "BALANCED" if 0.45 <= cr <= 0.55 else "not balanced"
        print(f"seed {seed}: selfplay={cr:.3f} ({band}) "
              f"random={res['pursuer_vs_random']['capture_rate']:.3f} "
              f"greedy={res['pursuer_vs_greedy']['capture_rate']:.3f}", flush=True)

    out_path = REPRO / "results_symmetric" / "eval_borderline_n300.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
