import sys
sys.path.insert(0, "/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024")
import numpy as np, torch, yaml
from pathlib import Path
from envs.pursuit_evasion_env import PursuitEvasionEnv
from agents.maddpg import MADDPGAgent

def eval_fast(env, p, e, n, fs, dev):
    cap = 0
    tl = 0
    for _ in range(n):
        op, oe = env.reset()
        bp = [op.copy() for _ in range(fs)]
        be = [oe.copy() for _ in range(fs)]
        done = False
        step = 0
        while not done and step < env.max_steps:
            with torch.no_grad():
                ap = p.actor(torch.FloatTensor(np.concatenate(bp[-fs:])).unsqueeze(0).to(dev)).cpu().numpy()[0]
                ae = e.actor(torch.FloatTensor(np.concatenate(be[-fs:])).unsqueeze(0).to(dev)).cpu().numpy()[0]
            op, oe, _, _, done = env.step(ap, ae)
            bp.append(op.copy())
            be.append(oe.copy())
            step += 1
        if step < env.max_steps:
            cap += 1
        tl += step
    return cap / n, tl / n

cfg = yaml.safe_load(open("config_symmetric.yaml"))
env = PursuitEvasionEnv(cfg)
fs = cfg["maddpg"]["frame_stack"]
obs_dim = 11 * fs
dev = "cuda" if torch.cuda.is_available() else "cpu"

seed = 5
cp = Path(f"./results_symmetric/seed_{seed}/checkpoints/episode_5000")
print(f"=== Seed {seed} ===")
p = MADDPGAgent(obs_dim, 2, obs_dim * 2, 4, False, dev)
e = MADDPGAgent(obs_dim, 2, obs_dim * 2, 4, False, dev)
p.actor.load_state_dict(torch.load(cp / "pursuer_actor.pt", map_location=dev))
e.actor.load_state_dict(torch.load(cp / "evader_actor.pt", map_location=dev))
p.actor.eval()
e.actor.eval()
for name, mode in [("vs_random", 0), ("vs_selfplay", 1), ("ema_vs_ema", 2)]:
    if mode == 2:
        ema_p = cp / "ema_pursuer.pt"
        ema_e = cp / "ema_evader.pt"
        if ema_p.exists() and ema_e.exists():
            p.actor.load_state_dict(torch.load(ema_p, map_location=dev))
            e.actor.load_state_dict(torch.load(ema_e, map_location=dev))
        else:
            print(f"  {name}: no EMA checkpoints")
            continue
    cr, ln = eval_fast(env, p, e, 50, fs, dev)
    print(f"  {name}: CR={cr:.3f}, len={ln:.1f}")
