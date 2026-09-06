"""Advantage-weighted regression (Peng, Kumar, Zhang & Levine 2019) on
Monte-Carlo returns — the retry round's family-independent arm (R-D).

Why this and not another actor-critic: every Tier-0 arm that bootstrapped a
critic either collapsed (D82's TD drift) or had to be pinned to the clone to
survive. AWR bootstraps NOTHING. The baseline V(s) is a regression onto the
Monte-Carlo return G_t = gamma^(L-t)·R of the episode the step came from
(`pe2.derisk.mc_returns`), and the actor is updated by weighted supervised
regression onto the actions it actually executed, with weights
exp((G_t − V(s_t)) / lam) clipped at w_max. Weights favour the actions that
led to better-than-expected outcomes; a flipped sign (the round's control)
favours the worse ones and must degrade the policy.

The actor is the SB3 SAC actor inside the clone's own model object, updated
the way `train_bc_actor` updates it (regression of tanh(mu), which is what
predict(deterministic=True) executes), so checkpoints, probes and the
evaluator are the ones every other arm uses. Rollouts are STOCHASTIC
(predict(deterministic=False)) — the training-time regime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import mc_returns, seed_torch
from pe2.exploitability import SingleAgentWrapper
from pe2.obs import obs_dim

LAM = 0.5  # temperature on our ±1 return range (AWR's 0.05 is for returns of tens)
W_MAX = 20.0  # AWR's weight clip


@dataclass(frozen=True)
class AwrConfig:
    lam: float = LAM
    w_max: float = W_MAX
    advantage_sign: float = 1.0  # −1.0 is the negative control (R-D-neg)
    gamma: float = 0.999
    initial_episodes: int = 300  # the offline phase: the clone's own rollouts
    buffer_episodes: int = 300  # FIFO, in episodes
    steps_per_iter: int = 50_000  # online: at least this many fresh steps per iteration
    value_epochs: int = 20
    value_lr: float = 1e-3
    actor_epochs: int = 10
    actor_lr: float = 3e-4
    batch_size: int = 512
    hidden: int = 256


@dataclass
class Episode:
    obs: np.ndarray  # (T, obs_dim) float32
    actions: np.ndarray  # (T, 2) float32 — the EXECUTED Box actions in [−1, 1]
    returns: np.ndarray  # (T,) float32 — G_t under the terminal-only reward
    won: bool
    length: int
    seed: int


def rollout_episode(model, wrapper, role, seed, gamma, deterministic=False) -> Episode:
    obs, _ = wrapper.reset(seed=seed)
    obs_l, act_l = [], []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_l.append(np.asarray(obs, dtype=np.float32))
        act_l.append(np.asarray(action, dtype=np.float32))
        obs, _reward, terminated, truncated, info = wrapper.step(action)
        if terminated or truncated:
            break
    length = wrapper.env.current_step
    won = (info["outcome"] == "capture") == (role == "pursuer")
    returns = np.asarray(
        mc_returns(length, 1.0 if won else -1.0, gamma), dtype=np.float32
    )
    return Episode(np.stack(obs_l), np.stack(act_l), returns, won, length, seed)


def collect(
    model, wrapper, role, seed0, min_steps, gamma, min_episodes=1
) -> list[Episode]:
    """Episodes on seeds seed0, seed0+1, … until >= min_steps AND >= min_episodes."""
    episodes, steps, i = [], 0, 0
    while steps < min_steps or len(episodes) < min_episodes:
        ep = rollout_episode(model, wrapper, role, seed0 + i, gamma)
        i += 1
        episodes.append(ep)
        steps += ep.length
    return episodes


def advantage_weights(returns, values, lam, w_max, sign) -> np.ndarray:
    adv = sign * (np.asarray(returns) - np.asarray(values))
    return np.minimum(np.exp(adv / lam), w_max)


def make_value_net(obs_dim_: int, hidden: int, device):
    from torch import nn  # lazy: optional dependency

    return nn.Sequential(
        nn.Linear(obs_dim_, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)


def fit_value(value_net, opt, obs, returns, epochs, batch_size, rng) -> list[float]:
    import torch

    n = len(obs)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i : i + batch_size], device=obs.device)
            loss = torch.nn.functional.mse_loss(
                value_net(obs[idx]).squeeze(-1), returns[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
    return losses


def awr_actor_update(
    model, opt, obs, actions, weights, epochs, batch_size, rng
) -> list[float]:
    """Weighted regression of tanh(mu(s)) onto the executed actions —
    `train_bc_actor`'s step with per-sample weights (normalised per batch)."""
    import torch

    n = len(obs)
    losses = []
    model.actor.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i : i + batch_size], device=obs.device)
            mean_actions, _log_std, _kwargs = model.actor.get_action_dist_params(
                obs[idx]
            )
            se = ((torch.tanh(mean_actions) - actions[idx]) ** 2).sum(dim=1)
            loss = (weights[idx] * se).sum() / weights[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
    model.actor.eval()
    return losses


def train_awr(
    cfg: EnvConfig,
    role: str,
    opponent_factory,
    clone_path,
    total_timesteps: int,
    seed: int,
    awr: AwrConfig,
    probe_every: int,
    probe,
    device: str = "auto",
    rollout_seed0: int = 820_000,
):
    """Offline AWR on the clone's rollouts, then iterated online AWR, until
    `total_timesteps` env steps have been collected (the offline rollouts
    count). `probe(model, grid_steps)` fires the first time the cumulative
    count reaches each multiple of `probe_every` (≤ total). Returns
    (model, stats)."""
    import torch
    from stable_baselines3 import SAC

    rng = np.random.default_rng(seed)
    model = SAC.load(clone_path, device=device)
    # SEED AFTER THE LOAD, never before. `SAC.load` re-applies the seed baked
    # into the zip through `_setup_model`, so seeding first is silently
    # overwritten and the declared seed never governs a single sampled
    # action (see scripts/derisk_t0_pilots.py's identical fix).
    torch_seed = seed_torch(seed)
    dev = model.device
    wrapper = SingleAgentWrapper(cfg, role, opponent_factory)
    value_net = make_value_net(obs_dim(cfg), awr.hidden, dev)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=awr.value_lr)
    a_opt = torch.optim.Adam(model.actor.parameters(), lr=awr.actor_lr)

    buffer: list[Episode] = []
    steps = 0
    n_episodes = 0
    positives = 0
    next_seed = rollout_seed0
    next_probe = probe_every
    probe_marks: list[dict] = []
    iterations: list[dict] = []

    def absorb(episodes: list[Episode]) -> None:
        nonlocal steps, n_episodes, positives, next_seed
        for ep in episodes:
            steps += ep.length
            n_episodes += 1
            positives += int(ep.won)
        next_seed += len(episodes)
        buffer.extend(episodes)
        del buffer[: max(0, len(buffer) - awr.buffer_episodes)]

    def update(phase: str) -> dict:
        obs = torch.as_tensor(np.concatenate([e.obs for e in buffer]), device=dev)
        acts = torch.as_tensor(np.concatenate([e.actions for e in buffer]), device=dev)
        rets_np = np.concatenate([e.returns for e in buffer])
        rets = torch.as_tensor(rets_np, device=dev)
        v_losses = fit_value(
            value_net, v_opt, obs, rets, awr.value_epochs, awr.batch_size, rng
        )
        with torch.no_grad():
            values = value_net(obs).squeeze(-1).cpu().numpy()
        w = advantage_weights(rets_np, values, awr.lam, awr.w_max, awr.advantage_sign)
        a_losses = awr_actor_update(
            model,
            a_opt,
            obs,
            acts,
            torch.as_tensor(w, dtype=torch.float32, device=dev),
            awr.actor_epochs,
            awr.batch_size,
            rng,
        )
        return {
            "phase": phase,
            "steps": steps,
            "n_samples": len(obs),
            "buffer_episodes": len(buffer),
            "value_loss_first": v_losses[0],
            "value_loss_last": v_losses[-1],
            "actor_loss_first": a_losses[0],
            "actor_loss_last": a_losses[-1],
            "advantage_mean": float((rets_np - values).mean()),
            "advantage_std": float((rets_np - values).std()),
            "weight_mean": float(w.mean()),
            "weight_clipped_fraction": float((w >= awr.w_max).mean()),
        }

    def fire_probes(iteration: int) -> None:
        nonlocal next_probe
        while next_probe <= total_timesteps and steps >= next_probe:
            probe(model, next_probe)
            probe_marks.append(
                {"grid": next_probe, "actual": steps, "iteration": iteration}
            )
            next_probe += probe_every

    absorb(
        collect(
            model,
            wrapper,
            role,
            next_seed,
            0,
            awr.gamma,
            min_episodes=awr.initial_episodes,
        )
    )
    iterations.append(update("offline"))
    fire_probes(0)
    while steps < total_timesteps:
        absorb(collect(model, wrapper, role, next_seed, awr.steps_per_iter, awr.gamma))
        iterations.append(update("online"))
        fire_probes(len(iterations) - 1)

    stats = {
        "torch_seed": torch_seed,
        "seed": seed,
        "rollout_seed0": rollout_seed0,
        "total_steps": steps,
        "n_episodes": n_episodes,
        "positives": positives,
        "positives_per_1M": positives / steps * 1e6,
        "probe_marks": probe_marks,
        "iterations": iterations,
        "config": asdict(awr),
    }
    return model, stats
