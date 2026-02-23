"""Stage 1: BiMDN Pre-training — collect data, train, validate.

Collects (obs_history, true_opponent_position) pairs from environment
rollouts, pre-trains the BiMDN belief encoder, and validates against
Gate 1 criteria.

Usage:
    ./venv/bin/python scripts/phase3_pretrain_bimdn.py \
        --n_episodes 500 --epochs 50 --seed 42 \
        --output results/stage1
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from agents.bimdn import BiMDN
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.wrappers import SingleAgentPEWrapper


def collect_data(n_episodes: int, history_length: int = 10, seed: int = 42):
    """Collect (obs_history, true_opponent_pos) from env rollouts.

    Uses random actions for both agents to get diverse states.
    """
    print(f"Collecting data from {n_episodes} episodes...")
    env = PursuitEvasionEnv()
    single_env = SingleAgentPEWrapper(env, role="pursuer")
    partial_env = PartialObsWrapper(single_env, role="pursuer",
                                    history_length=history_length)

    obs_histories = []
    target_positions = []
    n_visible = 0
    n_lost = 0

    np.random.seed(seed)
    for ep in range(n_episodes):
        obs, _ = partial_env.reset(seed=seed + ep)
        done = False
        while not done:
            action = partial_env.action_space.sample()
            obs, reward, terminated, truncated, info = partial_env.step(action)
            done = terminated or truncated

            # Record observation history and true opponent position
            obs_hist = obs["obs_history"]  # [K, 43]
            true_opp_pos = partial_env._base_env.evader_state[:2].copy()

            obs_histories.append(obs_hist)
            target_positions.append(true_opp_pos)

            # Count visible vs lost
            if obs["state"][5] >= 0:  # d_to_opp >= 0 means detected
                n_visible += 1
            else:
                n_lost += 1

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}, "
                  f"samples: {len(obs_histories)}, "
                  f"visible: {n_visible}, lost: {n_lost}")

    partial_env.close()

    obs_hist_arr = np.array(obs_histories, dtype=np.float32)
    target_pos_arr = np.array(target_positions, dtype=np.float32)

    print(f"Collected {len(obs_hist_arr)} samples "
          f"(visible: {n_visible}, lost: {n_lost}, "
          f"ratio: {n_visible/(n_visible+n_lost):.1%})")

    return obs_hist_arr, target_pos_arr


def pretrain_bimdn(obs_histories, target_positions, epochs, lr,
                   hidden_dim=64, n_mixtures=5, latent_dim=32,
                   val_split=0.1, seed=42):
    """Pre-train BiMDN on collected data."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    bimdn = BiMDN(obs_dim=43, hidden_dim=hidden_dim,
                  n_mixtures=n_mixtures, latent_dim=latent_dim).to(device)

    # Train/val split
    n = len(obs_histories)
    n_val = max(int(n * val_split), 1)
    n_train = n - n_val
    indices = np.random.RandomState(seed).permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Keep data on CPU, move to GPU per-batch to avoid OOM
    train_obs_np = obs_histories[train_idx]
    train_target_np = target_positions[train_idx]
    val_obs_np = obs_histories[val_idx]
    val_target_np = target_positions[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(train_obs_np), torch.tensor(train_target_np),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = TensorDataset(
        torch.tensor(val_obs_np), torch.tensor(val_target_np),
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    optimizer = Adam(bimdn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    print(f"Training BiMDN: {n_train} train, {n_val} val, {epochs} epochs")
    for epoch in range(epochs):
        # Training
        bimdn.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_obs, batch_target in train_loader:
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)
            _, (pi, mu, sigma) = bimdn(batch_obs)
            loss = bimdn.belief_loss(pi, mu, sigma, batch_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bimdn.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)
        scheduler.step()

        # Validation (batched to avoid OOM)
        bimdn.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for vb_obs, vb_target in val_loader:
                vb_obs = vb_obs.to(device)
                vb_target = vb_target.to(device)
                _, (pi_v, mu_v, sigma_v) = bimdn(vb_obs)
                val_loss_sum += bimdn.belief_loss(
                    pi_v, mu_v, sigma_v, vb_target,
                ).item()
                val_batches += 1
        val_losses.append(val_loss_sum / val_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_losses[-1]:.4f}, "
                  f"val_loss={val_losses[-1]:.4f}")

    return bimdn, train_losses, val_losses


def validate_bimdn(bimdn, obs_histories, target_positions, device,
                   batch_size=512):
    """Validate BiMDN against Gate 1 criteria."""
    print("\n=== Gate 1 Validation ===")
    bimdn.eval()
    results = {}

    # Split into visible vs lost based on last observation's d_to_opp
    visible_mask = obs_histories[:, -1, 5] >= 0  # d_to_opp in last step (index 5: x,y,theta,v,omega,d_to_opp,...)
    lost_mask = ~visible_mask

    # Process in batches to avoid OOM
    all_predicted = []
    all_n_eff = []
    n = len(obs_histories)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_obs = torch.tensor(obs_histories[start:end]).to(device)
            _, (pi_b, mu_b, sigma_b) = bimdn(batch_obs)
            pred = (pi_b.unsqueeze(-1) * mu_b).sum(dim=1)  # [B, 2]
            n_eff_b = bimdn.effective_n_components(pi_b)     # [B]
            all_predicted.append(pred.cpu())
            all_n_eff.append(n_eff_b.cpu())

    predicted_pos = torch.cat(all_predicted, dim=0)  # [N, 2]
    n_eff = torch.cat(all_n_eff, dim=0)              # [N]
    target_t = torch.tensor(target_positions)

    # 1. RMSE for visible targets
    if visible_mask.sum() > 0:
        vis_rmse = torch.sqrt(
            ((predicted_pos[visible_mask] - target_t[visible_mask]) ** 2)
            .sum(dim=-1).mean()
        ).item()
    else:
        vis_rmse = float("nan")
    results["in_fov_rmse"] = vis_rmse
    print(f"  In-FOV RMSE: {vis_rmse:.2f}m (target: < 2.0m, hard fail: > 4.0m)")

    if lost_mask.sum() > 0:
        lost_rmse = torch.sqrt(
            ((predicted_pos[lost_mask] - target_t[lost_mask]) ** 2)
            .sum(dim=-1).mean()
        ).item()
    else:
        lost_rmse = float("nan")
    results["out_fov_rmse"] = lost_rmse
    print(f"  Out-FOV RMSE: {lost_rmse:.2f}m (target: < 5.0m, hard fail: > 7.0m)")

    # 2. n_eff (effective number of components)
    if visible_mask.sum() > 0:
        n_eff_vis = n_eff[visible_mask].mean().item()
    else:
        n_eff_vis = float("nan")
    results["n_eff_visible"] = n_eff_vis
    print(f"  n_eff (visible): {n_eff_vis:.2f} (target: ~1.0, hard fail: > 3.0)")

    if lost_mask.sum() > 0:
        n_eff_lost = n_eff[lost_mask].mean().item()
    else:
        n_eff_lost = float("nan")
    results["n_eff_lost"] = n_eff_lost
    print(f"  n_eff (lost): {n_eff_lost:.2f} (target: > 1.5, hard fail: < 1.1)")

    # 3. Gate pass/fail
    gate_passed = True
    failures = []

    if vis_rmse > 4.0:
        gate_passed = False
        failures.append(f"In-FOV RMSE {vis_rmse:.2f} > 4.0 (HARD FAIL)")
    elif vis_rmse > 2.0:
        print(f"  WARNING: In-FOV RMSE {vis_rmse:.2f} > 2.0 (soft target)")

    if lost_rmse > 7.0:
        gate_passed = False
        failures.append(f"Out-FOV RMSE {lost_rmse:.2f} > 7.0 (HARD FAIL)")
    elif lost_rmse > 5.0:
        print(f"  WARNING: Out-FOV RMSE {lost_rmse:.2f} > 5.0 (soft target)")

    if n_eff_vis > 3.0:
        gate_passed = False
        failures.append(f"n_eff_visible {n_eff_vis:.2f} > 3.0 (HARD FAIL)")

    if n_eff_lost < 1.1:
        gate_passed = False
        failures.append(f"n_eff_lost {n_eff_lost:.2f} < 1.1 (HARD FAIL)")

    results["gate_passed"] = gate_passed
    results["failures"] = failures

    if gate_passed:
        print("\n  GATE 1: PASSED")
    else:
        print(f"\n  GATE 1: FAILED — {failures}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1: BiMDN Pre-training")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_mixtures", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/stage1")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Collect data (or load existing)
    t0 = time.time()
    data_path = os.path.join(args.output, "bimdn_pretrain_data.npz")
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        data = np.load(data_path)
        obs_hist = data["obs_histories"]
        target_pos = data["target_positions"]
        print(f"Loaded {len(obs_hist)} samples ({time.time()-t0:.1f}s)")
    else:
        obs_hist, target_pos = collect_data(
            args.n_episodes, args.history_length, args.seed,
        )
        np.savez(data_path, obs_histories=obs_hist, target_positions=target_pos)
        print(f"Data saved to {data_path} ({time.time()-t0:.1f}s)")

    # Step 2: Pre-train
    t1 = time.time()
    bimdn, train_losses, val_losses = pretrain_bimdn(
        obs_hist, target_pos,
        epochs=args.epochs, lr=args.lr,
        hidden_dim=args.hidden_dim, n_mixtures=args.n_mixtures,
        latent_dim=args.latent_dim, seed=args.seed,
    )
    model_path = os.path.join(args.output, "bimdn_pretrained.pt")
    torch.save(bimdn.state_dict(), model_path)
    print(f"Model saved to {model_path} ({time.time()-t1:.1f}s)")

    # Save loss curves
    loss_path = os.path.join(args.output, "loss_curves.npz")
    np.savez(loss_path, train_losses=train_losses, val_losses=val_losses)

    # Step 3: Validate
    results = validate_bimdn(bimdn, obs_hist, target_pos, device)

    # Save results
    results_path = os.path.join(args.output, "gate1_results.txt")
    with open(results_path, "w") as f:
        for key, val in results.items():
            f.write(f"{key}: {val}\n")

    # Summary
    print(f"\n=== Stage 1 Summary ===")
    print(f"Data: {len(obs_hist)} samples from {args.n_episodes} episodes")
    print(f"Training: {args.epochs} epochs, final train_loss={train_losses[-1]:.4f}")
    print(f"Loss reduction: {train_losses[0]:.4f} -> {train_losses[-1]:.4f} "
          f"({(1-train_losses[-1]/train_losses[0])*100:.1f}%)")
    print(f"Val loss: {val_losses[-1]:.4f} "
          f"(ratio to train: {val_losses[-1]/train_losses[-1]:.2f})")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"Gate 1: {'PASSED' if results['gate_passed'] else 'FAILED'}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
