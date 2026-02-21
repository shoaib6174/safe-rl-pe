#!/bin/bash
# Launch baseline PPO training on niro-2 for Phase 2.5 comparison.
# Usage: bash scripts/launch_baseline_ppo.sh

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p results/phase2_5/baseline_ppo

echo "Launching baseline PPO training..."
echo "Total timesteps: 2000000"
echo "Seed: 42"
echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

python3 -u scripts/train.py \
  total_timesteps=2000000 \
  seed=42 \
  wandb.mode=disabled \
  n_envs=4 \
  hydra.run.dir=results/phase2_5/baseline_ppo/hydra_output
