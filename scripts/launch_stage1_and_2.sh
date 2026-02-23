#!/bin/bash
# Launch Stage 1 (BiMDN pre-training) then Stage 2 (all 3 configs) sequentially.
# Run on niro-2 via: nohup bash scripts/launch_stage1_and_2.sh > results/training.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

mkdir -p results

echo "=========================================="
echo "Phase 3 Training Pipeline"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=========================================="

# Stage 1: BiMDN Pre-training
echo ""
echo "=== STAGE 1: BiMDN Pre-training ==="
echo "Start: $(date)"
./venv/bin/python scripts/phase3_pretrain_bimdn.py \
    --n_episodes 500 --epochs 100 --hidden_dim 128 --seed 42 \
    --output results/stage1
echo "Stage 1 done: $(date)"

# Stage 2: Single-Agent Validation (all 3 configs)
echo ""
echo "=== STAGE 2: Single-Agent Training ==="
echo "Start: $(date)"
./venv/bin/python scripts/phase3_train_stage2.py \
    --run all --steps 500000 --seed 42 \
    --n_envs 4 --gamma 0.2 \
    --output results/stage2
echo "Stage 2 training done: $(date)"

# Stage 2 Evaluation
echo ""
echo "=== STAGE 2: Evaluation ==="
./venv/bin/python scripts/phase3_evaluate_stage2.py \
    --input results/stage2 --n_episodes 200
echo "Stage 2 evaluation done: $(date)"

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
