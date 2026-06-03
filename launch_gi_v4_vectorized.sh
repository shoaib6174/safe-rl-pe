#!/bin/bash
# GI-v4-vectorized: BiMDN + parallel vectorized environments (8 envs)
# Fixes: twin critics, belief normalization, wheelbase=0.30, curriculum=2000ep
# New: 8 parallel vectorized envs for data diversity matching paper

cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024

PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SEED=${1:-7}
OUTPUT="results_belief_noamp/GI_v4_vectorized_s${SEED}"

mkdir -p "$OUTPUT"

PYTHONUNBUFFERED=1 nohup $PYTHON -u training/train.py config_belief_noamp.yaml \
  > "$OUTPUT/training.log" 2>&1 &

echo "Launched GI-v4-vectorized (seed $SEED, pid $!)"
echo "Monitor: tail -f $OUTPUT/training.log"
