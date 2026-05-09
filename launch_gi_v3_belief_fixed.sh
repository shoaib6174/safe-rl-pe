#!/bin/bash
# GI-v3-belief-fixed: BiMDN + AMP with normalized beliefs
# Fix: beliefs normalized to [-1, 1] before concatenation with observations
# Previous GI-v2-amp failed: 83.7% NaN losses due to scale mismatch (belief in [-8,8] vs obs in [-1,1])

cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024

# Use conda env on niro-2
PYTHON="/home/niro-2/anaconda3/envs/safe-rl/bin/python"

SEED=${1:-1}
OUTPUT="results/GI_v3_belief_fixed_s${SEED}"

mkdir -p "$OUTPUT"

nohup $PYTHON training/train.py config_belief.yaml \
  > "$OUTPUT/training.log" 2>&1 &

echo "Launched GI-v3-belief-fixed (seed $SEED, pid $!)"
echo "Monitor: tail -f $OUTPUT/training.log"
