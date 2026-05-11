#!/bin/bash
# GI-v3-belief-noamp: BiMDN + AMP disabled
# Fix: beliefs normalized to [-1, 1] before concatenation with observations
# Previous GI-v2-amp failed: 83.7% NaN losses due to scale mismatch (belief in [-8,8] vs obs in [-1,1])
# Previous GI-v3-belief-fixed failed: AMP causes NaN cascade in MADDPG losses
# This run: BiMDN + normalized beliefs + NO AMP

cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024

PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"

SEED=${1:-1}
OUTPUT="results_belief_noamp/GI_v3_belief_noamp_s${SEED}"

mkdir -p "$OUTPUT"

nohup $PYTHON training/train.py config_belief_noamp.yaml \
  > "$OUTPUT/training.log" 2>&1 &

echo "Launched GI-v3-belief-noamp (seed $SEED, pid $!)"
echo "Monitor: tail -f $OUTPUT/training.log"
