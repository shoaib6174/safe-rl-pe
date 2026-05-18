#!/bin/bash
# EMA vs Vanilla 5K-episode smoke test comparison
# Runs both vanilla SP and EMA (tau=0.001) for 5000 episodes each
# Usage: ./launch_ema_5k_comparison.sh [seed]
#
# Run this on niro-2 after pushing latest code.

set -e

SEED=${1:-0}
BASE_CONFIG="repro_gonultas_isler_2024/config_belief_noamp.yaml"
RUN_DIR="repro_gonultas_isler_2024"
PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"

cd ~/Codes/safe-rl-pe
export PYTHONPATH="${PYTHONPATH}:$(pwd)/${RUN_DIR}"

# Create temp configs by modifying base config with Python
VANILLA_CONFIG="/tmp/ema_vanilla_s${SEED}.yaml"
EMA_CONFIG="/tmp/ema_tau001_s${SEED}.yaml"

$PYTHON -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = ${SEED}
cfg['maddpg']['episodes'] = 5000
cfg['maddpg']['ema_tau'] = 0.0
with open('$VANILLA_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
cfg['maddpg']['ema_tau'] = 0.001
with open('$EMA_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print('Configs created')
"

# Launch vanilla
VANILLA_OUTPUT="${RUN_DIR}/results_ema_comparison/vanilla_5k_s${SEED}"
mkdir -p "$VANILLA_OUTPUT"
nohup $PYTHON ${RUN_DIR}/training/train.py "$VANILLA_CONFIG" \
  > "$VANILLA_OUTPUT/training.log" 2>&1 &
VANILLA_PID=$!
echo "Launched VANILLA (seed ${SEED}, pid ${VANILLA_PID})"
echo "  Log: tail -f ${VANILLA_OUTPUT}/training.log"

# Launch EMA
EMA_OUTPUT="${RUN_DIR}/results_ema_comparison/ema_tau001_5k_s${SEED}"
mkdir -p "$EMA_OUTPUT"
nohup $PYTHON ${RUN_DIR}/training/train.py "$EMA_CONFIG" \
  > "$EMA_OUTPUT/training.log" 2>&1 &
EMA_PID=$!
echo "Launched EMA tau=0.001 (seed ${SEED}, pid ${EMA_PID})"
echo "  Log: tail -f ${EMA_OUTPUT}/training.log"

echo ""
echo "Both runs started. To evaluate after completion:"
echo "  PYTHONPATH=... $PYTHON ${RUN_DIR}/evaluate_checkpoint_sweep.py ${VANILLA_CONFIG} ${VANILLA_OUTPUT}"
echo "  PYTHONPATH=... $PYTHON ${RUN_DIR}/evaluate_checkpoint_sweep.py ${EMA_CONFIG} ${EMA_OUTPUT}"
