#!/bin/bash
# EMA-only 5K-episode run with actor_lr=0.0005 (matching seed 0 config)
# Tests whether fixes (Q_MIN=-1500, actor Q-clipping, NaN recovery)
# enable stable training at the original learning rate.
# Usage: ./launch_ema_s3_lr0005.sh

set -e

SEED=3
BASE_CONFIG="config_belief_noamp.yaml"
RUN_DIR="repro_gonultas_isler_2024"
PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"

cd ~/Codes/safe-rl-pe

# Create EMA config with LR=0.0005 (original, not reduced)
EMA_CONFIG="/tmp/ema_tau001_s${SEED}_lr0005.yaml"

$PYTHON -c "
import yaml
with open('$RUN_DIR/$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = ${SEED}
cfg['maddpg']['episodes'] = 5000
cfg['maddpg']['actor_lr'] = 0.0005      # <-- Original LR from seed 0
cfg['maddpg']['ema_tau'] = 0.001
cfg['output_dir'] = 'results_ema_comparison/ema_tau001_5k_s${SEED}_lr0005'
with open('$EMA_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print('EMA config created: LR=0.0005, seed=3')
"

# Launch EMA only (vanilla proven to collapse — waste of compute)
EMA_OUTPUT="$RUN_DIR/results_ema_comparison/ema_tau001_5k_s${SEED}_lr0005"
mkdir -p "$EMA_OUTPUT"
cd "$RUN_DIR"
nohup env PYTHONPATH="$(cd .. && pwd)/$RUN_DIR" $PYTHON training/train.py "$EMA_CONFIG" \
  > "../$EMA_OUTPUT/training.log" 2>&1 &
EMA_PID=$!
cd ..
echo "Launched EMA tau=0.001 LR=0.0005 (seed ${SEED}, pid ${EMA_PID})"
echo "  Log: tail -f ${EMA_OUTPUT}/training.log"

echo ""
echo "To evaluate after completion:"
echo "  cd $RUN_DIR && PYTHONPATH=.. $PYTHON evaluate_checkpoint_sweep.py $EMA_CONFIG ../$EMA_OUTPUT"
