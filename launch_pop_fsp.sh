#!/bin/bash
# Pop-FSP smoke test launcher
# Usage: ./launch_pop_fsp.sh [seed] [episodes]
# Run on niro-2 after pushing latest code.

set -e

SEED=${1:-0}
EPISODES=${2:-500}
BASE_CONFIG="repro_gonultas_isler_2024/config_pop_fsp.yaml"
PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"

# Create temp config by modifying base config with Python
POP_FSP_CONFIG="/tmp/pop_fsp_s${SEED}.yaml"
OUTPUT_DIR="results/pop_fsp_s${SEED}_$(date +%Y%m%d_%H%M%S)"

cd ~/Codes/safe-rl-pe

$PYTHON -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = ${SEED}
cfg['maddpg']['episodes'] = ${EPISODES}
cfg['output_dir'] = '${OUTPUT_DIR}'
with open('$POP_FSP_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print('Pop-FSP config created:', '$POP_FSP_CONFIG')
print('Output dir:', '${OUTPUT_DIR}')
"

echo "Launching Pop-FSP: seed=$SEED, episodes=$EPISODES"
$PYTHON repro_gonultas_isler_2024/training/train.py "$POP_FSP_CONFIG"

echo "Done: $OUTPUT_DIR"
