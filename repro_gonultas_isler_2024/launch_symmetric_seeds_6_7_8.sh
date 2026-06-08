#!/usr/bin/env bash
# Launch symmetric point-mass self-play seeds 6, 7, 8
# Each ~3h runtime

set -e

cd ~/Codes/safe-rl-pe/repro_gonultas_isler_2024
PYTHON=~/safe-rl-pe/venv/bin/python

for SEED in 6 7 8; do
    # Create per-seed config
    CONFIG="config_symmetric_seed_${SEED}.yaml"
    sed "s/^seed: 0$/seed: ${SEED}/" config_symmetric.yaml > "${CONFIG}"

    LOG="symmetric_seed_${SEED}.log"

    echo "Launching symmetric seed ${SEED}..."
    nohup ${PYTHON} -u train.py "${CONFIG}" > "${LOG}" 2>&1 &
    echo "  PID=$!"
    sleep 2
done

echo "All 3 seeds launched. Monitor with:"
echo "  tail -f symmetric_seed_*.log"
