#!/usr/bin/env bash
# Launch symmetric point-mass self-play seeds 9-33 in batches of 5.
# Each seed ~3h runtime. GPU is shared RTX 5090; 5 seeds × 2 parallel envs = 10 envs.

set -e

cd /home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024
PYTHON=/home/niro-1/Codes/safe-rl-pe/venv/bin/python
export PYTHONPATH="/home/niro-1/Codes/safe-rl-pe/repro_gonultas_isler_2024:${PYTHONPATH}"

BATCH_SIZE=5
START_SEED=9
END_SEED=33

for ((batch_start=START_SEED; batch_start<=END_SEED; batch_start+=BATCH_SIZE)); do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    if ((batch_end > END_SEED)); then
        batch_end=$END_SEED
    fi

    echo "========================================"
    echo "Launching batch: seeds $batch_start-$batch_end"
    echo "========================================"

    pids=()
    for SEED in $(seq $batch_start $batch_end); do
        CONFIG="config_symmetric_seed_${SEED}.yaml"
        sed "s/^seed: 0$/seed: ${SEED}/" config_symmetric.yaml > "${CONFIG}"

        LOG="symmetric_seed_${SEED}.log"

        echo "Launching symmetric seed ${SEED}..."
        nohup ${PYTHON} -u training/train.py "${CONFIG}" > "${LOG}" 2>&1 &
        pids+=($!)
        echo "  PID=$!"
        sleep 2
    done

    echo "Waiting for batch $batch_start-$batch_end to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "Batch $batch_start-$batch_end complete."
done

echo "All symmetric seeds $START_SEED-$END_SEED complete."
