#!/bin/bash
# Pop-FSP v6 launcher — arena=19, evader=1.65 (intermediate between v4 and v5)
# Run on niro-2

set -e

PROJECT_DIR="/home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024"
VENV_PYTHON="/home/niro-2/safe-rl-pe/venv/bin/python"
SEEDS=(0 1 2)
EPISODES=5000

cd "$PROJECT_DIR"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

for SEED in "${SEEDS[@]}"; do
    OUTDIR="${PROJECT_DIR}/results_pop_fsp_v6/seed_${SEED}"
    mkdir -p "$OUTDIR"

    # Generate per-seed config
    CONFIG="/tmp/pop_fsp_v6_s${SEED}.yaml"
    sed "s/^seed:.*/seed: ${SEED}/; s|^output_dir:.*|output_dir: \"${OUTDIR}\"|" \
        "${PROJECT_DIR}/config_pop_fsp_v6.yaml" > "$CONFIG"

    echo "[$(date)] Launching Pop-FSP v6 seed ${SEED} → ${OUTDIR}"
    nohup "$VENV_PYTHON" "${PROJECT_DIR}/training/train.py" "$CONFIG" \
        > "${OUTDIR}/training.log" 2>&1 &
    echo "  PID: $!"
done

echo "[$(date)] All 3 seeds launched. Monitor with:"
echo "  tail -f ${PROJECT_DIR}/results_pop_fsp_v6/seed_*/training.log"
