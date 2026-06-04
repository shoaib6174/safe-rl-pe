#!/bin/bash
# Transfer Learning: 3 seeds with pre-train pursuer vs greedy evader

SEEDS=(0 1 2)
BASE_CONFIG="repro_gonultas_isler_2024/config_transfer.yaml"
RUN_DIR="repro_gonultas_isler_2024"
PYTHON="/home/niro-2/Codes/safe-rl-pe/.venv/bin/python"

cd ~/Codes/safe-rl-pe

for SEED in "${SEEDS[@]}"; do
    # Create temp config with seed and unique output_dir
    python3 -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = $SEED
cfg['output_dir'] = 'repro_gonultas_isler_2024/results_transfer/seed_${SEED}'
with open('/tmp/transfer_s${SEED}.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config created for seed $SEED')
"

    OUTPUT="$RUN_DIR/results_transfer/seed_${SEED}"
    mkdir -p "$OUTPUT"

    cd "$RUN_DIR"
    nohup env PYTHONPATH="$(cd .. && pwd)/$RUN_DIR" $PYTHON training/train.py "/tmp/transfer_s${SEED}.yaml" \
      > "../$OUTPUT/training.log" 2>&1 &
    cd ..

    echo "Launched seed $SEED (pid $!)"
done

echo ""
echo "All 3 transfer learning seeds launched."
echo "Monitor with: tail -f repro_gonultas_isler_2024/results_transfer/seed_*/training.log"
