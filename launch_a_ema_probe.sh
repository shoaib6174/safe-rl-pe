#!/bin/bash
# A-EMA Probe: 3 seeds with asymmetric LR + vis_reward

SEEDS=(0 1 2)
BASE_CONFIG="repro_gonultas_isler_2024/config_a_ema_probe.yaml"
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
cfg['output_dir'] = 'repro_gonultas_isler_2024/results_a_ema_probe/seed_${SEED}'
with open('/tmp/a_ema_probe_s${SEED}.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config created for seed $SEED')
"

    OUTPUT="$RUN_DIR/results_a_ema_probe/seed_${SEED}"
    mkdir -p "$OUTPUT"

    cd "$RUN_DIR"
    nohup env PYTHONPATH="$(cd .. && pwd)/$RUN_DIR" $PYTHON training/train.py "/tmp/a_ema_probe_s${SEED}.yaml" \
      > "../$OUTPUT/training.log" 2>&1 &
    cd ..

    echo "Launched seed $SEED (pid $!)"
done

echo ""
echo "All 3 probe seeds launched."
echo "Monitor with: tail -f repro_gonultas_isler_2024/results_a_ema_probe/seed_*/training.log"
