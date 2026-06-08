#!/bin/bash
# Launch PBT experiment on niro-2
# 3 seeds, 10000 episodes each, population of 5 agents

cd /home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024

for seed in 0 1 2; do
    echo "Launching PBT seed $seed"
    nohup /home/niro-2/safe-rl-pe/venv/bin/python -u training/train_pbt.py config_pbt.yaml \
        --seed $seed \
        > pbt_seed_${seed}.log 2>&1 &
    echo "PID: $!"
    sleep 5
done

echo "All PBT seeds launched. Monitor with:"
echo "  tail -f pbt_seed_*.log"
