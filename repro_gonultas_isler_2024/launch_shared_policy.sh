#!/bin/bash
# Launch shared policy experiment on niro-2
# 3 seeds, 10000 episodes each

cd /home/niro-2/Codes/safe-rl-pe/repro_gonultas_isler_2024

for seed in 0 1 2; do
    echo "Launching shared policy seed $seed"
    nohup /home/niro-2/safe-rl-pe/venv/bin/python -u training/train_shared.py config_shared_policy.yaml \
        --seed $seed \
        > shared_seed_${seed}.log 2>&1 &
    echo "PID: $!"
    sleep 5
done

echo "All seeds launched. Monitor with:"
echo "  tail -f shared_seed_*.log"
