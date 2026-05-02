#!/bin/bash
# Launch SP17b: additional seeds with earlier forced switch (2M instead of 3M)
# Usage: bash scripts/launch_sp17b.sh

set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

SEEDS=(46 47 48 49)

for seed in "${SEEDS[@]}"; do
    echo "Launching SP17b seed $seed..."
    nohup .venv/bin/python scripts/train_amsdrl.py \
        --algorithm sac \
        --fov_angle 90 \
        --sensing_radius 8.0 \
        --combined_masking \
        --arena_width 20 \
        --arena_height 20 \
        --max_steps 1200 \
        --n_obstacles_min 0 --n_obstacles_max 3 \
        --pursuer_v_max 1.0 --evader_v_max 1.0 \
        --lr 3e-4 --ent_coef 0.03 \
        --micro_phase_steps 2048 \
        --eval_interval 150 \
        --alternate_freeze \
        --freeze_switch_threshold 0.65 \
        --force_switch_steps 2000000 \
        --w_vis_pursuer 0.2 \
        --w_search 0.0001 \
        --t_stale 100 \
        --collapse_threshold 0.1 \
        --collapse_streak_limit 5 \
        --pfsp \
        --masking_curriculum \
        --p_full_anneal_steps 5000000 \
        --max_total_steps 30000000 \
        --seed "$seed" \
        --output "${RESULTS_DIR}/SP17b_s${seed}" \
        > "${RESULTS_DIR}/SP17b_s${seed}.log" 2>&1 &
    echo "  PID: $! -> SP17b_s${seed}.log"
done

echo "All 4 seeds launched. Check with: ps aux | grep train_amsdrl"
