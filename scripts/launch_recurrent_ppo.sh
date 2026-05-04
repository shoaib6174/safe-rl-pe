#!/bin/bash
# Launch RecurrentPPO with partial obs (SP17b config, algorithm=recurrent_ppo)
# Tests whether LSTM memory helps close the curriculum-induced exploitability gap.

set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Match SP17b config exactly, only change algorithm to recurrent_ppo
BASE="--algorithm recurrent_ppo \
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
    --max_total_steps 30000000"

# Launch 3 seeds for statistical validity
for seed in 100 101 102; do
    echo "Launching RecurrentPPO seed $seed..."
    nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
        --seed "$seed" \
        --output "${RESULTS_DIR}/REC_PO_s${seed}" \
        > "${RESULTS_DIR}/REC_PO_s${seed}.log" 2>&1 &
    echo "  PID: $! -> REC_PO_s${seed}.log"
done

echo "All RecurrentPPO seeds launched. Monitor with: ps aux | grep train_amsdrl"
