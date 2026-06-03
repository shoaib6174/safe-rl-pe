#!/bin/bash
cd ~/Codes/safe-rl-pe

BASE="--algorithm sac --fov_angle 90 --sensing_radius 8 --partial_obs_los --combined_masking --n_obstacles_min 0 --n_obstacles_max 3 --n_obstacle_obs 3 --masking_curriculum --p_full_anneal_steps 5000000 --p_full_residual 0.0 --alternate_freeze --freeze_switch_consecutive 1 --pfsp --opponent_pool_size 50 --learning_rate 3e-4 --ent_coef 0.03 --micro_phase_steps 2048 --eval_interval_micro 150 --max_total_steps 30000000 --collapse_threshold 0.1 --collapse_streak_limit 5"

BASE_NO_OBS="--algorithm sac --fov_angle 90 --sensing_radius 8 --partial_obs_los --n_obstacles_min 0 --n_obstacles_max 0 --n_obstacle_obs 0 --masking_curriculum --p_full_anneal_steps 5000000 --p_full_residual 0.0 --alternate_freeze --freeze_switch_consecutive 1 --pfsp --opponent_pool_size 50 --learning_rate 3e-4 --ent_coef 0.03 --micro_phase_steps 2048 --eval_interval_micro 150 --max_total_steps 30000000 --collapse_threshold 0.1 --collapse_streak_limit 5"

SCHED="--freeze_thr_schedule 0:0.8,2500000:0.75,5000000:0.70,8000000:0.65,11000000:0.60 --self_play_start_steps 15000000"

# SP15e_s42: EWC + gamma=0.9, fixed thr
nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
  --gamma 0.9 --capture_bonus 20 --timeout_penalty -20 \
  --freeze_switch_threshold 0.8 \
  --ewc_lambda 1000 --ewc_fisher_samples 1024 \
  --seed 42 --output results/SP15e_ewc_g09_s42 \
  > results/SP15e_ewc_g09_s42.log 2>&1 &
echo "Launched SP15e_ewc_g09_s42 (pid $!)"

# SP15g_s43: w_vis_pursuer=0.1 + gamma=0.9
nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
  --gamma 0.9 --capture_bonus 20 --timeout_penalty -20 \
  --freeze_switch_threshold 0.8 \
  --w_vis_pursuer 0.1 \
  --seed 43 --output results/SP15g_vis_g09_s43 \
  > results/SP15g_vis_g09_s43.log 2>&1 &
echo "Launched SP15g_vis_g09_s43 (pid $!)"

# SP15i: gamma=0.9, no obstacles
for SEED in 42 43; do
  nohup .venv/bin/python scripts/train_amsdrl.py $BASE_NO_OBS \
    --gamma 0.9 --capture_bonus 20 --timeout_penalty -20 \
    --freeze_switch_threshold 0.8 \
    --seed $SEED --output results/SP15i_noobs_g09_s${SEED} \
    > results/SP15i_noobs_g09_s${SEED}.log 2>&1 &
  echo "Launched SP15i_noobs_g09_s${SEED} (pid $!)"
done

# SP15j_s42: EWC + schedule + pure SP
nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
  --gamma 0.9 --capture_bonus 20 --timeout_penalty -20 \
  --freeze_switch_threshold 0.8 $SCHED \
  --ewc_lambda 1000 --ewc_fisher_samples 1024 \
  --seed 42 --output results/SP15j_ewc_sched_g09_s42 \
  > results/SP15j_ewc_sched_g09_s42.log 2>&1 &
echo "Launched SP15j_ewc_sched_g09_s42 (pid $!)"

# SP15k: w_vis_pursuer + schedule + pure SP
for SEED in 42 43; do
  nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
    --gamma 0.9 --capture_bonus 20 --timeout_penalty -20 \
    --freeze_switch_threshold 0.8 $SCHED \
    --w_vis_pursuer 0.1 \
    --seed $SEED --output results/SP15k_vis_sched_g09_s${SEED} \
    > results/SP15k_vis_sched_g09_s${SEED}.log 2>&1 &
  echo "Launched SP15k_vis_sched_g09_s${SEED} (pid $!)"
done

sleep 3
echo "---"
echo "Total SP15 processes:"
ps aux | grep train_amsdrl | grep -v grep | wc -l
echo "GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
