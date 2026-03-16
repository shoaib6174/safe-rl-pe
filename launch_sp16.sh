#!/bin/bash
cd ~/Codes/safe-rl-pe

# SP16: Vis reward + reachable threshold sweep
# Key finding from SP15: vis reward is the only thing that helps under PO.
# SP15 failed because thr=0.8 was unreachable — no role switches ever happened.
# SP16 lowers threshold to 0.65 and tests forced switching.

BASE="--algorithm sac --fov_angle 90 --sensing_radius 8 --partial_obs_los --combined_masking --n_obstacles_min 0 --n_obstacles_max 3 --n_obstacle_obs 3 --masking_curriculum --p_full_anneal_steps 5000000 --p_full_residual 0.0 --alternate_freeze --freeze_switch_consecutive 1 --pfsp --opponent_pool_size 50 --learning_rate 3e-4 --ent_coef 0.03 --micro_phase_steps 2048 --eval_interval_micro 150 --max_total_steps 30000000 --collapse_threshold 0.1 --collapse_streak_limit 5"

VIS="--w_vis_pursuer 0.1"
REWARDS="--gamma 0.9 --capture_bonus 20 --timeout_penalty -20"

# SP16a/b/c: vis reward + thr=0.65, multi-seed validation (PRIMARY)
for SEED in 42 43 44; do
  nohup .venv/bin/python scripts/train_amsdrl.py $BASE $VIS $REWARDS \
    --freeze_switch_threshold 0.65 \
    --seed $SEED --output results/SP16a_vis_thr65_s${SEED} \
    > results/SP16a_vis_thr65_s${SEED}.log 2>&1 &
  echo "Launched SP16a_vis_thr65_s${SEED} (pid $!)"
done

# SP16d: vis reward + thr=0.55 (aggressive alternation)
nohup .venv/bin/python scripts/train_amsdrl.py $BASE $VIS $REWARDS \
  --freeze_switch_threshold 0.55 \
  --seed 43 --output results/SP16d_vis_thr55_s43 \
  > results/SP16d_vis_thr55_s43.log 2>&1 &
echo "Launched SP16d_vis_thr55_s43 (pid $!)"

# SP16e: vis reward + forced switch at 3M steps
nohup .venv/bin/python scripts/train_amsdrl.py $BASE $VIS $REWARDS \
  --freeze_switch_threshold 0.65 \
  --force_first_switch_steps 3000000 \
  --seed 43 --output results/SP16e_vis_force3M_s43 \
  > results/SP16e_vis_force3M_s43.log 2>&1 &
echo "Launched SP16e_vis_force3M_s43 (pid $!)"

# SP16f: vis reward + gamma=0.99, thr=0.65 (untested combo)
nohup .venv/bin/python scripts/train_amsdrl.py $BASE $VIS \
  --gamma 0.99 --capture_bonus 10 --timeout_penalty -10 \
  --freeze_switch_threshold 0.65 \
  --seed 43 --output results/SP16f_vis_g099_thr65_s43 \
  > results/SP16f_vis_g099_thr65_s43.log 2>&1 &
echo "Launched SP16f_vis_g099_thr65_s43 (pid $!)"

# SP16g: stronger vis reward (0.2) + thr=0.65
nohup .venv/bin/python scripts/train_amsdrl.py $BASE \
  --w_vis_pursuer 0.2 $REWARDS \
  --freeze_switch_threshold 0.65 \
  --seed 43 --output results/SP16g_vis02_thr65_s43 \
  > results/SP16g_vis02_thr65_s43.log 2>&1 &
echo "Launched SP16g_vis02_thr65_s43 (pid $!)"

# SP16h: vis reward + thr=0.65 + EWC (re-test with actual switches)
nohup .venv/bin/python scripts/train_amsdrl.py $BASE $VIS $REWARDS \
  --freeze_switch_threshold 0.65 \
  --ewc_lambda 1000 --ewc_fisher_samples 1024 \
  --seed 43 --output results/SP16h_vis_ewc_thr65_s43 \
  > results/SP16h_vis_ewc_thr65_s43.log 2>&1 &
echo "Launched SP16h_vis_ewc_thr65_s43 (pid $!)"

echo ""
echo "Launched 8 SP16 runs. Monitor with:"
echo "  tensorboard --logdir results/ --bind_all"
