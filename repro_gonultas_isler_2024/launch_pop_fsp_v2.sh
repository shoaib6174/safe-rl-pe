#!/bin/bash
# Launch Pop-FSP v2 (recency + asymmetric + gradual mixing) on niro-2
# Multi-seed sweep

PROJECT_DIR="$HOME/Codes/safe-rl-pe/repro_gonultas_isler_2024"
cd "$PROJECT_DIR" || exit 1

SEEDS=(0 1 2)
EPISODES=5000

for SEED in "${SEEDS[@]}"; do
    OUTDIR="$PROJECT_DIR/results_pop_fsp_v2/seed_$SEED"
    mkdir -p "$OUTDIR"

    # Generate seed-specific config
    cat > /tmp/pop_fsp_v2_s${SEED}.yaml <<EOF
arena_size: 16.0
max_steps: 400
capture_radius: 0.15
sim_dt: 0.1

pursuer:
  wheelbase: 0.30
  v_max: 2.5
  v_min: -1.0
  delta_max: 0.34
  delta_min: -0.34
  accel_max: 2.0
  accel_min: -2.0
  steer_rate_max: 3.2
  steer_rate_min: -3.2
  fov_angle: 2.094
  fov_range: 6.0

evader:
  v_max: 1.5
  v_min: -1.5
  accel_max: 9.81
  accel_min: -9.81
  fov_angle: 6.283
  fov_range: 1000.0

reward:
  lambda_capture: 1000
  lambda_time: 1
  lambda_distance: 1

use_belief: false
use_amp: true
use_compile: false

maddpg:
  episodes: $EPISODES
  parallel_envs: 2
  replay_buffer_size: 100000
  batch_size: 1024
  actor_lr: 0.0005
  critic_lr: 0.0005
  gamma: 0.99
  tau: 0.005
  exploration_sigma: 0.1
  hidden_dim: 128
  frame_stack: 2
  frame_skip: 2
  update_freq: 1
  ema_tau: 0.0

  use_pop_fsp: true
  pop_size: 8
  pop_evader_size: 16
  pop_add_freq: 250
  pop_warmup: 500
  pop_recency_lambda: 0.15
  pop_mixing_start: 100
  pop_mixing_end: 500

bimdn:
  history_length: 200
  downsample_factor: 10
  lstm_hidden: 128
  feature_dim: 32
  num_components: 3
  belief_buffer_size: 25000
  belief_warmup: 10000
  belief_update_freq: 10
  belief_batch_size: 256
  belief_lr: 0.002

curriculum:
  t_end_ratio: 0.3
  fov_start: 6.283
  fov_end: 2.094
  evader_vmax_start: 0.0
  evader_vmax_end: 1.5

eval:
  checkpoint_freq: 250
  eval_episodes: 500

log_dir: "./runs"
seed: $SEED
output_dir: "$OUTDIR"
EOF

    echo "Launching Pop-FSP v2 seed $SEED -> $OUTDIR"
    nohup /home/niro-2/Codes/safe-rl-pe/venv/bin/python \
        "$PROJECT_DIR/training/train.py" \
        /tmp/pop_fsp_v2_s${SEED}.yaml \
        > "$OUTDIR/training.log" 2>&1 &
    echo $! > "$OUTDIR/pid.txt"
    sleep 5
done

echo "All seeds launched. PIDs:"
for SEED in "${SEEDS[@]}"; do
    OUTDIR="$PROJECT_DIR/results_pop_fsp_v2/seed_$SEED"
    echo "  seed $SEED: PID=$(cat $OUTDIR/pid.txt)"
done
