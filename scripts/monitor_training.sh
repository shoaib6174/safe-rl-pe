#!/bin/bash
# Monitor BarrierNet training on niro-2 and run evaluation when complete.
#
# Usage: bash scripts/monitor_training.sh
# Or run on niro-2: nohup bash scripts/monitor_training.sh > monitor_log.txt 2>&1 &

REPO_DIR="$HOME/Codes/safe-rl-pe"
VENV="$REPO_DIR/.venv/bin/activate"
SEED42_LOG="$REPO_DIR/results/phase2_5/barriernet/training_log.txt"
CHECKPOINT_42="$REPO_DIR/checkpoints/barriernet/barriernet_final.pt"
BASELINE_MODEL="$REPO_DIR/models/local_42/final_model.zip"

check_training_done() {
    local log_file=$1
    if [ ! -f "$log_file" ]; then
        return 1
    fi
    grep -q "Training complete" "$log_file" && return 0
    # Also check if the last line shows steps >= 2M
    local last_steps
    last_steps=$(tail -1 "$log_file" | grep -oP 'steps=\s*\K[0-9]+' || echo "0")
    [ "$last_steps" -ge 2000000 ] && return 0
    return 1
}

echo "=== BarrierNet Training Monitor ==="
echo "Started at $(date)"
echo ""

while true; do
    echo "[$(date +%H:%M)] Status:"

    if [ -f "$SEED42_LOG" ]; then
        LAST_42=$(tail -1 "$SEED42_LOG")
        echo "  Seed 42: $LAST_42"
    fi

    # Check if seed 42 is done
    if check_training_done "$SEED42_LOG"; then
        echo ""
        echo "=== Training complete! ==="
        echo "Running evaluation..."

        cd "$REPO_DIR"
        source "$VENV"

        # Evaluate BarrierNet
        if [ -f "$CHECKPOINT_42" ]; then
            echo "Evaluating BarrierNet..."
            python3 -u scripts/evaluate_comparison.py \
                --barriernet "$CHECKPOINT_42" \
                --barriernet-only \
                --n-episodes 200 \
                --obstacles 2 \
                --seed 0 \
                --output-dir results/phase2_5/comparison \
                > results/phase2_5/barriernet/eval_log_barriernet.txt 2>&1
            echo "BarrierNet evaluation done."
            cat results/phase2_5/barriernet/eval_log_barriernet.txt
        else
            echo "WARNING: Checkpoint not found at $CHECKPOINT_42"
        fi

        # Evaluate baseline PPO (with and without safety filter)
        if [ -f "$BASELINE_MODEL" ]; then
            echo ""
            echo "Evaluating baseline PPO..."
            python3 -u scripts/evaluate_comparison.py \
                --barriernet "$CHECKPOINT_42" \
                --cbf-beta "$BASELINE_MODEL" \
                --n-episodes 200 \
                --obstacles 2 \
                --seed 0 \
                --output-dir results/phase2_5/comparison \
                > results/phase2_5/barriernet/eval_log_comparison.txt 2>&1
            echo "Full comparison done."
            cat results/phase2_5/barriernet/eval_log_comparison.txt
        else
            echo "WARNING: Baseline model not found at $BASELINE_MODEL"
        fi

        echo ""
        echo "=== All done at $(date) ==="
        break
    fi

    sleep 300  # Check every 5 minutes
done
