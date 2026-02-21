#!/bin/bash
# Monitor BarrierNet training on niro-2 and run evaluation when complete.
#
# Usage: bash scripts/monitor_training.sh
# Or run on niro-2: nohup bash scripts/monitor_training.sh > monitor_log.txt 2>&1 &

REPO_DIR="$HOME/Codes/safe-rl-pe"
VENV="$REPO_DIR/.venv/bin/activate"
SEED42_LOG="$REPO_DIR/results/phase2_5/barriernet/training_log.txt"
SEED123_LOG="$REPO_DIR/results/phase2_5/barriernet/training_log_seed123.txt"
CHECKPOINT_42="$REPO_DIR/checkpoints/barriernet/barriernet_final.pt"
CHECKPOINT_123="$REPO_DIR/checkpoints/barriernet_seed123/barriernet_final.pt"

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

    if [ -f "$SEED123_LOG" ]; then
        LAST_123=$(tail -1 "$SEED123_LOG")
        echo "  Seed 123: $LAST_123"
    fi

    # Check if both done
    DONE_42=false
    DONE_123=false
    check_training_done "$SEED42_LOG" && DONE_42=true
    check_training_done "$SEED123_LOG" && DONE_123=true

    if $DONE_42 && $DONE_123; then
        echo ""
        echo "=== Both training runs complete! ==="
        echo "Running evaluation..."

        cd "$REPO_DIR"
        source "$VENV"

        # Evaluate seed 42
        if [ -f "$CHECKPOINT_42" ]; then
            echo "Evaluating seed 42..."
            python3 -u scripts/evaluate_comparison.py \
                --barriernet "$CHECKPOINT_42" \
                --barriernet-only \
                --n-episodes 200 \
                --obstacles 2 \
                --seed 0 \
                --output-dir results/phase2_5/comparison/seed42 \
                > results/phase2_5/barriernet/eval_log_seed42.txt 2>&1
            echo "Seed 42 evaluation done."
        fi

        # Evaluate seed 123
        if [ -f "$CHECKPOINT_123" ]; then
            echo "Evaluating seed 123..."
            python3 -u scripts/evaluate_comparison.py \
                --barriernet "$CHECKPOINT_123" \
                --barriernet-only \
                --n-episodes 200 \
                --obstacles 2 \
                --seed 1000 \
                --output-dir results/phase2_5/comparison/seed123 \
                > results/phase2_5/barriernet/eval_log_seed123.txt 2>&1
            echo "Seed 123 evaluation done."
        fi

        echo "=== All done at $(date) ==="
        break
    fi

    sleep 300  # Check every 5 minutes
done
