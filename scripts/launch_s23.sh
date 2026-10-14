#!/bin/zsh
# S23 launch -- same gates as S19's wrapper: frozen worktree, results/ linked to the ONE record root,
# literal cell lists, confirmation by record files not PIDs. The runner itself asserts cg_b_additive,
# the CG-S3 seeds file and record validity before its first episode.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s23_worktree
V=$MAIN/venv/bin/python
if pgrep -f "derisk_s19_planner.py --cells|replay_s18_branch_split|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
[ -f "$MAIN/results/derisk/s23_cg_s3_seeds.json" ] || { echo "REFUSED: CG-S3 seeds file missing"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s23_logs
# 4 chains = one row each (the MPC row is the slow one), OMP 4 each = 16 threads
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s23_strategy.py --cells PLAN-SHIELD_GREEDY PLAN-SHIELD_HUG PLAN-SHIELD_ATK > /home/niro-1/s23_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s23_strategy.py --cells LEAD-0.5-SHIELD_GREEDY LEAD-0.5-SHIELD_HUG LEAD-0.5-SHIELD_ATK > /home/niro-1/s23_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s23_strategy.py --cells LEAD-1.0-SHIELD_GREEDY LEAD-1.0-SHIELD_HUG LEAD-1.0-SHIELD_ATK > /home/niro-1/s23_logs/c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s23_strategy.py --cells MPC-2s-SHIELD_GREEDY MPC-2s-SHIELD_HUG MPC-2s-SHIELD_ATK > /home/niro-1/s23_logs/c3.log 2>&1 &
echo "4 chains started; confirm by results/derisk/s23_cells/*.json, not by PIDs"
