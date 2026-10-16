#!/bin/zsh
# S24-FLAT TRAINING launch -- FLAT x3 + STRUCT x3 from a frozen worktree. Same gates as the S24.1/3PR
# training launchers (live pgrep guard with plain |; launch from a command line that contains none of the
# pattern text). Refuses on an existing checkpoint dir (stop rule) or a missing trainer.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_flat_train_worktree
V=$MAIN/venv/bin/python
ARMS_FLAT=(FLAT-80 FLAT-81 FLAT-82 FLAT40-80 FLAT40-81 FLAT40-82)
ARMS_STRUCT=(STRUCT-80 STRUCT-81 STRUCT-82)
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
[ -f "$WT/scripts/derisk_s24_flat_train.py" ] || { echo "REFUSED: no flat trainer in the frozen worktree"; exit 1; }
[ -f "$WT/scripts/derisk_s24_struct_train.py" ] || { echo "REFUSED: no struct trainer in the frozen worktree"; exit 1; }
for a in $ARMS_FLAT; do [ -d "$MAIN/results/derisk/ckpts_S24-$a" ] && { echo "REFUSED: ckpts_S24-$a exists (stop rule)"; exit 1; }; done
for a in $ARMS_STRUCT; do [ -d "$MAIN/results/derisk/ckpts_S24-$a" ] && { echo "REFUSED: ckpts_S24-$a exists (stop rule)"; exit 1; }; done
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_flat_logs
for a in $ARMS_FLAT; do nohup env PYTHONPATH=$WT OMP_NUM_THREADS=2 $V -u scripts/derisk_s24_flat_train.py --arms $a > /home/niro-1/s24_flat_logs/$a.log 2>&1 & echo "started $a"; done
for a in $ARMS_STRUCT; do nohup env PYTHONPATH=$WT OMP_NUM_THREADS=2 $V -u scripts/derisk_s24_struct_train.py --arms $a > /home/niro-1/s24_flat_logs/$a.log 2>&1 & echo "started $a"; done
echo "9 arms started; confirm by results/derisk/ckpts_S24-{FLAT,STRUCT}-*/ checkpoints, not by PIDs"
