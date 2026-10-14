#!/bin/zsh
# S24.3-PR TRAINING launch -- six arms in parallel from a frozen worktree (prereg sections 3, 7).
# Same gates as launch_s24_1_train.sh: frozen worktree, results/ linked to the ONE record root, no
# competing jobs, literal arm names, confirmation by checkpoint files not PIDs. OMP 3 x 6 = 18 of 24
# cores. The runner itself asserts CG-P1 on each arm's BUILT env before its first step and aborts the
# launch (SystemExit, not an UNDETERMINED record) if the env is not at the declared ratio.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_3pr_train_worktree
V=$MAIN/venv/bin/python
ARMS=(SELr0.8-80 SELr0.8-81 SELr0.8-82 SELr1.0-80 SELr1.0-81 SELr1.0-82)
if pgrep -f "derisk_s2[34]_\|replay_s18_branch_split\|s24_tactic_game\|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_3pr_train.py" ] || { echo "REFUSED: $WT is frozen at a commit without the trainer"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in $ARMS; do
  [ -d "$MAIN/results/derisk/ckpts_S24-3PR-$a" ] && { echo "REFUSED: ckpts_S24-3PR-$a exists (stop rule: never re-run an arm)"; exit 1; }
  [ -f "$MAIN/results/derisk/s24_3pr_arm_$a.json" ] && { echo "REFUSED: s24_3pr_arm_$a.json exists (stop rule: never re-run an arm)"; exit 1; }
done
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_3pr_train_logs
for a in $ARMS; do
  nohup env PYTHONPATH=$WT OMP_NUM_THREADS=3 $V -u scripts/derisk_s24_3pr_train.py --arms $a > /home/niro-1/s24_3pr_train_logs/$a.log 2>&1 &
  echo "started $a"
done
echo "6 arms started; confirm by results/derisk/ckpts_S24-3PR-*/ checkpoints, not by PIDs"
