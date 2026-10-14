#!/bin/zsh
# S24.1 TRAINING launch -- six arms in parallel from a frozen worktree (prereg A1.4).
# Same gates as launch_s23.sh: frozen worktree, results/ linked to the ONE record root, no competing
# jobs, literal arm names, confirmation by checkpoint files not PIDs. OMP 3 x 6 = 18 of 24 cores.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_1_worktree
V=$MAIN/venv/bin/python
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in SEL-80 SEL-81 SEL-82 NOMEM-80 NOMEM-81 NOMEM-82; do
  [ -d "$MAIN/results/derisk/ckpts_S24-1-$a" ] && { echo "REFUSED: ckpts_S24-1-$a exists (stop rule: never re-run an arm)"; exit 1; }
done
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_1_logs
for a in SEL-80 SEL-81 SEL-82 NOMEM-80 NOMEM-81 NOMEM-82; do
  nohup env PYTHONPATH=$WT OMP_NUM_THREADS=3 $V -u scripts/derisk_s24_1_train.py --arms $a > /home/niro-1/s24_1_logs/$a.log 2>&1 &
  echo "started $a"
done
echo "6 arms started; confirm by results/derisk/ckpts_S24-1-*/ checkpoints, not by PIDs"
