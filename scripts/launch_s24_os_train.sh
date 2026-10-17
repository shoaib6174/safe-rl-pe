#!/bin/zsh
# S24-OS Step O.2 TRAINING launch -- three arms (SEL2-80/81/82, library v1) in parallel from a frozen
# worktree. Same gates as launch_s24_1_train.sh / launch_s24_3pr_train.sh: frozen worktree, results/
# linked to the ONE record root, no competing jobs, literal arm names, confirmation by checkpoint files
# not PIDs. OMP 4 x 3 = 12 of 24 cores. The runner itself asserts its construct gate (obs_len 59,
# actions[-1] ORBIT-SEARCH, 6th mask False when visible) on each arm's BUILT env before its first step
# and aborts the launch (SystemExit, not an UNDETERMINED record) if it does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_os_train_worktree
V=$MAIN/venv/bin/python
ARMS=(SEL2-80 SEL2-81 SEL2-82)
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_os_train.py" ] || { echo "REFUSED: $WT is frozen at a commit without the trainer"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
$V - "$WT" $ARMS <<'PY' || exit 1
# The checkpoint path is the TRAINER's own (_ckpt_dir), never a literal: the stop rule (never re-run an
# arm) is checked against whatever that function actually builds.
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("derisk_s24_os_train", pathlib.Path("scripts/derisk_s24_os_train.py"))
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)
bad = []
for a in sys.argv[2:]:
    if pathlib.Path(f"results/derisk/s24_os_arm_{a}.json").exists():
        bad.append(f"{a}: arm record exists")
    if TR._ckpt_dir(a).exists():
        bad.append(f"{a}: checkpoint dir {TR._ckpt_dir(a)} exists")
if bad:
    print("REFUSED (stop rule: never re-run an arm): " + "; ".join(bad))
    sys.exit(1)
print("stop-rule precondition clean")
PY
mkdir -p /home/niro-1/s24_os_logs
for a in $ARMS; do
  nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_os_train.py --arms $a > /home/niro-1/s24_os_logs/$a.log 2>&1 &
  echo "started $a"
done
echo "3 arms started; confirm by results/derisk/ckpts_S24-OS-*/ checkpoints, not by PIDs"
