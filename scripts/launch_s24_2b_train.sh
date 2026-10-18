#!/bin/zsh
# S24.2 Step E.3 TRAINING launch -- three arms (SEL3-80/81/82, the pursuer's tactic selector
# RE-TRAINED vs the per-episode evader mixture {TACTIC-V0, EVS-best, GREEDY, HUG}) in parallel from a
# frozen worktree. Same gates as launch_s24_2_train.sh / launch_s24_os_train.sh / launch_s24_1_train.sh:
# frozen worktree, results/ linked to the ONE record root, no competing jobs, literal arm names,
# confirmation by checkpoint files not PIDs. OMP 4 x 3 = 12 of 24 cores. The runner itself asserts its
# construct gate (obs_len 55, 5 actions, mask non-empty at reset, the mixture lists exactly the four
# seats) on each arm's BUILT env before its first step and aborts the launch (SystemExit, not an
# UNDETERMINED record) if it does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_2b_train_worktree
V=$MAIN/venv/bin/python
ARMS=(SEL3-80 SEL3-81 SEL3-82)
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_2b_train.py" ] || { echo "REFUSED: $WT is frozen at a commit without the trainer"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
$V - "$WT" $ARMS <<'PY' || exit 1
# The checkpoint path is the TRAINER's own (_ckpt_dir), never a literal: the stop rule (never re-run an
# arm) is checked against whatever that function actually builds. Importing this round's trainer also
# resolves EVS-best's checkpoint (E.2's own ckpt_path) and REFUSES here if it is absent.
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("derisk_s24_2b_train", pathlib.Path("scripts/derisk_s24_2b_train.py"))
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)
bad = []
if not TR.EVS_BEST_CKPT.exists():
    bad.append(f"EVS-best checkpoint missing at {TR.EVS_BEST_CKPT}")
for a in sys.argv[2:]:
    if pathlib.Path(f"results/derisk/s24_2b_arm_{a}.json").exists():
        bad.append(f"{a}: arm record exists")
    if TR._ckpt_dir(a).exists():
        bad.append(f"{a}: checkpoint dir {TR._ckpt_dir(a)} exists")
if bad:
    print("REFUSED (stop rule: never re-run an arm): " + "; ".join(bad))
    sys.exit(1)
print("stop-rule precondition clean")
PY
mkdir -p /home/niro-1/s24_2b_logs
for a in $ARMS; do
  nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_2b_train.py --arms ${a} > /home/niro-1/s24_2b_logs/${a}.log 2>&1 &
  echo "started ${a}"
done
echo "3 arms started; confirm by results/derisk/ckpts_S24-2B-*/ checkpoints, not by PIDs"
