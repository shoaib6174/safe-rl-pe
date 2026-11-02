#!/bin/zsh
# Paper B round N1 (internal S27) launch: docs/pathway/paper_b_prereg_n1_branch_attribution_2026-11-03.md.
# Same discipline as launch_paper_b.sh: frozen worktree, results/ linked to the ONE record root, no competing
# round on the host, literal cell lists, confirmation by record files. Evaluation only (replays; no training).
# 8 chains x OMP 3 = 24 cores; 40 cells x 500 episodes.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s27_worktree
LOGS=/home/niro-1/s27_logs
V=$MAIN/venv/bin/python
if pgrep -f "derisk_s2[3-6]_|paper_b_filter_sweep|paper_b_geometry|paper_b_branch_attribution|replay_s18_branch_split|s24_tactic_game" >/dev/null; then
  echo "REFUSED: another round is using the host"; exit 1
fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/paper_b_branch_attribution.py" ] || { echo "REFUSED: $WT is frozen at a commit without the runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
typeset -A CHAIN
CHAIN[0]="BASE-60_ATK4-45 BASE-62_GREEDY OBS-62_HUG7 GAM-62_ATK4-45 OBSGAM-61_GREEDY"
CHAIN[1]="BASE-60_HUG7 OBS-60_ATK4-45 OBS-62_GREEDY GAM-62_HUG7 OBSGAM-62_ATK4-45"
CHAIN[2]="BASE-60_GREEDY OBS-60_HUG7 GAM-60_ATK4-45 GAM-62_GREEDY OBSGAM-62_HUG7"
CHAIN[3]="BASE-61_ATK4-45 OBS-60_GREEDY GAM-60_HUG7 OBSGAM-60_ATK4-45 OBSGAM-62_GREEDY"
CHAIN[4]="BASE-61_HUG7 OBS-61_ATK4-45 GAM-60_GREEDY OBSGAM-60_HUG7 STR14_ATK4-45"
CHAIN[5]="BASE-61_GREEDY OBS-61_HUG7 GAM-61_ATK4-45 OBSGAM-60_GREEDY RC19_ATK4-45"
CHAIN[6]="BASE-62_ATK4-45 OBS-61_GREEDY GAM-61_HUG7 OBSGAM-61_ATK4-45 STR14_GREEDY"
CHAIN[7]="BASE-62_HUG7 OBS-62_ATK4-45 GAM-61_GREEDY OBSGAM-61_HUG7 RC19_GREEDY"
mkdir -p $LOGS
for k in 0 1 2 3 4 5 6 7; do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=$WT nohup $V scripts/paper_b_branch_attribution.py \
    --cells ${=CHAIN[$k]} > $LOGS/chain_$k.log 2>&1 &
  echo "chain $k: pid $! -> ${CHAIN[$k]}"
done
echo "launched 8 chains; records appear in $MAIN/results/derisk/s27_n1_cells/"
