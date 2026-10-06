#!/bin/zsh
# S19 launch — refuses unless CG-B (36/36 exact reproduction) holds and the host is free of Phase 0.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s19_worktree
V=$MAIN/venv/bin/python
if pgrep -f phase0_chain.sh >/dev/null; then echo "REFUSED: Phase 0 replay still running"; exit 1; fi
n=$(ls $MAIN/results/derisk/ | grep -c s18_branch_split)
pass=$($V - <<'PY'
import json, glob
ok = 0
for f in glob.glob("/home/niro-1/Codes/safe-rl-pe/results/derisk/s18_branch_split_*.json"):
    d = json.load(open(f)); v = d.get("reproduction", d.get("verdict", {}))
    ok += ((v.get("verdict") if isinstance(v, dict) else v) in ("PASS", True))
print(ok)
PY
)
echo "CG-B: $pass/$n cells reproduce S18 exactly"
if [ "$n" -lt 36 ] || [ "$pass" -lt 36 ]; then echo "REFUSED: CG-B requires 36/36"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
# 4 chains, disjoint literal cell lists; OMP 4 each = 16 threads
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s19_planner.py --cells LINE-OPEN_GREEDY LINE-OPEN_HUG LINE-OPEN_ATK > /home/niro-1/s19_logs/s19_c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s19_planner.py --cells LINE-SHIELD_GREEDY LINE-SHIELD_HUG LINE-SHIELD_ATK > /home/niro-1/s19_logs/s19_c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s19_planner.py --cells PLAN-OPEN_GREEDY PLAN-OPEN_HUG PLAN-OPEN_ATK > /home/niro-1/s19_logs/s19_c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s19_planner.py --cells PLAN-SHIELD_GREEDY PLAN-SHIELD_HUG PLAN-SHIELD_ATK > /home/niro-1/s19_logs/s19_c3.log 2>&1 &
echo "4 chains started; confirm by results/derisk/s19_cells/*.json, not by PIDs"
