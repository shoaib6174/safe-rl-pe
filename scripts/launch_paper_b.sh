#!/bin/zsh
# Paper B rounds S26 B1–B4 launch (docs/pathway/paper_b_prereg_rounds_2026-11-02.md). Same gates as launch_s25.sh:
# frozen worktree, results/ linked to the ONE record root, no competing round on the host (S25 included), literal
# cell lists, confirmation by record files not PIDs. Preconditions: the six frozen checkpoints and the S17 records
# the identity gate CG-B2-1 reproduces. No training. Five sweep chains (OMP 4) + two geometry chains (OMP 2) = 24 cores.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s26_worktree
V=$MAIN/venv/bin/python
if pgrep -f "derisk_s2[345]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary|paper_b_" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/paper_b_filter_sweep.py" ] && [ -f "$WT/scripts/paper_b_geometry.py" ] || { echo "REFUSED: $WT is frozen at a commit without the runners"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
PYTHONPATH=$WT $V - <<'PY' || exit 1
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("sweep", pathlib.Path("scripts/paper_b_filter_sweep.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
bad = [f"{p}: {s['ckpt']}" for p, s in M.POLICIES.items() if not s["ckpt"].exists()]
if not M.ATK_CKPT.exists(): bad.append(f"ATK4-45: {M.ATK_CKPT}")
for name, (path, _opp) in M.identity_pairs().items():
    if not path.exists(): bad.append(f"identity {name}: {path}")
if bad:
    print("REFUSED: missing inputs: " + "; ".join(bad)); sys.exit(1)
from stable_baselines3 import PPO
for p in M.POLICIES:
    M.check_width(p, PPO.load(M.POLICIES[p]["ckpt"], device="cpu"))
print(f"inputs confirmed; {len(M.CELLS)} sweep cells pre-registered; widths pinned")
PY
mkdir -p /home/niro-1/s26_logs
# five sweep chains, each a mix of B2 / B3 / B4 cells so the slower predictive cells are spread (86 cells once)
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/paper_b_filter_sweep.py --cells BASE-60_ATK4-45_PLAIN BASE-60_ATK4-45_RECF-5 BASE-60_ATK4-45_RECF-10 BASE-60_ATK4-45_RECF-20 BASE-60_ATK4-45_RECF-40 BASE-60_ATK4-45_RECF-80 BASE-60_ATK4-45_PERT-20 BASE-60_GREEDY_PLAIN BASE-60_GREEDY_RECF-5 BASE-60_GREEDY_RECF-10 BASE-60_GREEDY_RECF-20 BASE-60_GREEDY_RECF-40 BASE-60_GREEDY_RECF-80 BASE-60_GREEDY_PERT-20 BASE-60_ATK4-45_PRED-PLAIN BASE-60_GREEDY_PRED-PLAIN BASE-60_ATK4-45_PRED-RECF-20 BASE-60_GREEDY_PRED-RECF-20 > /home/niro-1/s26_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/paper_b_filter_sweep.py --cells BASE-61_ATK4-45_PLAIN BASE-61_ATK4-45_RECF-5 BASE-61_ATK4-45_RECF-10 BASE-61_ATK4-45_RECF-20 BASE-61_ATK4-45_RECF-40 BASE-61_ATK4-45_RECF-80 BASE-61_ATK4-45_PERT-20 BASE-61_GREEDY_PLAIN BASE-61_GREEDY_RECF-5 BASE-61_GREEDY_RECF-10 BASE-61_GREEDY_RECF-20 BASE-61_GREEDY_RECF-40 BASE-61_GREEDY_RECF-80 BASE-61_GREEDY_PERT-20 BASE-61_ATK4-45_PRED-PLAIN BASE-61_GREEDY_PRED-PLAIN BASE-61_ATK4-45_PRED-RECF-20 BASE-61_GREEDY_PRED-RECF-20 > /home/niro-1/s26_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/paper_b_filter_sweep.py --cells BASE-62_ATK4-45_PLAIN BASE-62_ATK4-45_RECF-5 BASE-62_ATK4-45_RECF-10 BASE-62_ATK4-45_RECF-20 BASE-62_ATK4-45_RECF-40 BASE-62_ATK4-45_RECF-80 BASE-62_ATK4-45_PERT-20 BASE-62_GREEDY_PLAIN BASE-62_GREEDY_RECF-5 BASE-62_GREEDY_RECF-10 BASE-62_GREEDY_RECF-20 BASE-62_GREEDY_RECF-40 BASE-62_GREEDY_RECF-80 BASE-62_GREEDY_PERT-20 BASE-62_ATK4-45_PRED-PLAIN BASE-62_GREEDY_PRED-PLAIN BASE-62_ATK4-45_PRED-RECF-20 BASE-62_GREEDY_PRED-RECF-20 > /home/niro-1/s26_logs/c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/paper_b_filter_sweep.py --cells STR14_ATK4-45_PLAIN STR14_ATK4-45_RECF-5 STR14_ATK4-45_RECF-10 STR14_ATK4-45_RECF-20 STR14_ATK4-45_RECF-40 STR14_ATK4-45_RECF-80 STR14_ATK4-45_PERT-20 STR14_GREEDY_PLAIN STR14_GREEDY_RECF-5 STR14_GREEDY_RECF-10 STR14_GREEDY_RECF-20 STR14_GREEDY_RECF-40 STR14_GREEDY_RECF-80 STR14_GREEDY_PERT-20 RC19_ATK4-45_PRED-PLAIN RC19_GREEDY_PRED-PLAIN > /home/niro-1/s26_logs/c3.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/paper_b_filter_sweep.py --cells RC19_ATK4-45_PLAIN RC19_ATK4-45_RECF-5 RC19_ATK4-45_RECF-10 RC19_ATK4-45_RECF-20 RC19_ATK4-45_RECF-40 RC19_ATK4-45_RECF-80 RC19_ATK4-45_PERT-20 RC19_GREEDY_PLAIN RC19_GREEDY_RECF-5 RC19_GREEDY_RECF-10 RC19_GREEDY_RECF-20 RC19_GREEDY_RECF-40 RC19_GREEDY_RECF-80 RC19_GREEDY_PERT-20 RC19_ATK4-45_PRED-RECF-20 RC19_GREEDY_PRED-RECF-20 > /home/niro-1/s26_logs/c4.log 2>&1 &
# the geometry round (B1): 8 cells, two chains (deterministic / stochastic)
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=2 $V -u scripts/paper_b_geometry.py --cells BASE-60_det BASE-61_det BASE-62_det RC19_det > /home/niro-1/s26_logs/b1_det.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=2 $V -u scripts/paper_b_geometry.py --cells BASE-60_stoch BASE-61_stoch BASE-62_stoch RC19_stoch > /home/niro-1/s26_logs/b1_stoch.log 2>&1 &
echo "7 chains started; confirm by results/derisk/s26_b2_cells/*.json and s26_b1_cells/*.json, not by PIDs"
echo "then: --gifs (CG-B2-3: 8; CG-B1-2: 4) and WATCH them BEFORE --summarize"
