#!/bin/zsh
# S25 value sweep launch -- 80 evaluation cells (4 pursuers x 4 evaders x 5 filter settings), five chains
# (one per setting, 16 cells each) from a frozen worktree. Same gates as launch_s24_2b.sh: frozen worktree,
# results/ linked to the ONE record root, no competing jobs, literal cell lists, confirmation by record
# files not PIDs. Preconditions: the three frozen checkpoints this round loads (SEL-81, SEL3-80, EVS-80),
# resolved through the runner's own functions, never a literal. No training. OMP 4 x 5 = 20 of 24 cores.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s25_worktree
V=$MAIN/venv/bin/python
if pgrep -f "derisk_s2[345]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s25_value_sweep.py" ] || { echo "REFUSED: $WT is frozen at a commit without the runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
PYTHONPATH=$WT $V - <<'PY' || exit 1
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("s25", pathlib.Path("scripts/derisk_s25_value_sweep.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
bad = [f"{r}: {M.ckpt_path(r)}" for r in ("SEL-81", "SEL3-80") if not M.ckpt_path(r).exists()]
if not M.T2.ckpt_path(M.EVS_BEST).exists():
    bad.append(f"{M.EVS_BEST}: {M.T2.ckpt_path(M.EVS_BEST)}")
if bad:
    print("REFUSED: missing frozen checkpoints: " + "; ".join(bad)); sys.exit(1)
print(f"checkpoints confirmed; {len(M.CELLS)} cells pre-registered")
PY
mkdir -p /home/niro-1/s25_logs
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s25_value_sweep.py --cells RULES-V0_GREEDY_NONE RULES-V0_HUG_NONE RULES-V0_TACTIC-V0_NONE RULES-V0_EVS-80_NONE SEL-81_GREEDY_NONE SEL-81_HUG_NONE SEL-81_TACTIC-V0_NONE SEL-81_EVS-80_NONE SEL3-80_GREEDY_NONE SEL3-80_HUG_NONE SEL3-80_TACTIC-V0_NONE SEL3-80_EVS-80_NONE FULLOBS_GREEDY_NONE FULLOBS_HUG_NONE FULLOBS_TACTIC-V0_NONE FULLOBS_EVS-80_NONE > /home/niro-1/s25_logs/c_NONE.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s25_value_sweep.py --cells RULES-V0_GREEDY_d0.00 RULES-V0_HUG_d0.00 RULES-V0_TACTIC-V0_d0.00 RULES-V0_EVS-80_d0.00 SEL-81_GREEDY_d0.00 SEL-81_HUG_d0.00 SEL-81_TACTIC-V0_d0.00 SEL-81_EVS-80_d0.00 SEL3-80_GREEDY_d0.00 SEL3-80_HUG_d0.00 SEL3-80_TACTIC-V0_d0.00 SEL3-80_EVS-80_d0.00 FULLOBS_GREEDY_d0.00 FULLOBS_HUG_d0.00 FULLOBS_TACTIC-V0_d0.00 FULLOBS_EVS-80_d0.00 > /home/niro-1/s25_logs/c_d000.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s25_value_sweep.py --cells RULES-V0_GREEDY_d0.05 RULES-V0_HUG_d0.05 RULES-V0_TACTIC-V0_d0.05 RULES-V0_EVS-80_d0.05 SEL-81_GREEDY_d0.05 SEL-81_HUG_d0.05 SEL-81_TACTIC-V0_d0.05 SEL-81_EVS-80_d0.05 SEL3-80_GREEDY_d0.05 SEL3-80_HUG_d0.05 SEL3-80_TACTIC-V0_d0.05 SEL3-80_EVS-80_d0.05 FULLOBS_GREEDY_d0.05 FULLOBS_HUG_d0.05 FULLOBS_TACTIC-V0_d0.05 FULLOBS_EVS-80_d0.05 > /home/niro-1/s25_logs/c_d005.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s25_value_sweep.py --cells RULES-V0_GREEDY_d0.10 RULES-V0_HUG_d0.10 RULES-V0_TACTIC-V0_d0.10 RULES-V0_EVS-80_d0.10 SEL-81_GREEDY_d0.10 SEL-81_HUG_d0.10 SEL-81_TACTIC-V0_d0.10 SEL-81_EVS-80_d0.10 SEL3-80_GREEDY_d0.10 SEL3-80_HUG_d0.10 SEL3-80_TACTIC-V0_d0.10 SEL3-80_EVS-80_d0.10 FULLOBS_GREEDY_d0.10 FULLOBS_HUG_d0.10 FULLOBS_TACTIC-V0_d0.10 FULLOBS_EVS-80_d0.10 > /home/niro-1/s25_logs/c_d010.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s25_value_sweep.py --cells RULES-V0_GREEDY_d0.20 RULES-V0_HUG_d0.20 RULES-V0_TACTIC-V0_d0.20 RULES-V0_EVS-80_d0.20 SEL-81_GREEDY_d0.20 SEL-81_HUG_d0.20 SEL-81_TACTIC-V0_d0.20 SEL-81_EVS-80_d0.20 SEL3-80_GREEDY_d0.20 SEL3-80_HUG_d0.20 SEL3-80_TACTIC-V0_d0.20 SEL3-80_EVS-80_d0.20 FULLOBS_GREEDY_d0.20 FULLOBS_HUG_d0.20 FULLOBS_TACTIC-V0_d0.20 FULLOBS_EVS-80_d0.20 > /home/niro-1/s25_logs/c_d020.log 2>&1 &
echo "5 chains started; confirm by results/derisk/s25_cells/*.json, not by PIDs"
echo "then: --gifs (CG-V2, 12 GIFs) BEFORE --summarize"
