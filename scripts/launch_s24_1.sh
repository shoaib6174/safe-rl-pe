#!/bin/zsh
# S24.1 GATE launch -- the 24 evaluation cells, four chains from a frozen worktree.
# Same gates as launch_s23.sh (frozen worktree, results/ linked to the ONE record root, no competing
# jobs, literal cell lists, confirmation by record files not PIDs) PLUS the Step 1.1 preconditions:
# all six arms' FINAL checkpoints exist, and every probe JSON reads inapplicable_picks == 0 and
# train_violations == 0 (prereg section 5 CG-T1, section 4 H-SAFE; A2.4 puts both in every probe row).
# The runner itself re-asserts record validity, the checkpoints and the construct gates before its
# first episode, and refuses the summary if CG-T1 or A2.1 does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_1_worktree
V=$MAIN/venv/bin/python
ARMS=(SEL-80 SEL-81 SEL-82 NOMEM-80 NOMEM-81 NOMEM-82)
if pgrep -f "derisk_s2[34]_\|replay_s18_branch_split\|s24_tactic_game\|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host (training arms included)"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_1_gate.py" ] || { echo "REFUSED: $WT is frozen at a commit without the gate runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in $ARMS; do
  [ -f "$MAIN/results/derisk/ckpts_S24-1-$a/S24-1-${a}_3000000.zip" ] \
    || { echo "REFUSED: no final 3M checkpoint for arm $a"; exit 1; }
  [ -f "$MAIN/results/derisk/s24_1_probe_$a.json" ] || { echo "REFUSED: no probe JSON for arm $a"; exit 1; }
done
$V - "$MAIN" $ARMS <<'PY' || exit 1
import json, pathlib, sys
main, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
bad = []
for arm in arms:
    rows = json.loads((main / f"results/derisk/s24_1_probe_{arm}.json").read_text())["checkpoints"]
    if not rows:
        bad.append(f"{arm}: no probe rows")
    for r in rows:
        for key in ("inapplicable_picks", "train_inapplicable_picks", "violations", "train_violations"):
            if int(r.get(key, 0)):
                bad.append(f"{arm} mark {r.get('mark')}: {key} = {r[key]}")
if bad:
    print("REFUSED: CG-T1 / H-SAFE evidence from training is not clean: " + "; ".join(bad))
    sys.exit(1)
print(f"CG-T1 and H-SAFE clean over {len(arms)} arms' probe rows")
PY
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_1_gate_logs
# 4 chains x 6 literal cells, OMP 4 each = 16 threads
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_1_gate.py --cells SEL-80_GREEDY SEL-80_HUG SEL-80_TACTIC-V0 SEL-81_GREEDY SEL-81_HUG SEL-81_TACTIC-V0 > /home/niro-1/s24_1_gate_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_1_gate.py --cells SEL-82_GREEDY SEL-82_HUG SEL-82_TACTIC-V0 NOMEM-80_GREEDY NOMEM-80_HUG NOMEM-80_TACTIC-V0 > /home/niro-1/s24_1_gate_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_1_gate.py --cells NOMEM-81_GREEDY NOMEM-81_HUG NOMEM-81_TACTIC-V0 NOMEM-82_GREEDY NOMEM-82_HUG NOMEM-82_TACTIC-V0 > /home/niro-1/s24_1_gate_logs/c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_1_gate.py --cells LEAD-1.0_GREEDY LEAD-1.0_HUG LEAD-1.0_TACTIC-V0 RULES-V0_GREEDY RULES-V0_HUG RULES-V0_TACTIC-V0 > /home/niro-1/s24_1_gate_logs/c3.log 2>&1 &
echo "4 chains started; confirm by results/derisk/s24_1_cells/*.json, not by PIDs"
echo "then: --gifs (CG-T3, 30 GIFs) BEFORE --summarize"
