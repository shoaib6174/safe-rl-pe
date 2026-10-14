#!/bin/zsh
# S24.3-PR GATE launch -- the 18 ratio-matched cells, three chains from a frozen worktree.
# Same gates as launch_s24_1.sh (frozen worktree, results/ linked to the ONE record root, no
# competing jobs, literal cell lists, confirmation by record files not PIDs) PLUS this round's Step 1
# preconditions: all six arms' FINAL 3 M checkpoints exist, and every probe JSON reads
# inapplicable_picks == 0 and train_violations == 0 AND carries the arm's declared ratio (prereg
# section 6 CG-P1/CG-P2, section 5 H-SAFE). The runner itself re-asserts record validity, the
# checkpoints, CG-Z1 on each built env and the re-read CG-P1 before its first episode, and refuses
# the summary if any construct gate does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_3pr_gate_worktree
V=$MAIN/venv/bin/python
ARMS=(SELr0.8-80 SELr0.8-81 SELr0.8-82 SELr1.0-80 SELr1.0-81 SELr1.0-82)
if pgrep -f "derisk_s2[34]_\|replay_s18_branch_split\|s24_tactic_game\|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host (training arms included)"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_3pr_gate.py" ] || { echo "REFUSED: $WT is frozen at a commit without the gate runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in $ARMS; do
  # Reading 1 of the trainer: the DIRECTORY is this round's, the file stem is S24.1's.
  [ -f "$MAIN/results/derisk/ckpts_S24-3PR-$a/S24-1-${a}_3000000.zip" ] \
    || { echo "REFUSED: no final 3M checkpoint for arm $a"; exit 1; }
  [ -f "$MAIN/results/derisk/s24_3pr_probe_$a.json" ] || { echo "REFUSED: no probe JSON for arm $a"; exit 1; }
done
$V - "$MAIN" $ARMS <<'PY' || exit 1
# CG-P2 / H-SAFE evidence from training, and the declared ratio in every probe row. A probe row that
# does not carry the counters is ABSENT EVIDENCE, not a zero (S24.1 A3.1), and a row whose ratio is
# not the arm's own would make the gate's pairing compare two different games.
import json, pathlib, sys
main, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
bad = []
for arm in arms:
    want = float(arm.removeprefix("SELr").split("-")[0])
    rows = json.loads((main / f"results/derisk/s24_3pr_probe_{arm}.json").read_text())["checkpoints"]
    if not rows:
        bad.append(f"{arm}: no probe rows")
    for r in rows:
        for key in ("inapplicable_picks", "train_inapplicable_picks", "violations", "train_violations"):
            if key not in r:
                bad.append(f"{arm} mark {r.get('mark')}: {key} absent (A3.1: absent evidence is an offender)")
            elif int(r[key]):
                bad.append(f"{arm} mark {r.get('mark')}: {key} = {r[key]}")
        if r.get("ratio") != want:
            bad.append(f"{arm} mark {r.get('mark')}: ratio = {r.get('ratio')}, not {want}")
if bad:
    print("REFUSED: CG-P2 / H-SAFE evidence from training is not clean: " + "; ".join(bad))
    sys.exit(1)
print(f"CG-P2 and H-SAFE clean over {len(arms)} arms' probe rows")
PY
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_3pr_gate_logs
# 3 chains x 6 literal cells, OMP 4 each = 12 threads
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3pr_gate.py --cells SELr0.8-80_GREEDY_r0.8 SELr0.8-80_HUG_r0.8 SELr0.8-80_TACTIC-V0_r0.8 SELr0.8-81_GREEDY_r0.8 SELr0.8-81_HUG_r0.8 SELr0.8-81_TACTIC-V0_r0.8 > /home/niro-1/s24_3pr_gate_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3pr_gate.py --cells SELr0.8-82_GREEDY_r0.8 SELr0.8-82_HUG_r0.8 SELr0.8-82_TACTIC-V0_r0.8 SELr1.0-80_GREEDY_r1.0 SELr1.0-80_HUG_r1.0 SELr1.0-80_TACTIC-V0_r1.0 > /home/niro-1/s24_3pr_gate_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3pr_gate.py --cells SELr1.0-81_GREEDY_r1.0 SELr1.0-81_HUG_r1.0 SELr1.0-81_TACTIC-V0_r1.0 SELr1.0-82_GREEDY_r1.0 SELr1.0-82_HUG_r1.0 SELr1.0-82_TACTIC-V0_r1.0 > /home/niro-1/s24_3pr_gate_logs/c2.log 2>&1 &
echo "3 chains started; confirm by results/derisk/s24_3pr_cells/*.json, not by PIDs"
echo "then: --gifs (CG-P3, 12 GIFs) BEFORE --summarize"
