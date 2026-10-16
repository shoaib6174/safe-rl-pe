#!/bin/zsh
# S24-FLAT GATE launch -- the 27 evaluation cells (FLAT/FLAT40/STRUCT x3 seeds x3 opponents), four
# chains from a frozen worktree. Same gates as launch_s24_3pr.sh (frozen worktree, results/ linked to
# the ONE record root, no competing jobs, literal cell lists, confirmation by record files not PIDs,
# the now-correct pgrep guard with plain |) PLUS this round's Step F.3 preconditions: all NINE arms'
# FINAL 3M checkpoints exist (A1.2 added FLAT40 -- nine arms, not the brief's older six), and every
# arm's probe JSON reads violations == 0 and train_violations == 0 (prereg section 5 H-SAFE; CG-F3:
# there is no mask, so inapplicable_picks is not checked here). The runner itself re-asserts record
# validity, the checkpoints and CG-F1 on each family's built env before its first episode, and
# refuses the summary if CG-F1 or H-SAFE does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_flat_gate_worktree
V=$MAIN/venv/bin/python
ARMS_FLAT=(FLAT-80 FLAT-81 FLAT-82 FLAT40-80 FLAT40-81 FLAT40-82)
ARMS_STRUCT=(STRUCT-80 STRUCT-81 STRUCT-82)
ARMS=($ARMS_FLAT $ARMS_STRUCT)
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host (training arms included)"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_flat_gate.py" ] || { echo "REFUSED: $WT is frozen at a commit without the gate runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in $ARMS; do
  # Both trainers agree on the `S24-{ARM}` checkpoint stem (reading 9 of the gate's docstring).
  [ -f "$MAIN/results/derisk/ckpts_S24-$a/S24-${a}_3000000.zip" ] \
    || { echo "REFUSED: no final 3M checkpoint for arm $a"; exit 1; }
done
for a in $ARMS_FLAT; do
  [ -f "$MAIN/results/derisk/s24_flat_probe_$a.json" ] || { echo "REFUSED: no probe JSON for arm $a"; exit 1; }
done
for a in $ARMS_STRUCT; do
  [ -f "$MAIN/results/derisk/s24_struct_probe_$a.json" ] || { echo "REFUSED: no probe JSON for arm $a"; exit 1; }
done
$V - "$MAIN" $ARMS_FLAT -- $ARMS_STRUCT <<'PY' || exit 1
# H-SAFE evidence from training and from the probe env (prereg section 5). A probe row that does not
# carry the counter is ABSENT EVIDENCE, not a zero (S24.1 A3.1, inherited by CG-F3). CG-F3: no mask, so
# inapplicable_picks is not a field of these probe rows and is not checked here.
import json, pathlib, sys
main = pathlib.Path(sys.argv[1])
rest = sys.argv[2:]
split = rest.index("--")
flat_arms, struct_arms = rest[:split], rest[split + 1:]
bad = []
for prefix, arms in (("s24_flat", flat_arms), ("s24_struct", struct_arms)):
    for arm in arms:
        path = main / f"results/derisk/{prefix}_probe_{arm}.json"
        rows = json.loads(path.read_text()).get("checkpoints", [])
        if not rows:
            bad.append(f"{arm}: no probe rows")
        for r in rows:
            for key in ("violations", "train_violations"):
                if key not in r:
                    bad.append(f"{arm} mark {r.get('mark')}: {key} absent (A3.1: absent evidence is an offender)")
                elif int(r[key]):
                    bad.append(f"{arm} mark {r.get('mark')}: {key} = {r[key]}")
if bad:
    print("REFUSED: H-SAFE evidence from training is not clean: " + "; ".join(bad))
    sys.exit(1)
print(f"H-SAFE clean over {len(flat_arms) + len(struct_arms)} arms' probe rows")
PY
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_flat_gate_logs
# 4 chains, 9/9/6/3 cells, OMP 4 each -- covers the 27 cells once.
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_flat_gate.py --cells FLAT-80_GREEDY FLAT-80_HUG FLAT-80_TACTIC-V0 FLAT-81_GREEDY FLAT-81_HUG FLAT-81_TACTIC-V0 FLAT-82_GREEDY FLAT-82_HUG FLAT-82_TACTIC-V0 > /home/niro-1/s24_flat_gate_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_flat_gate.py --cells FLAT40-80_GREEDY FLAT40-80_HUG FLAT40-80_TACTIC-V0 FLAT40-81_GREEDY FLAT40-81_HUG FLAT40-81_TACTIC-V0 FLAT40-82_GREEDY FLAT40-82_HUG FLAT40-82_TACTIC-V0 > /home/niro-1/s24_flat_gate_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_flat_gate.py --cells STRUCT-80_GREEDY STRUCT-80_HUG STRUCT-80_TACTIC-V0 STRUCT-81_GREEDY STRUCT-81_HUG STRUCT-81_TACTIC-V0 > /home/niro-1/s24_flat_gate_logs/c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_flat_gate.py --cells STRUCT-82_GREEDY STRUCT-82_HUG STRUCT-82_TACTIC-V0 > /home/niro-1/s24_flat_gate_logs/c3.log 2>&1 &
echo "4 chains started; confirm by results/derisk/s24_flat_cells/*.json, not by PIDs"
echo "then: --gifs (CG-F2, 24 GIFs) BEFORE --summarize"
