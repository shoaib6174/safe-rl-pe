#!/bin/zsh
# S24.2 Step E.3 GATE launch -- 18 cells (3 pursuer arms x 6 evader opponents), three chains from a
# frozen worktree. Same gates as launch_s24_2.sh / launch_s24_os.sh / launch_s24_1.sh: frozen worktree,
# results/ linked to the ONE record root, no competing jobs, literal cell lists, confirmation by record
# files not PIDs, PLUS this round's preconditions: all three arms' FINAL 3M checkpoints exist (resolved
# through the TRAINER's own ckpt_path, never a literal) and every probe JSON reads violations == 0 and
# train_violations == 0 and is present (A3.1: absent evidence is an offender, never a zero). The runner
# itself re-asserts record validity and the inherited CG-T1/A2.1 before any capture rate is read, and
# refuses the summary if any construct gate or H-SAFE does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_2b_gate_worktree
V=$MAIN/venv/bin/python
ARMS=(SEL3-80 SEL3-81 SEL3-82)
if pgrep -f "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host (training arms included)"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_2b_gate.py" ] || { echo "REFUSED: $WT is frozen at a commit without the gate runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
$V - "$WT" $ARMS <<'PY' || exit 1
# The checkpoint path is the TRAINER's own ckpt_path, never a literal; probe evidence is read exactly
# as the trainer wrote it (A3.1: absent evidence is an offender, never a zero).
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("derisk_s24_2b_train", pathlib.Path("scripts/derisk_s24_2b_train.py"))
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)
bad = []
for a in sys.argv[2:]:
    ckpt = TR.ckpt_path(a)
    if not ckpt.exists():
        bad.append(f"{a}: no final checkpoint at {ckpt}")
    probe_path = TR._probe_file(a)
    if not probe_path.exists():
        bad.append(f"{a}: no probe JSON at {probe_path}")
        continue
    rows = json.loads(probe_path.read_text()).get("checkpoints", [])
    if not rows:
        bad.append(f"{a}: no probe rows")
    for r in rows:
        for key in ("violations", "train_violations"):
            if key not in r:
                bad.append(f"{a} mark {r.get('mark')}: {key} absent (A3.1)")
            elif int(r[key]):
                bad.append(f"{a} mark {r.get('mark')}: {key} = {r[key]}")
if bad:
    print("REFUSED: " + "; ".join(bad))
    sys.exit(1)
print(f"checkpoints and clean probe rows confirmed over {len(sys.argv[2:])} arms")
PY
mkdir -p /home/niro-1/s24_2b_gate_logs
# one chain per pursuer arm: its 6 evader-opponent cells. OMP 4 each.
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_2b_gate.py --cells SEL3-80_TACTIC-V0 SEL3-80_EVS-80 SEL3-80_EVS-81 SEL3-80_EVS-82 SEL3-80_GREEDY SEL3-80_HUG > /home/niro-1/s24_2b_gate_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_2b_gate.py --cells SEL3-81_TACTIC-V0 SEL3-81_EVS-80 SEL3-81_EVS-81 SEL3-81_EVS-82 SEL3-81_GREEDY SEL3-81_HUG > /home/niro-1/s24_2b_gate_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_2b_gate.py --cells SEL3-82_TACTIC-V0 SEL3-82_EVS-80 SEL3-82_EVS-81 SEL3-82_EVS-82 SEL3-82_GREEDY SEL3-82_HUG > /home/niro-1/s24_2b_gate_logs/c2.log 2>&1 &
echo "3 chains started; confirm by results/derisk/s24_2b_cells/*.json, not by PIDs"
echo "then: --gifs (18 GIFs) BEFORE --summarize"
