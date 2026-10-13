#!/bin/zsh
# S24.3-ZS GATE launch -- the 36 zero-shot speed-sweep cells, four chains from a frozen worktree.
# Same gates as launch_s24_1.sh (frozen worktree, results/ linked to the ONE record root, no
# competing jobs, literal cell lists, confirmation by record files not PIDs, the learned arms'
# FINAL checkpoints present) PLUS this round's own precondition: the 1.2 COLUMN must already be
# on disk and measurable by S24.1's rule, because H-ZS-2 and H-ZS-3 join against it and nothing
# here re-runs it (prereg section 3). The runner itself re-asserts record validity, the
# checkpoints and CG-Z1 before its first episode, and refuses the summary if a gate does not hold.
set -u
MAIN=/home/niro-1/Codes/safe-rl-pe
WT=/home/niro-1/s24_3zs_gate_worktree
V=$MAIN/venv/bin/python
ARMS=(SEL-80 SEL-81 SEL-82 NOMEM-81)
if pgrep -f "derisk_s2[34]_\|replay_s18_branch_split\|s24_tactic_game\|derisk_s18_boundary" >/dev/null; then echo "REFUSED: another round is using the host (training arms included)"; exit 1; fi
[ -d "$WT" ] || { echo "REFUSED: no frozen worktree at $WT"; exit 1; }
[ -f "$WT/scripts/derisk_s24_3zs.py" ] || { echo "REFUSED: $WT is frozen at a commit without the gate runner"; exit 1; }
[ -e "$WT/results" ] || ln -s "$MAIN/results" "$WT/results"
[ -L "$WT/results" ] && [ "$(readlink -f "$WT/results")" = "$(readlink -f "$MAIN/results")" ] \
  || { echo "REFUSED: $WT/results is not a symlink to $MAIN/results"; exit 1; }
for a in $ARMS; do
  [ -f "$MAIN/results/derisk/ckpts_S24-1-$a/S24-1-${a}_3000000.zip" ] \
    || { echo "REFUSED: no final 3M checkpoint for arm $a"; exit 1; }
done
$V - "$MAIN" <<'PY' || exit 1
# The 1.2 column, checked by S24.1's own validity rule (N = 500 on stream 2 400 000) for every
# row x opponent this round joins. A partial or re-measured column is a REFUSAL, not a gap to be
# filled later: the ordering hypothesis would otherwise be computed against whatever is there.
import json, pathlib, sys
main = pathlib.Path(sys.argv[1])
rows = ["SEL-80", "SEL-81", "SEL-82", "NOMEM-81", "LEAD-1.0", "RULES-V0"]
opps = ["GREEDY", "HUG", "TACTIC-V0"]
bad = []
for row in rows:
    for o in opps:
        p = main / "results/derisk/s24_1_cells" / f"{row}_{o}.json"
        if not p.exists():
            bad.append(f"{p.name}: absent")
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception as exc:
            bad.append(f"{p.name}: unreadable: {exc}")
            continue
        if rec.get("verdict") == "UNDETERMINED":
            bad.append(f"{p.name}: UNDETERMINED")
        elif rec.get("n_episodes") != 500 or rec.get("seed0") != 2_400_000:
            bad.append(f"{p.name}: n_episodes={rec.get('n_episodes')} seed0={rec.get('seed0')}")
if bad:
    print("REFUSED: S24.1's 1.2 column is not measurable for every joined cell: " + "; ".join(bad))
    sys.exit(1)
print(f"the 1.2 column reads measurable over {len(rows) * len(opps)} S24.1 cells")
PY
cd $WT && echo "frozen at $(git rev-parse --short HEAD)"
mkdir -p /home/niro-1/s24_3zs_gate_logs
# 4 chains x 9 literal cells, OMP 4 each = 16 threads
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3zs.py --cells SEL-80_GREEDY_r0.8 SEL-80_HUG_r0.8 SEL-80_TACTIC-V0_r0.8 SEL-81_GREEDY_r0.8 SEL-81_HUG_r0.8 SEL-81_TACTIC-V0_r0.8 SEL-82_GREEDY_r0.8 SEL-82_HUG_r0.8 SEL-82_TACTIC-V0_r0.8 > /home/niro-1/s24_3zs_gate_logs/c0.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3zs.py --cells NOMEM-81_GREEDY_r0.8 NOMEM-81_HUG_r0.8 NOMEM-81_TACTIC-V0_r0.8 LEAD-1.0_GREEDY_r0.8 LEAD-1.0_HUG_r0.8 LEAD-1.0_TACTIC-V0_r0.8 RULES-V0_GREEDY_r0.8 RULES-V0_HUG_r0.8 RULES-V0_TACTIC-V0_r0.8 > /home/niro-1/s24_3zs_gate_logs/c1.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3zs.py --cells SEL-80_GREEDY_r1.0 SEL-80_HUG_r1.0 SEL-80_TACTIC-V0_r1.0 SEL-81_GREEDY_r1.0 SEL-81_HUG_r1.0 SEL-81_TACTIC-V0_r1.0 SEL-82_GREEDY_r1.0 SEL-82_HUG_r1.0 SEL-82_TACTIC-V0_r1.0 > /home/niro-1/s24_3zs_gate_logs/c2.log 2>&1 &
nohup env PYTHONPATH=$WT OMP_NUM_THREADS=4 $V -u scripts/derisk_s24_3zs.py --cells NOMEM-81_GREEDY_r1.0 NOMEM-81_HUG_r1.0 NOMEM-81_TACTIC-V0_r1.0 LEAD-1.0_GREEDY_r1.0 LEAD-1.0_HUG_r1.0 LEAD-1.0_TACTIC-V0_r1.0 RULES-V0_GREEDY_r1.0 RULES-V0_HUG_r1.0 RULES-V0_TACTIC-V0_r1.0 > /home/niro-1/s24_3zs_gate_logs/c3.log 2>&1 &
echo "4 chains started; confirm by results/derisk/s24_3zs_cells/*.json, not by PIDs"
echo "then: --gifs (CG-Z2, 12 GIFs) BEFORE --summarize"
