#!/usr/bin/env bash
# Copy SP17b s48/s49 milestone_phase2550 checkpoints into a read-only
# BR_frozen/ snapshot directory. Idempotent — safe to re-run.
#
# Usage (run on niro-2): bash scripts/snapshot_frozen_opponents.sh
set -euo pipefail

BASE="/home/niro-2/Codes/safe-rl-pe/results"
DEST="${BASE}/BR_frozen"
PHASE="milestone_phase2550"

mkdir -p "${DEST}"

for seed in 48 49; do
    for role in pursuer evader; do
        src="${BASE}/SP17b_s${seed}/checkpoints/${role}/${PHASE}_${role}"
        dst_dir="${DEST}/s${seed}/${role}"

        if [[ ! -d "${src}" ]]; then
            echo "ERROR: source missing: ${src}" >&2
            exit 1
        fi

        mkdir -p "${dst_dir}"
        cp -r "${src}/." "${dst_dir}/"
        chmod -R a-w "${dst_dir}"   # read-only — protect from accidental edit
        echo "  snapshot: ${src} -> ${dst_dir}"
    done
done

echo "Snapshot complete. Contents:"
find "${DEST}" -maxdepth 3 -type f | sort
