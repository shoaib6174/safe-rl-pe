#!/usr/bin/env python3
"""Backfill TensorBoard eval/ scalars from history.json for all runs."""

import json
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def backfill(results_dir: Path):
    for run_dir in sorted(results_dir.iterdir()):
        hist_file = run_dir / "history.json"
        if not hist_file.exists():
            continue

        data = json.loads(hist_file.read_text())
        entries = data.get("history", [])
        if not entries:
            continue

        # Skip if already backfilled
        eval_dir = run_dir / "tb" / "eval"
        if eval_dir.exists() and any(eval_dir.iterdir()):
            print(f"  {run_dir.name}: eval/ already exists, skipping")
            continue

        writer = SummaryWriter(str(eval_dir))
        for e in entries:
            step = e.get("total_steps", 0)
            if step == 0:
                continue
            if "capture_rate" in e:
                writer.add_scalar("eval/capture_rate", e["capture_rate"], step)
            if "escape_rate" in e:
                writer.add_scalar("eval/escape_rate", e["escape_rate"], step)
            cr = e.get("capture_rate", 0)
            er = e.get("escape_rate", 0)
            writer.add_scalar("eval/ne_gap", abs(cr - er), step)
            if "p_full_obs" in e:
                writer.add_scalar("eval/p_full_obs", e["p_full_obs"], step)
        writer.flush()
        writer.close()
        print(f"  {run_dir.name}: wrote {len(entries)} eval points")


if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results.is_dir():
        print(f"Error: {results} not found")
        sys.exit(1)
    print(f"Backfilling from {results}")
    backfill(results)
    print("Done.")
