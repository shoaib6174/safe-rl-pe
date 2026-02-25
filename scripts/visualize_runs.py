#!/usr/bin/env python3
"""Visualize and compare self-play training runs.

Generates a multi-panel figure comparing Run H (curriculum only)
vs Run I (curriculum + opponent pool).

Usage:
    ./venv/bin/python scripts/visualize_runs.py \
        --run_h results/stage3/run_h_curriculum/history.json \
        --run_i results/stage3/run_i_opponent_pool/history.json \
        --output results/stage3/run_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_history(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["history"], data


def plot_run_comparison(run_h_path: str, run_i_path: str | None, output: str):
    hist_h, data_h = load_history(run_h_path)

    has_run_i = run_i_path is not None and Path(run_i_path).exists()
    if has_run_i:
        hist_i, data_i = load_history(run_i_path)

    # Extract phase indices and metrics
    def extract(hist):
        phases = []
        capture = []
        escape = []
        ne_gap = []
        levels = []
        capture_time = []
        for entry in hist:
            p = entry["phase"]
            idx = 0 if p == "S0" else int(p[1:])
            phases.append(idx)
            capture.append(entry["capture_rate"])
            escape.append(entry["escape_rate"])
            ne_gap.append(abs(entry["capture_rate"] - entry["escape_rate"]))
            levels.append(entry.get("curriculum_level", 1))
            capture_time.append(entry.get("mean_capture_time", 0))
        return phases, capture, escape, ne_gap, levels, capture_time

    ph_h, cap_h, esc_h, ne_h, lev_h, ct_h = extract(hist_h)
    if has_run_i:
        ph_i, cap_i, esc_i, ne_i, lev_i, ct_i = extract(hist_i)

    n_panels = 4 if has_run_i else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=False)
    fig.suptitle("Self-Play Training: Run H (Curriculum) vs Run I (Curriculum + Opponent Pool)",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: Run H capture/escape rates with curriculum levels ──
    ax1 = axes[0]
    ax1.set_title("Run H: Curriculum Only (no opponent pool)", fontsize=12)
    ax1.plot(ph_h, cap_h, "o-", color="#d62728", linewidth=2, markersize=6, label="Capture rate (SR_P)")
    ax1.plot(ph_h, esc_h, "s-", color="#1f77b4", linewidth=2, markersize=6, label="Escape rate (SR_E)")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Balance (0.5)")
    ax1.set_ylabel("Rate")
    ax1.set_xlabel("Phase")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(ph_h)
    ax1.set_xticklabels([f"S{p}" for p in ph_h])
    ax1.legend(loc="upper left", fontsize=9)

    # Shade curriculum levels
    level_colors = {1: "#e6f5e6", 2: "#fff3cd", 3: "#fde2cd", 4: "#f5ccd0"}
    level_names = {1: "L1: Close", 2: "L2: Medium", 3: "L3: Obstacles", 4: "L4: Full"}
    prev_level = lev_h[0]
    start_idx = 0
    for j in range(1, len(lev_h)):
        if lev_h[j] != prev_level or j == len(lev_h) - 1:
            end_idx = j if lev_h[j] != prev_level else j + 1
            ax1.axvspan(ph_h[start_idx] - 0.4, ph_h[min(end_idx, len(ph_h)) - 1] + 0.4,
                       alpha=0.3, color=level_colors.get(prev_level, "#eee"))
            prev_level = lev_h[j]
            start_idx = j
    # Handle last segment
    if start_idx < len(lev_h) - 1 or lev_h[-1] == prev_level:
        ax1.axvspan(ph_h[start_idx] - 0.4, ph_h[-1] + 0.4,
                   alpha=0.3, color=level_colors.get(prev_level, "#eee"))

    # Add level legend
    level_patches = [mpatches.Patch(color=level_colors[k], alpha=0.5, label=level_names[k])
                     for k in sorted(set(lev_h))]
    ax1_leg2 = ax1.legend(handles=level_patches, loc="upper right", fontsize=8, title="Curriculum")
    ax1.add_artist(ax1.legend(loc="upper left", fontsize=9))

    # Annotate collapse
    collapse_start = None
    for j, c in enumerate(cap_h):
        if c >= 1.0 and collapse_start is None:
            collapse_start = j
    if collapse_start is not None:
        ax1.annotate("Collapse\n(100% pursuer)",
                    xy=(ph_h[collapse_start], cap_h[collapse_start]),
                    xytext=(ph_h[collapse_start] - 2, 0.7),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=9, color="red", fontweight="bold")

    # ── Panel 2: Run I capture/escape rates (if available) ──
    if has_run_i:
        ax2 = axes[1]
        ax2.set_title("Run I: Curriculum + Opponent Pool (size=5)", fontsize=12)
        ax2.plot(ph_i, cap_i, "o-", color="#d62728", linewidth=2, markersize=6, label="Capture rate (SR_P)")
        ax2.plot(ph_i, esc_i, "s-", color="#1f77b4", linewidth=2, markersize=6, label="Escape rate (SR_E)")
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Balance (0.5)")
        ax2.set_ylabel("Rate")
        ax2.set_xlabel("Phase")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xticks(ph_i)
        ax2.set_xticklabels([f"S{p}" for p in ph_i])
        ax2.legend(loc="upper left", fontsize=9)

        # Shade curriculum levels for Run I
        prev_level = lev_i[0]
        start_idx = 0
        for j in range(1, len(lev_i)):
            if lev_i[j] != prev_level or j == len(lev_i) - 1:
                end_idx = j if lev_i[j] != prev_level else j + 1
                ax2.axvspan(ph_i[start_idx] - 0.4, ph_i[min(end_idx, len(ph_i)) - 1] + 0.4,
                           alpha=0.3, color=level_colors.get(prev_level, "#eee"))
                prev_level = lev_i[j]
                start_idx = j
        if start_idx < len(lev_i) - 1 or lev_i[-1] == prev_level:
            ax2.axvspan(ph_i[start_idx] - 0.4, ph_i[-1] + 0.4,
                       alpha=0.3, color=level_colors.get(prev_level, "#eee"))

        level_patches_i = [mpatches.Patch(color=level_colors[k], alpha=0.5, label=level_names[k])
                          for k in sorted(set(lev_i))]
        ax2.legend(handles=level_patches_i, loc="upper right", fontsize=8, title="Curriculum")
        ax2.add_artist(ax2.legend(loc="upper left", fontsize=9))

    # ── Panel 3: NE Gap comparison ──
    ax3 = axes[2] if has_run_i else axes[1]
    ax3.set_title("Nash Equilibrium Gap Comparison", fontsize=12)
    ax3.plot(ph_h, ne_h, "o-", color="#d62728", linewidth=2, markersize=6,
             label=f"Run H (min={min(ne_h):.2f})")
    if has_run_i:
        ax3.plot(ph_i, ne_i, "s-", color="#2ca02c", linewidth=2, markersize=6,
                 label=f"Run I (min={min(ne_i):.2f})")
    ax3.axhline(y=0.10, color="green", linestyle="--", alpha=0.7, label="Target (η=0.10)")
    ax3.axhline(y=0.30, color="orange", linestyle="--", alpha=0.5, label="Hard fail (0.30)")
    ax3.set_ylabel("NE Gap |SR_P - SR_E|")
    ax3.set_xlabel("Phase")
    ax3.set_ylim(-0.05, 1.1)
    max_phase = max(ph_h[-1], ph_i[-1] if has_run_i else 0)
    ax3.set_xticks(range(0, max_phase + 1))
    ax3.set_xticklabels([f"S{p}" for p in range(0, max_phase + 1)])
    ax3.legend(fontsize=9)
    ax3.fill_between([0, max_phase], 0, 0.10, alpha=0.1, color="green", label="_nolegend_")

    # ── Panel 4: Capture time comparison ──
    ax4 = axes[3] if has_run_i else axes[2]
    ax4.set_title("Mean Capture Time", fontsize=12)
    ax4.plot(ph_h, ct_h, "o-", color="#d62728", linewidth=2, markersize=6, label="Run H")
    if has_run_i:
        ax4.plot(ph_i, ct_i, "s-", color="#2ca02c", linewidth=2, markersize=6, label="Run I")
    ax4.set_ylabel("Capture Time (s)")
    ax4.set_xlabel("Phase")
    ax4.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize self-play training runs")
    parser.add_argument("--run_h", type=str, required=True, help="Path to Run H history.json")
    parser.add_argument("--run_i", type=str, default=None, help="Path to Run I history.json (optional)")
    parser.add_argument("--output", type=str, default="results/stage3/run_comparison.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_run_comparison(args.run_h, args.run_i, args.output)


if __name__ == "__main__":
    main()
