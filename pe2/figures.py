"""Step 5: figures — eval dashboard regenerated from JSON alone (gate
requirement: no env access, no re-running duels; matplotlib Agg, so output
is deterministic given the same JSON).
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MATCHUP_ORDER = ["Gp_x_Ge", "Gp_x_FLe", "FLp_x_Ge", "FLp_x_FLe"]


def _cr_panel(ax, summaries: dict) -> None:
    """CR matrix with Wilson CIs (bar chart + CI whiskers + n annotations)."""
    names = [m for m in MATCHUP_ORDER if m in summaries]
    crs = [summaries[m]["cr"] for m in names]
    los = [summaries[m]["cr_ci"][0] for m in names]
    his = [summaries[m]["cr_ci"][1] for m in names]
    errs = [
        [c - lo for c, lo in zip(crs, los, strict=True)],
        [hi - c for c, hi in zip(crs, his, strict=True)],
    ]
    bars = ax.bar(names, crs, yerr=errs, capsize=5, color="#4c72b0", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("capture rate (pursuer)")
    ax.set_title("CR by matchup with Wilson 95% CI")
    for bar, cr, lo, hi, m in zip(bars, crs, los, his, names, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            hi + 0.03,
            f"{cr:.3f}\n[{lo:.3f}, {hi:.3f}]\nn={summaries[m]['n']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _paired_design_panel(ax, battery: dict) -> None:
    """Paired-design diagram: the same seed sequence feeds every matchup."""
    ax.axis("off")
    n, seed0 = battery["n"], battery["seed0"]
    names = [m for m in MATCHUP_ORDER if m in battery["summaries"]]
    lines = [
        "Paired evaluation design (D57)",
        "",
        f"seed sequence: reset(seed = {seed0} + i),  i = 0..{n - 1}",
        "",
        "every matchup sees the SAME episode inits:",
        "",
    ]
    for m in names:
        s = battery["summaries"][m]
        lines.append(
            f"  {m:<10} <- seeds {seed0}..{seed0 + n - 1}   "
            f"CR {s['cr']:.3f}, mean LOS breaks {s['mean_los_breaks']:.2f}"
        )
    lines += [
        "",
        "paired seeds cancel init-set luck: CR differences are",
        "attributable to the matchup, not the draw (D57/D55).",
        "",
        "regenerated from step_03_battery.json alone (no env access).",
    ]
    ax.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace"
    )
    ax.set_title("Paired-seed protocol")


def build_dashboard(
    battery_json_path: str | pathlib.Path, out_png: str | pathlib.Path
) -> None:
    """Render the Step 5 harness dashboard from a battery JSON file."""
    with open(battery_json_path) as f:
        battery = json.load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)
    _cr_panel(ax1, battery["summaries"])
    _paired_design_panel(ax2, battery)
    fig.suptitle(f"pe2 eval harness dashboard — N={battery['n']} per matchup")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
