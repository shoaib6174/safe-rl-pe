"""Nash Equilibrium verification and convergence analysis tools.

Provides:
  - compute_exploitability: Train best-response and measure gap
  - plot_ne_convergence: Visualize capture rates and NE gap over phases
  - compute_ne_gap: Simple NE gap from history
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_ne_gap(history: list[dict]) -> list[float]:
    """Compute NE gap from self-play history.

    NE gap = |capture_rate - escape_rate|.
    At Nash Equilibrium, both rates should be roughly equal.

    Args:
        history: List of phase evaluation dicts with 'capture_rate' and 'escape_rate'.

    Returns:
        List of NE gap values per phase.
    """
    return [
        abs(h["capture_rate"] - h["escape_rate"])
        for h in history
        if "capture_rate" in h and "escape_rate" in h
    ]


def analyze_convergence(history: list[dict], eta: float = 0.10) -> dict:
    """Analyze NE convergence from self-play history.

    Args:
        history: List of phase evaluation dicts.
        eta: Convergence threshold.

    Returns:
        Dict with convergence analysis results.
    """
    ne_gaps = compute_ne_gap(history)
    capture_rates = [h["capture_rate"] for h in history if "capture_rate" in h]
    escape_rates = [h["escape_rate"] for h in history if "escape_rate" in h]

    if not ne_gaps:
        return {"converged": False, "reason": "No history data"}

    final_gap = ne_gaps[-1]
    converged = final_gap < eta

    # Check if NE gap is trending down
    if len(ne_gaps) >= 3:
        first_half = np.mean(ne_gaps[:len(ne_gaps) // 2])
        second_half = np.mean(ne_gaps[len(ne_gaps) // 2:])
        trending_down = second_half < first_half
    else:
        trending_down = None

    return {
        "converged": converged,
        "final_ne_gap": final_gap,
        "min_ne_gap": min(ne_gaps),
        "mean_ne_gap": float(np.mean(ne_gaps)),
        "ne_gap_trend": "decreasing" if trending_down else "increasing/flat" if trending_down is not None else "insufficient_data",
        "final_capture_rate": capture_rates[-1] if capture_rates else None,
        "final_escape_rate": escape_rates[-1] if escape_rates else None,
        "n_phases": len(ne_gaps),
    }


def plot_ne_convergence(
    history: list[dict],
    eta: float = 0.10,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot NE convergence: capture/escape rates and NE gap.

    Args:
        history: List of phase evaluation dicts.
        eta: NE convergence threshold (drawn as horizontal line).
        save_path: Path to save figure. If None, not saved.
        show: Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    capture_rates = [h["capture_rate"] for h in history if "capture_rate" in h]
    escape_rates = [h["escape_rate"] for h in history if "escape_rate" in h]
    ne_gaps = compute_ne_gap(history)

    # Extract phase labels
    phases = []
    for h in history:
        if "capture_rate" in h:
            phases.append(h.get("phase", f"P{len(phases)}"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Capture and escape rates
    x = range(len(capture_rates))
    ax1.plot(x, capture_rates, "b-o", label="Capture Rate (Pursuer)", markersize=4)
    ax1.plot(x, escape_rates, "r-s", label="Escape Rate (Evader)", markersize=4)
    ax1.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="50% baseline")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="best")
    ax1.set_title("AMS-DRL Self-Play: Nash Equilibrium Convergence")
    ax1.grid(True, alpha=0.3)

    # Color phases by role
    for i, phase_label in enumerate(phases):
        if "role" in history[i]:
            color = "lightblue" if history[i]["role"] == "pursuer" else "lightyellow"
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.15, color=color)

    # Bottom: NE gap
    ax2.plot(x, ne_gaps, "g-o", markersize=4)
    ax2.axhline(eta, linestyle="--", color="red", alpha=0.7,
                label=f"Convergence threshold (eta={eta})")
    ax2.set_ylabel("NE Gap (|SR_P - SR_E|)")
    ax2.set_xlabel("Self-Play Phase")
    ax2.set_ylim(-0.05, max(ne_gaps) * 1.2 if ne_gaps else 1.0)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Set x-tick labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases, rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)
    return fig
