#!/usr/bin/env python3
"""Experiment 1b: Matrix-game pure-strategy commitment asymmetry.

In deep RL, policies are typically deterministic (pure strategies). The pursuer's
∃ problem means it only needs one deterministic strategy that catches some
evader; the evader's ∀ problem means its deterministic strategy must escape all
pursuers. We demonstrate this in a small zero-sum matrix game by computing:

1. Nash equilibrium value (mixed strategies allowed).
2. Pursuer-first pure-strategy commitment value: min_p max_e V(p,e).
   This is the value the pursuer can guarantee by committing to a pure strategy.
3. Evader-first pure-strategy commitment value: max_e min_p V(p,e).
   This is the value the evader can guarantee by committing to a pure strategy.

In a symmetric game the mixed Nash is balanced, but the pure-strategy
commitment values favor the pursuer, mirroring the structural bias observed in
deep RL Pop-FSP.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiment_1_tabular'
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# A 2x2 zero-sum game. Rows = pursuer pure strategies, columns = evader pure
# strategies. Payoff is capture probability from pursuer's perspective.
M = np.array([
    [0.70, 0.40],
    [0.30, 0.60],
])


def nash_value(M):
    """Compute Nash equilibrium value of a zero-sum matrix game."""
    from scipy.optimize import linprog
    n_p, n_e = M.shape
    c = [0.0] * n_p + [-1.0]
    A_ub = []
    for j in range(n_e):
        A_ub.append([-M[i, j] for i in range(n_p)] + [1.0])
    A_eq = [[1.0] * n_p + [0.0]]
    res = linprog(c, A_ub=A_ub, b_ub=[0.0] * n_e,
                  A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0.0, 1.0)] * n_p + [(None, None)],
                  method='highs')
    return float(res.x[-1])


def commitment_values(M):
    """Compute pure-strategy Stackelberg / commitment values.

    Returns:
        pursuer_first: min_p max_e M[p,e]  (pursuer commits, evader responds)
        evader_first:  max_e min_p M[p,e]  (evader commits, pursuer responds)
    """
    pursuer_first = float(np.min(np.max(M, axis=1)))
    evader_first = float(np.max(np.min(M, axis=0)))
    return pursuer_first, evader_first


def run_matrix_analysis():
    nash_v = nash_value(M)
    p_first, e_first = commitment_values(M)

    print(f'Nash value: {nash_v:.3f}')
    print(f'Pursuer-first pure commitment value: {p_first:.3f}')
    print(f'Evader-first pure commitment value: {e_first:.3f}')

    # Bar chart.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ['Nash\n(mixed)', 'Pursuer commits\nfirst (pure)', 'Evader commits\nfirst (pure)']
    values = [nash_v, p_first, e_first]
    colors = ['#2ca02c', '#d62728', '#1f77b4']

    bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
    ax.axhspan(0.45, 0.55, color='green', alpha=0.12, label='Balanced band')
    ax.axhline(0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.8)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_ylabel('Capture rate $CR$')
    ax.set_title('Pure-strategy commitment favors the pursuer')
    ax.set_ylim(0.0, 0.8)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'experiment_1b_matrix_commitment.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved figure: {fig_path}')
    plt.close()

    # Summary.
    summary_path = RESULTS_DIR / 'summary_matrix.md'
    with open(summary_path, 'w') as f:
        f.write('# Experiment 1b — Pure-Strategy Commitment Asymmetry\n\n')
        f.write('Payoff matrix (pursuer capture probability):\n\n')
        f.write('```\n')
        f.write('      E1   E2\n')
        f.write(f'P1   {M[0,0]:.2f}  {M[0,1]:.2f}\n')
        f.write(f'P2   {M[1,0]:.2f}  {M[1,1]:.2f}\n')
        f.write('```\n\n')
        f.write('| Equilibrium concept | Value | Interpretation |\n')
        f.write('|---------------------|-------|----------------|\n')
        f.write(f'| Nash (mixed strategies) | **{nash_v:.3f}** | Balanced; both players randomize. |\n')
        f.write(f'| Pursuer commits first (pure) | **{p_first:.3f}** | Pursuer can force a high CR by specializing. |\n')
        f.write(f'| Evader commits first (pure) | **{e_first:.3f}** | Evader cannot force balance with a single pure strategy. |\n')
        f.write('\n')
        f.write('## Interpretation\n\n')
        f.write('The mixed-strategy Nash equilibrium is balanced, but when players are ')
        f.write('restricted to pure strategies and one player commits first, the pursuer ')
        f.write(f'obtains a value of {p_first:.3f} while the evader can only guarantee {e_first:.3f}. ')
        f.write('This gap mirrors Pop-FSP in deep RL: the pursuer effectively commits to a ')
        f.write('specialized deterministic policy (the ∃ problem), while the evader must be ')
        f.write('robust to all pursuer policies (the ∀ problem) with a single policy.\n')

    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    run_matrix_analysis()
