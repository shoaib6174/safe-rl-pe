#!/usr/bin/env python3
"""Generate polished core figures for the RA-L / ICRA 2027 paper."""

import os
import re
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPRO_DIR = PROJECT_ROOT / 'repro_gonultas_isler_2024'
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def parse_training_log(log_path):
    """Extract (episode, cr) pairs from a training.log file."""
    episodes = []
    crs = []
    pattern = re.compile(r'Ep\s+(\d+)\s+\|\s+CR=([0-9.]+)')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                crs.append(float(m.group(2)))
    return np.array(episodes), np.array(crs)


def rolling_mean(eps, crs, window=50):
    """Compute rolling mean on a regular grid."""
    if len(eps) == 0:
        return np.array([]), np.array([])
    grid = np.arange(eps.min(), eps.max() + 1)
    crs_grid = np.interp(grid, eps, crs)
    if len(grid) < window:
        return grid, crs_grid
    weights = np.ones(window) / window
    smoothed = np.convolve(crs_grid, weights, mode='valid')
    return grid[window - 1:], smoothed


def plot_pop_fsp_trajectories():
    """Figure 1: Pop-FSP CR trajectories across v3-v7."""
    configs = [
        ('v3', '1.67x', [0, 1, 2]),
        ('v4', '1.43x', [0, 1, 2]),
        ('v5', '1.56x', [0, 1, 2]),
        ('v6', '1.52x', [0, 1, 2, 3, 4]),
        ('v7', '1.47x', [0, 1, 2]),
    ]

    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(2, 3, figure=fig, wspace=0.25, hspace=0.35)

    axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)

    balance_color = '#2ca02c'
    seed_color = '#aec7e8'
    mean_color = '#1f77b4'

    for ax, (config, ratio, seeds) in zip(axes, configs):
        all_crs = []
        max_ep = 0

        for seed in seeds:
            log_path = REPRO_DIR / f'results_pop_fsp_{config}' / f'seed_{seed}' / 'training.log'
            if not log_path.exists():
                continue
            eps, crs = parse_training_log(log_path)
            if len(eps) == 0:
                continue

            # Smooth per-seed trajectory
            grid_smooth, crs_smooth = rolling_mean(eps, crs, window=50)
            if len(grid_smooth) > 0:
                ax.plot(grid_smooth / 1000, crs_smooth, color=seed_color,
                        alpha=0.6, linewidth=0.9, zorder=1)
                all_crs.append((grid_smooth, crs_smooth))
                max_ep = max(max_ep, grid_smooth.max())

        # Population mean on common grid
        if all_crs:
            grid = np.arange(0, max_ep + 1, 2)
            cr_grid = []
            for gx, cr in all_crs:
                cr_grid.append(np.interp(grid, gx, cr, left=np.nan, right=np.nan))
            cr_grid = np.array(cr_grid)
            with np.errstate(invalid='ignore'):
                mean_cr = np.nanmean(cr_grid, axis=0)
            ax.plot(grid / 1000, mean_cr, color=mean_color, linewidth=2.5, zorder=3)

        ax.axhspan(0.45, 0.55, color=balance_color, alpha=0.12, zorder=0)
        ax.set_title(f'{config.upper()}  (speed ratio {ratio})')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 5.2)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[-3:]:
        ax.set_xlabel('Episode ($\\times 10^3$)')
    for ax in axes[::3]:
        ax.set_ylabel('Capture rate $CR$')

    # Use the sixth subplot for legend
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    # Custom legend entries
    mean_line = plt.Line2D([0], [0], color=mean_color, linewidth=2.5, label='Population mean')
    seed_line = plt.Line2D([0], [0], color=seed_color, alpha=0.8, linewidth=1.5, label='Per-seed trajectory')
    band_patch = mpatches.Patch(color=balance_color, alpha=0.25, label='Balanced band $[0.45, 0.55]$')

    ax_legend.legend(handles=[mean_line, seed_line, band_patch],
                     loc='center', frameon=True, fontsize=10,
                     title='Legend', title_fontsize=11)
    ax_legend.text(0.5, 0.15, f'Total: 17 seeds\\n0/17 balanced',
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = FIGURES_DIR / 'figure_1_pop_fsp_trajectories.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {output_path}')
    plt.close()


def plot_forall_exists_schematic():
    """Figure 2: ∀/∃ asymmetry schematic."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    pursuer_color = '#d62728'
    evader_color = '#1f77b4'
    highlight_color = '#ff7f0e'

    def draw_agent(ax, x, y, color, label, radius=0.11):
        circle = plt.Circle((x, y), radius, color=color, ec='black',
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=13,
                color='white', weight='bold', zorder=6)

    def draw_cloud(ax, cx, cy, n, color, label, radius=0.08):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        positions = []
        for angle in angles:
            x = cx + 0.42 * np.cos(angle)
            y = cy + 0.24 * np.sin(angle)
            circle = plt.Circle((x, y), radius, color=color, ec='black',
                                alpha=0.75, linewidth=1, zorder=3)
            ax.add_patch(circle)
            positions.append((x, y))
        ax.text(cx, cy + 0.58, label, ha='center', va='bottom',
                fontsize=11, weight='bold', zorder=7)
        return positions

    # Left panel: Pursuer (∃)
    ax_left.set_xlim(-0.15, 1.25)
    ax_left.set_ylim(-0.15, 1.05)
    ax_left.set_aspect('equal')
    ax_left.axis('off')
    ax_left.set_title('Pursuer solves $\\exists$', fontsize=13,
                      weight='bold', pad=10)

    draw_agent(ax_left, 0.15, 0.45, pursuer_color, 'P')
    evader_positions = draw_cloud(ax_left, 0.75, 0.45, 5, evader_color,
                                  'Evader population')

    sampled_idx = 1
    sx, sy = evader_positions[sampled_idx]
    circle = plt.Circle((sx, sy), 0.10, color=highlight_color, ec='black',
                        linewidth=2.5, zorder=4)
    ax_left.add_patch(circle)

    ax_left.annotate('', xy=(sx - 0.10, sy), xytext=(0.15 + 0.11, 0.45),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2,
                                    connectionstyle='arc3,rad=0.1'))
    ax_left.text(0.55, -0.02, 'Catch ANY one evader', ha='center',
                 fontsize=10, style='italic')
    ax_left.text(0.55, -0.12, r'$\max_{\pi_P}\, \mathbb{E}_{\pi_E \sim \Pi_E}[V]$',
                 ha='center', fontsize=12)

    # Right panel: Evader (∀)
    ax_right.set_xlim(-0.15, 1.25)
    ax_right.set_ylim(-0.15, 1.05)
    ax_right.set_aspect('equal')
    ax_right.axis('off')
    ax_right.set_title('Evader solves $\\forall$', fontsize=13,
                       weight='bold', pad=10)

    draw_agent(ax_right, 0.15, 0.45, evader_color, 'E')
    pursuer_positions = draw_cloud(ax_right, 0.75, 0.45, 5, pursuer_color,
                                   'Pursuer population')

    for px, py in pursuer_positions:
        ax_right.annotate('', xy=(px - 0.10, py), xytext=(0.15 + 0.11, 0.45),
                          arrowprops=dict(arrowstyle='->', color='black',
                                         lw=1.2, alpha=0.5,
                                         connectionstyle='arc3,rad=0.05'))

    ax_right.text(0.55, -0.02, 'Escape ALL pursuers', ha='center',
                  fontsize=10, style='italic')
    ax_right.text(0.55, -0.12, r'$\max_{\pi_E}\, \mathbb{E}_{\pi_P \sim \Pi_P}[V]$',
                  ha='center', fontsize=12)

    plt.tight_layout()
    output_path = FIGURES_DIR / 'figure_2_forall_exists_schematic.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {output_path}')
    plt.close()


def plot_symmetric_distribution():
    """Figure 3: Symmetric point-mass final CR distribution (full 34-seed cohort)."""
    # Seeds 0-8: canonical values from prior evaluation (S108)
    seeds_0_8 = np.arange(9)
    final_crs_0_8 = np.array([
        1.000,  # seed 0
        0.980,  # seed 1
        0.500,  # seed 2: balanced
        0.480,  # seed 3: balanced
        1.000,  # seed 4
        1.000,  # seed 5
        0.800,  # seed 6
        0.160,  # seed 7
        1.000,  # seed 8
    ])

    # Seeds 9-33: seeded paired evaluation (S117). Edge seeds (within ~2 SE of
    # the balance band: 12, 16, 20, 21, 30) use N=1000 (two init sets);
    # all others N=500. See reeval_seeded_cohort.py.
    eval_json_path = REPRO_DIR / 'results_symmetric' / 'eval_seeds_9_33_final.json'
    seeds_9_33 = np.arange(9, 34)
    final_crs_9_33 = np.full(len(seeds_9_33), np.nan)
    if eval_json_path.exists():
        import json
        with open(eval_json_path) as f:
            eval_data = json.load(f)
        for idx, seed in enumerate(seeds_9_33):
            seed_str = str(seed)
            if seed_str in eval_data and 'capture_rate' in eval_data[seed_str]:
                final_crs_9_33[idx] = eval_data[seed_str]['capture_rate']
    else:
        print(f'Warning: {eval_json_path} not found; using seeds 0-8 only.')

    seeds = np.concatenate([seeds_0_8, seeds_9_33])
    final_crs = np.concatenate([final_crs_0_8, final_crs_9_33])
    balanced = (final_crs >= 0.45) & (final_crs <= 0.55)
    n_total = len(final_crs)
    n_balanced = int(balanced.sum())

    # Categorize for summary counts
    pursuer_lock = np.sum(final_crs > 0.55)
    evader_lock = np.sum(final_crs < 0.45)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-seed strip plot: x=CR, y=seed index
    colors = np.where(balanced, '#2ca02c', '#d62728')
    for seed, cr, is_bal, color in zip(seeds, final_crs, balanced, colors):
        if np.isnan(cr):
            continue
        ax.plot(cr, seed, 'o', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1.0, zorder=5)
        # Seed label slightly offset to avoid overlap
        ha = 'left' if cr < 0.5 else 'right'
        offset = 0.02 if cr < 0.5 else -0.02
        ax.text(cr + offset, seed, str(seed), ha=ha, va='center',
                fontsize=7, color='black')

    # Balanced band
    ax.axvspan(0.45, 0.55, color='#2ca02c', alpha=0.12, zorder=0)
    ax.axvline(0.45, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(0.55, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Final capture rate $CR$ (pursuer vs self-play)', fontsize=12)
    ax.set_ylabel('Seed', fontsize=12)
    ax.set_title(f'Symmetric point-mass ablation: {n_balanced}/{n_total} seeds balanced',
                 fontsize=13, pad=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, n_total)
    ax.set_yticks(seeds)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

    # Summary text
    summary = (
        f'Balanced (CR∈[0.45,0.55]): {n_balanced}\n'
        f'Pursuer lock-in (CR>0.55): {pursuer_lock}\n'
        f'Evader lock-in (CR<0.45): {evader_lock}'
    )
    ax.text(0.02, 0.98, summary,
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = FIGURES_DIR / 'figure_3_symmetric_distribution.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {output_path}')
    plt.close()


if __name__ == '__main__':
    plot_pop_fsp_trajectories()
    plot_forall_exists_schematic()
    plot_symmetric_distribution()
    print(f'\nAll figures saved to {FIGURES_DIR}')
