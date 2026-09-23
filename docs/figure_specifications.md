# Figure Specifications — RA-L / ICRA 2027 Paper

**Purpose**: Specifications for the three core figures in the RA-L paper and the supplementary video.  
**Date**: 2026-07-14  
**Data source**: `docs/paper_data_summary.md`

---

## Figure 1 — Pop-FSP CR trajectories

**Type**: Line plot with translucent per-seed lines and thick population mean.

**Subplots**: One panel per configuration (v3, v4, v5, v6, v7) arranged horizontally or in a 2×3 grid.

**Axes**:
- x-axis: Episode / 1000 (range 0–5 for 5K episodes; v6 Seed 1 only has full 5K data)
- y-axis: Capture rate CR ∈ [0, 1]

**Data**:
- v3: 3 seeds, all converge to CR=1.0 quickly
- v4: 3 seeds, all converge to CR≈0.0
- v5: 3 seeds, all converge to CR=1.0
- v6: 5 seeds, Seed 1 has full 5000-episode trajectory with CR≈0.72 mean; others killed earlier
- v7: 3 seeds, lock-in at various points

**Annotations**:
- Horizontal shaded band: CR ∈ [0.45, 0.55] (balanced region)
- Label: "No seed stabilizes in balanced band"

**Style notes**:
- Per-seed lines: alpha=0.3, thin
- Population mean: alpha=1.0, thick, dark color
- Band: light green, alpha=0.2

**Script suggestion**: Use `matplotlib` with data from `results_pop_fsp_v{3-7}/seed_*/training.log` or equivalent CSV.

---

## Figure 2 — ∀/∃ asymmetry schematic

**Type**: Conceptual diagram (TikZ or vector graphics).

**Left panel — Pursuer (∃)**:
- Draw a pursuer agent.
- Draw a cloud of evader policies (3–5 icons).
- Show the pursuer connecting to ONE sampled evader (highlighted).
- Label: "Pursuer needs to catch \textbf{some} evader"
- Symbol: $\exists \pi_E \in \Pi_E$

**Right panel — Evader (∀)**:
- Draw an evader agent.
- Draw a cloud of pursuer policies (3–5 icons).
- Show the evader must escape ALL pursuers (all connected).
- Label: "Evader must escape \textbf{all} pursuers"
- Symbol: $\forall \pi_P \in \Pi_P$

**Bottom annotation**:
- "Existential objective is easier → systematic gradient imbalance → pursuer bias"

**Style notes**:
- Use consistent agent icons (circle/triangle for pursuer, square/diamond for evader).
- Highlight the sampled opponent in the pursuer panel.
- Use red "escape blocked" icon for evader panel if any pursuer catches it.

---

## Figure 3 — Symmetric point-mass CR distribution

**Type**: Strip plot (seed index vs CR) or histogram.

**Data**: Final CR (`pursuer_vs_selfplay`) for seeds 0–33:
- Seeds 0–8: hardcoded canonical values (see `scripts/generate_paper_figures.py`) — original N=50 unseeded protocol (S108)
- Seeds 9–33: `repro_gonultas_isler_2024/results_symmetric/eval_seeds_9_33_final.json` — **seeded paired re-evaluation (S117)**: N=500 (`reeval_seeded_cohort.py`, init seeds 700000+i); band-edge seeds 12/16/20/21/30 use N=1000 combined across two init sets (700000+i, 800000+i). Supersedes `eval_seeds_9_33.json` (N=50 unseeded, irreproducible).

**Axes**:
- x-axis: Final capture rate CR ∈ [0, 1]
- y-axis: Seed index 0–33

**Annotations**:
- Vertical shaded band: CR ∈ [0.45, 0.55] (balanced region)
- Vertical dashed lines at CR = 0.45 and CR = 0.55
- Text annotation with balance tally (e.g., "5/34 seeds balanced")
- Color-code markers: green if in band, red otherwise

**Style notes**:
- Marker size ~8
- Show seed number next to each marker
- Summary count box: balanced / pursuer lock-in / evader lock-in
- Keep layout readable for 34 seeds (taller figure than 9-seed version)

**Script**: `scripts/generate_paper_figures.py::plot_symmetric_distribution()`

---

## Figure 3 (paper numbering) — Controlled tabular analyses

**Added in S116.** In `ral_main.tex` this is Figure 3 (`fig:tabular`), placed in Results §IV-C. The symmetric-distribution figure becomes Figure 4 in the paper.

**Type**: Two-panel subfigure (side by side, `subcaption`).

**Panel (a) — Exact best-response regimes**:
- Source: `paper/figures/experiment_1_tabular_regimes.{pdf,png}`
- Data: `results/experiment_1_tabular/summary.md` (3×3 gridworld, γ=0.95, horizon 20)
- Shows: all five sampling regimes (Nash, ∃/∀, ∃/∀ myopic, ∃/∃, EMA-like) converging to Nash CR = 0.436 over 15 iterations.

**Panel (b) — Pure-strategy commitment**:
- Source: `paper/figures/experiment_1b_matrix_commitment.{pdf,png}`
- Data: `results/experiment_1_tabular/summary_matrix.md`
- Shows: bar chart of Nash 0.500 / pursuer-first 0.600 / evader-first 0.400.

**Scripts**: `scripts/experiment_1_tabular_forall_exists.py`, `scripts/experiment_1b_matrix_forall_exists.py`

---

## Supplementary Video — 2 minutes

**Purpose**: Show representative trajectories illustrating lock-in behavior for ICRA submission.

**Scene 1 — Balanced play (15 sec)**
- Use symmetric point-mass Seed 2 or 3.
- Show pursuer and evader trading advantage.
- CR oscillates around 0.5.

**Scene 2 — Pursuer lock-in (30 sec)**
- Use Pop-FSP v3 or v5.
- Show pursuer consistently catching evader.
- Highlight that evader cannot find an escape.

**Scene 3 — Evader lock-in (30 sec)**
- Use Pop-FSP v4 or symmetric Seed 7.
- Show evader consistently escaping.
- Highlight pursuer's inability to search.

**Scene 4 — ∀/∃ asymmetry animation (45 sec)**
- Animated schematic showing pursuer sampling one evader vs evader needing to escape all pursuers.
- Overlay with CR trajectory showing bias.

**Production notes**:
- Render at 1080p, 30 fps.
- Use the existing environment renderer.
- Add text overlays for CR and episode number.
- Keep total length under 2 minutes for ICRA supplementary limits.

---

## Data files needed

| Figure | Source path | Notes |
|---|---|---|
| Fig 1 | `results_pop_fsp_v{3-7}/seed_*/training.log` | Extract CR per episode |
| Fig 2 | N/A — conceptual | Create in TikZ / Illustrator / Inkscape |
| Fig 3 | `results_symmetric/seed_*/eval_results.csv` | Final CR per seed |
| Video | `results_pop_fsp_v{3-7}/seed_*/checkpoints/` | Load policy checkpoints and render |

---

## Suggested Python packages

- `matplotlib` for Figures 1 and 3
- `numpy` / `pandas` for data loading
- `tikz` or `draw.io` for Figure 2
- `moviepy` or `opencv-python` for video compilation
