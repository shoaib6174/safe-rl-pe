# BR Exploitability Test — Verdict

**Date:** 2026-05-02
**Spec:** [paper-strengthening-design](../superpowers/specs/2026-05-02-paper-strengthening-design.md)

## Verdict

- Cohort hypothesis: **H3**
- Framing decision:  **C**
- Note: Curriculum-induced evader-side exploitability, replicated.
- epsilon = 0.05

## Per-seed labels

- s48: **L3**
- s49: **L3**

## Per-run table

| Run | Frozen seed | Frozen role | BR role | Baseline | BR best | Gap | Per-seed |
|-----|-------------|-------------|---------|----------|---------|-----|----------|
| BR_1 | s48 | evader | pursuer | 0.510 | 0.085 | -0.425 | L3 |
| BR_2 | s48 | pursuer | evader | 0.510 | 0.410 | +0.100 | L3 |
| BR_3 | s49 | evader | pursuer | 0.600 | 0.115 | -0.485 | L3 |
| BR_4 | s49 | pursuer | evader | 0.600 | 0.520 | +0.080 | L3 |

## Next step

Open `paper/main.tex` and apply the abstract phrasing change for framing
**C** per spec §8.
