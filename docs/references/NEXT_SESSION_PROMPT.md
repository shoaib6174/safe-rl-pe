# Next Session Prompt

Copy-paste this into your next Claude Code session:

---

## Context

I'm working on a research project: **Safe Deep RL for 1v1 Ground Robot Pursuit-Evasion**. The research plan is in `claudedocs/pathway_A_safe_deep_RL_1v1_PE.md` (742 lines).

Over the previous sessions, I completed a comprehensive literature review and validation:

### What exists:
1. **Original plan**: `claudedocs/pathway_A_safe_deep_RL_1v1_PE.md` — Architecture: PPO + CBF-Beta (training) + RCBF-QP (deployment) + AMS-DRL self-play + BiMDN belief encoder for partial observability
2. **Original literature review**: `claudedocs/final_literature_review.md` — 36 papers, 7 research gaps (G1-G7)
3. **15 new papers** (N01-N15) found, downloaded, and fully read — all PDFs in `papers/safe_rl/pdfs/`
4. **Review documents** (all updated with PDF-verified findings):
   - `papers/safe_rl/new_papers.md` — Detailed summaries of all 15 new papers
   - `papers/safe_rl/key_methods_summary.md` — Equations and algorithms quick reference
   - `papers/safe_rl/pathway_A_validation_report.md` — Validation with 3 critical fixes, 5 improvements, updated risk assessment
   - `papers/safe_rl/bibtex_references.bib` — BibTeX for all papers
   - `papers/safe_rl/existing_papers_index.md` — Index of original 36 papers

### Key findings from validation:
1. **CRITICAL FIX**: Nonholonomic CBF relative degree problem — must use virtual control point CBF (N12: Zhang & Yang, Neurocomputing 2025). For unicycle: q = [x + d·cos(θ), y + d·sin(θ)], achieves uniform relative degree 1 in both v and ω. The M-matrix transformation prioritizes STEERING over BRAKING.
2. **CRITICAL FIX**: CBF-QP infeasibility handling — use 3-tier approach: (1) N13 learned feasibility constraints (reduces infeasibility 8.11%→0.21%), (2) hierarchical relaxation, (3) backup controller.
3. **CRITICAL FIX**: Update Isaac Sim → Isaac Lab throughout (N11 confirms native MARL support via PettingZoo).
4. **Closest prior work**: N15 (RMARL-CBF-SAM, Liu et al., Information Sciences 2025) combines robust MARL + neural CBFs + safety attention + reward shaping for multi-agent navigation. But differs in 5 ways: not PE, not nonholonomic, no self-play, no partial obs, no sim-to-real. **Novelty claim intact**.
5. **Recommended additions**: BarrierNet experiment (Phase 2.5), MACPO/CPO baselines, simultaneous self-play comparison (N06), safety-reward shaping term, PNCBF as neural CBF fallback, expanded metrics.

### Remote server:
- **niro-2**: 100.71.2.97, user: niro-2, password: 123456 (saved in CLAUDE.md)

## Task

Now I need you to **apply all the validated fixes and improvements to the actual plan document** (`claudedocs/pathway_A_safe_deep_RL_1v1_PE.md`). Specifically:

1. Read the validation report at `papers/safe_rl/pathway_A_validation_report.md`
2. Read the current plan at `claudedocs/pathway_A_safe_deep_RL_1v1_PE.md`
3. Apply ALL 3 critical fixes:
   - Replace position-based CBFs with virtual control point CBFs (Section 3.3.1) using the exact formulation from N12
   - Add 3-tier CBF infeasibility handling (Section 3.3.3) with N13 learned feasibility constraints
   - Update all Isaac Sim references to Isaac Lab with N11 details (PettingZoo MARL API, sensor suite, etc.)
4. Apply the 5 recommended improvements:
   - Add BarrierNet Phase 2.5 experiment
   - Add MACPO/MAPPO-Lagrangian + CPO to baselines (Section 5.2)
   - Add simultaneous self-play comparison to ablations (Section 5.3)
   - Add safety-reward shaping term w5 to reward design (Section 2.5)
   - Expand evaluation metrics (Section 5.1)
5. Update the risk assessment table with all new entries from the validation report
6. Update the phase plan timeline
7. Add N15 (RMARL-CBF-SAM) as the closest prior work in the related work discussion
8. Update all references/citations to include the 15 new papers

Do NOT rewrite the plan from scratch — make targeted edits to the existing document preserving its structure and voice. Read both documents fully before making any changes.

---
