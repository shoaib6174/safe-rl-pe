# Publishability Assessment — Safe Deep RL for 1v1 Pursuit-Evasion

**Date**: 2026-07-14 (Session 113)  
**Target venue (preliminary)**: ICRA 2027  
**Assessment method**: ARS-3W three-way scan + deep-research verification (101 agents, 19 sources, 25 claims adversarially verified)

---

## Bottom-line verdict

The project is **publishable at ICRA 2027** if framed as a **systematic diagnostic / negative-result study** rather than a new-method paper. The verified reproduction of Gonultas & Isler 2024 and the structural ∀/∃ quantifier asymmetry finding are the two strongest contributions. Symmetric point-mass and EMA ablations are useful but should remain supporting evidence.

---

## WHY the contribution is needed

1. **Concrete robotics formalism**: Gonultas & Isler (2024/2025) formalize 1v1 pursuit-evasion between car-like robots with sensor constraints as a partially observable stochastic zero-sum game. This aligns with ICRA's call for work that "captures the present and the future of robotics."
2. **Empirical gap**: Recent pursuit-evasion papers (AMSPBH quadrotors, ViPER visibility-based search, multi-UAV MAPPO) report capture-rate improvements but do not rigorously test balanced adversarial training or include independent reproduction audits.
3. **Theory-practice mismatch**: Recent literature confirms that population-based / PSRO methods suffer from expressiveness limits and high exploitability in heterogeneous/asymmetric settings, but this has not been specifically characterized for mobile-robot pursuit-evasion.

---

## HOW the methods compare to literature

| Project component | Literature context | Role in paper |
|---|---|---|
| **GI 2024 reproduction + 7 bug fixes** | Reproducibility is explicitly valued at ICRA; code audits with documented fixes are scarce in robotics MARL. | Baseline / credibility contribution. |
| **Pop-FSP ∀/∃ asymmetry** | Liu et al. (2024) prove Team PSRO cannot cover the full joint policy space in heterogeneous teams; FightLadder reports PSRO/League exploitability near 0.9–1.0. The project's 0/17 balanced configs is a concrete PE instance of this broader failure. | **Primary contribution.** |
| **Symmetric point-mass (2/9 balanced)** | Symmetry/optimal-value equivalence is proven only for cooperative games with exact invariance; zero-sum role-asymmetric games violate the preconditions. The result shows symmetry is necessary but insufficient. | Diagnostic ablation. |
| **EMA opponent stabilization (1/4 seeds)** | EMA-style smoothing is theoretically sound but sensitive; seed dependence matches the literature pattern. | Supporting ablation. |

---

## WHAT is publishable

### Strong claims
1. **Verified reproduction** of Gonultas & Isler 2024 with seven documented bug fixes (self-play CR = 0.550 matches paper; vs-random CR = 0.936).
2. **Structural ∀/∃ quantifier asymmetry** prevents competitive balance in population-based self-play for pursuit-evasion (0/17 seeds/configs balanced across five configurations).
3. **Symmetric dynamics are necessary but insufficient** for competitive balance (2/9 seeds balanced).

### Weaker / supporting claims
- EMA as a reliable standalone solution (too seed-dependent).
- Any claim that the point-mass ablation directly solves the full car-like, partially observable problem.

---

## Anticipated reviewer concerns and responses

| Concern | Response |
|---|---|
| "Where is the positive solution?" | Frame as a systematic / negative-result study. RSS 2022 accepted an explicit negative-result paper on learning from demonstration. |
| "Point-mass is too far from car-like dynamics." | Present it as a controlled ablation isolating symmetry effects, not the main result. |
| "Is the ∀/∃ asymmetry an artifact of approximate RL?" | Acknowledge as an open question; add a small exact/tabular illustration if feasible. |
| "Empirical scope is limited." | Emphasize depth over breadth: one domain, multiple methods, controlled ablations, verified reproduction. |

---

## Recommendations

1. **Lead with the ∀/∃ asymmetry** as the central contribution.
2. **Use the GI reproduction** to establish credibility and baseline dynamics.
3. **Position symmetric point-mass and EMA as controlled diagnostics**, not standalone methods.
4. **Frame the paper as a systematic study** with a title such as:
   > *"Why Population-Based Self-Play Fails to Balance Pursuit-Evasion: A Systematic Study of Structural Asymmetry"*
5. Keep the **PBT v3 result** separate until it completes; do not hinge the paper's main claim on it.

---

## Venue comparison and recommendation

As of 2026-07-14, the major 2026 robotics/robot-learning conference deadlines have already passed. The viable targets are ICRA 2027, IROS 2027, and a journal route (RA-L / T-RO).

| Venue | Deadline | Fit for this work | Verdict |
|---|---|---|---|
| **ICRA 2027** | Likely **September 2026** (not yet officially announced; follow [2027.ieee-icra.org/contribute](https://2027.ieee-icra.org/contribute/)) | Strong. Top robotics venue, explicitly values reproducibility, scope matches car-like PE with partial observability. | **Primary recommendation.** |
| **IROS 2027** | **March 1, 2027** (preliminary) | Good. Same RAS family as ICRA, slightly lower prestige but very relevant scope. Useful fallback if ICRA deadline is missed or paper is rejected. | **Secondary recommendation.** |
| **RA-L** | Rolling | Good for rapid archival publication. 6-page format fits a concise systematic study; accepted RA-L papers can be presented at ICRA/IROS if concurrently submitted. Does not explicitly solicit negative results but does not exclude them. | **Good parallel/alternative route**, especially if you want a journal citation before the next conference deadline. |
| **T-RO** | Rolling | Risky. Higher theoretical-bar; negative/systematic studies are not a standard article type and must demonstrate substantial novelty and technical depth. | **Not recommended** as first target unless the ∀/∃ asymmetry is proven analytically for a broad class of games. |
| **CoRL 2026** | May 28, 2026 | Relevant robot-learning venue, but deadline has passed. | Not available. |
| **RSS 2026** | January 30, 2026 | Explicitly accepts negative results, but deadline has passed. | Not available. |
| **IROS 2026** | March 2, 2026 | Deadline has passed. | Not available. |

### Recommended strategy

1. **Target ICRA 2027 first.** Prepare the full paper for the likely September 2026 deadline.
2. **Prepare a workshop pre-proposal** for ICRA 2027 if relevant (deadline September 1, 2026) — this could provide a venue for deeper discussion of reproducibility/negative results.
3. **Parallel-track RA-L** only if you can condense the story to 6 pages without losing the asymmetry argument.
4. **Keep IROS 2027 as fallback** if ICRA does not work out.

---

## Open questions for future work

1. Does the ∀/∃ asymmetry persist under exact best-response oracles, or is it an artifact of approximate deep RL and finite populations?
2. What minimal symmetry/role-exchange condition would make a shared population sufficient in this zero-sum setting, and can it be proven?
3. How does the structural finding transfer from point-mass abstractions to the full car-like, partially observable dynamics?
4. Can any simple architectural fix (separate role-specific populations, balanced sampling, diversity regularizer) break the asymmetry?

---

## Sources used in assessment

Key verified sources from the ARS-3W scan:
- Gonultas & Isler (2024/2025), *Pursuit-Evasion for Car-like Robots with Sensor Constraints* — problem formalization, baseline metrics.
- Liu et al. (2024), *Computing Ex Ante Equilibrium in Heterogeneous Zero-Sum Team Games* — Team PSRO expressiveness limits.
- Gopalan et al., RSS 2022, *Negative Result for Learning from Demonstration* — precedent for negative-result robotics papers.
- ICRA 2025 Call for Papers — scope confirmation.
- Li et al., AAAI 2023, *Solving Large-Scale Pursuit-Evasion Games Using Pre-trained Strategies* — PSRO scalability context.
- PPSRO 2024 / self-play survey — non-transitivity and correlated-best-response bottlenecks.

Full deep-research output: see workflow transcript `wf_515a2cba-f90` (2026-07-14).
