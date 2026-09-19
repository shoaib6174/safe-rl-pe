> # HISTORICAL — PAPER 1, DROPPED AT D117 (2026-09-17)
>
> This outlines the self-play ∀/∃ asymmetry paper, which the user dropped on 2026-09-17. It is unaffected by
> D125 (that correction concerns Paper 2's shielded chain). Kept for the evidence it indexes; §2 of
> `docs/reference/lessons_learned.md` carries the parts still cited as motivation.

---

# RA-L Paper Outline — 6 Pages

**Title**: *Why Population-Based Self-Play Fails to Balance Pursuit-Evasion: A Structural ∀/∃ Asymmetry*

**Framing**: Systematic diagnostic / negative-result study  
**Central claim**: Population-based self-play for 1v1 pursuit-evasion cannot reach competitive balance because the pursuer trains against an ∃-quantified population while the evader trains against a ∀-quantified population. This structural asymmetry is independent of dynamics symmetry and persists across 17 seeds and 5 configurations.

---

## Page 1 — Introduction (0.75 page)

**Paragraph 1 — Motivation**
- 1v1 pursuit-evasion (PE) between car-like robots under partial observability is a canonical robotics problem.
- Recent work (Gonultas & Isler 2024/2025) formalizes it as a partially observable stochastic zero-sum game and shows self-play can learn non-trivial policies.
- However, the community rarely asks: *does the training procedure actually converge to a balanced Nash equilibrium?*

**Paragraph 2 — The gap**
- Most PE papers report capture-rate improvements against fixed baselines, not adversarial balance.
- Population-based self-play / PSRO methods are known to suffer from expressiveness limits and high exploitability in asymmetric settings, but this has not been characterized for mobile-robot PE.
- Without balance checks, reported "strong" policies may simply reflect training-path bias.

**Paragraph 3 — Our contribution**
- We verify the GI 2024 reproduction and identify seven implementation issues (fixes summarized).
- We show that a standard Pop-FSP procedure fails to produce competitive balance in **0/17 seeds across 5 configurations**, despite symmetric point-mass dynamics giving balance in 2/9 seeds.
- We trace the failure to a structural ∀/∃ quantifier asymmetry: the pursuer best-responds to *some* evader (∃), while the evader best-responds to *all* pursuers (∀).
- We discuss why this makes symmetry of dynamics necessary but insufficient for balance.

**Paragraph 4 — Paper structure**
- Brief map: Methods → Results → Discussion.

---

## Page 1–2 — Related Work (0.75 page)

**Subsection A — Pursuit-evasion with learned policies**
- GI 2024/2025: car-like PE under partial observability; self-play baseline.
- Other recent work: AMSPBH quadrotors, ViPER visibility, multi-UAV MAPPO — focus on capture rate, not balance.
- Gap: no rigorous reproduction or balance audit.

**Subsection B — Population-based self-play and PSRO**
- PSRO / Pop-FSP / League training in MARL.
- Liu et al. 2024: Team PSRO cannot cover full joint policy space in heterogeneous teams.
- FightLadder / PPSRO: exploitability remains high; non-transitivity is common.
- Gap: these findings are not mapped to the mobile-robot PE setting.

**Subsection C — Symmetry and equilibrium in games**
- Symmetric games / cooperative games: symmetry implies equal value under exact invariance.
- Zero-sum role-asymmetric games: symmetry of dynamics does not imply symmetry of objective.
- We connect this to the ∃ vs ∀ opponent-sampling problem.

---

## Page 2–3.5 — Methods (1.5 pages)

### A. Problem formulation (0.25 page)
- Two car-like agents in a 10×10 arena with obstacles.
- Partial observability: combined radius + line-of-sight masking.
- Zero-sum objective: capture rate (CR) for pursuer, escape rate (1−CR) for evader.
- Desired equilibrium: CR ≈ 0.5 against the trained opponent.

### B. Verified reproduction of Gonultas & Isler 2024 (0.25 page)
- We reimplemented the GI 2024 self-play pipeline.
- Self-play CR = 0.550 matches the paper; vs-random CR = 0.936.
- We document seven implementation issues encountered and fixed (listed in supplement; two most critical mentioned here).
- This reproduction underpins all subsequent experiments.

### C. Pop-FSP algorithm (0.4 page)
- Maintain a population of pursuer policies Π_P and evader policies Π_E.
- Each learner is trained by best-response against a sampled opponent from the opposite population.
- **Pursuer objective**: maximize CR against *one* sampled evader → ∃ quantifier.
- **Evader objective**: maximize escape rate against *one* sampled pursuer → but to be unexploitable must escape *all* pursuers → ∀ quantifier.
- Explain why this mismatch is structural: a pursuer only needs one weakness in the evader population, while an evader must be robust to the entire pursuer population.

### D. Symmetric point-mass ablation (0.3 page)
- Replace car-like dynamics with symmetric, fully observable point-mass agents.
- Same Pop-FSP procedure; removes dynamics and observation asymmetry.
- 9 seeds, 5000 episodes each.

### E. Evaluation metric — competitive balance (0.15 page)
- Capture rate CR averaged over a final tournament.
- Balanced if CR ∈ [0.45, 0.55] over the last evaluation window.
- Robustness: fraction of evaluation episodes within the balanced band.

### F. EMA opponent stabilization baseline (0.15 page)
- Exponential moving average of opponent policies as a soft population baseline.
- 4 seeds; serves as comparison, not primary contribution.

---

## Page 3.5–5.5 — Results (2 pages)

### A. Pop-FSP fails to balance (0.7 page)
- **Figure 1**: CR trajectories for Pop-FSP v3–v7 (5 configurations, 17 seeds total).
- Observation: no seed reaches sustained CR ∈ [0.45, 0.55].
- Breakdown: X seeds pursuer-dominant (CR > 0.7), Y seeds evader-dominant (CR < 0.3), Z seeds oscillating but off-center.
- **Table 1**: configuration summary (speed ratio, population size, replacement interval, final CR range).

### B. The ∀/∃ asymmetry explains the bias (0.5 page)
- **Figure 2**: schematic of ∃ vs ∀ opponent sampling.
- Argument: even with perfectly symmetric reward structure, the evader's learning problem is strictly harder because a single pursuer exploit breaks balance, while the pursuer can specialize to any evader weakness.
- Show empirically: in seeds where the evader population is diverse, the pursuer still finds a single high-CR strategy; conversely, when pursuer population is diverse, evader cannot simultaneously counter all.

### C. Symmetric dynamics are necessary but insufficient (0.5 page)
- **Figure 3**: histogram / scatter of final CR for 9 point-mass seeds.
- Result: 2/9 seeds balanced (seeds 2, 3); 5/9 pursuer lock-in; 1/9 evader lock-in; 1/9 pursuer-biased.
- Interpretation: removing dynamics/observation asymmetry improves balance probability but does not guarantee it; the ∃/∀ structural issue remains.

### D. EMA does not reliably fix it (0.3 page)
- EMA seed 0: robustness = 0.698, balanced.
- EMA seeds 1–3: collapse or fail balance.
- Takeaway: EMA can help but is too seed-dependent to be a standalone solution.

---

## Page 5.5–6 — Discussion (0.75–1 page)

**Paragraph 1 — Summary of findings**
- Pop-FSP fails to balance car-like PE (0/17).
- The failure is not merely due to approximate RL or finite populations; it is structural (∃ vs ∀).
- Symmetric dynamics help but are insufficient (2/9).

**Paragraph 2 — Implications**
- Future PE work should report balance metrics, not just capture-rate improvements.
- Methods that explicitly address the ∀/∃ asymmetry (e.g., separate balanced objectives, adversarial populations with evader-side robustness constraints) may be needed.

**Paragraph 3 — Limitations and future work**
- Approximate RL and finite populations may amplify the asymmetry; exact/tabular analysis is future work.
- We focus on 1v1; multi-agent extensions may introduce additional asymmetries.
- PBT with explicit balance selection is currently under investigation.

**Paragraph 4 — Conclusion**
- Population-based self-play in pursuit-evasion is not guaranteed to produce competitive balance because of a structural ∀/∃ asymmetry.
- Recognizing this changes how we design and evaluate self-play methods for competitive robotics.

---

## Figures and Tables

### Figure 1 — Pop-FSP CR trajectories
- x-axis: episode / 1000
- y-axis: capture rate CR
- One subplot or overlay per configuration v3–v7
- All 17 seeds shown (translucent lines) with population mean (thick line)
- Highlight: no trajectory stabilizes in [0.45, 0.55] band

### Figure 2 — ∀/∃ asymmetry schematic
- Left: pursuer best-responds to sampled evader (∃)
- Right: evader must be robust to all pursuers (∀)
- Arrow thickness / color indicates strategy survival under selection

### Figure 3 — Symmetric point-mass balance distribution
- Histogram of final CR across 9 seeds
- Vertical dashed lines at 0.45 and 0.55 (balance band)
- Annotation: 2/9 balanced

### Table 1 — Pop-FSP configuration summary
| Config | Speed ratio | Pop size | Replacement interval | Final CR range | Balanced? |
|---|---|---|---|---|---|
| v3 | ... | ... | ... | ... | 0/N |
| v4 | ... | ... | ... | ... | 0/N |
| v5 | ... | ... | ... | ... | 0/N |
| v6 | ... | ... | ... | ... | 0/N |
| v7 | ... | ... | ... | ... | 0/N |

### Table 2 — Balance summary across experiments
| Experiment | Seeds | Balanced | Notes |
|---|---|---|---|
| Pop-FSP v3–v7 | 17 | 0/17 | Structural ∀/∃ asymmetry |
| Symmetric point-mass | 9 | 2/9 | Dynamics symmetry helps but insufficient |
| EMA | 4 | 1/4 | Seed-dependent |

---

## Appendices / Supplement (for RA-L supplementary material)

- Appendix A: Full GI 2024 reproduction details and the seven bug fixes
- Appendix B: Hyperparameters for Pop-FSP v3–v7
- Appendix C: Hyperparameters for symmetric point-mass ablation
- Appendix D: Hyperparameters for EMA baseline
- Appendix E: Statistical definitions of balance and robustness
- Appendix F: Trajectory visualizations for representative seeds

---

## Notes for draft generation

- Keep Methods concise; RA-L readers value brevity.
- Lead every Results paragraph with the empirical finding, then interpretation.
- Avoid claiming "Nash equilibrium" — use "competitive balance" or "CR ≈ 0.5" instead.
- The GI reproduction should be framed as a credibility anchor, not a separate contribution.
- PBT v3 results should not be mentioned unless they are available and positive before draft finalization.
