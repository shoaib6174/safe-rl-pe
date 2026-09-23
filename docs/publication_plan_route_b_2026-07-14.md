# Publication Plan — Route B: RA-L First, Then ICRA 2027

**Date**: 2026-07-14  
**Session**: S113  
**Decision basis**: `docs/publishability_assessment_2026-07-14.md`

---

## Strategy overview

Submit a **6-page RA-L rapid-communication paper** first, then expand to an **8-page ICRA 2027 paper** one month later. RA-L provides fast archival publication and a safety net; ICRA provides the top-tier conference presentation.

- **RA-L submission target**: Mid-August 2026 (~Aug 15)
- **ICRA 2027 submission target**: Early September 2026 (~Sep 1, buffer before likely Sep 15 deadline)
- **Core story**: Population-based self-play for pursuit-evasion suffers from a structural ∀/∃ quantifier asymmetry that prevents competitive balance.

---

## Week-by-week timeline

| Week | Dates | RA-L milestone | ICRA milestone |
|---|---|---|---|
| 0 | Jul 14–20 | Finalize paper plan; decide figure set | — |
| 1 | Jul 21–27 | RA-L outline + figure sketches | — |
| 2 | Jul 28–Aug 3 | Draft Intro + Related Work + Methods | — |
| 3 | Aug 4–10 | Draft Results + Discussion; cut to 6 pages | — |
| 4 | Aug 11–17 | Internal review; final RA-L submission | Start ICRA expansion outline |
| 5 | Aug 18–24 | Address RA-L reviewer comments if returned early | Expand Methods + Results |
| 6 | Aug 25–31 | — | Expand Discussion; add supplementary material |
| 7 | Sep 1–7 | — | Final ICRA submission |

---

## RA-L paper scope (6 pages)

**Single central claim**: The ∀/∃ quantifier asymmetry in population-based self-play makes competitive balance unattainable in 1v1 pursuit-evasion, even with symmetric dynamics.

**Sections**:
1. Introduction (0.75 page)
2. Related Work (0.75 page)
3. Problem and Methods (1.5 pages)
   - Pursuit-evasion as POSG
   - Pop-FSP algorithm
   - ∀/∃ asymmetry formalization
4. Results (2 pages)
   - Pop-FSP: 0/17 balanced
   - Symmetric point-mass: 2/9 balanced
   - EMA: 1/4 seeds, seed-dependent
5. Discussion and Future Work (1 page)

**Figures**:
- Fig 1: ∀/∃ asymmetry schematic
- Fig 2: Pop-FSP CR trajectories across configs
- Fig 3: Symmetric point-mass CR distribution

**Tables**:
- Table 1: Pop-FSP configuration summary
- Table 2: Balance metric summary

---

## ICRA 2027 expansion scope (8 pages)

**Additional material beyond RA-L**:
- Full GI 2024 reproduction details and the 7 bug fixes
- Expanded related work and literature positioning
- Additional ablations (EMA variants, A-EMA probe, shared-policy NO-GO)
- Supplementary video/trajectory figures
- PBT v3 result (if available and positive; otherwise omit)

**Sections**:
1. Introduction
2. Related Work
3. Background: GI 2024 Formulation and Reproduction
4. Methods
   - Pop-FSP and ∀/∃ asymmetry
   - Symmetric point-mass ablation
   - EMA stabilization
5. Results
6. Discussion
7. Conclusion

---

## Deliverables and effort estimate

| Deliverable | Effort | Owner | Due |
|---|---|---|---|
| RA-L 6-page outline | 1 day | — | Jul 21 |
| ∀/∃ asymmetry figure | 2 days | — | Jul 28 |
| Pop-FSP results table | 0.5 day | — | Jul 28 |
| RA-L first draft | 5 days | — | Aug 4 |
| Internal review pass | 2 days | — | Aug 11 |
| RA-L final submission | 1 day | — | Aug 15 |
| ICRA expansion draft | 5 days | — | Aug 25 |
| Supplementary material | 2 days | — | Aug 30 |
| ICRA final submission | 1 day | — | Sep 1 |

**Total writing effort**: ~18–20 focused days over 7 weeks.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ICRA deadline earlier than Sep 15 | Medium | High | Internal deadline Sep 1 |
| RA-L rejects the negative-result framing | Low | Medium | Frame as systematic diagnostic; cite RSS 2022 precedent |
| RA-L requires major revisions during ICRA crunch | Medium | High | Keep RA-L scope narrow; respond to minor revisions only before ICRA |
| PBT v3 fails or is late | Medium | Low | Do not include in either paper unless it succeeds before Aug 10 |
| Symmetric point-mass story is too thin | Low | Medium | Use as controlled ablation only; lead with Pop-FSP |

---

## Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D49 | 2026-07-14 | **Route B: RA-L first, then ICRA 2027** | Fastest credible route to publication; RA-L provides archival safety net and possible ICRA presentation; ICRA provides prestige. |

---

## Open questions

1. Should the RA-L paper include the GI 2024 reproduction as a credibility anchor, or keep it exclusively for ICRA?
2. Which ICRA 2027 workshop (if any) should we concurrently target for a short-paper/negative-results presentation?
3. Should we prepare a 2-minute supplementary video showing pursuer/evader lock-in behavior?

