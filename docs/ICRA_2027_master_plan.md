# ICRA 2027 Master Plan — Path to Submission

**Target Venue:** ICRA 2027 (deadline ~September 2026)
**Paper Title:** *A Study of Self-Play Stabilization Methods for Competitive Deep RL in Partially Observable Pursuit-Evasion*
**Framing:** Systematic study (not method paper) — EMA as partial positive result, Pop-FSP ∀/∃ asymmetry as structural contribution
**Current Date:** 2026-05-26
**Days to Deadline:** ~100 days

---

## Executive Summary

This plan structures the remaining work into 4 phases with explicit decision gates. Each phase has:
- **Deliverables** (what must be produced)
- **Experiments** (what must run)
- **Decision Gate** (what criteria determine the next phase)
- **Contingency** (what happens if the gate fails)

The plan is designed so that work in earlier phases is useful regardless of later outcomes — no wasted effort.

---

## Phase 1: Decision Gate — EMA Reproducibility (May 26 – June 7, ~2 weeks) ✅ COMPLETE

**Status:** NO-GO — EMA seeds 4-6 all collapsed (0/3 balanced). With seed 0, success rate is 1/5.

**Decision Date:** 2026-05-26

**Pivot:** Phase 2B (Structural Characterization Track) — Pop-FSP ∀/∃ asymmetry as primary contribution, EMA as secondary comparison.

**Goal:** Determine whether EMA is reliable enough to be the paper's primary result.

### Deliverables
| # | Deliverable | Owner | Deadline | Blocked On |
|---|------------|-------|----------|-----------|
| 1.1 | EMA seeds 4-6 evaluation with `evaluate_checkpoint_sweep.py` | User | May 27 | Seeds 4-6 completion |
| 1.2 | GI vanilla 20K baseline midpoint check (ep 5000-10000) | User | May 30 | Baseline running |
| 1.3 | Decision memo: EMA Go/No-Go | User | June 1 | 1.1 complete |
| 1.4 | Draft Methods section (LaTeX) | User | June 7 | None — can start now |

### Experiments
| Experiment | Seeds | Episodes | Purpose | Status |
|-----------|-------|----------|---------|--------|
| EMA τ=0.001 | 4, 5, 6 | 5000 | Reproducibility test | RUNNING on niro-2 |
| GI vanilla SP | 10, 11, 12 | 20000 | Baseline oscillation at paper duration | RUNNING on niro-2 |

### Decision Gate 1: EMA Reproducibility — RESULTS

**Date:** 2026-05-26

**Results:**

| Seed | LR | Final CR | Steps | Outcome |
|------|-----|----------|-------|---------|
| 4 | 0.0005 | 0.000 | 487,990 | Evader-dominant, NaN losses |
| 5 | 0.0005 | 1.000 | 149,057 | Pursuer-dominant collapse |
| 6 | 0.0005 | 0.000 | 491,671 | Evader-dominant |

**Complete EMA (all 7 seeds):**
| Seed | LR | Final CR | Outcome |
|------|-----|----------|---------|
| 0 | 0.0005 | 0.500 | Balanced (outlier) |
| 1 | 0.0005 | 0.000 | Evader collapse |
| 2 | 0.0001 | 1.000 | Pursuer collapse |
| 3 | 0.0005 | 0.000 | Evader collapse |
| 4 | 0.0005 | 0.000 | Evader collapse + NaN |
| 5 | 0.0005 | 0.000 | Pursuer collapse |
| 6 | 0.0005 | 0.000 | Evader collapse |

**Success rate with LR=0.0005: 1/5 seeds.**

**GI Vanilla 20K Results:**
| Seed | Final CR | Notes |
|------|----------|-------|
| 10 | 0.500 | Oscillated, NaN at end |
| 11 | 0.000 | Evader collapse, NaN |
| 12 | 0.000 | Evader collapse, NaN |

**Verdict: NO-GO** — EMA is not reproducible enough for a primary contribution.

**Action:** Proceed to Phase 2B (structural characterization as primary).

### Phase 1 Retrospective

**What was completed:**
- EMA seeds 4-6 finished and analyzed
- GI vanilla 20K baseline finished
- Decision made: NO-GO

**What to carry forward:**
- Seed 0 remains the best EMA result (robustness=0.698)
- All EMA data becomes comparison evidence
- GI 20K data characterizes vanilla oscillation at paper duration
- NaN loss finding is a new result worth documenting

---

## Phase 2A: EMA-Primary Track (June 1 – July 1, ~4 weeks) — NOT PURSUED

**Status:** Skipped due to Gate 1 NO-GO.

If Gate 1 had been GO, this track would have launched τ ablation + 20K extension. See original plan below for reference.

<details>
<summary>Original Phase 2A plan (for reference)</summary>

**Experiments:**
| Experiment | Seeds | Episodes | Purpose |
|-----------|-------|----------|---------|
| EMA τ=0.0005 | 3 seeds | 5000 | Slower EMA ablation |
| EMA τ=0.005 | 3 seeds | 5000 | Faster EMA ablation |
| EMA τ=0.001 20K | Best seed from 4-6 | 20000 | Long-horizon stability |

</details>

---

## Phase 2B: Structural-Characterization Track (June 1 – July 1, ~4 weeks) ✅ ACTIVE

**Goal:** Make the Pop-FSP ∀/∃ asymmetry and oscillation characterization the primary contribution.

**Rationale:** Gate 1 was NO-GO. EMA success rate is 1/5 — insufficient for a primary method claim. The ∀/∃ asymmetry analysis is rigorous, novel, and supported by 25+ GPU-hours of experiments. The paper becomes: "We systematically tested stabilization methods for competitive RL in PE. EMA shows promise but is fragile (1/5 seeds). Pop-FSP fails due to a structural ∀/∃ asymmetry we identify."

### Deliverables
| # | Deliverable | Deadline | Blocked On |
|---|------------|----------|-----------|
| 2B.1 | ∀/∃ asymmetry LaTeX section (formalized) | June 10 | None |
| 2B.2 | Complete Pop-FSP configuration sweep table | June 12 | None (already have v3-v7) |
| 2B.3 | EMA results (all 7 seeds: 0-6) compiled | June 1 | ✅ DONE |
| 2B.4 | Vanilla SP oscillation characterization (20K) | June 1 | ✅ DONE |
| 2B.5 | NaN instability analysis (new finding) | June 10 | None |
| 2B.6 | Publication-quality figures (8+) | July 1 | All data |

### Experiments
| Experiment | Seeds | Episodes | Purpose | Status |
|-----------|-------|----------|---------|--------|
| Vanilla SP 20K | 10, 11, 12 | 20000 | Oscillation characterization | ✅ COMPLETE |
| EMA τ=0.001 | 0-6 | 5000 | Comparison data | ✅ COMPLETE |

**Note:** Phase 2B requires NO new compute. All experiments are complete. Focus is entirely on analysis and writing.

### What to Do Each Week

**Week 1 (May 26 – June 1):**
- ✅ Check EMA seeds 4-6 results
- ✅ Check GI vanilla 20K results
- [ ] Run `evaluate_checkpoint_sweep.py` on seeds 4-6 (if not done)
- [ ] Start Methods section draft
- [ ] Start ∀/∃ asymmetry LaTeX section

**Week 2 (June 2–8):**
- [ ] Methods section v1.0 complete
- [ ] ∀/∃ asymmetry section complete with equations
- [ ] Generate FFT / phase portrait analyses from existing data
- [ ] Draft oscillation characterization section

**Week 3 (June 9–15):**
- [ ] Draft Related Work section
- [ ] Compile EMA results (all 7 seeds) into comparison table
- [ ] NaN instability analysis (new finding from seeds 4-6 and GI 20K)

**Week 4 (June 16–22):**
- [ ] Draft Results with "characterization" framing
- [ ] Draft Discussion section (why populations fail, why EMA is fragile)
- [ ] Generate all figures
- [ ] First draft of complete paper (v0.5)

---

## Phase 3: Paper Completion (July 1 – August 15, ~6 weeks)

**Goal:** Complete, review, and revise the full paper. This phase is IDENTICAL for both tracks.

### Deliverables
| # | Deliverable | Deadline | Blocked On |
|---|------------|----------|-----------|
| 3.1 | Complete draft v0.5 (all sections) | July 7 | Phase 2 deliverables |
| 3.2 | Internal review + revision | July 15 | 3.1 |
| 3.3 | Draft v0.9 (incorporating review) | July 25 | 3.2 |
| 3.4 | Final figures + tables | August 1 | 3.3 |
| 3.5 | Final draft v1.0 | August 10 | 3.4 |
| 3.6 | Supplement material | August 15 | 3.5 |

### Milestone Checkpoints

| Checkpoint | Date | Criteria |
|-----------|------|----------|
| Draft v0.5 | July 7 | All 8 pages written, figures placeholders filled |
| Draft v0.9 | July 25 | Review feedback incorporated, figures real |
| Camera-ready | August 10 | Within page limit, compliant with template |

### Weekly Tasks

**July 1–7 (Draft v0.5):**
- Write Introduction (lead with oscillation problem, not method)
- Write Related Work (G&I 2024, AlphaStar, PSRO, EMA literature)
- Write Preliminaries (game formulation, metrics)
- Write Methods (EMA, Pop-FSP, evaluation)
- Write Experiments (setup, baselines)
- Write Results (with available data)
- Write Discussion
- Write Conclusion

**July 8–15 (Review):**
- Self-review: Does every claim have evidence? Is the framing consistent?
- Check: Are we overclaiming EMA reliability? (Adjust if needed)
- Check: Is the ∀/∃ asymmetry presented as insight or failure? (Should be insight)

**July 16–25 (Revision):**
- Incorporate all review notes
- Tighten Introduction (most important section for reviewers)
- Ensure figures tell the story without reading text

**July 26 – Aug 1 (Figures):**
- Finalize all 6+ figures in publication quality (PDF)
- Regenerate if data changed

**Aug 2–10 (Polish):**
- Final proofreading
- Check IEEEconf template compliance
- Write supplement (hyperparameters, additional ablations)

**Aug 11–15 (Supplement):**
- Final supplement material
- Archive code for reproducibility

---

## Phase 4: Submission Sprint (August 15 – September 2026, ~2 weeks)

**Goal:** Submit to ICRA 2027.

### Deliverables
| # | Deliverable | Deadline |
|---|------------|----------|
| 4.1 | Format check (8 pages, double-blind) | Aug 18 |
| 4.2 | Final proofread by fresh eyes | Aug 20 |
| 4.3 | Supplement finalization | Aug 22 |
| 4.4 | Submit | Deadline day |

### Contingency: What If ICRA Is Not Ready?

| Scenario | Fallback Venue | Deadline | What We'd Need |
|----------|---------------|----------|---------------|
| Paper not ready by Sept | AAMAS 2027 | ~Nov 2026 | 2 extra months; good for multi-agent focus |
| Rejected from ICRA | ICRA 2028 | Sept 2027 | Full year; can add more experiments |
| Weak EMA + weak framing | CoLLAs 2027 | ~Mar 2027 | 6 months; reframe as learning theory |

---

## Resource Allocation (niro-2)

### Compute Budget

| Phase | Runs | GPU-Hours | Purpose |
|-------|------|-----------|---------|
| Phase 1 | 6 (already running) | ~12 | Seeds 4-6 + baseline |
| Phase 2A | 10 (τ ablation + 20K) | ~40 | Primary experiments |
| Phase 2B | 0 new | 0 | Analysis-only |
| Phase 3–4 | 0 | 0 | Writing-only |

**Total Phase 2A compute:** ~40 GPU-hours over 4 weeks = 10 hrs/week. Easily fits on niro-2.

### niro-2 Schedule

| Week | Runs on niro-2 | Notes |
|------|---------------|-------|
| May 26–Jun 1 | Seeds 4-6 completing | Evaluate immediately |
| Jun 1–Jun 7 | τ ablation (6 runs) | Launch if Gate 1 = GO |
| Jun 8–Jun 14 | τ ablation + EMA 20K | Monitor for collapse |
| Jun 15–Jun 21 | EMA 20K continuing | |
| Jun 22–Jun 28 | Baseline 20K completing | |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| EMA seeds 4-6 all collapse | Medium | High | Phase 2B track; paper pivots to characterization |
| 20K runs reveal late collapse | Medium | High | Frame as "long-horizon instability" finding |
| Writing takes longer than planned | High | Medium | Start writing NOW (Methods, Related Work) |
| niro-2 hardware failure | Low | High | No mitigation; accept delay |
| Figure generation bugs | Medium | Medium | Budget 1 week for figure debugging |

---

## This Week's Checklist (May 26 – June 1)

**Day 1 (May 26 — TODAY):**
- [ ] SSH to niro-2, check EMA seeds 4-6 status
- [ ] If complete: copy results back, run `evaluate_checkpoint_sweep.py`
- [ ] If still running: note current episode, estimate completion

**Day 2–3 (May 27–28):**
- [ ] Analyze EMA seed 4-6 results
- [ ] Write 1-page decision memo (Gate 1)
- [ ] Start Methods section draft

**Day 4–5 (May 29–30):**
- [ ] Continue Methods + draft ∀/∃ asymmetry section
- [ ] Check GI vanilla 20K baseline status

**Day 6–7 (May 31–June 1):**
- [ ] Methods v0.5 complete
- [ ] **DECISION GATE 1: Choose Phase 2A or 2B**
- [ ] Update this plan with chosen track

---

## Document History

| Date | Version | Change |
|------|---------|--------|
| 2026-05-26 | v1.0 | Initial plan created |

---

## How to Use This Plan

1. **At the start of each week:** Read the weekly tasks for your current phase.
2. **At the end of each week:** Check off completed deliverables. Update the Session Index in `docs/workflow_tracker.md`.
3. **At each Decision Gate:** Follow the criteria exactly. Do not rationalize a "GO" if criteria aren't met.
4. **If a risk materializes:** Follow the contingency, update the plan, and document in the worklog.
