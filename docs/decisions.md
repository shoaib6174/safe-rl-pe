# Decision Log

Searchable index of major architectural and design decisions. Each entry links to the session where it was made.

## Algorithm & Hyperparameters

| # | Date | Session | Decision | Rationale |
|---|------|---------|----------|-----------|
| D1 | 2026-03-02 | S66 | **SAC over PPO** for self-play | SAC hits 30-40% at same step count where PPO stalls. RecurrentPPO (LSTM) conclusively failed at 0-2% across 6 variants (SP13a-f). |
| D2 | 2026-03-03 | S67 | **ent_coef=0.03 fixed** (never auto) | SP11c collapsed from 24% to 6% with auto entropy. Fixed 0.03 is proven across all successful runs. |
| D3 | 2026-03-01 | S59 | **ent_coef=0.03 for variable speed evader** | 0.01 stalled at 12% escape; 0.03 exploded to 98%. Exploration critical for variable-speed learning. |
| D4 | 2026-03-04 | S71 | **w_search < 0.0003 if at all** | 0.003 was ~50% of distance reward magnitude, taught patrol instead of hunt. SP11g2/h2 stuck at 25% vs SP11e's 60%. |

## Game Design

| # | Date | Session | Decision | Rationale |
|---|------|---------|----------|-----------|
| D5 | 2026-03-02 | S64 | **1.2x evader speed** for self-play | Equal speed → degenerate (hide, no PE). 1.05x insufficient. 1.2x holds 35-50% escape vs full-obs greedy. |
| D6 | 2026-03-01 | S63 | **Combined masking (radius+LOS)** over alternatives | LOS-only too restrictive, radius-only sees through obstacles. Combined is most realistic and creates viable game. |
| D7 | 2026-02-26 | S55 | **2 obstacles in 10x10 arena** | 4 too cluttered, 0 gives no occlusion opportunity. 1-2 variable forces adaptive behavior. |
| D8 | 2026-02-26 | S55 | **micro_phase_steps=2048** (not 500K) | Long frozen phases are fundamentally wrong (S53 research). Rapid alternation prevents catastrophic forgetting. |

## Self-Play Strategy

| # | Date | Session | Decision | Rationale |
|---|------|---------|----------|-----------|
| D9 | 2026-03-02 | S66 | **Freeze→unfreeze curriculum** | SP7b: 10%→42% with frozen evader. 1:1 co-evolution (SP8) and adaptive boost (SP7a) both conclusively failed. |
| D10 | 2026-03-06 | S73 | **Multi-seed sweeps** (4+ seeds) | CUDA non-determinism dominates. SP11e (85%) vs SP11i (30%) = same config, different luck. Single seed is a gamble. |
| D11 | 2026-02-28 | S58 | **Always include collapse_rollback + PFSP** | Collapse rollback restores best checkpoint when SR drops. PFSP biases pool toward weaker opponents. Both are safety nets. |
| D12 | 2026-03-01 | S63 | **Cold-start works with partial obs** | SP4 bootstrapped to 53% with combined masking. Full-obs cold-starts (SP2+B+C+D) all failed. Information asymmetry is key. |

## Diagnostics & Evaluation

| # | Date | Session | Decision | Rationale |
|---|------|---------|----------|-----------|
| D13 | 2026-03-01 | S62 | **Always use `--greedy_full_obs`** | Without it, S1v5b showed 96% escape (pursuer was also blinded). S1v5c with full-obs greedy: 16%. Masked greedy measures nothing. |
| D14 | 2026-03-01 | S61 | **Best-model checkpointing** | S1v4a regressed -60pp, S1v4b -23pp from peak. Saving best model prevents losing peak performance to late regression. |

## Research Direction

| # | Date | Session | Decision | Rationale |
|---|------|---------|----------|-----------|
| D15 | 2026-03-06 | S75 | **Masking curriculum (PO-GRL style)** | +10-15pp over baseline at 5-8M steps. Annealing from full→partial obs gives agents time to learn basic skills before adding observability challenge. |
| D16 | 2026-03-05 | S72 | **Never kill runs for disk management** | AMSDRLSelfPlay has no resume. Use symlinks or change output paths instead. Disk full in S72 killed promising runs. |
| D17 | 2026-03-09 | S77 | **3m omnidirectional sensing abandoned** | 2.8% arena coverage at equal speed = impossible. SP12 (8 runs, 13.2M steps each) peaked 0.42-0.53, all declining. See `docs/analysis_sp12_postmortem.md`. |
| D18 | 2026-03-09 | S77 | **FOV 90° + 8m is primary sensing config** | SP13a peaked 93-97% CR at 5.5M steps. Directional sensing with longer range gives actionable geometry. |
| D19 | 2026-03-09 | S77 | **Alternate freeze threshold 0.6** for co-evolution | Pursuer hits 60%+ quickly under FOV. Lower threshold enables earlier evader training before pursuer overspecializes against random. |
| D20 | 2026-03-16 | S83 | **Vis reward (w=0.1) mandatory under PO** | SP15: 0.57 avg CR with vis reward vs 0.10 without. Under 90° FOV, pursuer gets no gradient signal without explicit reward for maintaining LOS. |
| D21 | 2026-03-16 | S83 | **Freeze threshold 0.65 (not 0.8) under full PO** | SP15: thr=0.8 unreachable — 0 switches across 10 runs, 19 iters each. Pursuer peaks at 0.75 against static evader under full PO. 0.65 is reachable but not trivial. |
| D22 | 2026-03-16 | S83 | **EWC useless without actual role switches** | SP15: EWC λ=1000 produced identical CRs to non-EWC runs. No switches = no forgetting = no effect. Re-testing in SP16 with lower threshold. |
| D23 | 2026-04-29 | S85 | **Periodic forced switch every 3M steps** | SP16e ran 23M steps (76 phases) with single forced switch at 3M — longest stable co-evolution. Prevents freeze-sticking where one role trains indefinitely. |
| D24 | 2026-04-29 | S85 | **Search reward w_search=0.0001** | S84 trajectory analysis showed systematic search is a critical missing skill. Staleness-based grid reward teaches area coverage without hard-coding patrol. |
| D25 | 2026-04-29 | S85 | **4-seed validation (42-45) minimum** | S73 established CUDA non-determinism dominates. Single seed is a gamble. 4 seeds for robust validation. |
| D26 | 2026-04-29 | S85 | **Forced switch must check every micro-phase** | Discovery: inside eval block meant check only every 150 micro-phases. Moved to per-micro-phase check in amsdrl.py. |
| D27 | 2026-04-30 | S86 | **GPU mandatory for SAC+Dict obs visualization** | CPU inference: ~1.8 steps/sec (11+ min/episode). GPU: ~20 sec/episode (60× speedup). SAC with PartialObsOpponentAdapter on CPU is unusable for analysis. |
| D28 | 2026-05-01 | S87 | **Target CoRL 2026 as stretch, ICRA 2027 as backup** | 27 days to deadline is aggressive but achievable. Worst case: reviewer feedback strengthens ICRA submission. Paper story: self-play stabilization under PO. |
| D29 | 2026-05-01 | S87 | **Hybrid machine allocation** | niro-2 for convergence cohort (same environment, identical config), niro-1 for ablations/baselines (isolated, protects cohort if issues arise). 8 parallel vs 4 = 2.7-day wall time savings. |
| D30 | 2026-05-01 | S87 | **Kill s43 (SP17, 3M switch)** | CR=0.25 at 11M steps — pursuer-dominant, not true NE. Gap never narrowed. s45 (same config) oscillating 0.47-0.69 — keep as comparison data only. |
| D31 | 2026-05-01 | S87 | **Paper framing: self-play stabilization under PO** | BarrierNet abandonment weakened safety angle. Self-play stabilization (visibility reward, forced switching, search reward) is empirically strong and novel. Safety filter becomes background/secondary contribution. |
| D32 | 2026-05-01 | S87 | **2M forced switch superior to 3M** | SP17b (2M): s48/s49 both converged to CR≈0.51-0.52. SP17 (3M): s43 collapsed to 0.25, s45 oscillating. Shorter interval prevents prolonged freeze-sticking. |
| D33 | 2026-05-01 | S87 | **Start LaTeX skeleton Day 1** | Writing is the long pole. 4 weeks from zero draft to polished submission is aggressive. Must parallelize writing with experiments. |
| D34 | 2026-05-02 | S88 | **BR baseline = M2550 snapshot CR, not tail mean** | The spec §3 assumed post-drift tail baselines (~0.31, ~0.37). But the BR opponents were extracted at M2550 (claimed NE point), so the correct baseline is the CR at that phase (0.51, 0.60). Comparing against post-drift would conflate "is the NE claim true?" with "did drift damage the policy?" |
| D35 | 2026-05-02 | S88 | **H3 verdict → Framing C** | BR exploitability test: both seeds L3 (evader exploitable, pursuer not). BR evader achieves CR=0.41-0.52 vs baseline 0.51-0.60 (gap +0.08 to +0.10 > ε=0.05). BR pursuer achieves CR≤0.115 (gap negative, frozen evader strong). Cohort hypothesis H3: curriculum-induced evader-side exploitability, replicated. Paper reframes to characterize this hole, not claim NE convergence. |
| D36 | 2026-05-02 | S88 | **Post-verdict evidence strengthening: ABL-3/4 + BR-5/6** | H3 verdict is empirically solid but needs more evidence for CoRL competitiveness. ABL-3 (no search) and ABL-4 (no curriculum) prove the remaining mechanisms. BR-5/6 on s50 replicate the exploitability test to 3 seeds. Decision gate May 10: if all show expected divergence, submit to CoRL; otherwise evaluate ICRA 2027. |
