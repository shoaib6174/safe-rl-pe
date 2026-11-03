# Paper B revision: archive and code facts (2026-11-03)

These are read-only lookups for roadmap items R3–R15 in `editorial_decision.md` §6. No cell was re-run. Every recomputation used archived per-entry or per-episode fields only, and the scripts are in the session scratchpad (`r3.py`, `r3b.py`, `r12.py`). "Registered" marks a number the round itself produced. "Recomputed" marks arithmetic on archived fields that was done today and is post hoc.

## 0. Two findings that change the text (read first)

**F-A. Remark 2 counts the discount in the wrong unit, and the corrected advantage is not small.**
- PPO discounts once per learner **transition**, not once per env step. One transition is `SearchMDP.step` (`pe2/structured.py:347–371`): up to `HOLD = 10` head steps, then `_advance` lets the law play until it hands back.
- Training records show an average of 15–16 env steps per transition. For example, `s16_arm_BASE-60.json` has `env_steps` 5 000 005 and `decisions` 313 662, which is 15.9. All 15 S16 PPO arms fall between 14.0 and 16.6.
- S15's `return_to_go` (`scripts/derisk_s15_mechanism.py:239–251`) uses `gamma ** steps_remaining` with the remaining time counted in **env steps**. That over-discounts by a factor of 10–16 in the exponent.
- On the learner's own clock the escape advantage is large:

| γ per transition | env steps per transition | Δ return-to-go, S15's form (+1 capture / −1 timeout) | A₀·E[γ^(T/k)], T = entry→capture on the 29 captured escapes |
|---|---|---|---|
| 0.99 | 1 (S15 as registered) | **0.0092** (registered) | 0.0136 |
| 0.99 | 10 (largest possible discounting at hold 10) | **0.294** | 0.183 |
| 0.99 | 15.6 (training mean) | 0.358 | 0.206 |
| 0.999 | 10 | 0.485 | 0.248 |

- **Consequence.** The claim that the reward for escaping is discounted away is not supported. At γ = 0.99 per decision, the discounted advantage is at least about 0.18–0.29, well above the 0.05 bar.
- **How this fits the registered verdict.** S15's registered P1 verdict is **FAILS**. The preregistration `loop_mechanism_prereg_2026-09-20.md:66` registered the direction for that outcome: "The signal was there and learning still missed it; the explanation is optimisation, not credit assignment." D128 then re-read the failure as a discount argument, and F-A undoes that re-reading. The registered direction stands.
- **Interpretive note.** One per-decision γ = 0.99 at about 10 env steps per decision is roughly γ = 0.999 per env step, so the Hor family at 0.999 per decision is roughly 0.9999 per env step.
- **Law-branch stalls carry no discount argument at all.** A law stall runs inside `_advance` within ONE transition. If it lasts to the timeout, the decision that handed over to the law receives the episode's terminal reward undiscounted. The learner takes no decision during the stall.

**F-B. The paper's opening example (Fig. 1) is a stall of the scripted law with the evader visible.**
- The episode is Fig. 1's own source: `figure_data.json` F1 = `s17_arm_OBS-61.json` / `RECF-OBS-61`, gate HUG7, `traces[3]`, seed 1 740 003.
- `results/derisk/s27_n1_cells/OBS-61_HUG7.json`, episode i = 3, records a first K-run starting at step 94 that lasts 1 106 steps, censored, with component `chase` and composition chase 1 106 / last_seen 0 / head 0.
- So the 1 106 constant full-speed requests come from `InterceptLaw` chasing a VISIBLE evader. They do not come from the "memoryless learned policy". The §1 sentence describing it is wrong.

Replacement hero episodes, all archived. Plain is Exp. 8 `*_PLAIN`, the recovery is `*_RECF-20`, and the stall owner is from N1:

| policy, opponent, seed | stall owner (first K-run) | run start / length | plain outcome | under recovery N = 20 |
|---|---|---|---|---|
| Free, GREEDY, 1 740 044 | head (not crossing) | 8 / 1 192 | timeout | capture |
| Free, GREEDY, 1 740 401 | head | 9 / 1 191 | timeout | capture |
| Free, GREEDY, 1 740 224 | last_seen | 96 / 1 104 | timeout | capture |
| Free, GREEDY, 1 740 053 | last_seen | 114 / 1 086 | timeout | capture |

## 1. R3: Remark 2 and Fig. 2 (S15, `s15_round.json` verdicts.m1; `s15_cells/*__h10.json` entries)

The pool is the hold-10 cells: FT-B-41 and RC19 (Aware), each against ATK4-45 and GREEDY. It has 383 entries, 97 escaped and 286 not. The table also needs the related-work row for Exp. 2, because Table 1's policies are "Aware; its anchored fine-tune (FT-B-41)".

| quantity | value | source |
|---|---|---|
| capture after escape / not | 0.299 (29/97) / 0.042 (12/286); A₀ = 0.257 | m1 |
| `mean_steps_remaining_escaped` = 629 | mean steps **from the stall ENTRY to episode end** over all 97 escaped entries (68 of them timeouts); median 549 | `steps_remaining = total − start`, s15 script line 366 |
| entry → capture, the 29 captured escapes | mean **346**, median **314** | recomputed |
| escape → capture, the 29 captured escapes | mean 327, median 295 | recomputed (`steps_remaining − steps_to_escape`) |
| steps from entry to escape | mean 18.3, median 17 | m1 |
| registered Δ return-to-go (γ = 0.99 per env step) | **0.0092** (P1's first conjunct held, bar 0.05) | m1 |
| γ^{E[T]}·A₀ as printed in the paper | 0.0005 (0.99) / 0.137 (0.999); this is a lower estimate of A₀·E[γ^T] by Jensen's inequality | `figure_data.json` F2 |
| correct A₀·E[γ^T], env-step clock | 0.0136 (0.99) / 0.183 (0.999) | recomputed |
| Δ return-to-go, env-step clock, γ = 0.999 | 0.295 | recomputed |

- **Fig. 2(b) pool.** It is all **8** cells, so hold 1 and hold 10 are mixed: 208 escapes, mean 677. The caption and text describe the hold-10 pool (97, 629). Its axis label reads "steps from escape to episode end", but the quantity is measured from the ENTRY and it includes timeouts.
- **Fig. 2(c).** It plots A₀·γ^629 and a "0.05 bar". That bar is P1's registered |Δ return-to-go| bar, `P1_RTG_BAR = 0.05` at s15 script line 79.
- **Single number for the abstract, the Remark and Fig. 2(c).** Under F-A, no "worth 0.009" claim survives.
  - Registered and correctly labelled: "Δ discounted return-to-go 0.0092 when the discount is applied per env step".
  - On the learner's clock: "≥ 0.29 per decision at γ = 0.99" (recomputed).
  - Recommendation: carry the registered P1 FAIL and its registered direction, and drop the discount mechanism as the explanation.

## 2. R5: the recovery (Proposition 3)

| fact | value | source |
|---|---|---|
| candidate order | (v_max, 0), (½v_max, 0), nominal, (0, +ω_max), (0, −ω_max); the first whose filtered speed is > `move_eps` is executed; else the larger \|ω\| rotation if \|ω\| ≥ 0.1 rad/s | `pe2/shield.py:662–676`; `RECOVERY_OMEGA_MIN = 0.1` at `shield.py:69` |
| the "nominal" candidate never wins | `policy` kind = 0 in every cell (S17 amendment A1) | D131; `recovery_kinds.policy` = 0 |
| rotation certified in Table B1 | \|ω\| ≥ **0.5 ω_max** | `supplement.tex:33` |
| T_turn | one full turn: 2π/2.84 = 2.21 s = **45** steps at ω_max, **89** at ½ω_max | arithmetic |
| displacement term | 0.10 m / (0.05·v_max·Δt = 0.0025 m) = **40** steps at the immobility threshold | arithmetic |
| registered bound | N + 45 (65 at N = 20) | `s26_b2_round.json` H-B2-2 `bound_recf20` = 65 |
| H-B2-2 verdict | **FAIL** (registered 95 % per cell; BASE-62 cells 0.945 / 0.923) | H-B2-2 |
| tail split (creep vs interrupted turn) | **not available cheaply.** `escape_times` holds only {t, remaining} per engagement; motion type would need traces (the cells store `traces` for a few episodes only) | `paper_b_filter_sweep.py:315, 440` |

Within-bound shares and median escape by N, pooled over 10 cells (H-B2-2 `all`):

| N | triggers | decidable | within N+45 | median escape (range over cells) |
|---|---|---|---|---|
| 5 | 15 306 | 15 232 | 0.998 | 4–5 |
| 10 | 14 867 | 14 779 | 0.991 | 4–8 |
| 20 | 10 560 | **10 475** | **0.978** | 4–8 |
| 40 | 8 754 | 8 652 | 0.952 | 4–10 |
| 80 | 7 378 | 7 216 | 0.963 | 4–9 |

- **R15.** The median escape is "4 to 10 steps" over all N, not "4 to 8".
- **Why 10 466 and 10 475 differ.** The record's "decidable" set means escaped, OR censored with at least the bound's worth of episode remaining (`paper_b_filter_sweep.py:628`). That gives 10 475, of which 0.9778 are within the bound. The figure (`paper_b_figures_s26.py:226`) counts escaped triggers only: 10 466 escaped, 94 censored, share 0.9786. The 9 triggers that were censored with ≥ 65 steps left are failures in the record and are absent from the figure.

## 3. R6: registered verdicts

| Exp. (round) | registered item | verdict | registered direction / numbers | source |
|---|---|---|---|---|
| 2 (S15) | P1 (escape advantage negligible: \|Δrtg\| < 0.05 AND capture after escape < 0.15) | **FAILS** (0.0092 passes, 0.299 fails) | "The signal was there and learning still missed it; the explanation is optimisation, not credit assignment." | `s15_round.json` m1; prereg `loop_mechanism_prereg_2026-09-20.md:66` |
| 2 | P2 (frozen action) | HOLDS (decided on hold 1: median Δ 0.000, max 0.0030) | world-obs Δ median 0.000; evader visible 2.4 % of stalled steps (all 8 cells; 3.4 % in the hold-10 pool) | m2 |
| 2 | P3 (entropy collapse, control) | N/A: σ is state-independent (A1) | | m3 |
| 2 | P4 (commitment, control) | HOLDS: hold 10 vs 1 changes immobility by +0.007 / −0.039 / −0.047 / −0.037 | | m4 |
| 3 (S16) | H-FIX-1, primary metric **immobile-step rate** | **ALL ARMS FAIL, including REC** | REC deadlock 0.000 in 9/9 cells by spinning; firings 47 205 / 414 928 / 133 616 per 1 500 episodes (seed 61 ≈ 277 per episode) | D129 |
| 3 | H-FIX-2 (no learned arm matches REC) | holds as predicted | | D129 |
| 3 | H-FIX-3 (Hor below Base by ≥ 0.05 immobility on ≥ 2 of 3 seeds) | PASSES (Δ −0.115 / −0.055 / −0.024) | | D129 |
| 5 (S18) | **H-CROSS** (penalty raises capture with the dense varying reward, lowers it with the sparse reward) | **FAIL** (sign wrong, not significant) | the penalty LOWERS capture in all three columns | `s18_round.json` summary.hypotheses |
| 5 | H-MECH | PASS | | same |
| 6 (S19) | H-PLAN-1 (planner raises shielded capture on ≥ 2/3 opponents) | **FAIL 1/3** | GREEDY +0.026 (p = 0.015), HUG −0.018 (p = 0.035), ATK −0.022 (p = 0.007) | `s19_round.json` |
| 6 | H-PLAN-2 | FAIL 1/3 | | same |
| 6 | H-PLAN-3 (law-pin episodes drop by ≥ 50 %) | PASS 3/3 | 0.170 → 0.002, 0.132 → 0.008, 0.368 → 0.004 | same |

Exp. 5 capture by reward column, BASE → PEN per seed 70/71/72 (`s18_round.json` H-CROSS / H-MECH):

| column | seed 70 | seed 71 | seed 72 | pooled paired p |
|---|---|---|---|---|
| SPARSE | 0.516 → 0.596 | 0.622 → 0.592 | 0.466 → 0.416 | 1.0 (0.535 vs 0.535) |
| DVARY ("dense-reward column") | 0.442 → 0.338 | 0.480 → 0.300 | 0.528 → 0.212 | 3.6e-51 |
| DRARE | 0.620 → 0.560 | 0.508 → 0.640 | 0.538 → 0.406 | 0.13 |

- **Table 1 mislabel.** Exp. 5's "three penalty schedules" are actually three REWARD columns, each run without and with the penalty (BASE / PEN rows).
- **Exp. 5 is not a Base-family experiment.** Its arms are S18 learners with obs width 16, trained from scratch at seeds 70–72.

## 4. R10: Experiment 6 reconciled (`s19_cells/{LINE,PLAN}-SHIELD_*.json`; `capture_ind`, `law_pin_ge_40`)

- **The search policy in every cell is the scripted `OrbitSearch`, not a learned head** (`planner_intercept_prereg_2026-10-06.md:45`). No learned parameter is in Exp. 6, so it is a scripted-controller result.
- **"Law-held"** means episodes of LINE-SHIELD with a law run of at least 40 steps.

| opponent | law-held episodes (LINE) | captured under LINE | same episodes captured under PLAN | other episodes LINE → PLAN | total LINE → PLAN | law-held under PLAN |
|---|---|---|---|---|---|---|
| GREEDY | 85 | 0 | **16** | 190 → 187 | 190 → 203 | 1 |
| HUG | 66 | 8 | 3 | 148 → 144 | 156 → 147 | 4 |
| ATK | 184 | 7 | 0 | 50 → 46 | 57 → 46 | 2 |

The editorial's "13 of 85" is the NET change 190 → 203. The planner converted 16 of the 85 law-held greedy episodes into captures and lost 3 elsewhere.

## 5. R11: artefacts and training rewards

| item | fact | source |
|---|---|---|
| Free = STR14 | best final of 5 confirmatory seeds STR11–15. "Artefact rule → STR14" is the highest gate rate, 0.738, with finals 0.692–0.738. Training: SearchMDP, PPO from scratch, terminal-only reward, γ 0.99, 10 M env steps, **no filter, no filter features** (obs width 11) | D111 |
| Aware = RC19 | best of the RC arm in "RC, RA, RB order; highest gate rate". **Filter-state features (3) + stall penalty 1e-3**, CBF in the training loop, 10 M steps, obs width 14 | D113; prereg `shield_remedy_prereg_2026-09-15.md:56–60` |
| Exp. 1 remedies, shielded capture vs greedy (stream 1 090 000+) | RA (features) 0.522 / 0.512 / 0.544; RB (penalty) 0.426 / 0.464 / 0.436; **RC (both) 0.564 / 0.550 / 0.444**; shielded Free 0.396; shielded teacher 0.426; unshielded Free 0.728 | D113, D112 |
| Free 0.396 vs 0.426 | 0.396 = Exp. 1 cell (S4 stream 1 090 000+, Greedy at ratio 1.2); 0.426 = Exp. 8 `STR14_GREEDY_PLAIN` (stream 1 740 000+). Different episodes, same opponent | D112; s26_b2 |
| **Base (and Obs, Hor, ObsHor, PEN, MEM)** | trained with the **S5 RC configuration**: CBF in the loop, **`shield_obs` = True (3 filter-state features), stall penalty 1e-3** (PEN 1e-2), R_ACQ 0, against GreedyEvader, 5 M env steps. Base obs width 14; Obs and ObsHor 16 (+2 escape features) | `derisk_s16_fixes.py:9, 94–97, 236–256`; `derisk_s17_fixes_b.py:102–104` |

Consequence for §3: "Base" is Aware's recipe at 5 M steps. Every trained family, Base included, sees the filter's state and is charged 1e-3 per stalled step.

## 6. R12: statistical reporting

The intervals are 95 % **Newcombe paired score intervals**, Newcombe 1998 method 10. Paired Wald intervals agree to 0.001. p is the exact two-sided binomial test on discordant pairs.

| delta | rates | Δ | 95 % CI | discordant (gain / loss) | p |
|---|---|---|---|---|---|
| Exp. 4 Base seed 1, greedy capture, bare → recovery | 0.466 → 0.640 | +0.174 | [+0.140, +0.207] | 88 / 1 | 3e-25 |
| Exp. 4 Base seed 2 | 0.366 → 0.718 | +0.352 | [+0.309, +0.393] | 176 / 0 | 2e-53 |
| Exp. 4 Base seed 3 | 0.402 → 0.442 | +0.040 | [+0.021, +0.059] | 22 / 2 | 3.6e-5 |
| Exp. 4 ObsHor seeds 1–3 | 0.560 / 0.634 / 0.636 → 0.624 / 0.668 / 0.686 | +0.064 / +0.034 / +0.050 | [+0.037, +0.091] / [+0.017, +0.051] / [+0.031, +0.070] | 40/8, 18/1, 25/0 | ≤ 7.6e-5 |
| Exp. 4 Base deadlock vs attacker, seeds 1–3 | 0.298 / 0.186 / 0.252 → 0 | −0.298 / −0.186 / −0.252 | [−0.340, −0.259] / [−0.222, −0.153] / [−0.292, −0.215] | | ≤ 2e-28 |
| Exp. 8 Free, greedy capture, plain → N = 20 | 0.426 → 0.728 | +0.302 | [+0.261, +0.341] | 151 / 0 | 7e-46 |
| Exp. 8 Aware | 0.560 → 0.654 | +0.094 | [+0.068, +0.120] | 47 / 0 | 1.4e-14 |
| Exp. 1 Free, filter cost (archived) | 0.728 → 0.396 | −0.33 | [−0.375, −0.289] | | 3e-44 |

- **Minimum detectable difference.** With N = 500 paired episodes, an exact two-sided test at α = 0.05 and power 0.8, the minimum detectable difference is about **0.05** at 10 % discordance, **0.06** at 20 % and **0.08** at 30 %. The calculation is a simulation-free exact power sum.
- **Fragile secondary p-values.** The weakest are the Exp. 6 deltas:
  - GREEDY: p = 0.015, from 19 / 6 discordant pairs.
  - HUG: p = 0.035, from 3 / 12.
  - ATK: p = 0.007, from 2 / 13.
  - Exp. 5 DRARE pooled: p = 0.13.
  - Exp. 1 RB seed 2: p = 0.16.
  - Each of these effects is at or below the minimum detectable difference.
- **Multiplicity.** No correction was registered in any round. Each prereg names its primary verdicts, so "a pre-specified primary per experiment; no correction within the registered family" describes the practice accurately.

## 7. R13: observation, reward, controller constants (code)

| item | value | source |
|---|---|---|
| base observation, 11 dims (`obs_dim = 7 + 2·n_obstacles`, n = 2) | [0–2] own x/5, y/5, θ/π; [3–5] evader dx, dy in the body frame /10, dθ/π, **zeroed when not visible**; [6] visible flag; [7–10] the two nearest obstacles' **centres** in the body frame /10 | `pe2/obs.py:1–20, 66–103` |
| obstacles | 2 disks per episode, radius U(0.3, 1.0) m, placed by rejection sampling at reset (seeded); **centres observed, radii NOT observed; arena walls only through own pose** | `pe2/env.py:305–328`; `pe2/config.py:72–73` |
| pressed-against obstacle observed? | its centre is in the observation (it is one of the two nearest); its radius and the clearance are not, except through `shield_obs`'s min_clearance | as above |
| `shield_obs` (+3; every trained family incl. Base, and Aware) | [intervention flag of the last executed step, min_clearance clipped to [−0.5, 2] m, stall_run / 40 clipped to [0, 2]]; the counter **saturates at 80 stalled steps** | `pe2/structured.py:302–313` |
| `escape_obs` (+2; Obs, ObsHor) | the admissible ω range at v = 0 at the CURRENT true pose | `structured.py:200–204, 687` |
| widths | Free 11; Base, Hor, Aware 14; Obs, ObsHor 16 | D131 (verified from checkpoints) |
| reward | terminal +1 capture / −1 timeout (env); minus 1e-3 per stalled env step inside a learner transition (stall penalty); R_ACQ = 0 for S5/S16/S17; summed over the transition | `pe2/env.py:17, 286–292`; `structured.py:325–333 (_shaped)` |
| discount | γ per **transition** (one transition = ≤ 10 head steps + the law's run); about 15–16 env steps per transition in training | `structured.py:347–371`; s16 records |
| intercept law | chase: v_max with ω = clip(3.0·bearing); go-to-last-seen until within **LAST_SEEN_TOL = 0.5 m**, else hand over | `structured.py:70–83`; `pe2/scripted.py:21, 25` |
| HOLD | 10 env steps; ends early if the evader becomes visible | `structured.py:44, 357` |
| capture | centre distance ≤ 2·0.15 = 0.30 m | `config.py:68, 78–80` |
| evader speed | 1.2 v_max | `config.py:67` |
| PPO | SB3 2.7.1 defaults apart from γ (MlpPolicy: 2 × 64 tanh, n_steps 2048, batch 64, 10 epochs, lr 3e-4, GAE 0.95, clip 0.2, entropy 0). MEM = RecurrentPPO MlpLstmPolicy | `derisk_s16_fixes.py:260–271`; `derisk_s5_remedy.py:346`; `derisk_s3_confirm.py:244` |

## 8. Q5: exact-tier feasibility in evaluation

- **Exp. 4.** Every one of the 63 S17 gate cells (21 arms × 3 opponents, 500 episodes each) records exactly **1** step off the exact tier. That is `infeasibility_rate × n_steps` = 1.000, about 1 step in 4 × 10⁵. `backup_rate` is 0 everywhere.
- **The one step is the same in every arm.** Its count does not depend on the arm, so it is very likely one seeded initial state common to all arms. The record does not say which episode, and finding out would need a replay.
- **Exp. 8 and 9.** The 86 S26 B2 cells have `backup_rate` 0 and no infeasibility field.
- **Exp. 7.** All 262 stalled states are on the exact tier.
- **Wording.** "exact-tier feasibility was lost on one step per 500-episode cell, never at a stall, and no step used the backup tier; zero contact steps".

## 9. R15: other items

- **Fig. 1's 1 106-step episode versus D125's 1 085-step episode.**
  - 1 106 is Obs seed 2 against HUG7, seed 1 740 003, Exp. 4. It is a **chase** stall (F-B).
  - 1 085 is the S14 clip D125 watched. That was an unregistered 100-episode diagnostic, with command σ 0.002 and the policy RC19 or FT-B-41 per D125.
  - They are different episodes.
- **Median escape** is 4–10 steps over N (§2).
- **Visible-fraction pool.** It is 2.4 % over all 8 S15 cells (decided on hold 1), and 3.4 % over the hold-10 pool.

## 10. N1 extras (`results/derisk/s27_n1_round.json`, `families`, `single_policies`)

The first three columns are the first-K owner of the deadlock episodes, from M1. The last three are stalled-step shares by component, from M2.

| policy | opponent | deadlock / N | first-K chase | last_seen | head | stalled chase | last_seen | head |
|---|---|---|---|---|---|---|---|---|
| Base | A | 368 / 1500 | 0.008 | 0.986 | 0.005 | 0.014 | 0.891 | 0.096 |
| Base | H | 187 / 1500 | 0.032 | 0.947 | 0.021 | 0.042 | 0.754 | 0.204 |
| Base | G | 153 / 1500 | 0.033 | 0.941 | 0.026 | 0.024 | 0.735 | 0.241 |
| Obs | A | 504 / 1500 | 0.002 | 0.966 | 0.032 | 0.006 | 0.962 | 0.032 |
| Obs | H | 294 / 1500 | 0.020 | 0.707 | 0.272 | 0.026 | 0.760 | 0.214 |
| Obs | G | 273 / 1500 | 0.033 | 0.736 | 0.231 | 0.035 | 0.751 | 0.214 |
| Hor | A | 643 / 1500 | 0.005 | 0.694 | 0.302 | 0.007 | 0.672 | 0.322 |
| Hor | H | 362 / 1500 | 0.014 | 0.486 | 0.500 | 0.019 | 0.470 | 0.511 |
| Hor | G | 373 / 1500 | 0.032 | 0.483 | 0.485 | 0.029 | 0.468 | 0.503 |
| ObsHor | A | 487 / 1500 | 0.010 | 0.926 | 0.064 | 0.009 | 0.929 | 0.062 |
| ObsHor | H | 231 / 1500 | 0.013 | 0.905 | 0.082 | 0.020 | 0.877 | 0.102 |
| ObsHor | G | 223 / 1500 | 0.036 | 0.874 | 0.090 | 0.033 | 0.862 | 0.105 |
| Free | A | 355 / 500 | 0.006 | 0.234 | 0.761 | 0.005 | 0.246 | 0.749 |
| Free | G | 242 / 500 | 0.021 | 0.132 | 0.847 | 0.024 | 0.126 | 0.851 |
| Aware | A | 293 / 500 | 0.007 | 0.642 | 0.352 | 0.007 | 0.684 | 0.309 |
| Aware | G | 142 / 500 | 0.021 | 0.394 | 0.585 | 0.017 | 0.385 | 0.598 |

- **Registered decision.** h_max is 0.302 (Hor), which selects **D-2**. Free's head share is 0.761 and Aware's is 0.352.
- **Registered pooled sentence.** Over the four trained families against the attacker there are 2 002 deadlock episodes:
  - 1 747 were owned by the law driving to its last-seen point.
  - 12 were owned by the law chasing.
  - 243 were owned by the learned head.
