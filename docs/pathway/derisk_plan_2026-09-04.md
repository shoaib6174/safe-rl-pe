# De-risk Plan — Four Pre-Registered Tests Before Steps 8–12 (2026-09-04, S121)

**Purpose**: the strong paper (plan v3 R1–R4) depends on facts we have not yet established. Each test below is pre-registered per D83 — hypothesis, pass/fail numbers, N with a power rationale, negative control, budget, and what each outcome does to the paper — **before any run starts**. Results cannot move these goalposts.

**Ordering**: T1 and T2 are GPU-bound and paper-deciding; they start first and run in parallel with T3 and T4, which are CPU-bound. Nothing in Steps 8–12 starts until the decision table in §6 has been applied.

**Protocol for every test**: seeded paired evaluation on the Step 5 harness (D57 seeds), frozen opponents (D55), Wilson CIs (D56), best-checkpoint selection (D14), paired McNemar as the primary test with CI non-overlap descriptive (D83), a mandatory negative control that must fail (D83), all numbers in a JSON under `results/derisk/`, visual gate GIF per test signed in the worklog.

**Power rationale (shared)**: N = 500 paired episodes. Wilson half-width at p ≈ 0.2 is ≈ 0.035, so a 5-point delta is resolvable on the CI alone; paired McNemar on the same seeds has strictly more power. N = 1000 for any result within 0.03 of a threshold (D57 band-edge rule).

---

## T1 — Can a pursuer LEARN in this environment? (GPU, blocking)

**Why**: RL from scratch (D68–D74) and RL fine-tuning of the BC clone (D78–D82) are refuted at 1M steps with single-step SAC at γ = 0.999; root cause is credit assignment over 1200 steps from a ±1 terminal. D84 names the retry: PPO + GAE (λ-return machinery already in SB3). If this fails too, the negative result becomes algorithm-family-independent and the "learning" half of "shielded learning game" is hollow.

**Hypothesis H-T1**: a PPO policy behaviour-cloned onto the FovLimitedPursuer demonstrations and fine-tuned with PPO+GAE at γ = 0.999 exceeds the gated SAC clone on the D76 battery.

**Arms**:
- (a) PPO-BC clone alone (no fine-tune) — establishes the PPO starting point.
- (b) PPO-BC → PPO+GAE fine-tune, λ = 0.95, γ = 0.999, 3 seeds, 2M env steps each, checkpoint every 100k, D14 selection.
- (c) **Negative control**: PPO+GAE from a random init, same budget, 1 seed — must FAIL the D76 gate (the LN-scratch SAC control was flat at 0.005; if random-init PPO passes, the gate is broken, not the hypothesis confirmed).

**Pass** (all three, pre-registered):
1. Arm (b) best checkpoint passes the D76 battery gate (beats every heuristic CI-hi and the pooled random-init CI-hi; McNemar p < 0.05 vs iid_random).
2. Arm (b) beats the **SAC clone (0.185)** on the same 500 paired seeds by paired McNemar p < 0.05, in ≥ 2 of 3 seeds.
3. Arm (b) beats **its own PPO-BC start (arm a)** by paired McNemar p < 0.05 — this is the actual learning claim; a fine-tune that merely preserves the clone is D82 again.

**Stretch (reported, not gated)**: D70 scenario suite — any nonzero capture in `occluded_start` or `shadow_adjacent`, where the clone scores 0/30, is the first evidence of learned search.

**Fail**: criterion 3 fails in ≥ 2 of 3 seeds at 2M steps, or the fine-tune collapses below the BC start (the D78/D82 signature).

**Prerequisite build**: a PPO behaviour-cloning path (`train_bc_actor` currently targets the SAC actor; PPO's MlpPolicy needs its own supervised head) and a `finetune_br` variant for PPO. Both are Step 7 scope, ~2 days.

**Budget**: 3 seeds × 2M + control 2M ≈ 8M env steps on niro-1 GPU.

---

## T2 — Can an EVADER be learned at all under FOV? (GPU, blocking)

**Why**: D79 — the scripted evader is non-Markov under a 90° FOV (it flees from memory 99.6 % of the time), so a memoryless clone fails (0.015 vs teacher 0.223). Every self-play, attacker and remedy result needs a learned evader. The headroom is known: FLe survives 22 % on speed alone.

**Hypothesis H-T2**: an evader policy conditioned on the Step 4 `LastSeenBelief` state (last-seen position, velocity, time-since-seen) is Markov enough to be cloned from FLe and then improved by PPO+GAE.

**Arms**:
- (a) Belief-conditioned PPO-BC clone of FLe (obs = 11-D observation ⊕ belief vector).
- (b) (a) → PPO+GAE fine-tune, 3 seeds, 2M steps, vs the frozen gated pursuer (T1 winner if available, else the SAC clone).
- (c) **Recurrent alternative**: SB3 RecurrentPPO (LSTM) clone + fine-tune, 1 seed — run only if (a) fails its gate, to separate "belief features insufficient" from "evader unlearnable".
- (d) **Negative control**: the memoryless clone from D79 re-evaluated on the same seeds — must remain ≈ 0.015 and fail the gate.

**Pass**:
1. Arm (a) clone escape rate ≥ 0.15 [CI-lo] on 500 paired seeds vs the frozen pursuer — i.e. recovers at least ~two-thirds of the teacher's 0.223.
2. Arm (b) beats arm (a) by paired McNemar p < 0.05 in ≥ 2 of 3 seeds.

**Fail**: arm (a) < 0.10 CI-hi AND arm (c) < 0.10 CI-hi → the evader is not learnable at this budget; record it and go to the opponent-family paper (§6).

**Budget**: ≈ 8M env steps GPU. Runs concurrently with T1 after the shared PPO-BC path exists.

---

## T3 — Does a shield shift the equilibrium of the EXACT game? (CPU, 2–3 days, not blocking but paper-shaping)

**Why**: plan v3 Step 3b. Jia et al. (D90) assert that safety projection alters learning dynamics and may prevent convergence; they never measure it. The exact gridworld from `scripts/experiment_1_tabular_forall_exists.py` (Experiment 1, D54) can compute it exactly, giving R1 a direction prediction before any deep run and a theory contribution that does not depend on T1/T2.

**Build**: add hazard cells to the Experiment-1 gridworld; define a **projection shield** as "any move into a hazard is replaced by the nearest non-hazard move" applied to BOTH players; compute (i) exact Nash value and (ii) exact best-response dynamics for the unshielded game, the shielded game, and a shield-on-pursuer-only game (the asymmetry confound in plan v3 Step 8).

**Hypothesis H-T3**: the shielded game's value and its BR-dynamics fixed point differ from the unshielded game's; the pursuer-only shield shifts the value toward the evader.

**Pass**: |V_shielded − V_unshielded| > 0.02 (2 points of capture probability) on ≥ 2 of 3 hazard layouts (sparse / medium / corridor), with the sign recorded as the **R1 direction prediction**. **Null result is also publishable**: "projection shields are benign in the exact game" contra the Jia et al. assertion — record it either way.

**Negative control**: a shield that never fires (hazards placed where no policy ever moves) must reproduce the unshielded value to numerical precision.

**Visual gate**: `results/derisk/t3_exact_shift.png` — value heat-maps, shielded vs unshielded, per layout.

---

## T4 — Does the shield hold under the pre-registered gates? (CPU, 2–3 days, not blocking)

**Why**: Step 6 gates G1–G4 were pre-registered in S119 and the metrics are now built (S120, D89). Running them is the system half of the paper and costs no GPU.

**Build**: `scripts/pe2_step06_gate.py` + `ObstacleHuggingEvader` (G3 probe; G4 needs it unshielded). Rules carried from the S120 review: shielded and control sweeps in **separate** `summarize_shield` calls; the control arm's stall/deadlock/pinned fields are structurally zero, not measured; G1's Wilson bound from `eval.wilson_ci`. Robust-margin arm (D87) run as a fifth gate G5 with the same criteria.

**Pass**: G1 0 violation steps / 1000 episodes (seeds 760000+); G2 shielded CR ≥ 0.5 × unshielded on paired seeds (McNemar reported); G3 0 violations vs the hugging probe (seeds 770000+); G4 unshielded probe ≥ 5 % episodes-with-violation (else the probe is too weak — fix the probe, not the gate); G5 robust-margin arm passes G1/G3 with intervention rate reported.

**Reported, not gated**: infeasibility rate, `min_clearance_floor`, tier histogram, immobile/stall/pinned rates and run-length distributions.

**Visual gate**: `step_06_shield.gif` per the S119 checklist.

---

## 5. Parallelism and calendar

```
GPU (niro-1):  [T1 build 2d] → T1 arms (b),(c) ──┐
               [T2 build 1d] → T2 arms (a),(b)  ──┤  ~2–3 weeks wall time
CPU:           T3 exact twin (2–3 d) ; T4 gates (2–3 d) ; T1/T2 evals as checkpoints land
```
Decision meeting when T1 and T2 have their 2M-step checkpoints evaluated. T3/T4 will be done first.

## 6. Decision table (pre-registered)

| T1 pursuer learns | T2 evader learns | Paper |
|---|---|---|
| pass | pass | **Strong**: plan v3 as written — shielded self-play (R1), learned attacker (R2), remedy (R3), hardware (R4) |
| pass | fail | **Asymmetric**: learned pursuer vs an evader **family** (scripted, belief-clone, speed ratios); R1 on the BR landscape; R2 attacker = learned pursuer exploiting a shielded evader family; R4 unchanged |
| fail | pass | **Mirror asymmetric**: learned evader vs pursuer family; R2 = learned evader as shield-attacker (the more natural direction anyway) |
| fail | fail | **Characterization paper**: exact-game shift (T3) + shield gates (T4) + hardware (R4) on cloned/scripted policies; the algorithm-family-independent learning failure is reported as a finding with paper 1's rigor. Still publishable; not the strong paper |

T3's sign becomes the R1 prediction in all four rows. T4 passing is required in all four rows.

**Fallback trigger, pre-committed**: the row is chosen by the pass/fail criteria above at the stated budgets. No additional seeds, budget extensions, or criterion changes after the results are seen. A change of algorithm beyond PPO+GAE and RecurrentPPO is a new pre-registration, not a retry.

## 7. What would change my odds

Strong-paper odds are 40–50 % today. T1 pass alone → ~60 %. T1 + T2 pass → ~70 %. T3 nonzero shift adds a theory contribution independent of the deep runs. T1 + T2 fail → the characterization paper at ~75 % publishable but not strong.
