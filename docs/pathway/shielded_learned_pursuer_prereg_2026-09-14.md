# S4 — Shielded Games on the learned pursuer (plan v3 Step 8-alt; 2026-09-14, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing decisions: D83 (pre-registration discipline), **D97** (shield v2: hard body rows +
soft VCP rows, exact one-step check, never-drop tiers; 0 violation steps over ≈ 3.5 M shielded
steps on scripted pursuers), D89 (reportable margin = `min_clearance` in metres; stall /
immobile / deadlock / pinned split; `NoOpShield` = the `"none"` control tier), D87
(robust-margin arm δ = 0.05 m is required), **D111** (the learned pursuer STR14, 0.738 vs the
teacher's 0.474), and the S122 exact-twin result **T3** (pursuer-only shielding uniformly hurts
the pursuer: ΔV −0.31 … −0.50 on 3/3 layouts — the pre-registered H1 direction).

## 1. Why this round exists

Paper 2's core question is what a hard safety filter does to an adversarial learning game.
Step 6 measured the shield on a *scripted* pursuer (G2: capture 0.451 → 0.375, a 7.6-point
drop, with stall rate 0.675 and deadlock episode rate 0.521). Step 3b predicted the sign in an
exact twin. A learned pursuer now exists (D111). This round puts the shield on it.

**R1 as tested here (Step 8-alt, plan v3):** the shield shifts the learned pursuer's outcome
across an opponent family, with the direction predicted in advance by T3, at zero violations,
and the costs of that shift are measured. **R3-lite (secondary):** training *under* the shield
recovers what shield-at-deploy loses — the CBF-RL question in the adversarial setting.

**Mechanism.** The shield filters every executed pursuer command inside `SearchMDP` — the fixed
chase/last-seen law's commands and the learned search's alike — using the true pursuer pose and
the static map (as in Step 6). The learner's observation is unchanged; only execution differs.
`NoOpShield` is the paired control: identical episodes, barriers measured, nothing filtered.

## 2. Components

- `pe2.structured.SearchMDP(…, shield=None)`: when a `CBFShield` (or `NoOpShield`) is given,
  `_step_inner` converts the normalised action to `(v, ω)`, calls `shield.filter((v, ω), pose,
  obstacles, v_max=cfg.v_max)`, executes the filtered command, and records
  `ShieldMetrics.record(info, violation=<env contact flag of the pursuer>)`. `play_episode`'s
  log gains `shielded_role = "pursuer"`, `violations` (contact steps of the pursuer) and
  `shield = metrics.summary()`, so `pe2.metrics.summarize_shield` applies unchanged. Training
  with a shield uses the same path (the reward is the unshaped terminal ±1; the shield never
  touches it).
- `pe2.structured.evaluate_structured(…, shield_factory=None)`: one fresh shield per episode.
- Runner `scripts/derisk_s4_shield.py`.

## 3. Protocol

- Role: pursuer. Policies: **STR14** (`results/derisk/ckpts_STR14/STR14_10000000.zip`, hold 10)
  and **TEACHER** (`OrbitSearch` at hold 1 inside `SearchMDP` — V1-identical to
  `SearchingPursuer`). Shield arms: **NONE** (`NoOpShield`), **SHIELD** (`CBFShield(cfg)`,
  δ = 0), **ROBUST** (`CBFShield(cfg, margin=0.05)`).
- Opponent family (six): `GreedyEvader` at speed ratio **1.2** (primary), `FovLimitedEvader`,
  `ObstacleHuggingEvader`, `OpenSpaceEvader` (all at 1.2), and `GreedyEvader` at **1.0** and
  **1.4** (`EnvConfig(evader_speed_ratio=…)`; STR14 was trained at 1.2 and is not retrained).
- **N = 500 paired episodes per (policy, opponent, shield arm)**, one fresh stream per
  opponent, shared by the three shield arms and both policies so every comparison is paired:
  Greedy-1.2 **1 090 000+**, FovLimited **1 091 000+**, ObstacleHugging **1 092 000+**,
  OpenSpace **1 093 000+**, Greedy-1.0 **1 094 000+**, Greedy-1.4 **1 095 000+** (fresh; ≤ 1 080 999
  consumed). TEACHER runs on Greedy-1.2 only (three arms). Total deployment: 21 (policy,
  opponent, arm) cells × 500 episodes.
- **Shield-in-training arms (R3-lite):** **STRSH-16, STRSH-17, STRSH-18** — SBLONG/S3's exact
  configuration (`SearchMDP(cfg, GreedyEvader, seed0 = seed × 1000, hold = HOLD, r_acq = 0)`,
  PPO from scratch, γ = 0.99, 10 M env steps, no selection) with **`shield = CBFShield(cfg)`
  inside the training env and inside every probe/gate rollout**; probes every 1 M on
  **1 100 000+** (N = 200, shielded); gate = the final checkpoint, shielded, on the primary stream
  1 090 000+ (N = 500). Thread count `OMP_NUM_THREADS=2` everywhere; provenance recorded.
- **Gate rule** (S2 §3): paired exact two-sided McNemar p < 0.05 with the majority of
  discordant pairs in the stated direction, AND Wilson bound separation as stated per
  hypothesis.

## 4. Hypotheses and verdicts (fixed now)

- **H-SAFE (kill criterion, as G1/G3):** on every SHIELD and ROBUST cell (deployment and
  training, all 10 M training steps included via the probe counters), **0 violation steps** —
  an env contact step (wall or obstacle) of the pursuer, magnitude-blind. Any violation → the
  round stops after the census is written; nothing else is read (Step 6 stop rule). The NONE
  arms report their violation-episode rate; if it is < 5 % on Greedy-1.2 the family member is
  too benign to test the guarantee and that is reported (G4's logic).
- **H1 (shift, primary):** on Greedy-1.2, **SHIELD STR14 captures less than NONE STR14** —
  McNemar p < 0.05 with more shielded losses than wins (c > b) AND the shielded Wilson upper
  bound below the unshielded point estimate. Verdict `CONFIRMED` / `NO SHIFT` (p ≥ 0.05) /
  `REVERSED` (significant in the other direction). The effect size Δ = rate(SHIELD) − rate(NONE)
  with a paired CI is reported and compared to the teacher's Δ on the same stream and to Step
  6's −7.6 points.
- **H1-family (secondary, reported):** the same test on each of the other five opponents and
  on ROBUST; "the shift holds on k of 6 opponents".
- **H-COST (reported, never gated, D89):** intervention, infeasibility, backup-tier, stall,
  immobile, deadlock-episode and pinned rates, tier histograms, `min_clearance` floors — for
  STR14 vs TEACHER under SHIELD and ROBUST. The question on record: does the learned pursuer
  trigger the shield more or less than the scripted one?
- **H3-lite (secondary):** **STRSH final (shielded) beats SHIELD STR14** on Greedy-1.2 — gate
  rule with Wilson-lo(STRSH) > point(SHIELD STR14) — on **≥ 2 of 3 seeds**; also reported: does
  it beat the unshielded TEACHER, and its own training-time violation count (must be 0 under
  H-SAFE). Verdict `RECOVERED` / `NOT RECOVERED`; INCOMPLETE if a seed record is missing.

## 5. What each outcome fixes

| result | direction |
|---|---|
| **H-SAFE holds, H1 CONFIRMED** | R1 is measured on a learned agent with the sign the exact twin predicted: the paper's central result. Magnitude and costs go into the results section; H1-family and H-COST are the characterisation. Next: Step 9 (shield-attacker evader) on STR14-SHIELD. |
| **H-SAFE holds, H1 NO SHIFT** | The shield is free for the learned pursuer at this margin — a result that contradicts both Step 6's scripted shift and T3's prediction on a learned policy; report with the cost table (an unshifted capture rate at high intervention is itself the finding). |
| **H-SAFE holds, H1 REVERSED** | The shield helps the learned pursuer; report as a refutation of T3's direction on this policy class, with the trajectories that explain it (visual gate). |
| **H3-lite RECOVERED** | Shield-in-training closes the deployment gap on the structured learner: Step 10's remedy claim has its first evidence; STRSH's best seed becomes the shielded artefact for Step 9. |
| **H3-lite NOT RECOVERED** | Deployment loss is not recoverable by training under the shield alone at this budget; Step 10 needs the feasibility-constraint / margin-shaping components. |
| **H-SAFE fails** | Stop. Census written. The learned nominal found a state the v2 construction does not cover; no further round without a user ruling. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is
seen. δ = 0.05, N, streams, seeds 16–18, the gate rule and the ≥ 2/3 rule are fixed here.

## 6. Visual gate

`step_08_shield.gif`: STR14 unshielded vs shielded on the same Greedy-1.2 episode, pursuer FOV
overlay and `ShieldOverlay`, one episode where the shield changes the outcome and one where it
does not; signed in the worklog.

## 7. What this round does NOT do

No learned evader (Step 8-alt by design), no NashConv, no shield on the evader, no attacker
(Step 9), no feasibility-constraint or margin-shaping remedy (Step 10), no change to STR14.

## 8. Artefacts

`results/derisk/s4_cells.json` (all 21 deployment cells: paired logs, `summarize_shield`
blocks, gate results), `s4_arm_STRSH{16,17,18}.json`, `s4_probe_STRSH*.json`,
`ckpts_STRSH*/`, `s4_round.{json,png}` (shift per opponent with CIs, cost table, STRSH
curves), `step_08_shield.gif`. Signed in the worklog.
