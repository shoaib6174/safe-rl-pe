# S24.1 — a learned pursuer selector against the scripted tactical evader (ratio 1.2)

**Status: PRE-REGISTERED. Committed BEFORE any round number exists.** S24.0 (D145–D149) is closed; its
measured v0b table is quoted below as the baseline, not as an outcome. Governing:
`tactics_plan_2026-10-13.md` (+ all rulings), `s24_orchestration_2026-10-13.md`, D131 (recovery shield),
D142/D144 (S19/S23: the best scripted laws), D83.

## 1. Claim

**Hypothesis T.** A selector that chooses among the pursuer's five tactics every ≤ 40 steps, with memory
(dead-reckoning estimate + age, own tactic history, the opponent's observable motion features, visibility,
geometry, speed ratio), can be learned by discrete PPO as a semi-MDP — the regime that worked in S1–S3 —
and beats the best fixed law under the shield. Its gain comes from *choosing*: its tactic profile differs
by opponent, and a selector denied the memory inputs does not match it.

## 2. What is frozen (one variable: the selector)

Everything from S24.0 after D149: the tactic library (`e5e16b4`), `LastSeenBelief` memory, feature-based
v0 rule tables (`231a362` + fix round), `RecoveryShieldV2`, coverage `SWEEP`, hold 10, T = 40 cap,
`EnvConfig()` at ratio **1.2**. The evader is the **scripted v0 tactical evader** throughout training and at
the gate; the learned selector never sees a learned evader in this round.

**Baseline, measured in S24.0 (v0b, 100 episodes per ratio, seeds 2 290 000+):** (`results/derisk/s24_tactic_game_v0b.json`, D150)

  | ratio | capture | mean steps | violations | evader top-3 | pursuer top-3 |
  |---|---|---|---|---|---|
  | 0.8 | 1.000 | 217.8 | 0 | HOLD-HIDDEN .32, HOP .32, ORBIT .25 | SWEEP .34, LEAD .34, INTERCEPT .26 |
  | 1.0 | 0.680 | 570.6 | 0 | ORBIT .67, HOLD-HIDDEN .15, HOP .14 | LEAD .69, SWEEP .18, INTERCEPT .11 |
  | 1.2 | 0.680 | 660.6 | 0 | ORBIT .52, HOLD-HIDDEN .27, HOP .16 | SWEEP .36, LEAD .35, INTERCEPT .16 |

  `AMBUSH` appears in no top-3; D150's watched episode 2 290 001 shows the predictable hop route it would punish.
  **Pre-registered expectation (not a bar):** if the learned selector chooses differently, `AMBUSH` against
  `TACTIC-V0`'s alternation is where it should show.

## 3. The learner

- Action space: the five pursuer tactics, **action-masked to `applicable()`** (D-ruling from the Step-1
  review: an inapplicable pick is impossible; the mask is logged per decision).
- Observation (all observable, speed-agnostic except the ratio itself, which is given): visibility flag;
  distance/bearing to the evader when visible; dead-reckoning point + age; own pose; nearest two obstacles
  (distance, bearing, radius); nearest wall distance; the recogniser's `features(t)` dict (`valid` flag
  included); one-hot of own current tactic + time in it; one-hot of own last 3 tactics; `speed_ratio`.
- Decision points: the current tactic's `DONE` or T = 40 steps; reward = the env's terminal ±1 summed over
  the span (sparse, as `SearchMDP`; no shaping — D138's lesson); γ = 0.99 per decision.
- PPO (SB3) with the S17 BASE hyper-parameters apart from the discrete head; **3 seeds (80, 81, 82)**;
  budget **3 M env steps** each; probe every 250 k on `2 350 000+` (N = 200, scripted evader only).
- **Negative control `NOMEM`:** identical, with the memory block (dead-reckoning, own history, opponent
  features) zeroed. 3 seeds. Pre-registered prediction: it does not match the full selector.

## 4. Gate and hypotheses — directions fixed now

Gate stream **2 400 000+**, N = **500** paired, opponents: `GREEDY`, `HUG`, and **`TACTIC-V0`** (the
scripted tactical evader; the training opponent, in-distribution and said so). Comparators on the same
seeds: **`LEAD-1.0`** (S23's best law under the shield) and **`RULES-V0`** (the scripted pursuer selector).

- **H-TAC-1 (per seed; ≥ 2/3 seeds):** shielded capture, learned > `LEAD-1.0`, `paired_indicator_test`
  p < 0.05, on ≥ 2/3 opponents. **H-TAC-1b (reported, not gated):** learned vs `RULES-V0`.
- **H-TAC-2 (mechanism):** the learned selector's tactic-usage distribution differs across the three
  opponents (χ² over the 5 tactics, p < 0.01, on ≥ 2/3 seeds) AND differs from `RULES-V0`'s profile on the
  same opponent (≥ 2/3 seeds). A selector that passes H-TAC-1 by always picking `LEAD` has not chosen.
- **H-TAC-3 (memory matters):** `NOMEM` capture < learned on ≥ 2/3 seeds against `TACTIC-V0` (p < 0.05).
- **H-SAFE:** 0 contact steps, every cell, training and gate.

**Refutations.** H-TAC-1 fails on all seeds → the discrete semi-MDP selector does not beat the best law;
report, and S24.2 is not licensed. H-TAC-1 passes but H-TAC-2 fails → the gain is a single-tactic default
(probably `LEAD`), reported as such; the selector is not the contribution. H-TAC-3 fails → memory is inert
at this ratio; the speed sweep is still run (memory may matter where the evader is slower) but with that
stated.

## 5. Construct gate — before any capture rate is read
- **CG-T1 (the mask holds):** 0 inapplicable picks over all training decisions (logged).
- **CG-T2 (it switches):** per seed, the learned selector's mean switches per episode on the gate ≥ 2 (a
  selector that never switches is a law, and is reported as one).
- **CG-T3 (behavioural):** 3 unselected gate episodes per seed vs `TACTIC-V0` and 3 vs `GREEDY`, frames
  labelled `EVADER: … | PURSUER: …` with the belief marker; watched before the table for: the pursuer
  choosing `COUNTER-ORBIT`/`AMBUSH` where a human would, pins (there should be none under the recovery
  shield), and whether `NOMEM` visibly behaves like a fixed law.

## 6. Stop rule and artefacts
As S23: nothing re-run after a number is seen; amendments only before the number they touch; UNDETERMINED
on crash; records measurable only at N = 500 on seed0 2 400 000; duplicates reject both; `--cells` prints no
rate; captions carry no rate; frozen worktree; gated launch wrapper. Artefacts `results/derisk/s24_1_*`.
