# S7 — Step 10, second iteration: a shielded pursuer trained against the structured attacker (2026-09-16, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
Governing: **D113** (S5: shield-aware features recover the shield's loss; artefact RC19), **D115** (S6b: the
structured evader — hug law within 4 m, any positioning beyond — pins STR14-SHIELD in 0.65–0.68 of episodes
rewarded or not; against RC19-SHIELD an unrewarded one manages 0.38–0.54 and a learned one 0.58–0.61; both beat
the scripted probes; zero violations throughout).

## 1. Why this round exists

RC19 was trained against `GreedyEvader` alone. The structured attacker defeats it because it never saw an
opponent that keeps solid geometry between them. The natural arms-race step is to train the shield-aware
pursuer against an opponent FAMILY that includes structured attackers, then measure (i) whether it resists
attackers it has never seen and (ii) whether fresh attackers trained against it recover their success.

**H3-2 (resistance):** a shield-aware pursuer trained against the mixture has a LOWER deadlock-episode rate
than RC19 against held-out learned attackers (the S6b CTL-D2 attackers, trained against RC19), seed-paired
on a fresh gate, on ≥ 2 of 3 held-out attackers, for ≥ 2 of 3 pursuer seeds — without losing capture
against `GreedyEvader` (guard, reported).
**H-RACE (stage 2, reported):** fresh attackers trained against the S7 artefact reach a lower deadlock rate
than S6b's attackers reached against RC19 (0.58–0.61).
**H-SAFE** (kill criterion): 0 pursuer contact steps everywhere (training, probes, gates, comparators,
stage 2).

## 2. Components

- `pe2.evade.StructuredEvaderPolicy(model, cfg, hold=10, d_pos=D_POS)` — the structured evader as an
  OPPONENT object for `SingleAgentWrapper(role="pursuer")`: `reset()`, `act(obs_evader, env) -> (v, ω)`:
  the `ObstacleHuggingEvader` law (full state) whenever the pursuer is within `d_pos`, else the model's
  decision on `full_state_observation` held `hold` env steps (ending early when the law applies) — exactly
  `EvadeMDP`'s semantics; the executed command takes the same float32 Box round-trip the learner seat
  applies. **Validity check V1-E:** inside `SingleAgentWrapper(role="pursuer")` driven by a frozen
  `StructuredPursuerPolicy` (D1), it must reproduce `EvadeMDP(D1).play_episode(model)` episode-for-episode
  (outcome, steps, pursuer stall count, longest stall run) on 200 seeds; a mismatch halts the round.
- `pe2.evade.MixtureOpponentFactory(members, seed)` — a factory returning, at each reset, a fresh opponent
  drawn uniformly (seeded RNG) from `members` (each a factory). Training only.
- Runner `scripts/derisk_s7_remedy2.py`.

## 3. Protocol — stage 1 (the pursuer)

- Learner: **exactly S5's RC configuration** — `SearchMDP(cfg, opponent, seed0 = seed × 1000, hold = HOLD,
  r_acq = 0, shield = CBFShield(cfg), shield_obs = True, stall_penalty = 1e-3)`, PPO from scratch, γ 0.99,
  SB3 defaults, 10 M env steps, no selection — with the training opponent replaced by the **mixture**:
  `GreedyEvader`, `ObstacleHuggingEvader`, the structured `SplitHugEvader`, and the three S6b attackers
  trained against STR14-SHIELD (`CTL-D125/126/127`, as `StructuredEvaderPolicy`), uniform per episode.
  Seeds **28, 29, 30**; probes every 1 M (N = 200) **vs `GreedyEvader`, shielded**, on **1 160 000+** (as
  S5, for comparability); `OMP_NUM_THREADS=2`.
- **Gate stream 1 170 000+** (N = 500), shielded, paired within opponent. Opponents: the **held-out
  attackers** `CTL-D225/226/227` (structured, trained against RC19 — never seen by the S7 pursuer),
  `GreedyEvader`, `ObstacleHuggingEvader`. **Reference RC19** is measured on the same stream against the same
  opponents BEFORE any arm (`s7_start.json`); an arm refuses to run without it.
- Gate readouts per opponent: pursuer deadlock-episode indicator, capture indicator, stall / intervention
  rates, violations.

## 4. Verdicts — stage 1 (fixed now)

- Deadlock test = `paired_indicator_test(rc19_ind, s7_ind)` (RC19 deadlocks MORE than the S7 pursuer,
  p < 0.05, b > c). Capture test = `gate_pass`-style McNemar on the capture indicator.
- **H3-2(seed) PASSES** iff the deadlock test passes on ≥ 2 of the 3 held-out attackers.
  **H3-2 PASSES** iff ≥ 2 of 3 seeds pass. Missing seed ⇒ `INCOMPLETE`.
- **Capture guard (reported):** vs `GreedyEvader`, McNemar between S7 and RC19 capture; a passing seed whose
  capture is significantly BELOW RC19's is labelled `PASS (robust but weaker)`.
- **Artefact:** the passing seed with the lowest mean deadlock rate over the three held-out attackers; ties →
  lowest seed. Written as `artefact`.
- H-SAFE fails ⇒ stop.

## 5. Protocol and verdict — stage 2 (the race)

Run only if H3-2 passes. Three fresh structured attackers (S6b protocol: `EvadeMDP`, CTL only, terminal
reward, seeds **31, 32, 33**, probes **1 180 000+**, 10 M env steps) trained against the **S7 artefact**
(shielded, frozen). Gate on **1 190 000+** (N = 500): each attacker's pursuer deadlock-episode rate with
Wilson CI, plus escape rate. **H-RACE (reported):** the pooled stage-2 deadlock rate's Wilson upper bound is
below **0.58** (the lowest S6b CTL-D2 rate against RC19). Also reported: the S7 artefact against the S6b
attackers (from the stage-1 gate) for the "before" number.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H3-2 PASSES, H-RACE holds** | Training against the structured family closes most of the attacker's gain: the arms race converges toward the defender — R3's strongest form. |
| **H3-2 PASSES, H-RACE fails** | Robust to unseen attackers of the old kind, but fresh attackers adapt: an open arms race, reported as such (the honest R2/R3 statement). |
| **H3-2 FAILS** | Family training does not buy resistance at this budget; the structured attack is a property of the shield's geometry, not of the pursuer's policy. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is seen.
The mixture, seeds 28–33, the streams, the tests and the ≥ 2/3 rules are fixed here.

## 7. Artefacts

`results/derisk/s7_preflight.json`, `s7_start.json`, `s7_arm_RCM{28,29,30}.json`, `s7_probe_*`,
`ckpts_RCM*/`, `s7_round.{json,png}`; stage 2: `s7_arm_ATK2-{31,32,33}.json`, `ckpts_ATK2-*/`,
`s7_race.json`. Signed in the worklog.
