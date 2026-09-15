# S6 — Step 9 shield-attacker: can a learning opponent weaponise the shield? (2026-09-15, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any arm of this round exists.**
(Supersedes the DRAFT of the same date after the user-approved review: full-state attacker, hugging
law fixed at close range, stall-only bonus, implicit-exploitation reading, zero-reward harness control.)
Governing decisions: D83, D97/D89 (shield v2, metrics), **D112** (S4: shielded STR14 deadlocks in 54 %
of episodes vs GreedyEvader, 61 % vs `ObstacleHuggingEvader`), **D113** (S5: RC19, the shield-aware
pursuer, 0.564 shielded, deadlock episodes 0.32). Plan v3 Step 9 / claim R2 / hypothesis H2.

## 1. Why this round exists

R2 asks whether a *learning* opponent can drive a hard-shielded pursuer into the shield's failure
modes — stalls, deadlocks, infeasibility — and by how much, against both the naive shielded pursuer
(STR14-SHIELD) and the remedied one (RC19-SHIELD). The scripted obstacle-hugger already raises
STR14-SHIELD's deadlock rate from 0.54 to 0.61 (S4). The question is whether learned positioning
does more, whether it does so even without being told to (a pinned pursuer cannot capture), and
whether the S5 remedy reduces the attacker's gain (R3's second half).

**The attacker is granted full state.** The safety guarantee must hold against any opponent, so the
worst case is the informative one; a partially-observed attacker is a hardware-realism variant for
later, not this round's question.

**Hypotheses.**
- **H2 (explicit weaponisation):** an evader rewarded per pursuer *stalled* step raises the
  pursuer's deadlock-episode rate above an identical learner without the bonus, against at least
  one defender, on ≥ 2 of 3 seeds.
- **H2-implicit (secondary):** the no-bonus learner alone raises the deadlock rate above the
  structure's own scripted positioning rule (a learned evader weaponises the shield merely by
  learning to escape).
- **R3-b (secondary):** the attacker's gain (ATTACK − CONTROL deadlock rate) is smaller against
  RC19-SHIELD than against STR14-SHIELD.
- **H-SAFE** (kill criterion): 0 pursuer contact steps in every arm's training, probes and gate.

## 2. Components

- `pe2.structured.StructuredPursuerPolicy(model, cfg, shield, hold, shield_obs)` — the frozen
  defender as an opponent object: `reset()`, `act(obs_pursuer, env) -> (v, ω)` (the executed,
  post-shield command; the intercept law when it applies, else the PPO decision held `hold` env
  steps, ending early when the law applies again — exactly `SearchMDP`'s semantics), `.last_sinfo`,
  `.metrics` (`ShieldMetrics`), `.record(contact: bool)` called by the environment after each step.
  **Validity check V1-P:** driving `SingleAgentWrapper(role="evader", opponent=StructuredPursuerPolicy(STR14, shield))`
  with a scripted evader must reproduce `SearchMDP(shield=CBFShield).play_episode(STR14)` against
  the same scripted evader **episode-for-episode** (outcome, steps, pursuer violation count, stall
  count) on 200 seeds. A mismatch halts the round.
- `pe2.obs.full_state_observation(own_pose, opponent_pose, obstacles, cfg)` — the 11-dim
  observation with the opponent's body-frame features always filled and the flag set to 1
  (equal to `build_observation` whenever the opponent is visible).
- `pe2.evade.EvadeMDP(cfg, defender_factory, seed0, hold=10, beta=0.0, zero_reward=False, d_pos=4.0)`:
  the evader's structured env. **Fixed law:** `ObstacleHuggingEvader.act` (obstacle hug → boundary
  hug → plain flee, full state) whenever the pursuer is within `d_pos = 4.0 m` (= `HUG_RANGE`, the
  distance at which the hug geometry engages). **Learner:** positioning whenever the pursuer is
  farther than `d_pos`, one decision held 10 env steps, ending early when the pursuer comes within
  `d_pos`. Learner observation = `full_state_observation` (11-dim). Reward per span = the evader's
  terminal (+1 timeout, −1 capture) summed over the span **+ β × (pursuer stalled steps in the
  span)**; `zero_reward=True` zeroes everything. `evaluate_evader(model, cfg, defender_factory, n,
  seed0, hold, d_pos) -> logs` carrying the pursuer's `summarize_shield` block and violation count
  per episode.
- `pe2.evade.SplitHugEvader` — the structure's own scripted positioning rule: the same `EvadeMDP`
  split with the fixed positioning "plain flee away from the pursuer" beyond `d_pos`. This is the
  scripted comparator for H2-implicit (identical structure, no learning).
- Runner `scripts/derisk_s6_attacker.py`.

**Amendment A1 (2026-09-15, before any comparator or arm; from Task 1's review).** V1-P compares two
routes that seat the scripted evader differently: as `SearchMDP`'s opponent its float64 command reaches the
env untouched, while inside `SingleAgentWrapper(role="evader")` it occupies the learner's seat and passes
through the float32 action Box (a ~4e-8 difference in the EVADER's ω that compounds over an episode;
pursuer commands were bit-identical). The reference side therefore uses `BoxRoundTripEvader` — `GreedyEvader`
with the same Box round-trip — so V1-P isolates the pursuer object. The compared tuple is outcome, steps,
pursuer violations, stall count, step count, intervention count and longest stall run. No threshold, seed,
stream, rule or budget changes.

## 3. Protocol

- Defenders (frozen, shielded δ = 0, pursuer role, `EnvConfig()` defaults, evader speed 1.2):
  **D1 = STR14-SHIELD** (`ckpts_STR14/STR14_10000000.zip`, hold 10, no shield features) and
  **D2 = RC19-SHIELD** (`ckpts_RC19/RC19_10000000.zip`, hold 10, `shield_obs=True`).
- Attacker learner: PPO from scratch on `EvadeMDP`, γ = 0.99, SB3 defaults, **10 M env steps**,
  probes every 1 M (N = 200) on **1 120 000+**, no selection — the final checkpoint is the arm's
  policy. Training seeds **22, 23, 24** (wrapper `seed0 = seed × 1000`). `OMP_NUM_THREADS=2`.
- **Gate stream 1 130 000+ (N = 500), paired within each defender** across every arm and every
  scripted comparator.
- **Bonus:** β = **1e-3 per pursuer stalled step** (executed speed ≤ 5 % of v_max while the
  nominal wanted to move — the shield's own `stalled` flag). Stalls only, not interventions:
  interventions are cheap and common (0.69 of steps for STR14-SHIELD), stalls are the exploit; at
  full pinning (≈ 500 stalled steps) the bonus is ≈ +0.5, below the terminal.
- **Scripted comparators, measured on the gate stream BEFORE any arm, per defender:**
  `SPLITHUG` (`SplitHugEvader`, the structure's own rule), `HUG7` (`ObstacleHuggingEvader`, the
  Step 6 probe with its 7 m threat range), `GREEDY` (`GreedyEvader`) — each with the pursuer's
  deadlock-episode indicator per episode, stall / intervention / backup rates, violation count,
  and the evader's escape rate. An arm refuses to run without `s6_start.json`.

## 4. The arms

| arm | defender | β | seeds | purpose |
|---|---|---|---|---|
| **ATK-D1**, **ATK-D2** | D1 / D2 | 1e-3 | 22, 23, 24 | **H2**: explicit weaponisation |
| **CTL-D1**, **CTL-D2** | D1 / D2 | 0 | 22, 23, 24 | the bonus's control; **H2-implicit** vs SPLITHUG |
| **EZ-D1**, **EZ-D2** | D1 / D2 | 0, all rewards zeroed | 22 | harness control: unrewarded positioning must not raise the deadlock rate above SPLITHUG |

Fourteen processes, ≈ 4 h concurrent (each step runs the shielded pursuer's QP).

## 5. Verdicts (fixed now)

Per arm × seed, on the gate: pursuer deadlock-episode rate, stall / intervention / backup /
infeasibility rates, `min_clearance` floor, violation steps; evader escape rate.

- **Deadlock test** between two evaders on the same defender: paired exact two-sided McNemar on the
  per-episode deadlock indicator, p < 0.05, with more discordant pairs in the first evader's favour.
- **H2(D) PASSES** iff ATK-D beats CTL-D on the deadlock test on ≥ 2 of 3 seeds (seed-paired:
  ATK seed s vs CTL seed s). **H2 overall** iff H2(D1) or H2(D2).
- **H2-implicit(D)** iff CTL-D beats SPLITHUG on ≥ 2 of 3 seeds (reported).
- **Beyond the best scripted probe (reported):** ATK-D vs HUG7 per seed.
- **R3-b (reported):** mean over seeds of (ATK − CTL deadlock rate) for D2 minus D1, with the
  three per-seed differences.
- **Harness:** EZ-D beating SPLITHUG on the deadlock test ⇒ H2(D) and H2-implicit(D)
  `UNDETERMINED (harness)`.
- **Escape trade-off (reported):** ATK's escape rate vs CTL's — an attacker that pins the pursuer at
  the cost of its own escapes is reported as such.
- Missing seed ⇒ `INCOMPLETE`. H-SAFE fails ⇒ stop, census, user ruling.
- **Intelligibility gate (visual):** `step_09_attack.gif` — the best ATK seed vs D1 on an episode
  with a deadlock, beside CTL on the same seed; the attack must be describable (which obstacle or
  wall, which bait) or the result is reported as statistically present but unexplained.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **H2 PASSES** | R2 measured: a learned opponent weaponises a hard shield; magnitude vs the scripted probe and the remedy's effect (R3-b) go to the paper. Figure 1 candidate. |
| **H2 FAILS, H2-implicit PASSES** | The bonus adds nothing but learning to escape already exploits the shield: R2 holds in its implicit form. |
| **Both fail, harness valid** | Learned positioning at this budget does not exceed the scripted probe; R2's honest statement is "no exploitation beyond the hugging geometry found". |
| **R3-b > 0** | The S5 remedy reduces the attacker's gain — R3's second half. |
| **H-SAFE fails** | Stop. |

**Stop rule.** Nothing is re-run, extended, retuned or re-seeded after any number of this round is
seen. β, `d_pos`, seeds 22–24, the streams, the deadlock test and the ≥ 2/3 rule are fixed here.

## 7. Not in scope

No partially-observed attacker, no evader shield, no self-play, no re-training of the defenders.

## 8. Artefacts

`results/derisk/s6_start.json` (per-defender comparator logs), `s6_arm_{ARM}.json`, `s6_probe_*`,
`ckpts_{ARM}/`, `s6_round.{json,png}`, `step_09_attack.gif`. Signed in the worklog.
