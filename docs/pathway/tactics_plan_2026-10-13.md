# Plan: the tactic game — hierarchical, memory-bearing, speed-agnostic agents (S24)

Written 2026-10-13 from the user's reading of the S23 episodes: the evader *already* plays a repertoire
(circle the obstacle, break, hide behind the next one); the pursuer only ever follows. Move the game up
one level: both agents own a **named library of tactics**, a **selector** that switches between them, and
**memory** — of the map, of the tactics they have used and are using, and of what the opponent has been
doing. Speed ratio is a parameter of the world and an input to the agents, never an assumption of a
tactic.

## 0. Why this is the right shape for THIS project

Flat continuous-action RL collapsed here every time (D108). The one learner that ever worked was the
structured one: a fixed intercept law with PPO choosing search decisions every ~20 steps (S1–S3, 5/5
seeds). A selector over ~5 tactics deciding every few seconds is that same object — a tiny discrete
semi-MDP — now with memory and for both roles. Every tactic is a scripted primitive we largely have
(`OrbitSearch`, the planner, `LEAD`, the hug law, Greedy flee, last-seen). The learning is in the
switching, and the switching is watchable: the frame label reads *EVADER: HOP→B / PURSUER: COUNTER-ORBIT*.

## 1. Architecture (both roles, symmetric)

**Tactic** = a closed-loop scripted controller with a name, preconditions, `act(obs, memory) → action`
and a termination predicate (`DONE` when its goal is reached or its precondition fails). Tactics must not
read the speed ratio; they are written to work at any ratio and to *fail honestly* where they cannot.

| role | tactic | what it does | primitive it reuses |
|---|---|---|---|
| evader | `FLEE` | radial flight from the pursuer | `GreedyEvader` |
| evader | `ORBIT(k)` | keep obstacle k between itself and the pursuer | hug law |
| evader | `HOP(k→j)` | plan to the far side of obstacle j so line of sight breaks | planner (A\*) on the known map |
| evader | `WALL-SLIDE` | slide along the nearest wall away from the pursuer | hug law's wall mode |
| evader | `HOLD-HIDDEN` | stop while out of sight (costs nothing; needs memory) | — |
| pursuer | `INTERCEPT` | plan to the evader's position | `PLAN` (S19) |
| pursuer | `LEAD` | plan to the evader's position + 1 s of velocity | `LEAD-1.0` (S23, the best measured) |
| pursuer | `COUNTER-ORBIT(k)` | go the *other* way round the obstacle the evader is orbiting | planner, mirrored goal |
| pursuer | `AMBUSH(gap)` | go to the gap between the evader's obstacle and the nearest other, and wait | planner + hold |
| pursuer | `SWEEP` | belief-driven search when the evader is out of sight | `OrbitSearch` now; belief-driven once the belief exists |

**Memory** (a small dataclass carried across steps, reset per episode):
- the map (static, known) and the arena;
- a **belief** over the opponent's position when out of sight — occlusion-aware particle filter (the S20
  design, never built because S19's gate withheld it; it is needed here for `SWEEP`, `HOLD-HIDDEN` and
  `AMBUSH`), validated against dead-reckoning (its median occlusion error must be lower);
- **own tactic history**: the last N (tactic, duration, outcome) and the current tactic + time in it;
- **opponent tactic history** via a **recogniser** that labels the opponent's last ~2 s of motion as one
  of the library's tactics. In self-play the opponent's true tactic is KNOWN, so the recogniser's accuracy
  is directly measurable — it is a construct and gets its own gate (≥ 0.8 accuracy on held-out episodes,
  confusion matrix reported) before any selector consumes it.

**Selector**: chooses a tactic at decision points (the current tactic's `DONE`, or every T = 40 steps).
- **v0, scripted**: a readable rule table (e.g. evader: pursuer within 3 m and an obstacle within 2 m →
  `ORBIT`; pursuer recognised as `COUNTER-ORBIT` → `HOP` to the next obstacle; out of sight → `HOLD-HIDDEN`).
  Exists so the tactic game can be WATCHED before anything is learned.
- **v1, learned**: discrete PPO over the library (≤ 5 actions), observation = [belief summary, own
  history (one-hot of last 3 tactics + time in current), recognised opponent history, **speed ratio**,
  distance/bearing/visibility, nearest-obstacle features]. A semi-MDP exactly like `SearchMDP`
  (`hold`-style spans), which is the regime our own results say is learnable.

**Safety**: the pursuer's executed command always passes through `CBFShield` (H-SAFE every round).
**Speed**: `evader_speed_ratio ∈ {0.8, 1.0, 1.2}` is an `EnvConfig` field and an observation; round 1
runs at **1.2** (every baseline exists there); round 4 sweeps.

## 2. Rounds — each pre-registered, construct-gated, watched before read

**S24.0 — the library, the memory, the recogniser, scripted selectors (no learning, no hypothesis).**
Construct gates: each tactic does what its name says, measured (e.g. `ORBIT` keeps the obstacle between
the agents on ≥ 90 % of its steps; `HOP` breaks LOS within its own duration; `AMBUSH` reaches the gap
before the evader on the cases where that is geometrically possible); belief beats dead-reckoning;
recogniser ≥ 0.8 on held-out self-play with the confusion matrix shown; the scripted v0 game watched
(12 GIFs with tactic labels, seeds by rule). Output: a tactic game that visibly plays.

**S24.1 — learned pursuer selector vs the scripted tactical evader, ratio 1.2.** H-TAC-1: shielded
capture > the best S23 law (`LEAD-1.0`) on ≥ 2/3 of {GREEDY, HUG, the scripted tactical evader}, p < 0.05
paired. H-TAC-2 (mechanism): the learned selector's tactic-usage profile differs by opponent (χ², and
reported per opponent) — it is choosing, not defaulting. Negative control: a selector with the memory
inputs zeroed must not match it. H-SAFE.

**S24.2 — learned evader selector vs the S24.1 pursuer, then one more alternation.** The arms race at
the tactic level, in a space of ≤ 5 × 5 tactics where it is legible. Report the empirical **tactic
meta-game** (S14's LP tooling): which tactics beat which, whether a pure or a cyclic equilibrium exists.
Kill: if either role's selector collapses to one tactic against every opponent, the library is too
narrow and the round says so.

**S24.3 — speed sweep {0.8, 1.0, 1.2}.** (i) zero-shot: the 1.2-trained selectors at the other ratios;
(ii) per-ratio training. H-SPEED: tactic profiles shift with the ratio in the pre-stated direction
(slower evader → more `ORBIT`/`HOLD-HIDDEN`, faster → more `FLEE`/`HOP`; pursuer the mirror). This is the
speed-agnostic claim made falsifiable.

## 3. Honest risks, stated now
- The library is hand-written: its bias is the price of legibility, and the kill rule in S24.2 is the
  check that it is wide enough.
- The recogniser is a construct; it is gated before use, and the selector's inputs are ablated (control).
- At 1.2 a hopping evader may be nearly uncatchable; capture may then come from its switching errors.
  Arena size and obstacle count are design parameters, not constants, and S24.0 reports the scripted
  capture rate at each ratio before anything is trained.
- Two learned selectors can cycle (Paper 1's lock-in, D37). The tiny discrete space, a population of past
  selectors as opponents, and the measured meta-game are the mitigation — and a measured cycle is itself
  a result.

## 4. Discipline and budget
Same ledger discipline as S23 (`s23_orchestration_2026-10-12.md` §Loop guards): one step at a time,
≤ 2 fix rounds, reviews on code that produces numbers, CG before reading, GIFs labelled and watched
first, pre-registration before any round number. **Proposed budget: 4.0 M tokens** (S24.0 ≈ 1.2 M,
S24.1 ≈ 1.2 M, S24.2 ≈ 1.0 M, S24.3 ≈ 0.6 M); hard stop at 3.6 M.

## 5. First action (now)
S24.0 Step 1: the tactic library for both roles as scripted controllers with termination predicates and
behavioural tests — no selector, no learning, no numbers that gate anything.

## S24.0 Step 1 rulings (2026-10-13, commit `3b28ca9`)
- **COUNTER-ORBIT's briefed bar was mis-specified.** "Clear LOS within 1.5 m" asked a flank to also close.
  Measured: direction committed and held 50/50 on the first segment, LOS cleared on 20/50 (at a median
  6.9 m), within 1.5 m on 2/50 — recorded as an honest xfail. The tactic's job is the flank; closing is
  `INTERCEPT`'s, reached by the selector's hand-off. No new bar is written after the number; S24.1's
  pre-registration states that COUNTER-ORBIT contributes only through the hand-off and reports its use.
- **`WALL_HUG_BAND` in `pe2/scripted.py` is NOT changed.** `ObstacleHuggingEvader` is a frozen gate
  opponent (S17/S19/S23); altering it would silently change three rounds' opponents. `WallSlide` carries
  its own band (0.10 m past the inset) inside `pe2/tactics.py`.
- ORBIT's keeping rate is measured against the S19 `PLAN` chaser; the chaser is named in the test so it
  cannot be swapped silently. Speed-agnostic assertion passed at 0.8 / 1.0 / 1.2.
- **For S24.1's pre-registration (from review):** neither adapter calls `applicable()` before running a tactic; a learned selector must be constrained (action mask) to applicable tactics, and an inapplicable pick is a logged event, never a silent `None`.
