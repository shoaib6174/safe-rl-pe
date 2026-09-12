# F1 — the family evaluation: pre-registration (2026-09-12, S125)

**Status: PRE-REGISTERED. Written and committed BEFORE any measurement of this round.**
Governing decisions: D83 (pre-registration discipline), **D104** (the benchmark-validity
finding this round responds to), D100/D102/D103 (the learner rounds whose interpretation is
at stake).

## 1. Why this round exists

D104 established, by measurement, that the project's yardstick does not work:

| policy | capture rate vs `GreedyEvader`, N = 500 |
|---|---|
| GreedyPursuer (full state) | 0.426 |
| **randomly-initialised net, seed 4** (v ≈ 0.505, ω ≈ −0.078, a 6.5 m arc) | **0.312** |
| FovLimitedPursuer — the hand-written teacher | 0.234 |
| BC clone | 0.185–0.194 |
| **the same non-policy, turn sign reversed** (seed 3) | **0.028** |

A network at initialisation is not a policy: it emits the zero action, which maps to half
speed and an almost-straight line, with action variability ~2 % of the mean. **A lazy arc
outscores active hunting by 8 points, and reversing the arc costs 28.** The score therefore
does not order policies by pursuit competence, and every threshold in Tier 0, T0.4, T0.5 and
T1.0 — the 0.185 bar, the 0.137 floor, every improvement rule — is defined against it.

The mechanism is visible in `GreedyEvader.act`: it flees directly away from the pursuer, or
toward an obstacle's shadow when threatened. **It has no wall awareness.** It corners itself,
and a slow sweep collects it while active pursuit pushes it back toward open space.

**This does not rescue the learners** — every RL arm moved *down*, failing to find even the
degenerate exploit, so "RL did not work here" stands. **It does invalidate the reading** that
0.185 measured competence or that beating it would have measured learning.

**Hypothesis F:** a single-opponent score is the wrong measure for a game. Evaluated against a
*family* of evaders with a worst-case readout, the degenerate arc collapses and the ordering
by competence is restored.

## 2. The family (fixed now)

| evader | role in the family | status |
|---|---|---|
| `GreedyEvader` | the incumbent — kept so every prior number stays comparable | exists |
| `FovLimitedEvader` | partial observability; memory-driven flight (D79) | exists |
| `ObstacleHuggingEvader` | the Step 6 adversarial probe; exploits occlusion | exists |
| **`OpenSpaceEvader`** | **new — the one that punishes circling.** Full-obs, flees the pursuer while refusing headings that end in a wall or an obstacle, so it cannot corner itself | to build |

`OpenSpaceEvader` is specified before any result: sixteen evenly spaced candidate headings;
each projected forward 1.0 s at the evader's top speed; headings whose projection leaves the
arena or enters an obstacle are rejected; among the survivors it takes the one maximising
distance from the pursuer at the projected point, and steers toward it with the same
proportional gain the other scripted policies use. Deterministic, full-observation, no memory.

## 3. Protocol

- Role under test: pursuer. **N = 200 paired episodes per (policy, evader) pair.**
- Seed streams, one per evader so every policy meets each opponent on identical episodes:
  `GreedyEvader` **930000+**, `FovLimitedEvader` **940000+**, `ObstacleHuggingEvader`
  **950000+**, `OpenSpaceEvader` **960000+**. All fresh (≤ 890999 consumed; 900000+ is the
  test range, avoided).
- Environment at its shipped defaults (`EnvConfig()`, 1200 steps) — this round changes the
  *measurement*, nothing else.
- **Primary readout: the worst-case capture rate across the four evaders.** Also reported per
  opponent, and as the mean across the family, never as a substitute for the worst case.
- Wilson intervals on every cell. No training of any kind occurs in this round.

## 4. The policy set (fixed now)

| policy | why it is in |
|---|---|
| `GreedyPursuer` (full state) | the information-unlimited reference |
| `FovLimitedPursuer` | the hand-written teacher; the BC source |
| BC clone `sac_bc_pursuer.zip` | the bar every learner round was judged against |
| **random-init seed 4** | **the validity check** — the 0.312 arc |
| random-init seed 3 | its mirror, the 0.028 arc |
| T0.5 `RE` final checkpoint | the best-behaved RL arm (PRESERVED, held ≥ 0.137 for 1M steps) |
| `ConstantPolicy` stand-still | the floor |

## 5. The validity check — stated in advance, and falsifiable

The family metric is only worth adopting if it demonstrably fixes what D104 broke. **Both of
the following must hold:**

1. **The arc collapses.** Random-init seed 4's worst-case across the family is strictly below
   `FovLimitedPursuer`'s worst-case.
2. **The ordering is restored.** On worst-case, `GreedyPursuer` ≥ `FovLimitedPursuer` ≥ BC
   clone, and both arcs and stand-still fall below the BC clone.

**If either fails, the family metric does not fix the problem.** Report that plainly, do not
adjust the family membership to make it pass, and stop — a third yardstick would need its own
pre-registration and a different idea behind it.

## 6. What each outcome fixes

| result | direction |
|---|---|
| **Both checks pass** | The worst-case-over-family score becomes the project's yardstick. Route C's negative claim is **restated against it** before it goes in a paper, and the belief-augmented clone (§7) is judged on it rather than on the single-opponent number. |
| **Either check fails** | The family metric is not the fix. Record it, and stop: no further evaluation or learning work until a different measurement idea is pre-registered. |

## 7. What this round deliberately does NOT do

It does not train anything, and it does not touch the belief-augmented clone — the obvious
next step given D104's companion finding (the BC teacher acts on `self.last_seen`, hidden
state absent from the observation, so a memoryless clone is regressing a non-function).
**That work waits for a yardstick that works.** Building a better clone and measuring it with
a broken score would tell us nothing, and the ordering of these two steps is the whole point
of this round.

## 8. Artefacts

`results/derisk/f1_family.{json,png}` — the full policy × evader matrix with Wilson intervals,
the worst-case column, and the two validity checks evaluated explicitly. Figure: a grouped bar
chart, one group per policy, one bar per evader, with the worst-case marked. Signed in the
worklog. Estimated wall time ≈ 2–3 h (7 policies × 4 evaders × 200 episodes, no training).
