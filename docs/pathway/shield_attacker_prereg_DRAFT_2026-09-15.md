# S6 — Step 9 shield-attacker (DRAFT for review; NOT pre-registered until S5 is read)

Governing: D112 (S4), S5's outcome (the remedied pursuer, if any). Plan v3 Step 9 / claim R2 /
hypothesis H2: *a learning opponent can (or provably cannot) weaponise the shield — drive it into
infeasibility, deadlock or high intervention — and by how much.*

## 1. The attacker

The project has never had a learned evader (D79: under a field of view the evader is non-Markov).
The pursuer's structure transfers: **`EvadeMDP`** fixes the evader's flee law when the pursuer is
visible (`FovLimitedEvader`'s visible branch and its stale dead-reckoning branch, verbatim) and
lets RL learn only **positioning** when the law has nothing to act on — where to go, and where
to wait, while unseen. Decision held 10 env steps, terminal reward +1 for a timeout, −1 for a
capture, PPO from scratch, 10 M env steps, three seeds.

- **ATTACK arm:** reward += β per env step on which the *pursuer's* shield reports an
  intervention, and += 2β per stalled step (β = 1e-3, the S5 scale). The evader's observation
  is its own 11-dim FOV-limited observation — it cannot see the pursuer's shield; it can only
  learn that certain positions make the pursuer stall.
- **CONTROL arm:** identical learner, no bonus (a plain learned evader).
- **Scripted probe:** `ObstacleHuggingEvader` (S4's HUG cell: pursuer deadlock-episode rate 0.61
  vs 0.54 against GreedyEvader).

## 2. The defenders (frozen, shielded, δ = 0)

`STR14-SHIELD` and, if S5 passes, the S5 artefact. Both arms of the attacker train and are gated
against each defender separately (2 defenders × 2 arms × 3 seeds = 12 training processes).

## 3. Readout and H2

Paired gate on a fresh stream (N = 500): the **pursuer's** deadlock-episode indicator,
infeasibility rate, backup-tier rate and intervention rate under ATTACK vs CONTROL vs the scripted
probe; plus the evader's own escape rate. **H2 PASSES** iff ATTACK raises the pursuer's
deadlock-episode rate above CONTROL's (paired McNemar on the indicator, p < 0.05, more discordant
pairs in the attack's favour) on ≥ 2/3 seeds against at least one defender, and the excess over
the scripted probe is reported. **Intelligibility gate (visual):** the attack must be visible in
`step_09_attack.gif` — which corner, which bait — or it is reported as statistically present
but unexplained.

## 4. What it settles

R2 either way: a measured, seeded exploitation of a hard shield by a learned opponent, or a
measured failure to exploit it. Against the S5 artefact it also gives R3's "remedy reduces attacker
gain" number. Open design question for review: whether the ATTACK bonus should also reward
pursuer *infeasibility* (backup tier) steps, which at δ = 0 the v2 construction makes unreachable
after a valid reset — the honest bonus is intervention + stall, as written.
