# Retrospective: a year of work, and why no discovery came out of it

Written 2026-09-23 from `docs/decisions.md` (D1–D132) and `docs/reference/lessons_learned.md`.
This is a diagnosis, not an apology. The purpose is to change the next six months, not to grade the last twelve.

## 1. What we actually tried

**Phase A — self-play stabilisation (Mar–Jun 2026, D1–D48).** Goal: balanced 1v1 pursuit–evasion under
partial observability. Mechanisms tried: SAC over PPO, fixed entropy, freeze/unfreeze curricula, masking
curricula, FOV sensing design, forced role switching at 2 M and 3 M, EWC, visibility and search shaping,
population FSP, EMA opponents, asymmetric learning rates, a shared role-conditioned policy, and PBT with a
balance fitness. **Result: 26 of 34 seeds lock in; Pop-FSP 0/17 across five configs; EMA 1/7; A-EMA 0/3;
shared policy 0/3; PBT 0/3.** Root cause found at D37 and it was structural — the ∀/∃ quantifier asymmetry —
not a tuning problem. This became Paper 1, dropped at D117.

**Phase B — the pe2 rebuild (Aug 2026, D60–D97).** A safety-first codebase: seed contracts, reproducibility,
a CBF shield, a scenario suite, an evaluation harness, CI. Excellent engineering, no scientific claims.

**Phase C — the learner problem (Aug–Sep 2026, D68–D103).** Behaviour-cloning then RL: distance shaping,
then no shaping, a speed curriculum, a horizon curriculum, demonstration-buffer seeding, critic warm-up,
LayerNorm critics, fixed entropy. **Eight rounds, every one refuted** (D77, D79, D82, D100, D102). Concluded
at D102 that RL adds nothing here ("Route C is final"). **D108 then showed the entire chain had been
following a diverging critic** — Q values of +3 to +21 on a problem bounded in [−0.3, 1], with no action
dependence. Ten rounds of conclusions invalidated at a stroke.

**Phase D — structured learner and "shielded games" (Sep 2026, D109–D124).** Restricting the learner to
search while fixing the intercept law worked immediately (D111, 5/5 seeds). Then thirteen rounds building an
adversarial arms race on top of it: the shield's cost, a learned attacker, a remedy, mixture training, an
anchored fine-tune, a selection procedure, sweeps over filter knobs, a second filter family, a full game
matrix. **All of it re-scoped by D125.**

**Phase E — the rebuild (Sep 19–22, D125–D132).** Correct the diagnosis, measure the mechanism in three
parts, compare five remedies, find a remedy that works. Then discover the phenomenon was published nine days
earlier by someone else.

## 2. Why it did not produce a discovery

Six causes, in order of how much damage each did.

**1. We gated inference and never gated meaning.** Four separate times a carefully measured quantity did not
mean what we thought: the diverging critic (D108, ten rounds lost), the shield's "attack surface" (D125,
thirteen rounds lost), the "geometric pocket" (D126/D130), and my choice of immobility as a primary metric
when the remedy's own mechanism was to rotate (D129). **In every case the statistics were impeccable and the
construct was wrong.** Pre-registration, paired tests, negative controls and stop rules all discipline what
you may conclude *from* a number. Not one of them asks whether the number measures the thing you named.

**2. The apparatus improved while the question stood still.** Most of D60–D97 and a large share of every
round since is infrastructure and method: seed contracts, atomic records, frozen worktrees, provenance,
amendment logs. All of it is good and most of it was necessary. But a year of decision entries is mostly
*how we measure*, not *what we found*.

**3. We fought hard problems with new tools instead of changing the problem.** Self-play balance: eight
mechanisms against a structural asymmetry that D37 had already identified as structural. The learner: eight
rounds against a critic that was broken from the start. In both cases the diagnosis was available early and
we kept adding methods on top of it.

**4. Negatives accumulated and were never converted.** We hold an unusual stock of well-established negative
results — population self-play cannot balance this game, a memoryless clone cannot imitate a memory-driven
teacher, a stall penalty teaches the agent to stop asking to move, learned remedies raise task success without
clearing a liveness failure. Each was logged and passed over. **None was ever treated as the finding.**

**5. The feedback loop was days long; the decisive check took one viewing.** D64 mandated visual gates in
August. We built them, produced GIFs at most milestones, and still used summary statistics as the primary
check. The correction that reset the project came from a human watching a clip once and saying the pursuer
looks stuck.

**6. We were in a crowded field with no differentiating asset.** No hardware in the loop (confirmed available
at D87, never used), no proprietary data, a standard simulator. In an area this active, novelty is won by
speed or by an asset nobody else has. We had neither, and our loop was slow.

## 3. What we have that is genuinely unusual

Stated plainly, because the plan should be built on it rather than on hope.

- **An evidence apparatus that almost nobody runs**: pre-registered specs with amendments timestamped before
  the numbers, seed-paired gates, negative controls, frozen-commit execution, and roughly half a billion
  shielded environment steps with zero safety violations.
- **A documented error trail**: four self-corrections, each with the measurement that forced it. This is
  ordinarily invisible and is the raw material of a credible methods section.
- **A large stock of clean negatives** on questions other people are still asking.
- **Two live contradictions in the published literature** that we are positioned to resolve.

## 4. The rule that changes the next six months

**Nothing gets interpreted before its construct is validated, and the validation is behavioural.** Before any
round's numbers are read: watch the policy act on unselected seeds, and check that the primary metric cannot
be satisfied by the remedy's own mechanism or by doing nothing. That single gate would have caught D108,
D125, D126 and D129 — four of the five collapses — within a day each instead of within months.

The second rule follows from cause 4: **when a negative result is clean, ask whether it is the paper before
moving past it.**
