> # SUBMISSION WITHDRAWN — SUPERSEDED BY D125 (2026-09-19)
>
> The AAMAS 2027 submission planned here is withdrawn: the central claim it would have made is not supported
> (D125). The factual content below — deadlines, tracks, what still needs verifying — stays accurate and will
> be useful if a multiagent venue is targeted again. The fit assessment assumes the shielded-games thesis and
> does not apply to the rebuilt absorbing-loop thesis; see `docs/rebuild_plan_2026-09-19.md` §6 for the
> current venue thinking (IROS 2027 or RA-L).

---

# Venue assessment — AAMAS 2027 (2026-09-19)

Supersedes the ICRA-2027 planning docs, which belong to the dropped Paper 1 (D117).

## AAMAS 2027 — the facts I verified

From `https://warwick.ac.uk/fac/sci/dcs/aamas2027/` on 2026-09-19:

| item | value |
|---|---|
| Conference | 26th International Conference on Autonomous Agents and Multiagent Systems |
| Where / when | JW Marriott, Hanoi, Vietnam, 2027-05-03 to 2027-05-07 |
| Abstract deadline | **2026-10-01** |
| Full paper deadline | **2026-10-08** |
| Other tracks | Blue Sky Ideas, AAAI Fast Track, Workshops (2026-10-29), Tutorials (2026-12-10) |

**NOT verified — check before committing to a format:** page limit, ACM template, double-blind policy,
supplementary/appendix rules, topic areas. The overview page says each track has its own limits and points at
a per-track call; the main-track call URL I tried returned 404. Someone must open the Calls page and read the
main-track call.

## Fit

**Good, with a reframe.** The contribution is a two-player zero-sum game solved by iterated best response, with
an empirical exploitability gap as the headline metric and an arms race that reaches a defender-side fixed
point (D116, D120, D122). That is AAMAS's native vocabulary. The safety filter should be introduced as a
mechanism that restricts one player's action set state-dependently and thereby creates a strategic
vulnerability — a multiagent framing, and more novel at AAMAS than at a robotics venue where shields are
routine and the game is the unusual part.

**Reframe required (no new experiments):**
- Lead with the game, not the control theory. The CBF-QP is a mechanism, not a contribution.
- State the solution concept honestly: a computed best response at a fixed budget, not a proven equilibrium.
  Every deadlock-gap number is a lower bound on exploitability (already written that way in
  `docs/paper_exploitability_table.md`).
- The self-play motivation paragraph (why frozen best-response iteration and not simultaneous learning) carries
  more weight at AAMAS than elsewhere: cite the Pathway A evidence (26/34 lock-in, the ∀/∃ population
  asymmetry, seed-fragile single-opponent smoothing — `docs/reference/lessons_learned.md` §2).

**Expected objections and our answers:**
- *One environment.* Two filter families (D118, D121), three filter/geometry knobs, and the speed ratio (S13)
  are the breadth we have. Say so explicitly rather than hoping nobody asks.
- *No theory.* A formal statement of the pocket condition is the one thing that would materially help and is
  already listed as open in `docs/paper_mechanism_stall.md` §5.
- *Blinding.* The pre-registration story points at a named repository; an anonymised supplement is needed.

## Plan

1. Submit the abstract on 2026-10-01 (cheap, reversible).
2. Write against the AAMAS main-track format once its call is read.
3. Fallback if the draft is not strong by 2026-10-06: RA-L (rolling, no deadline) or IROS 2027 (~March).
4. S13 (speed ratio) runs in the background and does not gate the writing.
