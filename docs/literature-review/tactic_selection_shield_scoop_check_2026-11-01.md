# Scoop check — hierarchical tactic selection under a certified filter, and the learned evader as minimax (2026-11-01)

Scope: the claims of D151/D152/D154 (learned pursuer tactic selector beats flat and structured RL under a CBF-QP
recovery shield; zero-shot speed transfer) and D156/D157 (learned evader tactic selector drives every pursuer to
0.09–0.22; the pursuer's alternation fails; the tactic meta-game has a pure saddle at value 0.138 set by the
filter's obstacle margin). Previous check: `adversarial_shield_deadlock_2026-09-19.md` (Yeke et al., USENIX
Security 25, stays the intro's near-scoop for the deadlock-as-objective line). Six web searches + four abstracts
/ one full text read on 2026-11-01. Not exhaustive; arXiv dates taken from the listings.

## Verdict: no scoop; two must-cite neighbours; one framing adjustment

**1. Wu, Yang, Oh, Nguyen, Hu — "Turning Safety into Competence: Minimally Exploitable Robot Policies via
Safety-Filtered RL" (S2C), arXiv 2609.27312, Sep 2026.** The closest paper. Two-player zero-sum "touchdown"
game between quadrupeds; a LEARNED robust safety filter (Robust Q-CBF from adversarial RL, MAGICS) embedded in
the environment during task training; exploitability = game value minus worst-case payoff, measured against the
strongest policy in a checkpoint pool (net win 0.11 vs S2C, 0.54–0.75 vs eight baselines); Theorem 1: perfect
filtering preserves non-exploitability when all players commit to safe manoeuvres. **Delta:** no obstacles, no
pursuit–evasion, no hierarchy, continuous joint targets; the filter is learned, not a certified CBF-QP; the paper
never studies an opponent exploiting the filter's margin, and never states or measures that the filter sets the
game's value. Our D156/D157 is the complementary finding: with a CERTIFIED model-based filter and a hand-written
tactic library, the evader's learned exploitation of the filter's obstacle margin IS the minimax strategy, and the
value (0.138) is a property of the filter, not of either learner. Cite in intro and related work; adopt their
exploitability definition for E.4's table (ours is the same quantity, value − worst-case payoff, over a 4×4 pool).

**2. Ahmad et al. — HMARL-CBF, arXiv 2507.14850 (NeurIPS 2025 per the search listing).** High-level policy over
skills, low-level skills executed under CBFs; cooperative road-network navigation. The architecture is the nearest
relative of our selector (discrete skill choice above a CBF-guarded primitive). **Delta:** cooperative, no
adversary, no measured cost of the filter, skills' provenance unstated. Cite as the hierarchical-CBF precedent;
our contribution relative to it is the adversarial two-sided game and the construct discipline (bit-identical
replays, pre-registered gates).

**3. Acharya et al. — "Engagement-Aware Agentic Pursuit-Evasion", arXiv 2607.10986, Jul/Sep 2026.** LLM planners
assign discrete tactical roles per robot; MPC + discrete-time CBF filter executes them; both teams LLM-driven.
**Delta:** roles chosen by an LLM, not learned by RL against the filter; no capture-rate or value measurement; no
exploitation analysis. Cite as "tactic switching above a CBF filter" precedent — it makes the discrete-tactic
layer a recognised design, which helps our framing.

**4. Wang, Zhang, Xu, Wang, Guay — safe robust RL in PE differential games, arXiv 2507.19516, TNNLS 2026.**
Zero-sum PE with obstacles and disturbances; CBF as Stackelberg leader over an SMC follower; min-max learned
online. No tactic layer, no cost-of-safety measurement. Cite in the CBF-PE related work.

**5. Bao et al. — "Not an Accident: Exploiting Safety Filter Inadequacy in Humanoid Robots", arXiv 2610.31923,
Oct 2026 (cs.CR).** Goal-insertion attacks steer a humanoid into states its filter cannot rescue; attacker = user
inputs, not an embodied opponent; 3× the attacks of a baseline, real G1 collisions. Related to the filter-failure
line (D112/D125), orthogonal to the game-theoretic claim. Cite beside Yeke et al. as the second attack-on-filters
paper; our delta remains embodied / certified filter / learned zero-sum opponent.

**6. Still unobtained:** *Control Engineering Practice* 2026 "Safe exploration for pursuit-evasion games based on
CBFs and RL" (S0967066126000961, 403 again; no arXiv version found). Its abstract claims CBF-filtered RL converges
to "conservative safe equilibria" rather than the true saddle point. That is the one statement in the field that
touches "the filter sets the value" — obtain through institutional access before submission and quote it exactly.

## Framing adjustment for the write-up
Lead with the two-sided result as **exploitability under a certified filter** (S2C's vocabulary) rather than
"arms race": our pursuer side is non-exploitable within its library (exploitability 0–0.05, D157) and the
evader's learned margin-orbit is the minimax strategy; the value is set by the filter's margin — which the planned
margin sweep (item 3 of the 2026-11-01 plan) would measure directly and S2C does not.

## Searches run (2026-11-01)
hierarchical RL tactic selection PE CBF shield; learned evader exploits safety filter; option selection scripted
behaviours shielded meta-game; safety-filter cost / game value; adversary exploits filter conservativeness;
shielded RL deadlock/livelock learned adversary. Secondary hits noted, not cited: SMARL probabilistic logic shields
for equilibrium selection (2411.04867); Gameplay Filters (2405.00846); CBF-RL (2510.14959); predictive shielding
MARL livelock (2609.07618); Zhou et al. visibility CBFs ICRA 25 (2411.01321).
