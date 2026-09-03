# Novelty Audit — Safe Adversarial Learning in Pursuit-Evasion (2026-09-03)

**Purpose**: Check the Pathway B claim space against literature published after the Feb 2026 review (`docs/literature-review/`). Web-search level only; the two closest neighbours (marked **READ FULL TEXT**) must be read before any claim is committed.

## Nearest neighbours found (post-Feb-2026 unless noted)

| # | Work | What it does | What it does NOT do | Threat to us |
|---|---|---|---|---|
| A | **Safe exploration for PE games via HOCBF + RL**, Control Engineering Practice, May 2026 (S0967066126000961) | Embeds HOCBF safety into the game formulation via an auxiliary system; claims convergence to the *original* saddle point, explicitly contrasting with projection/shielding methods that converge to "conservative safe equilibria" | Appears analytical/ADP-style, full observability, no deep self-play, no sensing constraints, no exploitability measurement | **HIGH — READ FULL TEXT.** Names the exact phenomenon we would measure (shield shifts the equilibrium). Becomes our theoretical hook and a baseline, not a blocker, if it has no deep-RL/PO experiments |
| B | **CBF-RL: Safety Filtering RL in Training with CBFs** — Yang, Werner, de Sa, Ames — **ICRA 2026** (arXiv 2510.14959) ✅verified | Shield-during-training vs shield-at-deploy, single-agent; filtered rollouts + minimal CBF policy term; proves continuous-time filters deploy in closed form on discrete rollouts; navigation ablations + **Unitree G1 humanoid hardware** (no runtime filter needed) | Single agent, no opponent, no game, no exploitability | **HIGH for Step 7** — our single-agent 3-arm study is already done. Reuse as baseline; novelty must be the adversarial setting |
| C | **Wang, Zhang, Xu, Wang, Guay — RL in PE differential game: safety, stability, robustness** — **IEEE TNNLS 2026** (arXiv 2507.19516; our [01]) ✅verified | Zero-sum PE, CBF (leader) + sliding-mode (follower) in a Stackelberg hierarchy, min-max, sufficient condition under conflicting constraints; **simulation only** | No PO, no deep self-play, no hardware, no exploitability | Medium — occupies "safe PE RL" naming space |
| D | **Safe MARL with adversarial strategy + CBF feasible set**, Neurocomputing, Mar 2026 (S0925231226008854) | Neural CBF trained treating non-target agents as worst-case adversaries; feasible-set Q-CBF | Collision safety among agents, not a strategic game objective; no PE, no PO | Medium — "adversary vs safety" framing exists; ours is strategic opponent exploiting an obstacle shield |
| E | **Gameplay Filters** (Hu, Fisac et al., CoRL 2024) | Safety filter synthesized by self-play vs virtual *disturbance* adversary | Adversary is uncertainty, not a strategic opponent | Low–Medium — framing overlap ("adversarial safety") |
| F | **Maximal robust CBF via adversarial RL** (arXiv 2604.13192, Apr 2026) | Isaacs-equation safety value as robust discrete-time Q-CBF | Disturbance game, not PE task game | Low — useful tool for a robust shield arm |
| G | **Capture, Shield, or Neutralize** (arXiv 2607.10986, Jul 2026) | Hierarchical minimax MPC with DCBF inner loop, multi-agent | Model-based, no learning | Low — cite as model-based counterpart |
| H | **Robust Shielding for Safe RL** (arXiv 2606.00270, Jun 2026); Melcer et al. shield decomposition (RLC 2024); ElSayed-Aly et al. (AAMAS 2021) | Discrete/LTL shields, robust MDPs, PO multi-agent | Not continuous dynamics, not PE | Low — related-work paragraph |
| I | **Xiao, Cassandras, Belta — learned feasibility constraints** (ECC 2023; N13) | Learns H(z)≥0 to keep HOCBF-QP feasible; infeasibility 0.081→0.002 | Single/cooperative agents; no adversary | Tool for the remedy arm; adversarial infeasibility remains open |
| J | **Gonultas & Isler**, IROS 2025 ([02]) | 1v1 car-like PE, sensor constraints, BiMDN, MADDPG, real F1TENTH | No hard safety, self-play fragile (our paper 1) | The system baseline; paper 1 is our published lever |

## What is still open (as far as search shows)

1. **Empirical measurement of the "conservative safe equilibrium"**: nobody has measured, with exploitability, how a projection shield shifts the learned equilibrium of a deep-RL adversarial game. (A) predicts it analytically; (B) studies shield-in-training only single-agent.
2. **Strategic shield exploitation**: an opponent that *learns* to drive the shielded agent into infeasible or high-intervention states. (D), (E), (F) treat adversaries as disturbances or collision partners, not as a task opponent weaponizing the obstacle shield.
3. **Shield-aware co-adaptation**: shield-during-training + learned feasibility constraints + margin shaping in *self-play* (combining B + I in the adversarial setting).
4. **Safety under partial observability in a game** (review gap G5) — shield on true state vs on belief state.
5. **Real-robot hard-safe deep-RL PE** (review gap G1) — still open; (J) has hardware without hard safety, (C)/(A) have safety without hardware.

## Verdict

The original six-element "first to combine" framing is now weak: (A) + (C) + (J) between them cover safe PE RL, PE with sensing on hardware, and saddle-point-preserving safety. The defensible, still-open contribution is **the effect of hard safety filtering on adversarial learning itself** (items 1–3), with items 4–5 as multipliers.

**Mandatory before committing**: read (A) and (B) in full — see `docs/reference/reading_list_shielded_games.md` Tier 0.

## Update 2026-09-03 (S120, after verification pass)

- (B) is **ICRA 2026** with humanoid hardware — stronger prior art than first logged. Our Step 7 is explicitly a replication of it; the contribution must live in the adversarial setting (R1–R3) and the PE hardware (R4).
- (C) is **IEEE TNNLS 2026**, simulation only — hardware gap G1 stays open, which is what R4 fills.
- (A) and (D) are ScienceDirect-paywalled and returned HTTP 403; niro-2 (the download machine) is offline since 2026-07-14. Access plan in the reading list §7. (A) remains the only item that can force a scope change.
- Two further neighbours added to the reading list rather than here, because they threaten framing rather than novelty: **Neurocomputing Jul-2025 `S0925231225016984`** (safe RL for discrete-time nonlinear zero-sum games, neural CBF in the *reward*, critic-only value iteration) and **Grover/Liu/Sycara deadlock theory** (CBF controllers provably stall at force-equilibria — the citable mechanism for shield-induced standoffs in our game).
- **Hardware confirmed available (user, 2026-09-03)**: R4 and Step 12 are in scope (D87).


## T0.1 READ IN FULL — 2026-09-03 (S120). Verdict: R1 survives and strengthens; R4 must be reworded.

**Full citation**: Yupeng Jia, Xiran Cui, Xiumei Han, Zhipeng Shen, "Safe exploration for pursuit-evasion games based on control barrier functions and reinforcement learning", *Control Engineering Practice* **171** (2026) 106852. Accepted 10 Feb 2026, online 18 Feb 2026. Dalian Scientific Test and Control Technology Institute / Tongji / Dalian Maritime. PDF now in `papers/original/`.

**What it actually is**: a nonlinear zero-sum PE differential game recast through an auxiliary system into an HJI equation; a single critic network approximates the value function (Weierstrass higher-order approximation, basis-function weights, Bellman-error weight update laws) — i.e. **adaptive dynamic programming / critic-only RL, NOT deep RL**. A high-order CBF enforces safety for both players. Control-affine dynamics, **full state**, no sensing model, no partial observability, no occlusion. Single run per experiment; no seeds, no cohort, no exploitability measurement, no self-play in our sense.

**Their hardware**: two Qbot 2 UGVs indoors, 8 circular obstacles (26 cm) in roughly a 3.5 x 3 m arena, **16-camera optical motion capture** for state, and **all HOCBF computation on an off-board host computer** (i7-9700K) which sends commands to the robots. One demonstration trajectory (their Figs. 13-16), not a statistical campaign.

**The sentence that matters for R1** (their §1, as motivation — asserted, never measured): existing approaches "impose safety constraints directly during policy optimization, which restricts the exploration space of both players, alters the iterative learning dynamics, and may prevent convergence to equilibrium strategies", and their framework "preserv[es] convergence to the saddle-point solution of the original game" unlike "conservative control projection or policy shielding".

**Their own admission** (§5): "Since most of the compared approaches are RL-based methods, the convergence rate of our algorithm does not show a significant advantage. However, the main strength of our method lies in its ability to safely learn an optimal policy."

### Consequences

- **R1 SURVIVES, and is now better positioned.** The equilibrium-shift claim is a *published, recent, named prediction* that its authors motivate with but never test. No measurement of it exists — not in deep RL, not under partial observability, not with exploitability. Our R1 becomes "the first empirical test of the Jia et al. (2026) prediction", a stronger framing than an unattributed gap. Their formulation is also our natural comparator: the alternative that claims not to shift.
- **R4 MUST BE REWORDED — a bare "first hard-safe RL pursuit-evasion on real ground robots" is now refuted by this paper.** The defensible claim is narrower and must be stated precisely every time: first **deep-RL**, **onboard-sensing / partially-observable**, **model-free-policy** hard-safe PE on real ground robots, evaluated as a **seeded statistical campaign** rather than a demonstration. Every qualifier is load-bearing: they have motion-capture full state, off-board computation, a critic-ADP policy, and a single run.
- **Their limitations name our territory** (§5): "pursuit-evasion game problems under high-order safety constraints with **unknown system dynamics** and multi-agent settings" — unknown dynamics is our model-free element.
- Not a blocker for Step 3b. The exact-game twin proceeds; its proposition should be framed against their claim.
