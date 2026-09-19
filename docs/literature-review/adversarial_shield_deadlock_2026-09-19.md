> # PARTLY SUPERSEDED BY D125 (2026-09-19)
>
> The **searches and the papers found remain valid and reusable**, and two of the clean negatives are now more
> important than when they were recorded: Q3 found no work measuring a filter's task cost under adversarial
> pressure, and Q4 found none on shield-aware remedies or transfer across filter families.
>
> **The Related-work positioning paragraph is obsolete.** It argues our delta for a claim — an embodied
> adversary manufacturing filter deadlock — that D125 shows was an artefact of a missing escape behaviour.
> The rebuilt thesis (a safety filter creating an absorbing *observation* loop for a memoryless learner) needs
> its own novelty check; the Q3/Q4 negatives cover part of it, the learning-pathology literature does not
> appear here at all and must be searched (absorbing states, frozen observations, credit assignment under
> action masking, deadlock in shielded RL).

---

# Literature check: adversarially induced safety-filter deadlock

**Date:** 2026-09-19
**Scope:** novelty check for the claim that a hard safety filter *relocates* rather than removes an adversary's leverage — from collision to stall/deadlock — and that a *learned* adversary discovers this unaided.
**Method:** WebSearch + WebFetch over arXiv, IEEE, ACM/USENIX, Springer, ScienceDirect, PMLR, Sept 2026. PDFs of the three highest-risk candidates were downloaded and grepped directly rather than trusted to summarisation.

---

## Verdict

**The core claim survives, but it is no longer untouched: there is one genuine near-scoop.** Yeke et al., *Automated Discovery of Semantic Attacks in Multi-Robot Navigation Systems* (USENIX Security '25), explicitly introduces **"deadlock" as a new adversarial objective** — immobilising a victim robot by exploiting its collision-avoidance module — and verifies it on ORCA and GLAS in sim and on real Crazyflies. That is adversarially *induced* deadlock, and it is prior art we must cite prominently and distinguish. It is **not** a full scoop, for four separable reasons: (i) the attack vector is **false-data injection into the victim's sensing/comms channel** (a cyber attack), not an embodied opponent moving legally in the world; (ii) the victim module is **reciprocal collision avoidance (ORCA/GLAS), not a certified safety filter** — there is no QP, no admissible-set collapse, no class-K gain or margin; (iii) the attacker is an **STL-robustness-guided stochastic search over injection parameters**, not a learner that discovers the vulnerability from reward alone; and (iv) the setting is **cooperative multi-robot navigation**, not a zero-sum game, so the framing "the shield relocates the adversary's leverage" has no analogue there.

Beyond that, **nothing was found that pairs a hard CBF/lookahead safety filter with an adversary whose objective is stall.** Every attack-on-safety-filter paper I found aims at the *opposite* failure — deactivating the filter to cause a violation (Arnström & Teixeira; Escudero et al.; GUARDIAN). Every deadlock paper I found is **cooperative and self-inflicted** (symmetry, force equilibrium among co-operating robots), never adversarially sought. One verified anecdote comes close to our claim (2): Gavin et al. (AMIAD 2025) report a *learned* evader that "fl[ies] close to the arena walls to stay safe, exploiting the pursuer's fear to avoid boundary violations" — but that is a soft reward penalty, it is reported in one sentence as a nuisance and **shaped away with a buffer zone rather than studied**. Claims (3) (stall surface scaling with margin and class-K gain, obstacle dependence, second filter family) and (4) (shield-aware resist-and-hunt fixed point) returned **no hits at all** and appear novel.

**Sharpest residual risk:** a reviewer who knows Raven will say "adversarially induced deadlock is done." The delta must be stated in the abstract, not buried: *embodied* adversary under no sensing compromise, against a *certified* filter, with the attack *learned*, in a *zero-sum* game.

---

## Q1 — Has anyone adversarially INDUCED safety-filter deadlock?

### The near-scoop

**Automated Discovery of Semantic Attacks in Multi-Robot Navigation Systems**
Doguhan Yeke, Kartik A. Pant, Muslum Ozgur Ozmen, Hyungsub Kim, James M. Goppert, Inseok Hwang, Antonio Bianchi, Z. Berkay Celik. USENIX Security Symposium 2025.
https://www.usenix.org/conference/usenixsecurity25/presentation/yeke

Introduces **Raven**, which expresses five adversarial goals in Signal Temporal Logic and finds stealthy false-data-injection parameters achieving them against multi-robot collision avoidance (ORCA, GLAS) across a Python simulator, PX4/Gazebo, and a Crazyflie swarm. Two goals are claimed as new: **deadlock** — STL `□[t,mis_time](total_pos_diff_of_victim > x)`, "immobilizes robots by maintaining their positions for a certain duration" — and **herding**. Success rates 90–100% with five injections; the paper attributes vulnerability to five design flaws including "high reactivity" and a planning-time trade-off, and notes the ripple effect by which one compromised robot stalls others.

> **Threat to us:** This is the paper that comes closest to scooping the headline concept, and it names the phenomenon with the same word. Our delta is the threat model (embodied, no sensing compromise), the victim (a certified CBF/lookahead filter with a tunable margin and class-K gain, not ORCA), the attacker (learned from reward, not searched over injection parameters), and the game (zero-sum, where "relocated leverage" is meaningful). Cite it in the intro, not only in related work.

### Attacks on safety filters that aim at violation, not stall

**Stealthy Deactivation of Safety Filters** — Daniel Arnström, André M. H. Teixeira. ECC/IEEE 2024. https://arxiv.org/abs/2403.17861
False-data injection on the measurement channel biases the observer's state estimate toward the interior of the safe set, so the filter passes unsafe actions while the anomaly detector stays quiet. Requires knowledge of dynamics, safe set, and observer gain.
> Relation: establishes the safety filter as an attack surface — the framing we inherit — but the adversary's goal is a *collision*. It is the exact complement of our claim, and useful as the contrast case: when the filter is sound and the sensing honest, this attack is unavailable and the adversary is pushed toward stall instead.

**Data-Driven and Stealthy Deactivation of Safety Filters** — Daniel Arnström, André M. H. Teixeira. L4DC 2025 (PMLR v283). https://arxiv.org/abs/2412.01346
Relaxes the above: the adversary identifies the observer dynamics and approximates the safe set from observed input–output data alone, with no a priori model knowledge. Demonstrated on an inverted pendulum.
> Relation: same complement; shows the attacker-side literature is maturing toward weaker assumptions, which raises the bar for how we motivate the embodied threat model.

**Safety Filters Against Actuator Attacks** — Escudero et al. *International Journal of Robust and Nonlinear Control*, 2025. https://onlinelibrary.wiley.com/doi/10.1002/rnc.7871
Safety filters as a *defence* against actuator-channel attacks.
> Relation: defensive framing; assumes the filter is the thing being protected, never the thing being exploited.

**GUARDIAN: Safety Filtering for Systems with Perception Models Subject to Adversarial Attacks** — 2026. https://arxiv.org/pdf/2602.06026
Certifies a safety filter against adversarial perturbation of an NN state estimator, via NN verification bounds plumbed into a modified HJ reachability formulation.
> Relation: hardens the filter against perception attacks; says nothing about an adversary who accepts the filter's correctness and attacks liveness instead.

### Adversarial *stress testing* of safety filters (checked directly — not a scoop)

**Adversarial Stress Testing of SPARK Humanoid Safety Filters** — Saurav Ghosh, Abdou Sow, Luke Zhang. arXiv:2605.19009, May 2026 (preprint, not peer reviewed). https://arxiv.org/abs/2605.19009
Replicates the SPARK humanoid benchmark in MuJoCo and compares RSSA, RSSS, SSA, CBF, PFM, SMA under **obstacle crowding, perception noise, and sensor latency**. Reports a safety–performance trade-off and repeated **"No Solution"** QP outputs, explicitly conceding "we did not systematically measure the frequency of infeasibility."
> **Caution logged:** an automated summary of this paper claimed it studies a *learned adversary deliberately inducing deadlock*. I downloaded the PDF and grepped it: the strings "deadlock", "stall", "freeze" appear **zero times**. The "adversary" is environmental perturbation, not an agent. Not a scoop — but cite it as the closest existing work on stress-testing filters to infeasibility, and as evidence that the community has noticed the symptom without naming the mechanism.

### Action-freezing attacks in other modalities

**FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models** — arXiv:2509.19870 (ICLR-cycle OpenReview). https://arxiv.org/abs/2509.19870
Min–max bi-level optimisation generates adversarial *images* that make a VLA ignore subsequent instructions and stop acting; 76.2% mean attack success across three VLAs and four benchmarks, with strong image transferability. Notes that inaction "can be mistaken for normal operation" and delay intervention.
> Relation: independent confirmation that *inducing inaction* is a recognised adversarial objective with real safety consequences — good supporting framing for why stall is a legitimate attack goal — but the vector is observation perturbation of a learned policy, with no safety filter anywhere in the loop.

**SWARMFLAWFINDER: Discovering and Exploiting Logic Flaws of Swarm Algorithms** — Chijung Jung et al., IEEE S&P 2022. https://yonghwi-kwon.github.io/data/swarmflawfinder_sp22.pdf
Greybox fuzzing with a "degree of causal contribution" feedback metric, using an external attack drone to find logic flaws in four swarm algorithms; 42 confirmed flaws.
> Relation: the attacker here **is** embodied (an attack drone), which is closer to our threat model than Raven's, but the targets are swarm logic bugs, the search is fuzzing rather than learning, and deadlock is not an articulated objective.

### Exploiting conservative agents (no filter, but the same economic logic)

**Stackelberg Punishment and Bully-Proofing Autonomous Vehicles** — Matt Cooper, Jun Ki Lee, Jacob Beck, Joshua D. Fishman, Michael Gilbert, Jonathan Baxter, Kobi Lachmy, Michael L. Littman et al. ICSR 2019. https://arxiv.org/abs/1908.08641
Human drivers exploit a conservative AV by forcing it to yield; the paper proposes a Stackelberg punishment policy that deters bullying while maximising the AV's own return.
> Relation: the closest *conceptual* ancestor of our thesis in a non-filter setting — conservatism is exploitable leverage — and a precedent for our claim (4), since "resist-and-hunt" is structurally a learned Stackelberg punishment against a stall attacker.

**Ready to bully automated vehicles on public roads?** / **Is it OK to bully automated cars?** — *Accident Analysis & Prevention*, 2020 / 2022. https://www.sciencedirect.com/science/article/abs/pii/S0001457522001506
Human-factors evidence that road users exploit risk-averse AV behaviour.
> Relation: real-world motivation only; no formal filter, no deadlock analysis.

### Clean negatives for Q1
- **"deadlock attack" against a CBF/HJ safety filter by a physically embodied opponent** — *no hits*. Searched across CBF/safety-filter/liveness/deadlock/attack/exploit phrasings; every return was either cooperative deadlock or a spoofing attack aimed at unsafety.
- **A learned (RL) adversary whose reward is the victim's filter activation, stall time, or QP infeasibility** — *no hits*. Raven searches; FreezeVLA optimises pixels; SwarmFlawFinder fuzzes. None learns a *policy* in the victim's state space.
- **Stall/deadlock rate reported as a metric for a safety-filtered agent under adversarial pressure** — *no hits*.

---

## Q2 — Cooperative deadlock literature (must cite and position against)

**The Before, During, and After of Multi-Robot Deadlock** — Jaskaran Grover, Changliu Liu, Katia Sycara. *International Journal of Robotics Research* 42(5), 2023 (arXiv:2206.01781). https://arxiv.org/abs/2206.01781
Analyses the closed-loop dynamics of N robots under CBF-QP collision avoidance and characterises the controller parameters, initial conditions and goal locations that provably lead to deadlock. Uses duality and a mechanics reading of the KKT conditions to show deadlock is a **force equilibrium** among robots, derives the geometry of deadlocked configurations, and gives a resolution scheme.
> **Difference:** deadlock is an emergent property of *mutually* filtered, cooperating agents whose constraints are reciprocal; there is no agent that *wants* it, and the resolution is a design fix. Ours is a **single** filtered agent, the constraint is against static geometry and a non-reciprocal opponent, and deadlock is the opponent's *objective*, produced on demand.

**Why Does Symmetry Cause Deadlocks?** — Jaskaran Grover, Changliu Liu, Katia Sycara. IFAC World Congress 2020, *IFAC-PapersOnLine*. https://www.sciencedirect.com/science/article/pii/S2405896320334042
Proves geometric symmetry in initial conditions and goals *necessarily* yields deadlock, and that heterogeneity in controller parameters is insufficient to avoid it.
> **Difference:** the cause is symmetry between peers. Our stall has no symmetry requirement and no peer — it is produced by an asymmetric opponent interposing geometry.

**Deadlock Analysis and Resolution for Multi-Robot Systems** — Jaskaran Grover, Changliu Liu, Katia Sycara. WAFR 2020, Springer SPAR (arXiv:1911.09146). https://arxiv.org/abs/1911.09146
Formal analysis of 2- and 3-robot CBF systems; provably correct decentralised deadlock-resolution algorithm.
> **Difference:** as above; resolution assumes willing participants.

**Multiagent Systems with CBF-Based Controllers: Collision Avoidance and Liveness from Instability** — Ross/Zhu et al. (arXiv:2207.04915), *IEEE Transactions on Control Systems Technology*, 2023; and the earlier **Collision Avoidance and Liveness of Multi-agent Systems with CBF-based Controllers**, CDC 2021 (arXiv:2012.10261). https://arxiv.org/abs/2207.04915
Studies the safety/liveness trade-off across centralised, decentralised, PCCA and CCS policies; shows deliberately induced *instability* restores liveness, and proves feasibility for several policies.
> **Difference:** the safety–liveness trade-off is framed as a *design* question among cooperating agents; nobody is trying to push the system to the liveness-losing side.

**Deadlock-free, Safe, and Decentralized Multi-Robot Navigation in Social Mini-Games via Discrete-Time Control Barrier Functions** — Rohan Chandra et al. *Autonomous Robots*, 2025 (arXiv:2308.10966). https://arxiv.org/abs/2308.10966 · see also **Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms**, arXiv:2508.13459
DCBF-based decentralised navigation for doorways, intersections and corridors, with explicit deadlock-freedom.
> **Difference:** closest in *mechanism* (discrete-time CBF, and the same intuition that constrained geometry is where filters fail), but the mini-games are cooperative-competitive negotiations over shared space with congruent goals, and deadlock is the failure the paper removes, not the one it seeks. Our DCBF is the same tool used as the *victim*.

**LiveNet: Robust, Minimally Invasive Multi-Robot Control for Safe and Live Navigation in Constrained Environments** — arXiv:2412.04659. https://arxiv.org/abs/2412.04659
Differential-CBF layer that looks ahead and emits accelerations preserving both collision avoidance and liveness.

**LivePoint: Fully Decentralized, Safe, Deadlock-Free Multi-Robot Control in Cluttered Environments with High-Dimensional Inputs** — arXiv:2503.13098. https://arxiv.org/abs/2503.13098
A universal CBF that computes safety *and* liveness certificates simultaneously; an interaction function breaks deadlock-causing symmetry minimally invasively.

**Adaptive Deadlock Avoidance for Decentralized Multi-agent Systems via CBF-inspired Risk Measurement** — arXiv:2503.09621. https://arxiv.org/abs/2503.09621

**Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock Prevention** — arXiv:2503.05848. https://arxiv.org/abs/2503.05848
Temporary circular "roundabout" reference paths as a lightweight deadlock-prevention layer.
> **Difference (all four):** these are *remedies*, and every one of them assumes deadlock is an unwanted accident arising from symmetry or congestion among cooperating agents. None asks whether the remedy survives an agent that is actively constructing the deadlock configuration. That is precisely the experiment our claim (3) runs, and these papers are the natural targets for a future "does the fix hold under attack?" follow-up.

**Addressing the deadlock problem of ORCA in distributed multi-robot navigation** — *Robotica*, Cambridge. https://www.cambridge.org/core/journals/robotica/article/abs/addressing-the-deadlock-problem-of-the-optimal-reciprocal-collision-avoidance-algorithm-in-distributed-multirobot-navigation/70CEA27CFAE13D30DD3ABCD189F9019D
ORCA's symmetry/reciprocity assumptions produce stuck configurations.
> Useful bridge: Raven's deadlock attack is exactly a deliberate violation of ORCA's reciprocity assumption — which is why it does *not* transfer to a unilateral CBF filter, and why our threat model needs its own treatment.

---

## Q3 — Safety–performance / liveness tension under shields

**Provably Optimal Reinforcement Learning under Safety Filtering** — arXiv:2510.18082. https://arxiv.org/abs/2510.18082
Argues the perceived safety–performance trade-off is not inherent, and proves that enforcing safety with a **sufficiently permissive** filter does not degrade asymptotic performance.
> **Threat to us — take seriously.** A reviewer may cite this to argue our capture-rate cost is an artefact of an over-conservative filter. Our answer must be claim (3): we *sweep* the margin and the class-K gain and show the stall surface grows monotonically with conservativeness, i.e. we characterise the permissiveness axis rather than picking one point on it. Their theorem concerns asymptotic performance against a *fixed* environment; it says nothing about performance against an adversary who re-optimises against whatever permissiveness you chose.

**Safe Control Algorithms Using Energy Functions: A Unified Framework, Benchmark, and New Directions** — Tianhao Wei, Changliu Liu. CDC 2019 / arXiv:1908.01883. https://arxiv.org/abs/1908.01883
Unifies potential-field, safe-set, barrier-function and sliding-mode filters via a perpendicular decomposition of control into a safety component along the Lie derivative and an efficiency component orthogonal to it; benchmarks the safety–efficiency trade-off and derives the sublevel safe-set algorithm.
> **Highly relevant to claim (3):** this is the standard reference for "a second, non-CBF filter family," and it supplies the vocabulary (safety vs efficiency component) for arguing that our stall result is a property of *filters as a class*, not of CBFs specifically. Their benchmark measures efficiency loss, but only against a passive environment.

**Persistently Feasible Robust Safe Control by Safety Index Synthesis and Convex Semi-Infinite Programming** — Wei, Kang, Zhao, Liu. *IEEE L-CSS*, 2022.
Synthesises a safety index that is *persistently feasible* under control limits and bounded uncertainty.
> Relation: the constructive answer to "just make the filter never infeasible." Worth citing as the defence our attacker should be evaluated against; note that persistent feasibility of the *QP* is not the same as non-zero admissible motion toward the goal — a filter can be strictly feasible and still admit only the zero command, which is exactly the stall we measure. This distinction is worth a sentence in the paper.

**The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems** — Kai-Chieh Hsu, Haimin Hu, Jaime F. Fisac. *Annual Review of Control, Robotics, and Autonomous Systems*, 2024 (arXiv:2309.05837). https://arxiv.org/abs/2309.05837
The canonical survey unifying CBF, HJ-reachability and predictive safety filters; discusses conservativeness and the filter's interference with task performance.
> Relation: our framing reference; supports the two-axis reporting rule, and its taxonomy justifies testing a second filter family.

**CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions** — arXiv:2510.14959. https://arxiv.org/abs/2510.14959
Enforces CBFs during training rather than only at deployment, motivated explicitly by the conservatism that arises when the RL policy is unaware of the constraint.
> Relation: the standard remedy for the "policy doesn't know about the shield" problem — the same intuition as our shield-state observation features. Directly relevant to claim (4) and to our earlier finding that shield-in-training alone did not recover the loss.

**Adversarial Stress Testing of SPARK Humanoid Safety Filters** (above) also reports a filter-by-filter safety/goal-tracking trade-off with collision-step and goal-distance metrics.
> Relation: an independent instance of two-axis reporting, though not framed as a rule.

> **Summary for Q3:** safety-vs-liveness for CBFs is well recognised (Q2 papers) and efficiency cost has been benchmarked passively (Wei & Liu; SPARK). **No work found measures a safety filter's task-performance cost under an opponent optimising to maximise that cost.** Our two-axis reporting rule under adversarial pressure appears to be new.

---

## Q4 — Shielded RL and safe RL in adversarial / game settings

**Safe Reinforcement Learning via Shielding** — Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum, Ufuk Topcu. AAAI 2018 (arXiv:1708.08611). https://arxiv.org/abs/1708.08611
The origin of the shield abstraction: a reactive system synthesised from a safety automaton corrects the learner's action, decoupling safety from reward.
> Relation: baseline citation; the shield's correctness is assumed to settle the safety question, which is exactly the assumption we complicate.

**Safe Reinforcement Learning via Probabilistic Shields** — Nils Jansen, Bettina Könighofer, Sebastian Junges, Alex Serban, Roderick Bloem. CONCUR 2020 (arXiv:1807.06096). https://arxiv.org/abs/1807.06096
Shielding with probabilistic model checking; discusses the safety/performance trade-off explicitly.

**Robust Shielding for Safe Reinforcement Learning** — Edwin Hamel-De le Court et al., arXiv:2606.00270 (2026). https://arxiv.org/abs/2606.00270
Shielding for robust MDPs, posed as a game between the agent and an adversary choosing transition probabilities from an uncertainty set; the shield blocks any policy unsafe under *any* adversarial choice.
> **Relation — note carefully:** the adversary is a *disturbance* over transition kernels, and the shield is proved to make the adversary's *safety* attack infeasible. Nobody asks what the adversary does next once safety is closed off. That unexamined next step is our paper.

**ISAACS: Iterative Soft Adversarial Actor-Critic for Safety** — Kai-Chieh Hsu, Duy Phuong Nguyen, Jaime F. Fisac. L4DC 2023, PMLR v211 (arXiv:2212.03228). https://arxiv.org/abs/2212.03228
Co-trains a safety-seeking fallback policy against a learned worst-case disturbance agent; converts the result into a rollout-based robust safety filter with zero violations under the oracle worst-case disturbance, at moderate conservativeness relative to the HJ optimum.
> Relation: the strongest precedent for *learning* an adversary in a safety-filter pipeline — but the learned adversary's objective is **to cause a violation**, used to *build* the filter. Ours inverts both: the adversary is learned to cause a *stall*, against a filter that already holds.

**Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination** — Duy P. Nguyen, Kai-Chieh Hsu, Wenhao Yu, Jie Tan, Jaime F. Fisac. CoRL 2024 (arXiv:2405.00846). https://arxiv.org/abs/2405.00846
Monitors safety by rapidly simulating adversarial futures, pitting a best-effort safety strategy against a learned virtual adversary that exploits uncertainty and sim-to-real error.
> Relation: same inversion as ISAACS; the adversary is imagined and internal, its goal is to tip the robot over.

**Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees** — Kai-Chieh Hsu, Allen Z. Ren, Duy P. Nguyen, Anirudha Majumdar, Jaime F. Fisac. *Artificial Intelligence*, 2023 (arXiv:2201.08355). https://arxiv.org/abs/2201.08355

**Safe Reinforcement Learning in Black-Box Environments via Adaptive Shielding** — arXiv:2405.18180. https://arxiv.org/abs/2405.18180
Reports that a **pre-trained shield cuts safety violations by over 50% when transferred across environments** with comparable reward.
> **This is the only transfer-adjacent result I found**, and it is shield-across-*environments*, not shield-across-*filter-families*. It does not answer the transfer question we pose.

**Safe exploration for pursuit-evasion games based on control barrier functions and reinforcement learning** — *Control Engineering Practice*, 2026. https://www.sciencedirect.com/science/article/abs/pii/S0967066126000961
High-order CBFs for player safety plus an RL safe-exploration scheme for PE games. (Publisher blocked automated retrieval — 403 — so details come from indexing snippets only; see Gaps.)
> Relation: the most directly overlapping PE+CBF+RL paper; claims to avoid the "conservative safe equilibria" that CBF-filtered RL converges to. **Must be read in full before submission** — the phrase "conservative safe equilibria" is uncomfortably close to our stall, though framed as a learning-convergence artefact rather than an adversarial objective.

**Reinforcement learning in pursuit-evasion differential game: safety, stability and robustness** — Xinyang Wang, Hongwei Zhang, Jun Xu, Shimin Wang, Martin Guay. *IEEE TNNLS*, 2026 (arXiv:2507.19516). https://arxiv.org/abs/2507.19516
Stackelberg-inspired hierarchy with a CBF leader and sliding-mode follower for PE in obstacle-rich environments. **Verified: does not study an adversary inducing deadlock.**

**Capture, Shield, or Neutralize: Engagement-Aware Pursuit-Evasion** — Ananya Acharya, Trenton Goyette, Masoud Ataei, Adrian Stoica, Vikas Dhiman, Mohammad Javad Khojasteh. arXiv:2607.10986, 2026. https://arxiv.org/abs/2607.10986
Receding-horizon minimax MPC with a **discrete-time CBF inner-loop safety filter** for PE and perimeter defence on 2D double-integrator and 3D quadrotor dynamics. **Verified:** reports capture rates but **no MPC-only vs MPC+CBF ablation**, uses QP slack variables to force strict feasibility "even in dense multi-agent clusters," and assumes the evader solves a symmetric problem — explicitly not an evader exploiting the filter.
> Relation: our nearest architectural neighbour (same DCBF-as-inner-filter design, same PE game) and a clean demonstration that the community *does not yet measure* the filter's capture-rate cost. Their slack-variable move is also a good foil: slack restores QP feasibility while silently surrendering the guarantee.

**Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility and Safety Barrier Functions** — Minnan Zhou et al., ICRA 2025 (arXiv:2411.01321). https://arxiv.org/abs/2411.01321
Signed-distance-function CBFs for both field-of-view maintenance and obstacle avoidance for a pursuer under occlusion.
> Relation: closest work on PE + CBF + obstacles + partial observability, i.e. our exact ingredient list. The evader is scripted and non-exploitative; obstacles occlude rather than serve as stall furniture.

**Safe Control for Pursuit-Evasion with Density Functions** — arXiv:2505.15718. https://arxiv.org/abs/2505.15718

**Ensuring Safety in Target Pursuit Control: A CBF-Safe Reinforcement Learning Approach** — arXiv:2411.17552. https://arxiv.org/abs/2411.17552
QP safety filter over RL actions covering sensing, collision avoidance and input constraints for pursuers.

**Agile Interception of a Flying Target using Competitive Reinforcement Learning** — Timothée Gavin, Simon Lacroix, Murat Bronz. AMIAD (Conference on AI for Defence), Nov 2025; arXiv:2603.16279. https://arxiv.org/abs/2603.16279
Competitive self-play for drone interception. **Verified quote (§III):** "the evader can quickly learn to fly close to the arena walls to stay safe, **exploiting the pursuer's fear to avoid boundary violations**. To discourage this behaviour … a narrow buffer zone is added adjacent to every wall; only the evader is penalized for entering this zone."
> **This is the best independent evidence for our claim (2)** — a learned evader spontaneously discovering that the pursuer's safety constraint is exploitable leverage, in a paper with no interest in the phenomenon. Crucially: the constraint is a **soft reward penalty**, not a hard filter; the behaviour is reported in two sentences and **engineered away rather than measured**; there is no stall, no deadlock, no margin sweep. Cite it as prior anecdote — it strengthens our claim's plausibility while leaving the contribution intact.

**Adversarial Policies: Attacking Deep Reinforcement Learning** — Adam Gleave, Michael Dennis, Cody Wild, Neel Kant, Sergey Levine, Stuart Russell. ICLR 2020 (arXiv:1905.10615). https://arxiv.org/abs/1905.10615
Physically realistic threat model: the attacker cannot perturb observations, only choose a policy in a shared zero-sum environment. Finds adversarial policies that reliably beat self-play-hardened victims using under 3% of the victim's training budget.
> Relation: the **methodological template for our threat model** and the right citation for "embodied, no sensing compromise." Their victims have no safety filter; the novelty we add is that the victim carries a *guarantee*, which redirects the adversarial policy from defeat-by-collision to defeat-by-stall.

### Clean negative for Q4
- **Shield transfer across filter families** (a policy hardened against a CBF-QP filter evaluated under an exact-lookahead filter, or vice versa) — *no hits*. The nearest is shield transfer across environments (arXiv:2405.18180) and zero-shot *safety* generalisation (Sim-to-Lab-to-Real, gameplay filters). The cross-family transfer question appears **unasked**, which makes it a cheap and defensible contribution.

---

## Related-work positioning (adaptable paragraph)

> Deadlock under control-barrier-function control is well understood as a *cooperative* pathology. Grover et al. characterise it as a force equilibrium among mutually filtered robots, prove that geometric symmetry in initial conditions and goals makes it unavoidable, and give decentralised resolution schemes; a substantial remedy literature follows, adding liveness certificates, interaction terms that break symmetry, risk-based adaptive relaxation, and roundabout reference paths. Across all of this work deadlock is an accident — an unwanted equilibrium produced by congestion or symmetry among agents whose goals are congruent — and the contribution is a design change that removes it. The security literature has shown, separately, that safety machinery is attackable: false-data injection can bias a state estimator so that a safety filter passes unsafe actions, and Raven's robustness-guided search over injection parameters can drive ORCA- and GLAS-controlled robots into an immobilised state, introducing deadlock as an explicit adversarial objective. Those attacks presuppose a compromised sensing or communication channel and target reciprocal collision-avoidance heuristics rather than a certified filter. Work that *learns* an adversary against a safety pipeline — ISAACS, gameplay filters, robust shielding for RMDPs — trains that adversary to cause a violation, and uses it to construct or certify the filter; once safety is proved robust, the adversary's remaining options go unexamined. We occupy the gap those three literatures leave open. Our adversary is embodied and legal: it perturbs no sensor, injects no data, and merely moves, in the manner of Gleave et al.'s adversarial policies. Its victim is a single agent behind a filter whose guarantee genuinely holds — no collision occurs in any episode we report. And its objective is the victim's *liveness*: it manoeuvres to place a surface between itself and the pursuer so that the filter's admissible command set collapses and the pursuer must stand still. Deadlock here is not an accident to be designed out but a resource to be manufactured, and the pursuer's own guarantee is what manufactures it. We show the attack is discovered by a learner that is never told a filter exists, that the stall region grows with the filter's margin and with a more permissive class-K gain, that it requires obstacles rather than walls alone, and that it survives a change of filter family — so the effect is a property of hard filtering, not of one certificate. This reframes the cooperative-deadlock remedies as untested under attack, and it motivates reporting safety and liveness on two axes rather than treating a satisfied safety constraint as the end of the analysis.

---

## Citation list (BibTeX)

```bibtex
@inproceedings{yeke2025raven,
  title     = {Automated Discovery of Semantic Attacks in Multi-Robot Navigation Systems},
  author    = {Yeke, Doguhan and Pant, Kartik A. and Ozmen, Muslum Ozgur and Kim, Hyungsub and Goppert, James M. and Hwang, Inseok and Bianchi, Antonio and Celik, Z. Berkay},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  year      = {2025},
  address   = {Seattle, WA},
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/usenixsecurity25/presentation/yeke}
}

@article{grover2023before,
  title   = {The Before, During, and After of Multi-Robot Deadlock},
  author  = {Grover, Jaskaran and Liu, Changliu and Sycara, Katia},
  journal = {The International Journal of Robotics Research},
  volume  = {42},
  number  = {5},
  year    = {2023},
  doi     = {10.1177/02783649221074718},
  note    = {arXiv:2206.01781}
}

@article{grover2020symmetry,
  title   = {Why Does Symmetry Cause Deadlocks?},
  author  = {Grover, Jaskaran and Liu, Changliu and Sycara, Katia},
  journal = {IFAC-PapersOnLine},
  volume  = {53},
  number  = {2},
  year    = {2020},
  note    = {21st IFAC World Congress},
  url     = {https://www.sciencedirect.com/science/article/pii/S2405896320334042}
}

@inproceedings{grover2020deadlock,
  title     = {Deadlock Analysis and Resolution for Multi-Robot Systems},
  author    = {Grover, Jaskaran and Liu, Changliu and Sycara, Katia},
  booktitle = {Algorithmic Foundations of Robotics XIV (WAFR)},
  year      = {2020},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-66723-8_18},
  note      = {arXiv:1911.09146}
}

@article{chandra2025socialminigames,
  title   = {Deadlock-free, safe, and decentralized multi-robot navigation in social mini-games via discrete-time control barrier functions},
  author  = {Chandra, Rohan and others},
  journal = {Autonomous Robots},
  year    = {2025},
  doi     = {10.1007/s10514-025-10194-8},
  note    = {arXiv:2308.10966}
}

@article{cbf_liveness_instability,
  title   = {Multiagent Systems With CBF-Based Controllers: Collision Avoidance and Liveness From Instability},
  journal = {IEEE Transactions on Control Systems Technology},
  year    = {2023},
  note    = {arXiv:2207.04915; earlier version at CDC 2021, arXiv:2012.10261},
  url     = {https://arxiv.org/abs/2207.04915}
}

@misc{livenet2024,
  title        = {LiveNet: Robust, Minimally Invasive Multi-Robot Control for Safe and Live Navigation in Constrained Environments},
  year         = {2024},
  howpublished = {arXiv:2412.04659},
  url          = {https://arxiv.org/abs/2412.04659}
}

@misc{livepoint2025,
  title        = {LivePoint: Fully Decentralized, Safe, Deadlock-Free Multi-Robot Control in Cluttered Environments with High-Dimensional Inputs},
  year         = {2025},
  howpublished = {arXiv:2503.13098},
  url          = {https://arxiv.org/abs/2503.13098}
}

@misc{merrygoround2025,
  title        = {Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock Prevention},
  year         = {2025},
  howpublished = {arXiv:2503.05848},
  url          = {https://arxiv.org/abs/2503.05848}
}

@misc{adaptive_deadlock2025,
  title        = {Adaptive Deadlock Avoidance for Decentralized Multi-agent Systems via CBF-inspired Risk Measurement},
  year         = {2025},
  howpublished = {arXiv:2503.09621},
  url          = {https://arxiv.org/abs/2503.09621}
}

@inproceedings{arnstrom2024stealthy,
  title     = {Stealthy Deactivation of Safety Filters},
  author    = {Arnstr{\"o}m, Daniel and Teixeira, Andr{\'e} M. H.},
  booktitle = {European Control Conference (ECC)},
  year      = {2024},
  note      = {arXiv:2403.17861},
  url       = {https://arxiv.org/abs/2403.17861}
}

@inproceedings{arnstrom2025datadriven,
  title     = {Data-Driven and Stealthy Deactivation of Safety Filters},
  author    = {Arnstr{\"o}m, Daniel and Teixeira, Andr{\'e} M. H.},
  booktitle = {Proceedings of the 7th Annual Learning for Dynamics and Control Conference (L4DC)},
  series    = {PMLR},
  volume    = {283},
  year      = {2025},
  note      = {arXiv:2412.01346},
  url       = {https://proceedings.mlr.press/v283/arnstrom25a.html}
}

@inproceedings{jung2022swarmflawfinder,
  title     = {SWARMFLAWFINDER: Discovering and Exploiting Logic Flaws of Swarm Algorithms},
  author    = {Jung, Chijung and others},
  booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
  year      = {2022},
  doi       = {10.1109/SP46214.2022.9833685}
}

@misc{freezevla2025,
  title        = {FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models},
  year         = {2025},
  howpublished = {arXiv:2509.19870},
  url          = {https://arxiv.org/abs/2509.19870}
}

@misc{ghosh2026spark,
  title        = {Adversarial Stress Testing of SPARK Humanoid Safety Filters},
  author       = {Ghosh, Saurav and Sow, Abdou and Zhang, Luke},
  year         = {2026},
  howpublished = {arXiv:2605.19009},
  note         = {Preprint, not peer reviewed},
  url          = {https://arxiv.org/abs/2605.19009}
}

@inproceedings{gleave2020adversarial,
  title     = {Adversarial Policies: Attacking Deep Reinforcement Learning},
  author    = {Gleave, Adam and Dennis, Michael and Wild, Cody and Kant, Neel and Levine, Sergey and Russell, Stuart},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  note      = {arXiv:1905.10615},
  url       = {https://arxiv.org/abs/1905.10615}
}

@inproceedings{alshiekh2018shielding,
  title     = {Safe Reinforcement Learning via Shielding},
  author    = {Alshiekh, Mohammed and Bloem, Roderick and Ehlers, R{\"u}diger and K{\"o}nighofer, Bettina and Niekum, Scott and Topcu, Ufuk},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2018},
  note      = {arXiv:1708.08611}
}

@inproceedings{jansen2020probabilistic,
  title     = {Safe Reinforcement Learning via Probabilistic Shields},
  author    = {Jansen, Nils and K{\"o}nighofer, Bettina and Junges, Sebastian and Serban, Alex and Bloem, Roderick},
  booktitle = {31st International Conference on Concurrency Theory (CONCUR)},
  year      = {2020},
  note      = {arXiv:1807.06096}
}

@inproceedings{hsu2023isaacs,
  title     = {ISAACS: Iterative Soft Adversarial Actor-Critic for Safety},
  author    = {Hsu, Kai-Chieh and Nguyen, Duy Phuong and Fisac, Jaime F.},
  booktitle = {Proceedings of the 5th Annual Learning for Dynamics and Control Conference (L4DC)},
  series    = {PMLR},
  volume    = {211},
  year      = {2023},
  note      = {arXiv:2212.03228}
}

@inproceedings{nguyen2024gameplay,
  title     = {Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination},
  author    = {Nguyen, Duy P. and Hsu, Kai-Chieh and Yu, Wenhao and Tan, Jie and Fisac, Jaime F.},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2024},
  note      = {arXiv:2405.00846}
}

@article{hsu2024safetyfilter,
  title   = {The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems},
  author  = {Hsu, Kai-Chieh and Hu, Haimin and Fisac, Jaime F.},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  volume  = {7},
  year    = {2024},
  doi     = {10.1146/annurev-control-071723-102940},
  note    = {arXiv:2309.05837}
}

@inproceedings{wei2019energy,
  title     = {Safe Control Algorithms Using Energy Functions: A Unified Framework, Benchmark, and New Directions},
  author    = {Wei, Tianhao and Liu, Changliu},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2019},
  note      = {arXiv:1908.01883}
}

@misc{provably_optimal_rl_filtering,
  title        = {Provably Optimal Reinforcement Learning under Safety Filtering},
  year         = {2025},
  howpublished = {arXiv:2510.18082},
  url          = {https://arxiv.org/abs/2510.18082}
}

@misc{cbfrl2025,
  title        = {CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions},
  year         = {2025},
  howpublished = {arXiv:2510.14959},
  url          = {https://arxiv.org/abs/2510.14959}
}

@misc{adaptive_shielding_blackbox,
  title        = {Safe Reinforcement Learning in Black-Box Environments via Adaptive Shielding},
  year         = {2024},
  howpublished = {arXiv:2405.18180},
  url          = {https://arxiv.org/abs/2405.18180}
}

@inproceedings{cooper2019stackelberg,
  title     = {Stackelberg Punishment and Bully-Proofing Autonomous Vehicles},
  author    = {Cooper, Matt and Lee, Jun Ki and Beck, Jacob and Fishman, Joshua D. and Gilbert, Michael and Baxter, Jonathan and Lachmy, Kobi and Littman, Michael L.},
  booktitle = {International Conference on Social Robotics (ICSR)},
  year      = {2019},
  doi       = {10.1007/978-3-030-35888-4_34},
  note      = {arXiv:1908.08641}
}

@misc{gavin2026agile,
  title        = {Agile Interception of a Flying Target using Competitive Reinforcement Learning},
  author       = {Gavin, Timoth{\'e}e and Lacroix, Simon and Bronz, Murat},
  year         = {2026},
  howpublished = {arXiv:2603.16279},
  note         = {Conference on Artificial Intelligence for Defence (AMIAD), Rennes, France, Nov. 2025},
  url          = {https://arxiv.org/abs/2603.16279}
}

@article{wang2026pursuitevasion,
  title   = {Reinforcement Learning in Pursuit-Evasion Differential Game: Safety, Stability, and Robustness},
  author  = {Wang, Xinyang and Zhang, Hongwei and Xu, Jun and Wang, Shimin and Guay, Martin},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  note    = {arXiv:2507.19516}
}

@misc{acharya2026capture,
  title        = {Capture, Shield, or Neutralize: Engagement-Aware Pursuit-Evasion},
  author       = {Acharya, Ananya and Goyette, Trenton and Ataei, Masoud and Stoica, Adrian and Dhiman, Vikas and Khojasteh, Mohammad Javad},
  year         = {2026},
  howpublished = {arXiv:2607.10986},
  url          = {https://arxiv.org/abs/2607.10986}
}

@inproceedings{zhou2025occlusion,
  title     = {Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility and Safety Barrier Functions},
  author    = {Zhou, Minnan and others},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  note      = {arXiv:2411.01321}
}

@article{safe_exploration_pe_cbf_2026,
  title   = {Safe exploration for pursuit-evasion games based on control barrier functions and reinforcement learning},
  journal = {Control Engineering Practice},
  year    = {2026},
  note    = {Author list not retrieved --- publisher blocked automated access (HTTP 403)},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0967066126000961}
}

@misc{guardian2026,
  title        = {GUARDIAN: Safety Filtering for Systems with Perception Models Subject to Adversarial Attacks},
  year         = {2026},
  howpublished = {arXiv:2602.06026},
  url          = {https://arxiv.org/abs/2602.06026}
}
```

---

## Gaps / what I could not determine

1. **`Control Engineering Practice` 2026, "Safe exploration for pursuit-evasion games based on CBFs and RL"** — ScienceDirect returned HTTP 403 to automated retrieval, so I have **no author list, no abstract in full, and no verification** of its claims. An indexing snippet attributes to it the statement that conventional CBF-filtered RL "converge[s] to conservative safe equilibria" while their method reaches the true saddle point. That phrase is close enough to our stall that **this paper must be obtained and read in full before submission** — it is the single largest unresolved risk in this review. Retrieve via niro-2 or institutional access.
2. **Raven's deadlock attack details.** I confirmed the STL specification, the five goals, the ORCA/GLAS targets, and the Figure 10/12 deadlock illustrations. I did **not** extract their deadlock-specific success rate or whether the 90–100% figure covers the deadlock goal specifically — the aggregate is reported across attack types. Worth pinning down before writing the delta paragraph, since "Raven achieves X% deadlock" is a number a reviewer will ask us to compare against.
3. **Cooperative-deadlock remedies under attack.** None of LiveNet, LivePoint, Merry-Go-Round, or the adaptive risk-measure method has been evaluated against an adversary; I could not determine whether any of them would resist our attacker, only that nobody has tried. This is stated as an opportunity, not a result.
4. **Non-English and pre-2019 control-theory literature** on adversarial liveness denial was not searched. Classical differential-game "blocking" and "barrier" results may contain an analytic ancestor of the stall surface that a control-theory reviewer will know; a targeted search of Isaacs-lineage work on constrained pursuit would be prudent.
5. **Venue/year fields marked in BibTeX as preprints** (arXiv-only entries from 2026) may have since been accepted somewhere; I did not verify publication status for each. Check before the camera-ready.
6. **One automated-summary error was caught and corrected** (the SPARK paper was wrongly summarised as studying learned adversarial deadlock induction; direct PDF inspection found zero occurrences of "deadlock", "stall", or "freeze"). Other summaries in this document that were *not* backed by direct PDF inspection — chiefly the shorter Q2/Q3 entries — carry the same residual risk and should be spot-checked before any claim in the paper leans on them. Entries verified against the source PDF or publisher page are: Raven, SPARK, Gavin et al., Acharya et al., Wang et al.
