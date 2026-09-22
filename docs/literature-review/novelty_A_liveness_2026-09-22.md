# Novelty Check A: Liveness/Freezing Under Safety Filters — Search Log (2026-09-22)

## The claim being checked (for reference)
An RL policy runs behind a hard safety filter (CBF-QP, or any minimal-intervention filter that
returns the admissible action nearest the commanded one). The filter stops the agent and it
**never restarts**: immobile forever, the filter's guarantee holds perfectly (zero violations,
ever), and the task silently fails. Escaping requires a multi-step manoeuvre (turn, then move)
that the filter's nearest-admissible objective never proposes and the policy never discovers.

---

## A1. Liveness / deadlock / freezing of a SINGLE agent under a safety filter or shield

### 1. BIG-CBF: Behavior-Imagination-Guided Control Barrier Function with Shared Uncertainty for Mobile Robot Navigation
- **Authors**: Shibo Li, Zhongcheng Wang, Jiahe Cao, Jianhua Yang, Ke Wu (Northwestern Polytechnical University, MBZUAI)
- **Year/Venue**: arXiv:2609.14343, submitted 13 Sep 2026 (preprint)
- **URL**: https://arxiv.org/abs/2609.14343
- **What it does**: Names and formalizes exactly our mechanism for a single robot: "a minimum-intervention CBF filter lacks task-level maneuver awareness and may fail to select a productive avoidance direction when multiple distinct maneuvers are locally viable, leading to safe but stalled behavior in geometrically ambiguous environments." Their fix is a two-rate architecture — a low-rate "behavior imagination" layer that evaluates six closed-loop maneuver modes (GOAL, PASS-L/R, SIDE-L/R, YIELD) over a 1.5 s horizon, accounting explicitly for "task progress, freezing, smoothness, and switching," with a hard CBF still the final safety authority. 3,600-episode benchmark: 99.78% success, reduced CBF intervention; 15/15 contact-free physical trials.
- **Relation to our claim**: **Closest match found.** Same phenomenon ("safe yet stalled"), same root cause (nearest-admissible objective can't select a productive multi-step turn-then-move maneuver), same single-agent framing. Differs critically: (a) **no RL** — the underlying commanded controller is a generic nominal/planner controller, not a learned policy; (b) the paper's contribution is a *fix* (an architecture that prevents freezing), not a diagnosis of an RL policy's silent failure; (c) it does not claim the robot is stuck "forever" as an emergent RL training/deployment pathology — freezing is treated as an architectural design flaw to engineer around, not a behavior an RL agent exhibits and fails to learn its way out of.

### 2. Preventing Inactive CBF Safety Filters Caused by Invalid Relative Degree Assumptions
- **Authors**: Lukas Brunke, Siqi Zhou, Angela P. Schoellig
- **Year/Venue**: arXiv:2409.11171 (2024)
- **URL**: https://arxiv.org/abs/2409.11171
- **What it does**: Identifies a failure mode where a CBF safety filter becomes permanently inactive/infeasible because relative-degree assumptions about the underlying system are violated, immobilizing the controller rather than merely allowing a violation. Proposes filter modifications so the constraint stays well-posed.
- **Relation to our claim**: A genuine liveness failure of a single agent under a CBF filter, but the mechanism is a **modeling/relative-degree bug**, not a geometric multi-step-maneuver blind spot; no RL; the fix is a better-posed filter, not a study of what an RL policy does (or fails to do) behind it.

### 3. On Safety and Liveness Filtering Using Hamilton-Jacobi Reachability Analysis
- **Authors**: Javier Borquez, Kaustav Chakraborty, Hao Wang, Somil Bansal
- **Year/Venue**: IEEE Transactions on Robotics, 2024 (arXiv:2312.15347, first posted Dec 2023, revised Aug 2024)
- **URL**: https://arxiv.org/abs/2312.15347 / https://dl.acm.org/doi/abs/10.1109/TRO.2024.3454470
- **What it does**: Foundational unifying HJ-reachability framework that constructs both *safety* filters and *liveness* filters (each minimally modifying a nominal controller within the maximal admissible input set), applied to navigation and rocket landing on hardware. Liveness here means guaranteeing the system *can* reach a goal set, as a formal dual to the safety guarantee.
- **Relation to our claim**: Establishes "liveness filtering" as a studied formal concept for single agents, which is adjacent territory, but it is classical optimal control/reachability (no RL), and the paper is about *constructing* filters that provably preserve liveness by design, not about diagnosing an RL policy that silently fails to discover a multi-step escape behind an already-deployed filter.

### 4. Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in Reinforcement Learning
- **Authors**: Tiberiu-Andrei Georgescu, Alexander W. Goodall, Dalal Alrajeh, Francesco Belardinelli, Sebastian Uchitel
- **Year/Venue**: arXiv:2511.02605, submitted 4 Nov 2025
- **URL**: https://arxiv.org/abs/2511.02605
- **What it does**: Addresses shielded RL where a static logical (GR(1)/LTL) shield can trap the agent when runtime environment assumptions are violated, preventing goal-reaching. Proposes online Inductive-Logic-Programming repair of the GR(1) spec to restore liveness while keeping safety, tested on Minepump and Atari Seaquest.
- **Relation to our claim**: **Second-closest match, and the only RL + liveness-preserving-shield paper found.** It explicitly frames the shield-traps-the-agent problem for an RL policy and requires "sequences of steps beyond what a greedy nearest-safe-action policy would propose" to escape — structurally the same shape as our claim. Differs on: (a) the shield is a **discrete symbolic/LTL automaton**, not a continuous CBF-QP nearest-admissible-action filter; (b) the trigger is **violation of an environment assumption** (a changing world), not a fixed geometric configuration that a static filter's objective is blind to; (c) domains are Minepump/Atari, not continuous single/multi-agent physical navigation or pursuit-evasion; (d) it is not framed as "zero violations forever, guarantee holds perfectly, task silently fails" — it is framed as "static shields can make liveness unreachable, so make the shield adaptive."

### 5. Escape-Aware Control Barrier Functions for Quadrotor Safety under Body-Rate Limits
- **Authors**: Lei Shi, Haosong Wen, Qichao Liu
- **Year/Venue**: arXiv:2609.17292, Sept 2026
- **URL**: https://arxiv.org/abs/2609.17292
- **What it does**: Proves that state-only CBF certificates can certify states from which no body-rate-feasible escape maneuver exists (a "certification gap") for quadrotors that can't instantaneously reorient thrust; proposes an escape-aware, horizon-embedded certificate.
- **Relation to our claim**: Conceptually adjacent ("the filter may certify a state from which escape is impossible / requires a maneuver the filter doesn't see"), but purely control-theoretic (no RL), single-vehicle quadrotor dynamics, and about actuator/body-rate constraints rather than geometric ambiguity needing a discrete turn-then-move plan.

### 6. Cheng, Orosz, Murray, Burdick — End-to-End Safe RL through Barrier Functions (older, foundational, flagged for completeness)
- **Note**: Multiple recent search results reference the well-known finding that a CBF-QP filter "can stop the agent at the border of the safety zone where the agent gets stuck" (RSS/AAAI-era CBF-RL literature, ~2019-2020, e.g. Cheng et al. AAAI 2019 and the Berkeley Hybrid Robotics RSS 2020 RL-CBF-CLF line). This is older than the 2024-2026 window but establishes that "RL + CBF-QP → stuck at boundary" is not itself a new observation in the field.
- **Relation to our claim**: Prior awareness of "gets stuck at the boundary" exists, but none of the exposed text frames this as a *permanent, guarantee-holding, silently-failing* deadlock requiring a *multi-step* manoeuvre — it reads as a known nuisance addressed by tuning/annealing, not a headline liveness result.

### 7. Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control
- **Year/Venue**: arXiv:2601.10233 (2026)
- **URL**: https://arxiv.org/pdf/2601.10233
- Surfaced by search but not deeply verified; addresses local minima in reactive safe controllers generally (same neighborhood as BIG-CBF). Not RL-specific per title. Flagged as adjacent but not fetched in full — recommend a follow-up read if scoping this exact corner further.

---

## A2. Safety-vs-task-performance under filters, measured

### 1. Evaluation Metrics for Safe Reinforcement Learning
- **Authors**: Lindsay Spoor, Aske Plaat, Thomas Moerland
- **Year/Venue**: arXiv:2609.15315, submitted 14 Sep 2026
- **URL**: https://arxiv.org/abs/2609.15315
- **What it does**: Argues current safe-RL evaluation (mean cost vs. a threshold) hides how often/severely the safety bound is violated and whether training-time behavior matches the converged policy; proposes an aggregate + distributional + task-specific reporting protocol, a "safety tier" system, and (per related abstracts in the same line) metrics like Expected Maximum Consecutive Cost (EMCC) that distinguish prolonged from occasional violations. Ships `SafeRLEval` toolkit.
- **Relation to our claim**: Directly in "safety metrics alone are insufficient" territory, but its insufficiency argument is about violation *severity/consistency*, not about distinguishing "conservative/slower" from "permanently stopped/task fails" — it does not appear to define a liveness/progress metric or discuss agents that freeze. Worth citing as the state of the art in safe-RL evaluation critique, but does not scoop the liveness angle.

### 2. Provably Optimal Reinforcement Learning under Safety Filtering
- **Authors**: Donggeon David Oh, Duy P. Nguyen, Haimin Hu, Jaime F. Fisac
- **Year/Venue**: arXiv:2510.18082; accepted IASEAI 2026
- **URL**: https://arxiv.org/abs/2510.18082
- **What it does**: Proves that if a safety filter is "sufficiently permissive" (formalized via a safety-critical MDP), enforcing it does not degrade the asymptotic optimal return — i.e., argues there is *no inherent* safety/performance tradeoff under the right filter design.
- **Relation to our claim**: This is a **potential threat to the framing**, not to the specific liveness claim: it argues theoretically that a well-designed filter costs nothing asymptotically. Our claim depends on the filter being a *minimal-intervention, nearest-admissible-action* filter that is (by construction) not permissive in the sense this paper requires for its "no tradeoff" result to hold — worth checking their permissiveness assumption carefully, since a reviewer could cite this paper to ask "why isn't your filter permissive?" It does not discuss liveness, stuck states, or multi-step maneuvers.

### 3. Guided by Guardrails: Control Barrier Functions as Safety Instructors for Robotic Learning
- **Authors**: Maeva Guerrier, Karthik Soma, Hassan Fouad, Giovanni Beltrame
- **Year/Venue**: arXiv:2505.18858 (May 2025)
- **URL**: https://arxiv.org/abs/2505.18858
- **What it does**: Compares three CBF-RL integration strategies (Filter, Decay, Reward). Shows the plain "CBF Filter" approach lets the RL agent reach the goal while the filter is active but the agent never internalizes obstacle avoidance — when the filter is removed it collides, achieving lower reward. Proposes CBF Decay (anneal filter strength as a curriculum) to fix this.
- **Relation to our claim**: Measures a safety-vs-performance cost, but the cost is "unsafe once the filter is removed," not "permanently immobile while the filter stays on." Different failure mode from ours (unlearned safety vs. undiscovered escape maneuver); does not discuss freezing/liveness.

### 4. CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions
- **Year/Venue**: arXiv:2510.14959 (2025)
- **URL**: https://arxiv.org/abs/2510.14959
- **What it does**: Uses a CBF filter only during training so the RL policy internalizes safety and can run filter-free at deployment (>90% success without runtime filter). Motivated partly by the same "policy doesn't know about the CBF → conservative/degraded behavior" concern.
- **Relation to our claim**: Addresses conservatism/performance cost, not liveness/freezing; doesn't discuss permanent immobility.

### 5. Silent Failures in Physical AI: A Literature Review of Runtime Action Authorization for Autonomous Systems
- **Author**: Barak Or
- **Year/Venue**: arXiv:2606.00090 (2026)
- **URL**: https://arxiv.org/abs/2606.00090
- **What it does**: A literature-review taxonomy of "runtime action authorization" guardrails across manipulators/mobile/legged/aerial robots. Defines "silent physical-action failure" as an autonomy stack **authorizing and executing an unsafe action** because the world-state estimate was stale/occluded/corrupted (the guardrail's evidence was wrong), not caught until a hardware controller detects the violation.
- **Relation to our claim**: **False-friend / name collision only.** Their "silent failure" is the *opposite* direction from ours — a filter silently *letting an unsafe action through* due to bad state evidence, versus our filter silently *freezing task progress* while remaining perfectly safe. Worth a footnote to disambiguate terminology if this term is used in our writeup, but it does not touch liveness/freezing at all. Clean negative for the liveness question.

### 6. Optimal Runtime Assurance via Reinforcement Learning (ICCPS 2024)
- **Venue**: ICCPS 2024 (Illinois RTARL group)
- **URL**: https://mitras.ece.illinois.edu/research/2024/RTARL_ICCPS24.pdf
- **What it does**: Per its listing, investigates the effect of Runtime Assurance (RTA) on RL training in terms of training efficiency, safety-constraint satisfaction, control efficiency, task efficiency, and training duration — i.e., a multi-axis measurement of RTA's cost to RL performance.
- **Relation to our claim**: Could not extract full text (malformed PDF in fetch); flagged as a strong A2 candidate to re-check manually — it appears to measure exactly the "task efficiency cost of the filter" axis, but there is no indication from available metadata that it separates "slower" from "permanently stuck."

---

## Clean negatives (search directions that returned nothing relevant)

- `"freezing robot problem" control barrier function` — no paper using this exact named term in combination with an RL policy; the closest hits (BIG-CBF) describe the same phenomenon without using this label, and use it for a generic nominal controller, not RL.
- `safety filter reinforcement learning agent gets stuck deadlock CBF QP` — returned general QP-infeasibility-causes-deadlock commentary, no paper centered on an RL policy specifically failing to discover a multi-step escape.
- `safe RL "task success rate" drop safety filter conservative behavior benchmark` — returned only general shielding-benchmark claims ("orders-of-magnitude reduction in violations," "shields prevent deadlock by construction" for *proper* shields) — no paper isolating "permanent stop" as a distinct, measured failure category from "conservative."
- `"never restarts" OR "permanently stuck" reinforcement learning safety filter zero violations task failure` — no direct hits; returned adjacent zero-violation-achieved papers (Koopman-CBF, Safety Gymnasium) that report success, not freezing.
- `multi-step maneuver escape local minimum control barrier function nearest safe action myopic` — surfaced only the general "CBFs are myopic" critique and MPC/MPPI-integration fixes, none framed around an RL policy failing to find the maneuver.

---

## Bottom line for the writeup
No exact scoop was found: nobody combines (a) an RL policy, (b) a hard minimum-intervention/CBF-QP-style filter, (c) permanent single-agent freezing with the safety guarantee holding perfectly and zero violations ever, and (d) the diagnosis that escape needs an undiscovered multi-step (turn-then-move) manoeuvre, framed as a *silent task failure*. The two papers that come closest each miss on a different axis: BIG-CBF (2609.14343) has the mechanism exactly right but no RL and frames it as an engineering fix, not a diagnosed RL failure mode; the GR(1) adaptive-shielding paper (2511.02605) has the RL + liveness-trap framing right but uses a discrete logical shield in a different problem class, triggered by assumption violation rather than static geometric ambiguity.
