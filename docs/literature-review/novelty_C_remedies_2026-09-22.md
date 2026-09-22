# Novelty Check C — Remedies for Safety-Filter-Induced Stalling (2026-09-22)

Scope: adversarial literature search for prior work that could scoop three claims about
remedies for an RL agent permanently stopped by its own hard safety filter:

1. **Escape/recovery via candidate-action enumeration** drives permanent-stop rate to ~0, raises task success, no retraining, no loss of safety guarantee.
2. **Penalizing the stall backfires** — reward hacking, agent learns to stop asking to move.
3. **Exposing the filter's admissible action set as an observation** helps task success but does NOT fix the permanent-stop rate; neither does a longer discount horizon.

Areas covered: C1 (recovery/fallback/escape mechanisms for shielded agents) and C2 (reward
hacking of inaction penalties; feasibility/admissible-action info as policy input).

---

## C1 — Recovery, fallback, and escape mechanisms for safety-filtered agents

### Dynamic Model Predictive Shielding (DMPS)
Arko Banerjee, Kia Rahmani, Joydeep Biswas, Isil Dillig. NeurIPS 2024. arXiv:2405.13863.
https://arxiv.org/abs/2405.13863
Classical Model Predictive Shielding (MPS) reverts to a static, task-oblivious "backup policy"
(often just halting) whenever the learned policy proposes a risky action; the paper's own
motivating example is a navigation agent that "instructs halting, rather than trying to find a
more optimal recovery approach." DMPS replaces the fixed backup policy with a local planner
that uses Monte Carlo Tree Search over candidate recovery-action sequences, selecting the one
that maximizes finite-horizon progress plus the policy's learned long-term value, while
retaining MPS's formal safety guarantee.
**Relation to claims: THREATENS CLAIM 1 (near-scoop).** Same problem framing (static backup ⇒
halting; formal safety must be preserved) and the same high-level fix (search over candidate
recovery actions instead of a fixed stop). Differs from claim 1's mechanism in three ways: (a)
the search is MCTS/value-guided rather than a short fixed priority list (full speed → half
speed → rotations); (b) it is triggered reactively on every unsafe-action proposal, not after
N steps of measured immobility; (c) it does not report a "permanent-stop rate" metric driven to
~0 — it reports reduced shield-intervention frequency over training. This is the single closest
paper found for claim 1 and should be discussed/distinguished explicitly.

### Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in RL
Georgescu, Goodall, Alrajeh, Belardinelli, Uchitel. NeSy 2026 (accepted). arXiv:2511.02605.
https://arxiv.org/abs/2511.02605
Static shields built from GR(1)/LTL specifications can leave an agent permanently stuck when
environment assumptions are violated at runtime. The paper uses Inductive Logic Programming to
repair the specification online (minimally weakening goals only when necessary) so the shield
adapts and liveness is restored, tested on Minepump and Atari Seaquest.
**Relation to claims: THREATENS CLAIM 1 (near-scoop, different mechanism).** Same top-level
problem (shield ⇒ agent permanently stuck) and same top-level promise (fix without losing the
safety guarantee), but the mechanism is formal-methods specification repair, not candidate
physical-action enumeration, and no reward penalty / reward-hacking analysis (claim 2) or
observation-augmentation ablation (claim 3) is present.

### Policy Library CBF (PL-CBF)
Taekyung Kim, Hideki Okamoto, Bardh Hoxha, Georgios Fainekos, Dimitra Panagou. arXiv, 2026
(2605.16588). https://arxiv.org/html/2605.16588
A single fixed fallback maneuver in a CBF safety filter becomes infeasible as constraints
change (e.g., sudden friction loss, new obstacle). PL-CBF instead maintains a library of
diverse fallback policies (stopping, evasive maneuvers, nominal control), evaluates all of them
in parallel via finite-horizon rollouts, and certifies safety whenever any library member
remains feasible, switching to the least-restrictive option. Tested on double integrator,
highway driving, and 3D quadrotor navigation; 0% failure on a scenario where single-fallback
baselines fail 22–80%.
**Relation to claims: adjacent to claim 1, weaker overlap.** Conceptually the same idea as
"enumerate several candidate maneuvers and pick one that works," but it operates at the level
of *filter design* (which backup controller certifies the QP) in general safety-critical
control, not specifically as a remedy for an RL agent that has already been driven to a
standstill by its own filter; no permanent-stop / task-success metric, no RL training, no
reward-hacking or observation-augmentation discussion.

### Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones
Brijen Thananjeyan et al. arXiv:2010.15920 (2020/2021), IEEE RA-L.
https://arxiv.org/pdf/2010.15920
Foundational two-policy architecture: a task policy and a learned recovery policy that takes
over when the agent nears a constraint-violation zone, driving it back to a safe region.
**Relation to claims: background only, no scoop.** Establishes the general "recovery policy"
category claim 1 sits inside, but recovery is a *learned* policy (requires training), not a
retraining-free candidate-action-enumeration fallback, and the paper does not study permanent
immobility, reward hacking of stall penalties, or admissible-set observations.

### SALT — Robots that Suggest Safe Alternatives
Hyun Joe Jeong, Rosy Chen, Andrea Bajcsy. arXiv:2409.09883 (v2 March 2025).
https://arxiv.org/abs/2409.09883
When a goal-conditioned policy cannot safely reach a requested goal, SALT uses offline
Hamilton-Jacobi reachability to suggest the *most similar alternative goal* that is safely
reachable, rather than trying to execute the original one.
**Relation to claims: adjacent, not a scoop.** The "offer an alternative" idea rhymes with
claim 1's "offer candidate motions," but SALT substitutes goals, not low-level motor commands,
and does not address an agent stuck at a standstill or measure permanent-stop rate.

### LiveNet
Srikar Gouru, Siddharth Lakkoju, Rohan Chandra (UVA). arXiv:2412.04659 (Dec 2024).
https://arxiv.org/pdf/2412.04659
Learned, minimally-invasive velocity modulation with differentiable CBFs for deadlock-free
multi-robot navigation through doorways/intersections; prevents deadlock proactively rather
than recovering from it.
**Relation to claims: not a scoop.** Proactive avoidance, not a remedy applied after the agent
is already stopped; no candidate-action enumeration, no reward-penalty or observation study.

### Reducing Safety Interventions in Provably Safe RL
Jakob Thumm, Guillaume Pelat, Matthias Althoff. arXiv:2303.03339 (2023).
https://arxiv.org/abs/2303.03339
Proposes "proactive replacement" and "proactive projection," which modify the RL agent's
proposed action *before* the failsafe controller would need to intervene, reducing the number
of jarring failsafe interventions (evaluated on Safety Gym and a human-robot task).
**Relation to claims: adjacent, not a scoop.** Reduces the frequency of interventions but is
proactive/preventive, not a reactive escape mechanism for an agent already immobilized; no
permanent-stop metric, no reward-hacking or observation-augmentation study.

### Classical robotics practice — ROS navigation recovery behaviors
`rotate_recovery`, `clear_costmap_recovery` (ROS Navigation stack, ~2010s; see e.g.
https://medium.com/@patiladitya1309/recovery-behaviours-used-in-navigation-stack-ros-5df991f408c7,
http://library.isr.ist.utl.pt/docs/roswiki/rotate_recovery.html).
When `move_base` detects the robot is stuck (oscillating / no valid plan), it runs an escalating
sequence of recovery behaviors: clear local costmap → rotate 360° in place → clear costmap
aggressively → rotate again → give up.
**Relation to claims: CLEAN NEGATIVE, but flag as prior art in spirit for claim 1.** This is
the same *behavioral pattern* as claim 1 (escalating candidate maneuvers tried until one
succeeds), but it is (a) pre-2024 classical robotics engineering, not an RL/safety-filter paper,
(b) not gated by a formal hard-safety-guarantee filter in the sense used here, and (c) not
accompanied by any measurement of permanent-stop rate, task success, or a comparison against a
reward-penalty alternative. Worth citing as related engineering practice / motivation, not as a
scientific scoop.

---

## C2 — Reward hacking of stall/inaction penalties; admissible-action-set as observation

### Safety Filtering While Training
Federico Pizarro Bejarano, Lukas Brunke, Angela P. Schoellig. IEEE RA-L, Jan 2025.
arXiv:2410.11671. https://arxiv.org/abs/2410.11671 (full text read via
https://www.dynsyslab.org/wp-content/papercite-data/pdf/pizarro-ral25.pdf)
Studies penalizing the *magnitude of the safety filter's correction* `‖u_uncert − u_cert‖²`
(weight α) during RL training of a quadrotor trajectory-tracking controller, versus a constant
constraint-violation penalty β. Ablates α ∈ {0.1,1,10,100} and β ∈ {0,0.01,0.1,1} on simulated
and real Crazyflie 2.0 flight.
**Relation to claims: IMPORTANT CONTRASTING EVIDENCE for claim 2 (near-miss, not a scoop).**
This paper penalizes almost exactly the quantity claim 2 penalizes ("how much the agent wanted
an unsafe/infeasible action"), yet finds the *opposite* qualitative result within its tested
range: increasing the correction penalty α monotonically reduces chattering and does not
collapse return or increase immobility — return is *maximized* at α=1 and stays comparable up
to α=100, with no report of the agent "learning to stop asking to move." The likely reason the
result diverges from claim 2 is structural: their reward is continuous proximity to a moving
reference trajectory, so "doing nothing" is never a rewarding local optimum the way it can be
in a stop/immobility-prone pursuit or navigation task. This paper is worth citing explicitly to
pre-empt a reviewer who might otherwise cite it as a counterexample — the divergence should be
explained by task structure (tracking a moving reference vs. discrete stall-prone navigation),
not treated as a refutation.

### The "freezing robot problem" (general phenomenon)
Originates with Trautman & Krause (2010) for dense-crowd navigation; recent instances:
- "Don't Freeze, Don't Crash: Extending the Safe Operating Range of Neural Navigation in Dense
  Crowds," Jiefu Zhang, Yang Xu, Vaneet Aggarwal (Purdue). arXiv:2603.06729 (March 2026).
  https://arxiv.org/html/2603.06729 — addresses ORCA-style analytical filters halting entirely
  when the collision-free velocity set vanishes; fixes it via density-invariant observation
  encoding + potential-based proxemic reward shaping (not via penalizing/un-penalizing stalls,
  and not via candidate-action enumeration or admissible-set observation).
**Relation to claims: background confirmation of claim 2's premise, not a scoop.** Confirms the
general "heavy safety pressure ⇒ agent freezes" phenomenon is well recognized and actively
studied in 2026, but neither paper isolates and ablates a stall-specific reward penalty the way
claim 2 does, nor reports the specific "10x penalty halves the stall metric while nearly
doubling immobility" reward-hacking signature.

### Reward hacking / "agent learns to do nothing" (generic AI-safety background)
Lilian Weng, "Reward Hacking in Reinforcement Learning," Nov 2024,
https://lilianweng.github.io/posts/2024-11-28-reward-hacking/; Amodei et al., "Concrete
Problems in AI Safety," arXiv:1606.06565 (cleaning-robot-closes-its-eyes example).
**Relation to claims: generic background, not a scoop.** These establish that "penalize a
proxy for danger ⇒ agent avoids the proxy by doing nothing" is a known general reward-hacking
pattern, but neither source studies a hard safety-filter setting, a stall-specific penalty, or
reports quantitative before/after metrics resembling claim 2's numbers.

### Action Mapping for RL in Continuous Environments with Constraints
Mirco Theile, Lukas Dirnberger, Raphael Trumpp, Marco Caccamo, Alberto Sangiovanni-Vincentelli.
arXiv:2412.04327 (Dec 2024). https://arxiv.org/abs/2412.04327
Pretrains a "feasibility policy" that generates the feasible-action set from a feasibility
model, decoupling feasible-action generation from the task-objective policy, which then selects
only among the reduced feasible set. Improves sample efficiency/training performance in
constrained continuous-action environments, especially with imperfect feasibility models.
**Relation to claims: adjacent to claim 3, not a scoop.** Shares the motivation "give the
policy explicit knowledge of what's currently feasible," but the mechanism is architectural
action-space reduction (the policy only ever proposes feasible actions), not appending the
admissible set as an extra observation input to a policy that still outputs into the full
action space through a separate filter. No discount-horizon ablation, no permanent-stop metric.

### Feasibility Consistent Safe RL (FCSRL)
Zhepeng Cen, Yihang Yao, Zuxin Liu, Ding Zhao (CMU). ICML 2024. arXiv:2405.11718.
https://arxiv.org/html/2405.11718v1
Uses a "feasibility score" (max discounted cost along a trajectory) as an auxiliary
self-supervised representation-learning signal, improving the state embedding fed to both
policy and value networks; not feeding the admissible action set itself as an observation.
**Relation to claims: adjacent, not a scoop.** Different mechanism (representation learning
target vs. raw admissible-set observation) and no permanent-stop or reward-hacking analysis.

### Dynamic/constraint-aware action masking literature
E.g., "Imperative Action Masking for Safe Exploration in RL" (dl.acm.org/10.1007/978-3-031-40878-6_8);
Masked Constrained Policy Optimization, AAMAS 2021 (ifaamas.org/Proceedings/aamas2021/pdfs/p1344.pdf).
These mask out unsafe actions from the sampling distribution directly (the agent literally
cannot select a masked action), reporting faster convergence and improved safety.
**Relation to claims: distinguishable from claim 3, not a scoop.** Action masking constrains
*which action can be taken*; claim 3 is about exposing the admissible set as *information* in
the observation while a separate downstream filter still enforces safety — and claim 3's
finding is explicitly that this does NOT fix the permanent-stop rate. Because masking papers
structurally prevent infeasible actions from ever being chosen, they are not positioned to
observe (or report) the specific negative result claim 3 reports (task success up, stall
unchanged), and none of the papers found ablate discount-horizon length as an alternative fix.

### "Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL"
arXiv:2609.20715 (Sept 2026). https://arxiv.org/html/2609.20715
Superficially keyword-matches ("mask," "observation," "explore," "RL") but is unrelated: it is
about LLM-agent SFT/GRPO training on terminal/coding benchmarks (Terminal-Bench 2.0,
aider-polyglot), using environment-observation prediction as an auxiliary training signal — not
about safety filters, admissible action sets, or physical/robotic agents at all.
**Relation to claims: CLEAN NEGATIVE (false-positive keyword match).**

---

## Clean negatives (searched, found not to threaten any claim)

- Recovery RL (Thananjeyan et al. 2020) — general recovery-policy background, pre-2024, no
  permanent-stop/reward-hacking/observation study.
- SafeExplorer / MSMAR-RL (2025/2026 recovery-RL variants) — recovery policies for physical
  falls/failures during training, not for a filter-induced standstill.
- CBF-RL (Yang, Werner, de Sa, Ames, arXiv:2510.14959) — penalizes filter-correction magnitude
  as a reward-shaping term to internalize safety; does not discuss or ablate passivity/inaction
  as a failure mode.
- High-Speed Vision-Based Flight in Clutter with Safety-Shielded RL (Zhang et al., RA-L 2026,
  arXiv:2602.08653) — search snippet suggested the shielded quadrotor "reaches a dead-end where
  it remains in hover," but the full text contains no discussion of this as a named failure mode
  and proposes no recovery/escape mechanism or reward-penalty ablation.
- "Mo' States Mo' Problems: Emergency Stop Mechanisms from Observation" (arXiv:1912.01649) —
  learns *when* to trigger an e-stop from demonstrations; unrelated to penalizing stalls or
  admissible-set observations.
- Discount-horizon-length literature (offline RL conservatism papers, e.g. arXiv:2512.04341) —
  discusses horizon length and conservatism generically but not in a safety-filter/stall context
  matching claim 3's specific ablation.
- ROS navigation recovery behaviors — see C1 section above; genuine prior art in spirit but not
  a scientific/empirical scoop of claim 1.

---

## Bottom line

No paper found combines all three claims, or even two of them, in one study. The closest
individual threats are narrow and mechanism-distinct:
- **Claim 1** is closest to being scooped in spirit by DMPS (NeurIPS 2024) and the GR(1)
  spec-repair paper (NeSy 2026), both of which fix "shield ⇒ permanent halt" without losing the
  safety guarantee — but via MCTS/value-guided search or formal specification repair, not via a
  short fixed list of candidate motion primitives filtered until one passes.
- **Claim 2** has no direct empirical match; the closest paper (Pizarro Bejarano et al., RA-L
  2025) penalizes almost the same quantity and finds the opposite result, which should be
  explained rather than ignored.
- **Claim 3** has no direct match; nearby work either changes the action space itself (action
  mapping, action masking) rather than adding an observation, or uses feasibility as a
  representation-learning signal rather than a raw input.
