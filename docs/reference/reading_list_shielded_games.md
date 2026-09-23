# Reading List — Paper 2 "Shielded Games" (D86)

**Created**: 2026-09-03 (S120). **Scope**: what to read for the shielded-game study + safety-first system + two-robot validation.
**Companions**: `docs/reference/novelty_audit_2026-09-03.md` (threat analysis), `docs/literature-review/` (36 papers, Feb 2026), `docs/references/new_papers.md` (N01–N15), `docs/pathway/from_scratch_build_plan.md` (plan v3).

**Legend**: ✅ abstract/details verified this session · 🔎 identified, details unverified · 📕 paywalled (access note in §7) · 📗 open access · ⭐ already summarized in our Feb-2026 review.

**How to use**: Tier 0 decides what the paper can claim — read first, in order. Tier 1 is the related-work spine you must cite to survive review. Tier 2 is what you implement. Tier 3 is how you measure. Tier 4 is hardware. Tier 5 is re-reading our own library with the new question in mind.

---

## Tier 0 — Read first: these can change the claim (5 papers, ~1 week)

| # | Paper | Why it decides something | Status |
|---|---|---|---|
| T0.1 | **Safe exploration for pursuit-evasion games based on control barrier functions and reinforcement learning** — Control Engineering Practice, May 2026, `S0967066126000961` | **The single biggest threat.** Establishes a unified HOCBF PE-game framework and argues that CBF-based safe RL enforcing safety by "conservative control projection or policy shielding" converges to conservative safe equilibria, whereas embedding safety in the game preserves the original saddle point. If it contains deep-RL and/or partial-observability experiments, our R1 narrows to PO + hardware. If it is analytical/ADP + full-obs (likely), it becomes our **theoretical hook**: they predict, we measure. | 📕 (403 on fetch) |
| T0.2 | **CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions** — Yang, Werner, de Sa, Ames — **ICRA 2026**, arXiv 2510.14959 | Already does shield-in-training vs deploy-time filtering, single-agent, with a Unitree G1 hardware demo. **Our Step 7 is a replication of this**, so it must be cited as the method we extend, and our contribution must be the adversarial setting. Also gives the closed-form discrete-rollout safety-filter result we can reuse. | ✅ 📗 |
| T0.3 | **Reinforcement learning in pursuit-evasion differential game: safety, stability and robustness** — Wang, Zhang, Xu, Wang, Guay — **IEEE TNNLS 2026**, arXiv 2507.19516 (= our paper ⭐[01]) | Nearest "safe PE RL" neighbour: CBF + sliding-mode + Stackelberg hierarchy, zero-sum, simulation only. Needed to state our delta precisely (they hierarchically decouple safety; we measure what safety does to the learning game). | ✅ 📗 ⭐ |
| T0.4 | **Safe multi-agent reinforcement learning based on adversarial strategy and control barrier function feasible set** — Neurocomputing, Mar 2026, `S0925231226008854` | The other "adversarial + CBF feasible set" paper. Treats non-target agents as worst-case adversaries and casts safety as a **maximal feasible set** problem with a CBF Q-function. Closest prior art to our R2 (infeasibility under adversarial pressure) — must read to draw the line: their adversary attacks *safety*, ours is a task opponent exploiting the shield. | 📕 |
| T0.5 | **The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems** — Hsu, Hu, Fisac — Annual Review of Control, Robotics, and Autonomous Systems 7:47–72 (2024), arXiv 2309.05837 | The taxonomy your Methods section should speak in (CBF / HJ / MPC / predictive filters as one family). Cheapest way to make the framing look field-fluent and to place our projection shield precisely. | ✅ 📗 |

**Deliverable after Tier 0**: update `novelty_audit_2026-09-03.md` verdict + fix R1's scope in plan v3 §1, before Step 3b.

---

## Tier 1 — Related-work spine (cite or die)

### 1a. Shielding and safety filters as a field
| # | Paper | Relevance |
|---|---|---|
| T1.1 | **Safe Reinforcement Learning via Shielding** — Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu — AAAI 2018, arXiv 1708.08611 | The origin of "shielding"; our vocabulary comes from here. 🔎📗 |
| T1.2 | **Safe Multi-Agent Reinforcement Learning via Shielding** — ElSayed-Aly, Bharadwaj, Amato, Ehlers, Topcu, Feng — AAMAS 2021 | First multi-agent shielding; centralized/factored shields. 🔎 |
| T1.3 | **Shield Decentralization for Safe Multi-Agent Reinforcement Learning** — Melcer, Amato, Tripakis — NeurIPS 2022 | Decentralized shields; the MARL-shielding line we extend to a *competitive* game. 🔎📗 |
| T1.4 | **Shield Decomposition for Safe RL in General Partially Observable Multi-Agent Environments** — Melcer, Amato, Tripakis — RLC 2024 | PO + multi-agent shielding — closest formal analogue to our setting (discrete, not continuous control). 🔎📗 |
| T1.5 | **Safe Reinforcement Learning via Shielding under Partial Observability** — Carr, Jansen, Junges, Topcu — AAAI 37(12):14748–14756, 2023 | The PO-shielding reference; justifies our "shield sees true state, policy sees sensors" split as a deliberate design choice against a known alternative. ✅📗 |
| T1.6 | **Robust Shielding for Safe Reinforcement Learning** — Hamel-De le Court, Badings, Abate, Belardinelli, Fabiano — arXiv 2606.00270 (May 2026) | Shields for robust MDPs framed as an agent-vs-adversary game; sound + optimal. Adjacent framing to ours, discrete/LTL. ✅📗 |
| T1.7 | **Realizable Continuous-Space Shields for Safe Reinforcement Learning** — arXiv 2410.02038 | Continuous-space shielding — the bridge from LTL shields to our QP projection. 🔎📗 |

### 1b. Safety in games / adversarial safety
| # | Paper | Relevance |
|---|---|---|
| T1.8 | **Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination** — Hu, Nguyen, Fisac et al. — CoRL 2024 (PMLR), arXiv 2405.00846 | Safety filter *synthesized* by self-play against a virtual adversary; 36-D quadruped hardware. Their adversary is uncertainty; ours is a strategic opponent — the contrast is a paragraph in Related Work and a talking point in reviews. ✅📗 |
| T1.9 | **Synthesis and Deployment of Maximal Robust Control Barrier Functions through Adversarial Reinforcement Learning** — arXiv 2604.13192 (Apr 2026) | Isaacs-equation safety value as a robust discrete-time **Q-CBF**; removes explicit dynamics. Candidate upgrade for our robust-margin shield arm and a strong "state of the art in adversarial CBF synthesis" cite. ✅📗 |
| T1.10 | **MAGICS: Adversarial RL with Minimax Actors Guided by Implicit Critic Stackelberg** — arXiv 2409.13867 | Convergent minimax adversarial RL for robot safety; relevant to how we train the shield-attacker (Step 9) without cycling. 🔎📗 |
| T1.11 | **Safe reinforcement learning for discrete-time nonlinear zero-sum games with unknown state constraints and asymmetric input constraints** — Neurocomputing, Jul 2025, `S0925231225016984` | Zero-sum + neural CBF in the *reward* (value-iteration, critic-only). **Possibly the "other paper" you were thinking of.** Contrast: reward-augmentation vs our hard projection. 📕✅(abstract) |
| T1.12 | **Capture, Shield, or Neutralize: Engagement-Aware Pursuit-Evasion** — Acharya, Goyette, Ataei, Stoica, Dhiman, Khojasteh — arXiv 2607.10986 (Jul 2026) | Minimax receding-horizon PE with a DCBF inner-loop safety filter, model-based. The model-based counterpart to our learned system; also uses "shield" in the PE context (terminology collision worth knowing). ✅📗 |

### 1c. Safe MARL context
| # | Paper | Relevance |
|---|---|---|
| T1.13 | **A Survey of Safe RL and Constrained MDPs: Single-Agent and Multi-Agent Safety** — arXiv 2505.17342 (rev. Apr 2026) | Current umbrella survey; use for positioning and for its open-problem list. ✅📗 |
| T1.14 | **Multi-Agent Constrained Policy Optimisation (MACPO / MAPPO-Lagrangian)** — Gu et al. (⭐N03) | The Lagrangian baseline family for H5. Take the algorithm, not the cooperative setting. ⭐📗 |
| T1.15 | **Safe MARL through Decentralized Multiple CBFs (MADDPG-CBF)** — arXiv 2103.12553 | Early CBF+MARL; frequently cited, cheap to cite correctly. 🔎📗 |
| T1.16 | **Safe robust MARL with neural CBFs and safety attention (RMARL-CBF-SAM)** — Information Sciences 2024 (⭐N15) | H∞-style adversary + neural CBF; double-integrator, cooperative. ⭐📕 |

---

## Tier 2 — Methods you will implement

| # | Paper | What you take from it | Step |
|---|---|---|---|
| T2.1 | **Discrete Control Barrier Functions for Safety-Critical Control of Discrete Systems with Application to Bipedal Robot Navigation** — Agrawal & Sreenath — RSS 2017 (DOI 10.15607/RSS.2017.XIII.073) | **The correct citation for our γ formulation** h(x_{k+1}) ≥ (1−γ)h(x_k). Also flags that the general discrete-time case is an NLP, not a QP — our 2-D exactness (D85) is worth stating against that. ✅ | 6 |
| T2.2 | **Learning Feasibility Constraints for Multi-Agent CBF Controllers** — Xiao, Cassandras, Belta — ECC 2023 (⭐N13) | The R3 remedy: learn H(z)≥0, add to the QP, resample ×3; infeasibility 0.081→0.002 on the *same unicycle dynamics*. | 10 |
| T2.3 | **Dynamic Obstacle Avoidance for Car-like Mobile Robots via Neurodynamic Optimization with CBFs** — Zhang & Yang — Neurocomputing 654 (2025) (⭐N12) | Virtual-control-point CBF (uniform relative degree 1) + "steer, don't brake" weighting — already implemented in `pe2/shield.py`. Re-read only if the anisotropic weights need justification. | 6 |
| T2.4 | **How to Train Your Neural CBF (PNCBF)** — So, Serlin, Mann, Gonzales, Rutledge, Roy, Fan — ICRA 2024 (⭐N02) | If hand-designed h proves too conservative, this learns a valid CBF from a nominal policy and provably enlarges the safe set. Optional arm. | 10 |
| T2.5 | **Measurement-Robust Control Barrier Functions: Certainty in Safety with Uncertainty in State** — Cosner, Yue, Ames et al. — arXiv 2104.14030; and **Guaranteeing Safety of Learned Perception Modules via MR-CBFs** — Dean, Taylor et al. — CoRL 2020 (PMLR v155) | **Directly justifies the robust-margin shield arm** for hardware, where AMCL/lidar give an uncertain state. Gives a principled δ instead of a hand-tuned margin. ✅📗 | 6, 12a |
| T2.6 | **Robust CBF-QP + GP disturbance estimation** — Emam et al. (⭐[06]) | The GP residual + cold-start protocol already specced in `docs/phases/phase4_sim_to_real_transfer.md` §2.6–2.7. | 12a |
| T2.7 | **Deadlock Analysis and Resolution for Multi-Robot Systems** (WAFR 2020) + **The Before, During, and After of Multi-Robot Deadlock** (IJRR 2023) — Grover, Liu, Sycara | CBF controllers provably deadlock at force-equilibria. **This is the mechanism by which a shield could create standoffs in our game** — the "degenerate standoff" risk in the plan now has a citable theory and a diagnostic (KKT force balance). ✅📗 | 6, 8, 9 |
| T2.8 | **Learning CBFs and their Application in RL: A Survey** — Guerrier, Fouad, Beltrame — arXiv 2404.16879 (⭐N01) | Taxonomy for the CBF+RL half of Related Work. | writing |

---

## Tier 3 — Measurement, exploitability, and adversarial evaluation

| # | Paper | What you take from it | Step |
|---|---|---|---|
| T3.1 | **Approximate Exploitability: Learning a Best Response** — Timbers, Bard, Lockhart, Lanctot, Schmid, Burch, Schrittwieser, Hubert, Bowling — IJCAI 2022, pp. 3487–3493 | **The methodological citation for our NashConv-by-approximate-BR protocol**, including that it is a *lower bound*. Fixes the D83 weakness ("the gate certifies learned, not approximate BR") with a standard reference. ✅📗 | 5, 8, 9 |
| T3.2 | **Adversarial Policies: Attacking Deep RL** — Gleave, Dennis, Wild, Kant, Levine, Russell — ICLR 2020 | **The template for Step 9.** Adversarial policies in zero-sum robot games beat self-play-trained victims via natural observations, and win *without* playing well. Our shield-attacker is an adversarial policy whose objective is the victim's *safety filter* — a genuinely new target. ✅📗 | 9 |
| T3.3 | **Adversarial Policies Beat Superhuman Go AIs** — Wang, Gleave, Tseng et al. — ICML 2023 (oral); follow-up **Can Go AIs be adversarially robust?** — Tseng, McLean, Wang et al., arXiv 2406.12843 | >97% win rate vs superhuman KataGo; attacks are human-comprehensible; **the vulnerability persists after adversarial training.** Sets the expectation for our R3 remedy (do not overclaim a fix) and legitimizes "intelligible exploit" as a result format. ✅📗 | 9, 10 |
| T3.4 | **Neural Fictitious Self-Play** — Heinrich & Silver 2016; **PSRO** — Lanctot et al. NeurIPS 2017 | Only for the Related Work sentence explaining why we do *not* claim convergence (and paper 1's ∀/∃ result). Do not re-litigate. 🔎📗 | writing |
| T3.5 | **Robust Adversarial RL (RARL)** — Pinto, Davidson, Sukthankar, Gupta — ICML 2017 | The canonical "train against a learned adversary for robustness" cite; frames Step 10's remedy as adversarial training. 🔎📗 | 10 |
| T3.6 | **Safety-Gymnasium** — arXiv 2310.12567 | Benchmark conventions for reporting cost/violation vs return frontiers (our H5 figure). 🔎📗 | 7, 11 |

---

## Tier 4 — Hardware and sim-to-real (now in scope: 2× TurtleBot3 confirmed)

| # | Paper | What you take from it | Step |
|---|---|---|---|
| T4.1 | **Sim-to-Real Transfer for Mobile Robots: Isaac Sim → Gazebo → Real ROS2** — Salimpour et al. 2025 (⭐N10) | The exact pipeline our phase-4 spec is built on: ONNX export, ROS 2 inference node, ~1.6–5 ms on RPi4, 80–100% success, curriculum. | 12a–c |
| T4.2 | **Isaac Lab** — (⭐N11) | Only if we decide to re-train instead of using the similarity transform. Currently **not** needed (plan v3 assumption 2). | — |
| T4.3 | **Pursuit-Evasion for Car-like Robots with Sensor Constraints** — Gonultas & Isler — IROS 2025 (⭐[02]) | The system baseline: 1v1 PE, FOV constraints, BiMDN, real F1TENTH/JetRacer. Our hardware protocol should mirror theirs where possible so numbers are comparable. | 12c, writing |
| T4.4 | **MADR: MPC-guided Adversarial DeepReach** — Teoh et al. 2025 (⭐[22]) | Game-theoretically optimal PE **on TurtleBots** — the closest hardware precedent and a candidate real-robot baseline. Read the experimental protocol (arena, localization, engagement lengths). | 12c |
| T4.5 | **Vision-based PE with privileged teacher-student** — Bajcsy et al. — ICRA 2024 (⭐[08]) | Only for the hardware related-work paragraph (quadruped, camera-only). | writing |
| T4.6 | **Robust CBF under state uncertainty using extended state observer and output measurement** — arXiv 2308.13943 | Alternative to T2.5 if AMCL error dominates; less conservative bounds from output measurements only. 🔎📗 | 12a |

---

## Tier 5 — Re-read from our own library with the new question in mind

These are ⭐ already summarized in `docs/literature-review/`; the point is to re-read **asking "what does the shield do to learning?"**, which nobody asked in Feb 2026.

- **[16] CBF-constrained Beta policy** — Suttle et al., AISTATS 2024. Hard safety *inside* the policy with convergence guarantees — the third architecture between our two arms. Now a plausible fourth arm in Step 7.
- **[05] Closed-form CBF filter + reward shaping** — Yang et al. (Caltech) 2025. Predecessor of CBF-RL (T0.2); the "filter + shaping" dual is our R3 margin-shaping component.
- **[25] HJ Reachability in RL: A Survey** — Ganai et al. 2024. Names shielding (task policy + HJ backup) as the practical architecture; our alternative-shield arm if CBF conservatism becomes the story.
- **[13]/[14] Kokolakis & Vamvoudakis** — CDC 2022 + IEEE TNNLS 35:3130 (2024). Barrier-in-the-HJI-cost for PE, GPs, finite-time convergence, **bounded rationality**. The "safety inside the game objective" lineage that T0.1 belongs to.
- **[34] SHADOW** — La Gatta et al. 2025. 1v1 unicycle PE with information trade-offs; our closest sim sibling on dynamics + PO.
- **[37] RL for PE games: a 235-reference survey** — Yang et al., Chinese Journal of Aeronautics 2025. Use to check we are not missing a whole PE sub-literature.
- **Phase 2.5 decision report** (internal) — the BarrierNet failure (2% capture, 88% intervention) is now *evidence for the paper's thesis*, not just a dead end: it is our own measurement of a safety layer destroying learning.

---

## 6. Reading order (suggested, ~3 weeks part-time)

1. **Week 1 (blocking)**: T0.2 → T0.5 → T0.1 → T0.4 → T0.3. Then rewrite the audit verdict and R1 scope. Nothing in Step 3b or 8 should start before this.
2. **Week 2 (implementation)**: T2.1, T2.5, T2.7, T2.2 — these change `pe2/shield.py` and the Step 6 gate JSON (margins, infeasibility, deadlock diagnostics).
3. **Week 2–3 (measurement)**: T3.1, T3.2, T3.3 — these fix the Step 9 design and the exploitability prose.
4. **Week 3 (context, skim)**: T1.1–T1.7 for the shielding spine, T1.13 survey, T4.1/T4.4 for the hardware protocol.
5. **Ongoing**: Tier 5 re-reads while writing.

## 7. Access notes

- niro-2 (the paywalled-download machine, CLAUDE.md) has been **offline since 2026-07-14** — the three 📕 ScienceDirect items (T0.1, T0.4, T1.11) need another route: institutional login, `arxiv.org` preprint search under the same title/authors, direct author request by email, or an interlibrary request.
- **T0.1 is worth real effort** — it is the only item that can force a scope change. If it stays inaccessible, write the paper so the claim survives either way: cite it from the abstract for the *prediction*, and present our result as the first measurement, explicitly noting we could not verify their experimental scope.
- Everything marked 📗 is on arXiv, PMLR, OpenReview, or AAAI/IJCAI open proceedings.

## 8. What is deliberately NOT on this list

- Convergence-mechanism literature beyond one citation (NFSP/PSRO/league play): we are not claiming convergence (D86).
- Differentiable-QP / end-to-end safety layers beyond our own BarrierNet evidence (D31, L5.1).
- Population-based training and self-play stabilization tricks: paper 1's territory (D37–D47).
- Vision-based perception: out of scope (lidar + software cone mask).
