# Novelty Check B — Absorbing/Frozen-Observation Self-Loops & Delayed-Escape Credit Assignment

**Date**: 2026-09-22
**Scope**: Literature search (WebSearch/WebFetch only, no code review) against the claim:

> A deterministic, memoryless RL policy enters a self-perpetuating loop: its action is blocked by an
> external mechanism, so the agent stops moving, so its observation stops changing, so it re-emits the
> same blocked action forever. Measured: observable world changes by exactly zero, opponent visible on
> only 2.4% of such steps, emitted action's median change is zero even under per-step re-decision.
> Exploration does not break it (stochastic sampling at converged noise = identical immobility).
> Entropy collapse excluded (state-independent noise by architecture). Escape unlearned because its
> payoff arrives ~629 steps later, discounted to ~0.009 at gamma=0.99 (gamma~0.9974 needed).

Two sub-areas: **B1** (absorbing/self-perpetuating states from frozen observations, livelock, shield-induced
loops, action masking repetition) and **B2** (discount-horizon limits, credit assignment for rare long-delayed
escapes).

---

## B1 — Absorbing / self-perpetuating states from frozen observations

### 1. Kim, Corsi, Rodríguez, Lanier, Parellada, Baldi, Sánchez, Fox — "Realizable Continuous-Space Shields for Safe Reinforcement Learning"
- **Year/venue**: arXiv:2410.02038v2, posted Dec 2, 2024 (preprint; associated with AAMAS-style shielding line of work).
- **URL**: https://arxiv.org/html/2410.02038
- **What it does**: Proposes continuous-space safety shields for RL that guarantee collision avoidance. Reports that collision-avoidance-only shielding produces "overly conservative behavior, often characterized by repetitive or looping actions" — success rate dropped (e.g. 0.40 without vs measurably worse with naive shielding in their Table 2) because agents get stuck rather than reaching goals. They fix it with a **non-Markovian loop-avoidance constraint**: a queue of recently visited discretized (pose, action) pairs that forbids repeating the same pair within a window `L_Q`, explicitly to "encourage the agent to explore new actions and potentially escape from being stuck."
- **Relation to claim**: **Closest match found.** It documents the same qualitative failure mode (shield blocks action → agent loops/repeats) and treats it as a known, nameable pathology requiring a dedicated fix. However, it does NOT perform the mechanistic analysis in the claim — no measurement of observation freezing, no quantification of "world changes by exactly zero," no opponent-visibility statistic, no isolation of exploration vs. entropy-collapse vs. discount-horizon causes, and no adversarial (pursuit-evasion) setting. Their diagnosis stays at "repetitive/looping actions reduce success rate"; the claim's contribution is the causal chain (frozen observation → deterministic re-emission → provably zero exploration benefit → discount-horizon explanation) plus hard measurements. Treat as a **near-scoop**, not a scoop.

### 2. "One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments"
- **Year/venue**: arXiv:2412.09989, Dec 2024.
- **URL**: https://arxiv.org/html/2412.09989
- **What it does**: Studies a safety filter (Backward Reachable Tube-based) wrapped around an a priori unknown locomotion policy; notes that in cluttered/walled environments overly conservative BRTs "severely inhibit progress" and can manifest as **stalling behavior**, and discusses mitigations (training with wall-like obstacles, observation-conditioned calibration).
- **Relation to claim**: Independently confirms that safety filters/shields on learned locomotion policies can cause stalling in obstacle-dense settings, but it is an engineering mitigation paper (calibration, filter tuning), not a mechanistic/causal study of the frozen-observation self-loop or its discount-horizon root cause. No opponent/adversarial setting, no quantitative "zero world-state-change" measurement.

### 3. "No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning" (Zhong, Yang, Zhao)
- **Year/venue**: AAAI 2024, Proceedings of AAAI 38(15), pp. 17078–17086. Also arXiv:2312.06258.
- **URL**: https://arxiv.org/abs/2312.06258
- **What it does**: Shows that large/redundant action spaces cause agents to repeatedly attempt actions that have no effect (because they map to the same next-state distribution), sometimes causing task failure; proposes a redundant-action-filtering method using a similarity factor over state distributions, no prior knowledge required.
- **Relation to claim**: Addresses "the agent keeps re-selecting an action that doesn't change anything," but the mechanism is a redundant/no-op **action-space design** problem (many actions collapse to the same transition), not an externally imposed blocking mechanism (shield/obstacle) causing observation-freeze under a deterministic memoryless policy. Different causal story; does not threaten the claim's novelty, but is adjacent.

### 4. Classic background (pre-2024, establishes the theoretical mechanism but not novel post-2024 work)
- **Singh (1992)**, "Reinforcement Learning with a Hierarchy of Abstract Models," AAAI-92 — and the broader **perceptual-aliasing** literature (POMDP folklore): a deterministic memoryless policy that sees identical observations for distinct underlying states can loop forever on the wrong action; the textbook remedy is a *stochastic* memoryless policy to "roll the dice" out of the ambiguous state.
- **Relation to claim**: This is the closest *theoretical* precedent for "deterministic memoryless policy + repeated observation → self-loop." It is decades old and generic (POMDP folklore, not specific to safety shields, discount-horizon analysis, or measured world-state statistics). It does NOT anticipate the claim's specific finding that stochastic exploration *fails* to break the loop (because the blocking mechanism, not aliasing per se, keeps the observation literally frozen at the converged noise level — a stronger and different condition than generic perceptual aliasing, where different underlying states alias to the same observation but the observation itself is not static across time).

### 5. Broader shielding literature (Alshiekh et al. 2018 "Safe RL via Shielding," Jansen et al. 2020 probabilistic shields, Pranger & Könighofer 2026 "Easy-to-Use Shielding," ElSayed-Aly 2021 multi-agent shielding, "Fear Field" framework 2025)
- **What they do**: General framework papers on shields blocking unsafe actions at runtime; some note shields can cause conservative/oscillatory behavior at the boundary of the safe set, or that a shield failing "non-blocking" properties can trap agents in local minima/deadlock.
- **Relation to claim**: Background/context only — establishes that shield-induced pathological behavior is a recognized open issue in the field, strengthening the plausibility and importance of the claim, but none perform the specific frozen-observation/discount-horizon diagnosis.

---

## B2 — Discount-horizon limits & credit assignment for rare, long-delayed escapes

### 6. Laidlaw, Russell, Dragan — "Bridging RL Theory and Practice with the Effective Horizon"
- **Year/venue**: NeurIPS 2023 (Oral). arXiv:2304.09853.
- **URL**: https://arxiv.org/abs/2304.09853 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html
- **What it does**: Introduces BRIDGE, 155 tabular-reducible MDPs from deep RL benchmarks, and a new complexity measure — the *effective horizon* — that predicts when deep RL succeeds/fails based on how many steps of lookahead (evaluated via random rollouts) are needed to find the greedy-optimal action from the random policy's Q-values. Explains many deep RL failures as effective-horizon problems rather than pure exploration problems.
- **Relation to claim**: General theoretical grounding for "gamma sets an effective horizon beyond which the agent can't see the payoff," which underlies the claim's ~629-step / gamma-needed-0.9974 argument, but it is a benchmark/complexity-measure paper (2023, pre-window but foundational) — it does not address the specific frozen-observation self-loop mechanism, nor compute anything about opponent visibility or a 629-step pursuit-evasion escape. No overlap on the causal story of B1; only supports the generic "1/(1-gamma) effective horizon" arithmetic used in B2.

### 7. "FracTrace: Long-horizon credit assignment in deep reinforcement learning via fractional eligibility traces"
- **Year/venue**: Neurocomputing (ScienceDirect), 2026 (article number suggests a 2026 volume; exact issue not confirmed — WebFetch blocked by paywall, so bibliographic details below are from search snippets only and should be verified against the publisher page before citing formally).
- **URL**: https://www.sciencedirect.com/science/article/pii/S0925231226023222 (fetch returned HTTP 403; not independently verified beyond search-snippet text)
- **What it does** (per search snippets): Proposes a Fractional Trace Actor-Critic that replaces exponentially-decaying eligibility traces with power-law (Mittag-Leffler) memory kernels, explicitly targeting **reward delays of 500–800 steps** in robotic manipulation / autonomous navigation, where it says "conventional exponentially-decaying eligibility traces allocate negligible credit beyond moderate horizons... causal attribution effectively impossible at extended lags."
- **Relation to claim**: **Most numerically adjacent B2 paper found** — its 500–800 step delay range brackets the claim's ~629-step escape delay almost exactly, and its framing ("credit assignment intractable when reward is hundreds of steps from the causal action") is essentially the same phenomenon the claim invokes to explain why the escape is never learned. It is a *mitigation* paper (proposes fractional traces as fix), not a diagnostic one, and it is not set in a pursuit-evasion or shielded/blocked-action context — but it is the nearest B2 analogue and should be checked carefully (full text not accessible via WebFetch in this session; recommend a manual/institutional-access read before finalizing novelty claims against it).

### 8. RUDDER: Return Decomposition for Delayed Rewards (Arjona-Medina et al., 2018/2019)
- **Year/venue**: arXiv:1806.07857 (NeurIPS 2019).
- **URL**: https://arxiv.org/pdf/1806.07857
- **What it does**: Classic method for redistributing delayed/sparse rewards along a trajectory via LSTM-based return decomposition, explicitly built to fix exponential discounting's inability to assign credit to actions whose consequence is far in the future.
- **Relation to claim**: Old (pre-2024) but the canonical prior art for "discounting kills credit assignment over long delays." Does not overlap with the claim's specific numbers or the shield/frozen-observation setting; useful only as required background citation, not a scoop threat.

### 9. "Memoryless Exact Solutions for Deterministic MDPs with Sparse Rewards" (2018, arXiv:1805.07220)
- **What it does**: Theoretical analysis of when memoryless (stateless) policies can be exact/optimal for deterministic MDPs with sparse terminal rewards.
- **Relation to claim**: Tangential theoretical background; too old and abstract to threaten novelty.

---

## Clean negatives (searched, returned nothing on point)

The following query families were run and returned **no paper matching the claim's specific mechanism or its exact quantitative framing** (zero-world-state-change measurement, 2.4% opponent-visibility statistic, median-zero-action-delta-despite-redecision, or a computed gamma≈0.9974-for-629-steps argument):

- "absorbing state reinforcement learning policy repeats action deployment" — only textbook/theoretical absorbing-state definitions, nothing on deployment-time self-loops from external blocking.
- "frozen observation deterministic policy loop reinforcement learning livelock" — returned unrelated "Frozen Policy Iteration" / FrozenLake environment hits; no genuine livelock-in-RL paper.
- "livelock learned control policy stuck repeating blocked action 2024 2025" — returned general CS livelock definitions and one unrelated drone-replanning GitHub issue (RACER), not RL-policy-specific, not peer-reviewed.
- "self-perpetuating loop learned policy safety shield reinforcement learning robot" — no paper uses this framing; closest is the Kim et al. 2024 "repetitive/looping actions" language (already captured above as near-scoop).
- "pursuit evasion reinforcement learning pursuer stuck wall shield exploit stall" — no paper documents a pursuer or evader entering a frozen-observation self-loop against a shield/wall in a PE setting specifically.
- Searches for the exact quantitative signatures (opponent visible on 2.4% of steps, median action-delta of zero under re-decision, gamma≈0.9974 for a specific horizon) returned nothing — these appear to be original measurements not present anywhere in the searched literature.
- "deterministic policy fixed point stationary state RL deployment stuck" — only generic DPG/Bellman-fixed-point theory, not the deployment self-loop phenomenon.

---

## Summary judgment

- **No paper found performs the same causal/mechanistic decomposition** (blocked action → stalled observation → deterministic re-emission → exploration provably inert → discount-horizon explains why escape was never learned) with matched quantitative evidence.
- **Kim et al. 2024** (shield-induced "repetitive/looping actions," fixed via non-Markovian loop-avoidance) is the nearest **qualitative** precedent for B1 and should be cited and distinguished explicitly in any write-up.
- **FracTrace (2026)** is the nearest **quantitative** precedent for B2 (500–800 step delay range vs. the claim's 629 steps) but is a mitigation-method paper, not a diagnostic one, and its full text could not be verified in this session (paywalled; only search snippets available) — recommend obtaining full text before finalizing any novelty claim against it.
