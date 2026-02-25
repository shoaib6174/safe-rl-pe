# Research Report: Preventing Collapse in Adversarial Self-Play Training

**Date**: 2026-02-24 | **Session**: S46
**Context**: Phase 3 — Our 1v1 pursuit-evasion self-play training collapses when obstacle levels are introduced. The pursuer (faster agent) achieves 98-100% capture rate, the evader cannot learn to use obstacles, and the curriculum advances too fast because it only checks pursuer capture rate.

---

## Table of Contents
1. [Problem Diagnosis](#1-problem-diagnosis)
2. [Approach 1: OpenAI Hide-and-Seek Style Autocurricula](#2-approach-1-openai-hide-and-seek-style-autocurricula)
3. [Approach 2: Asymmetric Self-Play Training](#3-approach-2-asymmetric-self-play-training)
4. [Approach 3: Population-Based Training (PBT)](#4-approach-3-population-based-training-pbt)
5. [Approach 4: Curriculum Design for Adversarial Games](#5-approach-4-curriculum-design-for-adversarial-games)
6. [Approach 5: Fictitious Self-Play (FSP) and PSRO](#6-approach-5-fictitious-self-play-fsp-and-psro)
7. [Approach 6: Exploitability-Based and Regret-Based Curriculum](#7-approach-6-exploitability-based-and-regret-based-curriculum)
8. [Comparative Summary](#8-comparative-summary)
9. [Recommended Action Plan](#9-recommended-action-plan)
10. [References](#10-references)

---

## 1. Problem Diagnosis

Our training collapse has three interacting root causes:

1. **Asymmetric difficulty**: The pursuer is inherently faster (speed advantage), so in open space or simple obstacle layouts, raw speed dominates and the evader has no viable strategy. The evader needs to discover obstacle-exploitation strategies to compete, which is a much harder learning problem than "chase the target."

2. **Unilateral curriculum gate**: The curriculum only checks pursuer capture rate (>X%) to advance. This means the curriculum advances whenever the pursuer is winning, regardless of whether the evader has learned anything useful at the current level. The evader never gets enough training time at any level to discover obstacle strategies.

3. **Non-stationarity + catastrophic forgetting**: Even if the evader briefly learns something, the pursuer adapts and the evader's old strategy becomes useless. Without a mechanism to retain diverse strategies, both agents cycle or the stronger one dominates permanently.

---

## 2. Approach 1: OpenAI Hide-and-Seek Style Autocurricula

### Source
- Baker et al., "Emergent Tool Use From Multi-Agent Autocurricula," ICLR 2020 ([arXiv:1909.07528](https://arxiv.org/abs/1909.07528))
- [OpenAI Blog Post](https://openai.com/index/emergent-tool-use/)

### How It Works

OpenAI trained hiders and seekers in a 3D physics environment with movable objects (boxes, ramps). Key training details:

- **Shared PPO optimization**: All agents are optimized simultaneously with PPO + GAE using a centralized-training-decentralized-execution (CTDE) paradigm. Each agent has its own policy network and value network (separate parameters), but all are trained in the same optimization loop.
- **Opposing reward signals**: Hiders get +1 if all hidden, -1 if any seen. Seekers get the exact opposite. No curriculum was explicitly designed.
- **No separate training schedules**: Hiders and seekers train at the same rate with the same batch sizes. The autocurriculum emerges naturally from the zero-sum reward structure.
- **Massive scale was critical**: The default model uses batch size 64,000 with 1.6M parameters and required 132.3 million episodes (31.7 billion frames). Batch sizes below 32k never reached advanced strategy stages.
- **Preparation phase**: Hiders are given a head start while seekers are immobilized, providing an initial asymmetric advantage to the weaker side.
- **Entity-centric observations**: Permutation-invariant observation of objects and other agents.

The training produced 6 emergent strategy phases: random movement -> running -> fort building -> ramp use -> ramp locking -> box surfing -> box locking.

### Core Mechanism That Prevents Collapse

The autocurriculum relies on the zero-sum reward structure plus **massive scale**. There is no explicit anti-collapse mechanism. The key insight is that with enough scale (billions of frames), the losing side eventually stumbles onto counter-strategies through exploration. The preparation phase gives hiders an initial information/positioning advantage that partially compensates for their inherent disadvantage.

### Application to Our PE Setting

**Partially applicable, with caveats:**
- Our setting differs fundamentally: the speed asymmetry in PE is permanent and physical (the evader is slower), whereas in hide-and-seek the asymmetry comes from the game rules (hiding vs seeking), not from physical capabilities.
- The "preparation phase" idea IS directly applicable: giving the evader a head start (spawn further from the pursuer, give the evader a few timesteps to move before the pursuer activates) is a simple mechanism that could help.
- The scale requirement is concerning. OpenAI used 31.7 billion frames. We will not have that compute budget. Without that scale, the autocurriculum may not emerge naturally.

### Implementation Complexity: **Easy** (for the applicable parts)
- Adding a preparation phase / head start: trivial to implement.
- Matching their full scale: infeasible for our compute budget.

### Expected Effectiveness: **Low-Medium**
- The head start helps but does not solve the core problem (pursuer is faster, will catch up).
- Without massive scale, the autocurriculum alone is unlikely to produce obstacle-exploitation behavior.

---

## 3. Approach 2: Asymmetric Self-Play Training

### Sources
- Rao et al., "A Multi-party Asymmetric Self-play Algorithm (MASP)," MLPRAE 2024 ([ACM DL](https://dl.acm.org/doi/10.1145/3696687.3696712))
- [Hugging Face Deep RL Course: Self-Play](https://huggingface.co/learn/deep-rl-course/unit7/self-play)
- Bansal et al., "Emergent Complexity via Multi-Agent Competition," ICLR 2018 ([arXiv:1710.03748](https://arxiv.org/abs/1710.03748))

### How It Works

The core idea: **give the weaker agent more training updates** or **reduce the update frequency of the stronger agent**. Several variants exist:

1. **Asymmetric update ratios (MASP)**: Dynamically adjust the training update frequency based on ELO ratings. The stronger party's policy is updated less frequently, while the weaker party gets more updates. An improved ELO scoring system tracks each party's relative strength and determines update scheduling.

2. **Alternating freezing**: Train one agent while freezing the other. In pursuit-evasion, this is implemented as Synchronized Alternating Freezing Adversarial Training (SAFAT): train the evader for N steps while the pursuer is frozen, then train the pursuer for M steps while the evader is frozen. The ratio N:M can be set to favor the weaker agent (e.g., 3:1 evader:pursuer).

3. **Opponent sampling from history**: Instead of always playing against the latest (strongest) opponent, sample opponents from a pool of past checkpoints. This ensures the weaker agent sometimes faces easier opponents, preventing total discouragement. The `play_against_latest_model_ratio` parameter controls this.

4. **Regret Matching+ reweighting**: Reweight training data to emphasize underperforming role pairings, aligning the strength of a generalist model across all possible roles.

### Core Mechanism That Prevents Collapse

The weaker agent (evader) gets proportionally more learning signal relative to the stronger agent (pursuer). This prevents the "the evader never has time to learn" problem by explicitly allocating more training budget to the agent that needs it.

### Application to Our PE Setting

**Highly applicable and directly addresses our core problem:**
- **Alternating freezing with asymmetric ratio**: Train evader for 3x as many steps as pursuer. While the evader trains, it faces a fixed pursuer policy and can gradually learn obstacle-exploitation strategies without the pursuer simultaneously adapting to counter them.
- **Opponent pool sampling**: Sample pursuer opponents from a history of checkpoints (not just the latest, strongest version). The evader sometimes faces weaker pursuers, giving it a chance to practice and refine obstacle strategies.
- **ELO-based scheduling**: Track each agent's relative strength and automatically allocate more training to whichever agent is falling behind.

### Implementation Complexity: **Easy-Medium**
- Alternating freezing: Easy. Just freeze one network during the other's training loop.
- Asymmetric ratios: Easy. Train evader for N updates, pursuer for M updates (N > M).
- Opponent pool: Medium. Need checkpoint management and sampling logic (but we already have an OpponentPool class from S43).
- ELO tracking: Medium. Need to implement ELO computation and dynamic ratio adjustment.

### Expected Effectiveness: **High**
- This directly addresses the "evader never learns" problem.
- The alternating freeze + asymmetric ratio is probably the single most impactful change we can make.
- Combining with opponent pool sampling makes it even more effective.

---

## 4. Approach 3: Population-Based Training (PBT)

### Sources
- Czempin & Gleave, "Reducing Exploitability with Population Based Training," 2022 ([arXiv:2208.05083](https://arxiv.org/abs/2208.05083), [GitHub](https://github.com/HumanCompatibleAI/reducing-exploitability))
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (AlphaStar), Nature 2019 ([DeepMind Blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/))
- Huang et al., "A Robust and Opponent-Aware League Training Method for StarCraft II," NeurIPS 2023

### How It Works

Instead of training a single pursuer and a single evader, maintain a **population** of each:

**AlphaStar League Structure (the gold standard):**
- **Main Agents**: Train against the entire league using Prioritized Fictitious Self-Play (PFSP). Opponent selection probability is proportional to the opponent's win rate against this main agent (harder opponents are selected more often).
- **Main Exploiters**: Specifically trained to find weaknesses in the main agents.
- **League Exploiters**: Trained against the entire league to find general weaknesses.
- Exploiters are periodically reset to prevent overfitting. The league ran for 44 days, producing 900+ unique agents.

**Simpler Population-Based Approach (Czempin & Gleave 2022):**
- Maintain a pool of N opponent policies (e.g., N=5-20).
- The victim (agent being trained) plays against opponents randomly sampled from the pool.
- Pool diversity is the key: a larger pool means more diverse opponents, which means the agent cannot overfit to a single strategy.
- Robustness scales with population size.
- **Explicit diversity mechanisms** like RPPO (reward-randomized PPO) can be used to ensure population members develop different strategies rather than converging.

### Core Mechanism That Prevents Collapse

Population diversity prevents overfitting to a single opponent strategy. If the pursuer population contains diverse strategies (some fast-direct, some cut-off, some wait-at-obstacles), the evader must learn general obstacle-exploitation rather than a single counter-move. Similarly, diverse evader strategies prevent the pursuer from learning only one chase pattern.

### Application to Our PE Setting

**Applicable but computationally expensive:**
- Maintain 5-10 pursuer policies and 5-10 evader policies.
- Each evader trains against a random mix of pursuers. Each pursuer trains against a random mix of evaders.
- Some evaders will naturally develop obstacle-seeking behavior because they face diverse pursuers.
- The computational cost scales linearly with population size.

### Implementation Complexity: **Medium-Hard**
- We already have an OpponentPool class (S43) with FIFO eviction. This would need to be extended to maintain and train a full population simultaneously.
- Training N agents simultaneously requires N times the compute.
- Need to manage checkpoint storage, ELO/matchmaking between pool members, and diversity metrics.
- AlphaStar-style league is too expensive; a simpler 5-agent pool is feasible.

### Expected Effectiveness: **Medium-High**
- Strong theoretical backing and proven in practice (AlphaStar, Go, Dota).
- The diversity benefit directly addresses our collapse problem.
- But the computational cost is significant for our setting.

---

## 5. Approach 4: Curriculum Design for Adversarial Games

### Sources
- Dennis et al., "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design" (PAIRED), NeurIPS 2020 ([arXiv:2012.02096](https://arxiv.org/abs/2012.02096), [Google Research Blog](https://research.google/blog/paired-a-new-multi-agent-approach-for-adversarial-environment-generation/))
- Jiang et al., "Replay-Guided Adversarial Environment Design," NeurIPS 2021 ([arXiv:2110.02439](https://arxiv.org/abs/2110.02439))
- Narvekar et al., "Curriculum Learning for Reinforcement Learning Domains," JMLR 2020
- AAAI 2024: "Accelerate Multi-Agent RL in Zero-Sum Games with Subgame Curriculum Learning" ([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29011))

### How It Works

**The key insight for our problem: our curriculum advancement gate is broken.** It only checks pursuer performance, not evader performance. Here are the fixes:

**Fix 1: Dual-Criteria Curriculum Gate (immediate fix)**
- Do NOT advance to the next obstacle level unless BOTH agents meet performance criteria.
- For pursuit-evasion: advance only when (a) pursuer capture rate > threshold AND (b) evader survival time > threshold (or evader "near-miss" rate indicates it's actually using obstacles).
- Add a **minimum episodes per level** requirement (e.g., at least 50k episodes at each level) to prevent premature advancement.

**Fix 2: PAIRED (Protagonist Antagonist Induced Regret Environment Design)**
- Three RL agents: protagonist (evader), antagonist (reference evader), and adversary (environment designer).
- The adversary generates environment configurations (obstacle placements) to maximize the **regret** of the protagonist: the difference between the antagonist's performance and the protagonist's performance.
- Regret-based objective ensures the adversary creates environments that are *challenging but not impossible*: it finds the easiest environments the protagonist cannot yet solve.
- This naturally produces a curriculum that stays at the boundary of the protagonist's capabilities.

**Fix 3: Subgame Curriculum Learning**
- Adaptively sample training scenarios based on difficulty estimation.
- Start with large capture radius and small speed ratio, gradually tighten.
- For obstacle levels: start with simple obstacle layouts where the evader has obvious hiding spots, then gradually increase complexity.

**Fix 4: Reverse curriculum / fallback**
- If performance drops, regress to a previous level.
- Never "lock in" advancement; allow the curriculum to go backwards if the weaker agent's performance degrades.

### Core Mechanism That Prevents Collapse

The dual-criteria gate ensures both agents have learned meaningful strategies before advancing. PAIRED ensures environment difficulty matches the weaker agent's current capability. Both prevent the "curriculum races ahead because one agent is dominating" failure mode.

### Application to Our PE Setting

**Directly applicable and addresses our exact failure mode:**
- **Dual-criteria gate**: The single most important fix. Change the curriculum from "advance when pursuer capture rate > 90%" to "advance when pursuer capture rate > X% AND evader mean survival time > Y seconds AND minimum Z episodes at this level."
- **PAIRED**: Instead of hand-designing obstacle levels, train an environment generator to place obstacles in configurations that are challenging-but-solvable for the evader.
- **Reverse curriculum**: Allow the curriculum to drop back a level if the evader's performance falls below a floor.

### Implementation Complexity
- Dual-criteria gate: **Easy** (modify the existing CurriculumManager)
- Minimum episodes per level: **Easy** (add a counter)
- Reverse curriculum / fallback: **Easy** (add regression logic to CurriculumManager)
- PAIRED: **Hard** (requires training a third agent, the environment designer, and defining the environment parameterization)

### Expected Effectiveness: **High** (for dual-criteria) / **Very High** (for PAIRED)
- The dual-criteria gate alone would likely have prevented our collapse.
- PAIRED is the theoretically strongest approach for curriculum generation but has high implementation cost.

---

## 6. Approach 5: Fictitious Self-Play (FSP) and PSRO

### Sources
- Heinrich & Silver, "Fictitious Self-Play in Extensive-Form Games," ICML 2015 ([PMLR](https://proceedings.mlr.press/v37/heinrich15.html))
- Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games," 2016 ([arXiv:1603.01121](https://arxiv.org/abs/1603.01121))
- Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (PSRO), NeurIPS 2017 ([PDF](https://mlanctot.info/files/papers/nips17-psro.pdf))
- McAleer et al., "Self-Play PSRO: Toward Optimal Populations in Two-Player Zero-Sum Games," 2022 ([arXiv:2207.06541](https://arxiv.org/abs/2207.06541))
- IJCAI 2024: "Policy Space Response Oracles: A Survey" ([arXiv:2403.02227](https://arxiv.org/html/2403.02227v1))

### How It Works

**Fictitious Self-Play (FSP / NFSP):**
- Each agent maintains an **average policy** (mixture of all past policies) alongside its current best-response policy.
- At each step, the agent best-responds to the opponent's average policy, not their current policy.
- The average policy converges to a Nash equilibrium in two-player zero-sum games.
- Neural FSP (NFSP) implements this with two neural networks: one for the best response (trained via RL) and one for the average policy (trained via supervised learning on a reservoir buffer of past actions).
- Key property: **prevents cyclic strategies**. Vanilla self-play can get stuck in rock-paper-scissors cycles. FSP avoids this because the average policy "remembers" all past strategies.

**Policy Space Response Oracles (PSRO):**
- Maintains an explicit **population** of policies for each player.
- At each iteration: (1) compute a meta-game (payoff matrix for all policy matchups), (2) solve the meta-game for a Nash equilibrium mixture, (3) train a new best-response policy against this Nash mixture, (4) add it to the population.
- The meta-game Nash mixture converges to the true game Nash equilibrium as the population grows.
- **Variants**: Nash-PSRO (use Nash solver for meta-game), Uniform-PSRO (uniform mixture, equivalent to fictitious play), Self-Play PSRO (SP-PSRO, incorporates self-play for faster convergence).

### Core Mechanism That Prevents Collapse

Both FSP and PSRO prevent the **forgetting problem** in self-play. In vanilla self-play, when agent A adapts to beat agent B's new strategy, A may forget how to beat B's old strategy. FSP/PSRO retain all historical strategies (either via the average policy or the explicit population), so an agent can never "forget" a past counter-strategy.

They also prevent **cyclic strategies** (A beats B which beats C which beats A). By best-responding to the average/mixture rather than the latest policy, the agents converge monotonically toward equilibrium rather than cycling.

### Application to Our PE Setting

**Applicable but with nuances:**
- **NFSP is directly applicable**: Train both pursuer and evader with NFSP. The evader's average policy would retain obstacle-exploitation strategies even as the pursuer adapts. This prevents the "evader briefly learns to use obstacles, then pursuer adapts, then evader forgets obstacles" cycle.
- **PSRO is applicable but heavier**: Maintain a population of pursuer and evader policies, compute payoff matrices, and find Nash mixtures. Provides stronger convergence guarantees but requires more compute and infrastructure.
- **OpenSpiel** (by Lanctot et al.) provides reference implementations of NFSP and PSRO, but these are designed for discrete-action games (poker, etc.). Adapting to continuous-action PE would require custom implementation.

### Implementation Complexity
- NFSP: **Medium-Hard** (need two networks per agent, reservoir buffer, supervised learning loop alongside RL loop)
- PSRO: **Hard** (need population management, payoff matrix computation, meta-game Nash solver, best-response oracle training)

### Expected Effectiveness: **Medium-High**
- Theoretically strong guarantees (converges to NE in two-player zero-sum).
- However, our game has continuous actions and states, which makes the convergence slower and less clean than in tabular games.
- NFSP's average policy might be "mushy" (average of many different strategies) rather than a single coherent obstacle-exploitation strategy.
- The forgetting-prevention property is highly valuable for our setting.

---

## 7. Approach 6: Exploitability-Based and Regret-Based Curriculum

### Sources
- Lockhart et al., "Computing Approximate Equilibria in Sequential Adversarial Games by Exploitability Descent," IJCAI 2019 ([arXiv:1903.05614](https://arxiv.org/abs/1903.05614))
- Timbers et al., "Approximate Exploitability: Learning a Best Response in Large Games," IJCAI 2022
- Balcan et al., "Nash Equilibria and Pitfalls of Adversarial Training in Adversarial Robustness Games," AISTATS 2023 ([arXiv:2210.12606](https://arxiv.org/abs/2210.12606))

### How It Works

**Exploitability Descent (ED):**
- Instead of using win rate to control training, directly optimize **exploitability** (the maximum payoff gain an opponent could achieve by switching to a best response).
- At each step, compute or approximate each agent's exploitability, then update policies via gradient descent on exploitability.
- When both players independently run ED, NashConv (sum of exploitabilities) is locally minimized, converging toward Nash equilibrium.
- Unlike fictitious play and CFR, convergence pertains to the *actual policies* being optimized, not the average policies.

**Approximate Exploitability:**
- Computing exact exploitability requires solving a full best-response problem, which is expensive. Approximate exploitability uses a learned approximate best response (trained via RL) as a proxy.
- This makes exploitability-based methods practical for large games.

**Regret-Based Curriculum Control:**
- Instead of advancing curriculum based on win rate, advance based on **exploitability gap** or **regret**.
- Only advance when the exploitability of both agents is below a threshold (meaning neither agent has an obvious counter-strategy it hasn't learned yet).
- This is more informative than win rate: a 90% win rate for the pursuer tells you the pursuer is winning, but not whether the evader has any unexploited strategies. Low exploitability tells you that neither agent has unexploited opportunities.

### Core Mechanism That Prevents Collapse

Exploitability directly measures "how much could be gained by switching strategies." If the evader's exploitability is high (meaning there exists a strategy the evader could use that would significantly improve its outcomes), the curriculum does NOT advance. This prevents advancing past levels where the evader has undiscovered useful strategies (like obstacle exploitation).

### Application to Our PE Setting

**Theoretically ideal, practically challenging:**
- **Exploitability as curriculum gate**: Instead of "advance when pursuer capture rate > X%", use "advance when max(exploitability_pursuer, exploitability_evader) < epsilon." This would perfectly prevent premature curriculum advancement.
- **Approximate exploitability**: Train a "best response evader" and a "best response pursuer" alongside the main agents. The best response evader would naturally discover obstacle-exploitation strategies. If the best response evader achieves much better survival than the main evader, that means the main evader still has significant room for improvement (high exploitability), and the curriculum should NOT advance.
- The main challenge: computing approximate exploitability requires training additional best-response agents, which roughly doubles compute.

### Implementation Complexity: **Hard**
- Computing exploitability requires training approximate best-response agents.
- Exploitability descent requires differentiating through the exploitability computation.
- Approximate exploitability (using a learned best response) is more practical but still requires training extra agents per curriculum level.

### Expected Effectiveness: **Very High** (in theory) / **Medium** (in practice)
- Theoretically, this is the "right" metric to optimize.
- In practice, approximate exploitability can be noisy and expensive to compute.
- A simplified version (train a separate "oracle evader" with hand-crafted obstacle-exploitation reward to bound exploitability) could capture most of the benefit at lower cost.

---

## 8. Comparative Summary

| Approach | Prevents Collapse How | Implementation | Effectiveness | Compute Cost |
|----------|----------------------|----------------|---------------|-------------|
| **OpenAI Autocurricula** | Scale + emergence | Easy (head start) | Low-Medium | Very High |
| **Asymmetric Training** | More updates for weaker agent | Easy-Medium | **High** | Low |
| **Population-Based (PBT)** | Diversity of opponents | Medium-Hard | Medium-High | High |
| **Dual-Criteria Curriculum** | Both agents must pass | **Easy** | **High** | None |
| **PAIRED Env Design** | Regret-based environment generation | Hard | Very High | Medium |
| **NFSP** | Average policy prevents forgetting | Medium-Hard | Medium-High | Medium |
| **PSRO** | Population + Nash meta-solver | Hard | High | High |
| **Exploitability Descent** | Direct NE gap optimization | Hard | Very High | High |

### For Our Specific Problem (ranked by expected impact / effort ratio):

1. **Dual-criteria curriculum gate + minimum episodes** (Easy, High impact) -- FIX THIS FIRST
2. **Asymmetric training ratio** (Easy, High impact) -- evader gets 2-3x training
3. **Alternating freeze training** (Easy, High impact) -- train evader with frozen pursuer
4. **Opponent pool sampling** (Medium, we already have the class) -- use historical opponents
5. **Reverse/fallback curriculum** (Easy, Medium impact) -- allow regression
6. **Evader head start** (Easy, Low-Medium impact) -- preparation phase
7. **NFSP** (Medium-Hard, Medium-High impact) -- if simpler fixes are insufficient
8. **Population-based training** (Hard, Medium-High impact) -- if we have compute budget
9. **PAIRED** (Hard, Very High impact) -- if we want optimal curriculum generation
10. **Exploitability descent** (Hard, Very High theoretical impact) -- research-grade solution

---

## 9. Recommended Action Plan

### Phase A: Immediate Fixes (Easy, implement now)

**A1. Fix the curriculum gate (CRITICAL)**
```
Current: advance when pursuer_capture_rate > threshold
Fixed:   advance when ALL of:
         - pursuer_capture_rate > pursuit_threshold
         - evader_mean_survival_time > survival_threshold
         - evader_obstacle_proximity_rate > obstacle_threshold  (evader spends time near obstacles)
         - episodes_at_current_level >= min_episodes_per_level  (e.g., 50k)
         - curriculum can regress if evader performance drops below floor
```

**A2. Asymmetric training ratio**
- Train evader for 3 PPO updates per 1 pursuer PPO update
- Or: train evader for 3000 steps, then pursuer for 1000 steps, alternating

**A3. Alternating freeze**
- Freeze pursuer for N steps while evader trains
- Freeze evader for M steps while pursuer trains
- N >> M initially (e.g., 3:1 or 4:1)

**A4. Evader head start**
- Give evader 10-20 timesteps of movement before the pursuer activates
- Allows evader to reach an obstacle before the chase begins

### Phase B: Medium Effort (implement if Phase A insufficient)

**B1. Enhanced opponent pool**
- Extend existing OpponentPool to maintain historical checkpoints for both agents
- Evader trains against mix of current and historical (weaker) pursuers
- Pursuer trains against mix of current and historical evaders
- Ratio: 50% latest opponent, 50% sampled from pool

**B2. Oracle exploitability approximation (simplified)**
- Train a separate "oracle evader" with hand-crafted obstacle-seeking reward
- If oracle evader survives much longer than main evader, main evader's exploitability is high
- Use this as an additional curriculum gate: don't advance until main evader's performance is within 50% of oracle evader's

### Phase C: Heavy Lift (if Phase A+B insufficient)

**C1. NFSP adaptation**
- Implement Neural Fictitious Self-Play with reservoir buffer
- Two networks per agent: best response (RL) + average policy (supervised)
- Prevents strategy cycling and forgetting

**C2. PAIRED for obstacle placement**
- Train a third agent (environment designer) to place obstacles
- Regret-based objective ensures obstacles are placed at the boundary of evader's current capability

---

## 10. References

### Directly Cited Papers

1. Baker, B. et al. (2020). "Emergent Tool Use From Multi-Agent Autocurricula." ICLR 2020. arXiv:1909.07528.
2. Bansal, T. et al. (2018). "Emergent Complexity via Multi-Agent Competition." ICLR 2018. arXiv:1710.03748.
3. Rao, Y. et al. (2024). "A Multi-party Asymmetric Self-play Algorithm (MASP)." MLPRAE 2024.
4. Czempin, P. & Gleave, A. (2022). "Reducing Exploitability with Population Based Training." arXiv:2208.05083.
5. Vinyals, O. et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature.
6. Dennis, M. et al. (2020). "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED)." NeurIPS 2020. arXiv:2012.02096.
7. Jiang, M. et al. (2021). "Replay-Guided Adversarial Environment Design." NeurIPS 2021. arXiv:2110.02439.
8. Narvekar, S. et al. (2020). "Curriculum Learning for Reinforcement Learning Domains." JMLR.
9. Heinrich, J. & Silver, D. (2015). "Fictitious Self-Play in Extensive-Form Games." ICML 2015.
10. Heinrich, J. & Silver, D. (2016). "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games." arXiv:1603.01121.
11. Lanctot, M. et al. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO)." NeurIPS 2017.
12. McAleer, S. et al. (2022). "Self-Play PSRO: Toward Optimal Populations in Two-Player Zero-Sum Games." arXiv:2207.06541.
13. Lockhart, E. et al. (2019). "Computing Approximate Equilibria by Exploitability Descent." IJCAI 2019. arXiv:1903.05614.
14. Timbers, F. et al. (2022). "Approximate Exploitability: Learning a Best Response in Large Games." IJCAI 2022.
15. Balcan, M-F. et al. (2023). "Nash Equilibria and Pitfalls of Adversarial Training." AISTATS 2023.
16. Huang, R. et al. (2023). "A Robust and Opponent-Aware League Training Method for StarCraft II." NeurIPS 2023.
17. Lanctot, M. et al. (2019). "OpenSpiel: A Framework for Reinforcement Learning in Games." arXiv:1908.09453.

### Related Surveys
18. Survey on Self-play Methods in Reinforcement Learning, 2024. arXiv:2408.01072.
19. Policy Space Response Oracles: A Survey, IJCAI 2024. arXiv:2403.02227.
20. Review of RL approaches for pursuit-evasion games, Chinese Journal of Aeronautics, 2025.
