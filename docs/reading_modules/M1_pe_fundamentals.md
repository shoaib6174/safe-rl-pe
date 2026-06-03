# Module 1: Pursuit-Evasion as a Differential Game

**Estimated reading time:** 3–4 hours  
**Prerequisites:** None (this is the foundation)  
**Goal:** Understand why pursuit-evasion is fundamentally hard, what a "solution" means in game-theoretic terms, and how the G\&I 2024 environment instantiates the problem.

---

## 1.1 What Is a Pursuit-Evasion Game?

At its core, a pursuit-evasion (PE) game is a **two-player differential game** played in continuous time (or fine-grained discrete time) over a state space. One player — the **pursuer** — wants to capture the other — the **evader**. The evader wants to avoid capture indefinitely (or escape).

Formally:
- State: $\mathbf{x}(t) \in \mathbb{R}^n$ describes the positions, velocities, and orientations of both agents.
- Pursuer control: $\mathbf{u}_p(t) \in \mathcal{U}_p$
- Evader control: $\mathbf{u}_e(t) \in \mathcal{U}_e$
- Dynamics: $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}_p, \mathbf{u}_e)$

The game ends when the state enters the **capture set** $\mathcal{C}$ — typically when the distance between pursuer and evader falls below a threshold $d_{\text{capture}}$.

**Why this matters for deep RL:** Unlike single-agent RL where the environment is stationary, in PE the "environment" is an **adaptive opponent** whose policy changes as it learns. This makes the learning dynamics fundamentally non-stationary.

---

## 1.2 Two-Player Zero-Sum Games

PE is naturally modeled as a **zero-sum game**: the pursuer's gain is the evader's loss. We capture this with a **payoff functional**:

$$J(\mathbf{u}_p, \mathbf{u}_e) = \int_0^{T} g(\mathbf{x}(t), \mathbf{u}_p(t), \mathbf{u}_e(t)) \, dt + h(\mathbf{x}(T))$$

where $T$ is the terminal time (capture or timeout). In the simplest formulation:
- If capture occurs: pursuer wins (+1), evader loses (−1)
- If timeout occurs: draw (0)

The pursuer wants to **maximize** $J$; the evader wants to **minimize** it.

### Saddle-Point Equilibrium

In a two-player zero-sum game, a **saddle-point equilibrium** (or Nash equilibrium) is a pair of strategies $(\mathbf{u}_p^*, \mathbf{u}_e^*)$ such that neither player can improve their outcome by unilaterally changing strategy:

$$J(\mathbf{u}_p^*, \mathbf{u}_e) \leq J(\mathbf{u}_p^*, \mathbf{u}_e^*) \leq J(\mathbf{u}_p, \mathbf{u}_e^*) \quad \forall \, \mathbf{u}_p, \mathbf{u}_e$$

**Intuition:** The pursuer's optimal strategy guarantees at least the value $J^*$, regardless of what the evader does. The evader's optimal strategy guarantees at most $J^*$, regardless of what the pursuer does. If both play optimally, the outcome is $J^*$.

**Why this matters for RL:** Deep RL does not guarantee finding the saddle point. Self-play can oscillate (pursuer improves → evader improves → pursuer adapts → ...) without converging. A capture rate of 0.5 (50% capture, 50% escape) suggests the policies are near a saddle point — neither side can consistently dominate.

### The Value Function

In optimal control, the value function $V(\mathbf{x})$ gives the best achievable outcome from state $\mathbf{x}$ assuming both players play optimally. For PE:

- $V(\mathbf{x}) > 0$: pursuer can guarantee capture from $\mathbf{x}$
- $V(\mathbf{x}) < 0$: evader can guarantee escape from $\mathbf{x}$
- $V(\mathbf{x}) = 0$: the boundary — the **barrier**

The set $\{\mathbf{x} : V(\mathbf{x}) \geq 0\}$ is the **capture set** (or more precisely, the set from which capture is guaranteed). Its boundary is the **barrier surface**.

---

## 1.3 Isaacs' Equations and the Barrier

Rufus Isaacs (1965) derived the fundamental PDE for differential games, analogous to the Hamilton-Jacobi-Bellman (HJB) equation for optimal control:

$$\min_{\mathbf{u}_e} \max_{\mathbf{u}_p} \left[ \nabla V(\mathbf{x}) \cdot f(\mathbf{x}, \mathbf{u}_p, \mathbf{u}_e) + g(\mathbf{x}, \mathbf{u}_p, \mathbf{u}_e) \right] = 0$$

This is the **Isaacs equation** (or Hamilton-Jacobi-Isaacs, HJI).

### A Concrete Example

Consider two players on a line. The pursuer moves with velocity $u_p \in [-1, 1]$, the evader with $u_e \in [-1, 1]$. The state is the distance $x$ between them. The dynamics are:

$$\dot{x} = u_e - u_p$$

The evader wants to maximize $x$ (escape); the pursuer wants to minimize it (capture). The value function is simply $V(x) = x$ because:
- If $x > 0$, the evader moves right ($u_e = 1$), the pursuer moves left ($u_p = -1$), and $\dot{x} = 2$ — escape is guaranteed.
- If $x = 0$, capture has occurred.

This toy example shows the structure: the value function captures "who wins from here." The HJI equation generalizes this to complex dynamics.

**The order of min and max matters.** In the equation above, we assume the pursuer can respond to the evader's choice. This is the **closed-loop** formulation. If the order were reversed (max then min), it would represent the **open-loop** case where the evader responds to the pursuer.

### Reach-Avoid Sets

A closely related concept is the **reach-avoid set**: the set of states from which the pursuer can guarantee reaching the capture set while avoiding some forbidden region (e.g., obstacles). In PE without obstacles, the reach-avoid set coincides with the capture set.

**Computationally:** Reach-avoid sets are computed by solving the HJI PDE backward in time from the terminal condition (capture). The zero level set of the solution defines the boundary.

### Why the Barrier Is Hard to Compute

For simple dynamics (e.g., both players are single integrators with speed limits), the barrier can be computed analytically. For car-like dynamics with partial observability, the state space is high-dimensional and the HJI PDE becomes intractable. This is why **approximate methods** — including deep RL — are necessary.

**Key insight:** Deep RL does not solve the HJI equation directly. Instead, it learns an approximate policy that implicitly captures some region of the value function.

---

## 1.4 Open-Loop vs Closed-Loop Strategies

An important distinction in differential games:

- **Open-loop:** Each player commits to a trajectory at the start. No feedback from the opponent's actual movements.
- **Closed-loop (feedback):** Each player observes the current state and chooses their control based on it. This allows reacting to the opponent.

**Deep RL learns closed-loop policies.** The policy network $\pi(\mathbf{o})$ maps observations to actions, enabling real-time reactions. This is both a strength (adaptability) and a challenge (non-stationarity — if the opponent changes, the policy may become suboptimal).

---

## 1.5 Why Equal-Speed PE Is Fundamentally Hard

Consider the simplest case: both players are single integrators with the **same maximum speed** $v_{\max}$, in an **unbounded** domain.

**The evader can always escape.** If the evader simply moves directly away from the pursuer, the distance between them never decreases. The pursuer can at best maintain the distance, never close it. Capture requires either:
1. The pursuer being faster ($v_p > v_e$), or
2. A bounded domain where the evader eventually runs out of room.

In a **bounded domain** (like the 16×16m arena in G\&I 2024), equal speeds make capture possible but far from guaranteed. The evader can use the boundary to create "shadows" — regions the pursuer cannot reach before the evader escapes.

### The Mirroring Argument

Imagine both players on an infinite plane with equal speed. The evader can guarantee non-capture with a simple strategy: move directly away from the pursuer. The distance never decreases.

In a **bounded arena**, the evader cannot run forever. But the evader can use the boundary to create **shadows** — regions the pursuer cannot reach before the evader escapes. One such strategy is **mirroring**: position yourself so the pursuer is between you and the nearest boundary. Now the pursuer must approach from a disadvantaged angle, giving you more escape routes.

With **partial observability**, the mirroring strategy becomes even stronger. The pursuer may not know exactly where the evader is, making it harder to cut off escape routes.

**Implication:** If the evader has equal or greater speed, capture requires one of:
- **Speed advantage** (faster pursuer)
- **Information advantage** (better sensing)
- **Maneuverability advantage** (tighter turns)
- **Boundary exploitation** (trapping the evader in a corner)

---

## 1.6 The G\&I 2024 Environment

So far we have discussed differential games in the abstract. Now let's look at the specific instantiation used in Gonultas \& Isler 2024 — the environment that all our experiments (EMA, Pop-FSP, and the reproduction) use.

The G\&I environment makes the following design choices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Arena | 16×16m square | Large enough for non-trivial strategies |
| Episode length | 400 steps (40s at dt=0.1) | Long enough for extended chase |
| Pursuer dynamics | Bicycle model, wheelbase 0.30m | Realistic car-like constraints |
| Pursuer v_max | 2.5 m/s | Faster than evader |
| Evader dynamics | Point mass | Simpler, more agile |
| Evader v_max | 1.5 m/s | Slower but can change direction instantly |
| Capture radius | 0.15m (effective 0.30m) | Small relative to arena |
| Sensing | FOV 120° / 6m range (pursuer), omnidirectional (evader) | Partial observability |

**Critical observation:** The pursuer is **faster** (2.5 vs 1.5 m/s) but has **more constrained dynamics** (bicycle model vs point mass) and **worse sensing** (120° FOV vs omnidirectional). This creates a nuanced tradeoff, not a trivial pursuer advantage.

### Why This Design Is Interesting

1. **Speed asymmetry** means capture is possible in principle, but not guaranteed.
2. **Sensing asymmetry** (FOV vs omnidirectional) means the evader has information advantage.
3. **Dynamics asymmetry** (bicycle vs point mass) means the evader is more agile.

The combination makes the game genuinely competitive. If the pursuer were faster *and* had better sensing *and* better dynamics, capture would be trivial. If the evader were faster, escape would be trivial.

---

## 1.7 Our Journey: Speed Ratios and What We Learned

Our early work (Phase 1–3, S11–S84) used a different environment: 10×10m arena, 200 steps, unicycle dynamics for both agents, **equal top speeds**. We experimented with giving the evader speed advantages (1.05×, 1.1×, 1.2×) based on the literature survey (S54).

**What we found (S59–S64):**
- 1.05× evader advantage: insufficient, pursuer still dominated
- 1.2× evader advantage: evader could sustain 35–50% escape rate, but strategies were degenerate (pure speed running, no obstacle use)
- Equal speeds with **partial observability** (combined radius + LOS masking): the evader adopted a "hide and don't move" strategy — detection became nearly equivalent to capture. Information asymmetry alone was insufficient to create rich pursuit dynamics without some speed differential.

**Key lesson:** The G\&I environment's combination of speed, sensing, and dynamics asymmetry is carefully balanced. Changing any one parameter significantly alters the game's character.

---

## 1.8 From Theory to Deep RL

The HJI framework assumes:
1. Perfect knowledge of the state
2. Optimal play from both sides
3. Continuous, instantaneous control

Deep RL relaxes all three:
1. **Partial observability:** Agents see only FOV-limited observations
2. **Suboptimal play:** Policies are learned, not computed
3. **Discrete control:** Actions are taken at 10Hz (dt=0.1s)

This relaxation makes the problem tractable but introduces new challenges:
- **Non-stationarity:** As the opponent learns, the "environment" changes
- **Exploration:** Agents must discover strategies without knowing the value function
- **Generalization:** Policies must work against diverse opponents, not just one

### From Continuous to Discrete: The RL Bridge

The HJI framework assumes continuous time. Deep RL operates in **discrete time** with a fixed step size $\Delta t$ (in G\&I: $\Delta t = 0.1$s). The state evolves as:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot f(\mathbf{x}_t, \mathbf{u}_{p,t}, \mathbf{u}_{e,t})$$

The **reward** at each step is the discrete approximation of the running cost: $r_t = \Delta t \cdot g(\mathbf{x}_t, \mathbf{u}_{p,t}, \mathbf{u}_{e,t})$. In G\&I, this is:
- Dense reward: combination of distance penalty, time penalty, and capture bonus
- Terminal reward: large bonus/penalty at capture or timeout

The **discount factor** $\gamma$ in RL approximates the continuous-time exponential discounting. With $\gamma = 0.99$ and $\Delta t = 0.1$s, the effective time horizon is roughly $\frac{\Delta t}{1-\gamma} = 10$s, or about 100 steps.

These challenges are the subject of the remaining modules.

---

## 1.9 Summary

| Concept | Meaning for PE |
|---------|---------------|
| Differential game | Continuous-time competitive optimization |
| Value function $V(\mathbf{x})$ | Best achievable outcome from state $\mathbf{x}$ |
| Barrier $V(\mathbf{x}) = 0$ | Boundary between capture and escape regions |
| Isaacs equation | PDE for the value function (intractable for complex dynamics) |
| Equal-speed difficulty | Evader can mirror/escape; bounded domain helps pursuer |
| G\&I environment | Speed + sensing + dynamics asymmetry creates balanced game |

---

## Self-Check

1. **Why is equal-speed PE in an unbounded domain trivial for the evader?** Draw the distance dynamics when both move at $v_{\max}$ directly away from each other.

2. **What does $V(\mathbf{x}) = 0$ mean physically?** If the state is exactly on the barrier, what happens under optimal play from both sides?

3. **Why does the G\&I environment give the evader better sensing (omnidirectional) despite the pursuer being faster?** What would happen if the pursuer had omnidirectional sensing and the evader had FOV?

4. **Our Phase 1–3 framework used equal top speeds with partial observability, but the evader adopted a "hide and don't move" strategy. What does this tell you about the role of information asymmetry relative to speed asymmetry?**

---

## Papers to Read

- **(ESSENTIAL)** Gonultas \& Isler 2024, Section II: Problem formulation and environment description.
- **(REFERENCE)** Isaacs, R. (1965). *Differential Games*. (The original — read Chapter 1 for intuition.)
- **(REFERENCE)** Paper 25 (HJ reachability survey, 2024): For understanding how HJI methods relate to RL.
- **(OPTIONAL)** Paper 19 (self-play in aircraft PE, 2024): Different domain, same differential game structure.

---

## Deep Dive Sessions

- **S54** (PE design choices survey): Comprehensive survey of arena sizes, speed ratios, episode lengths across 15+ papers. Shows G\&I's choices are representative but not universal.
- **S64** (speed diagnostics): Detailed analysis of why 1.2× evader advantage produced degenerate strategies.
