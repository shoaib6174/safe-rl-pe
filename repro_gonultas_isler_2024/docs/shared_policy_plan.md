# Shared Policy with Role Conditioning — Implementation Plan

## Theoretical Foundation

### Core Insight
Both agents share the **same neural network** π(s, role). This creates a symmetry by construction that no previous method achieved.

### Why Shared Policy Balances CR

**Standard self-play (separate networks):**
- L_p = -Q_p(s, π_p(s_p), a_e) → improve pursuer
- L_e = -Q_e(s, a_p, π_e(s_e)) → improve evader
- **No coupling** between π_p and π_e
- Winner pulls ahead → positive feedback → lock-in

**Shared policy:**
- L_p = -Q_p(s, π(s_p, 0), a_e) → improve pursuer-role weights
- L_e = +Q_p(s, a_p, π(s_e, 1)) → improve evader-role weights
- **Same network** receives both gradients

**The negative feedback mechanism:**
- If CR > 0.5 (pursuer dominates): L_p gradient is weak (already near-optimal). L_e gradient is strong (evader needs improvement). Network shifts toward evader → CR decreases.
- If CR < 0.5 (evader dominates): L_e gradient is weak. L_p gradient is strong. Network shifts toward pursuer → CR increases.
- **Result: restoring force toward CR=0.5**

### Nash Equilibrium Guarantee

In a symmetric zero-sum game with shared policy π, the fixed point satisfies:

π = argmax_π [ J_pursuer(π(·,0), π(·,1)) + J_evader(π(·,1), π(·,0)) ]

At fixed point: π(·,0) is best response to π(·,1), and vice versa. This is a **Nash equilibrium**.

## Architecture Changes

### Actor Network

```
Input: [obs (11 dims), role (1 dim)]  → 12 dims total
Hidden: fc1(12→128), ReLU, fc2(128→128), ReLU
Output: fc3(128→2), tanh  → 2D action [ax, ay]

role = 0.0: pursuer
role = 1.0: evader
```

**Implementation:** Modify `ActorNetwork.__init__` to accept `obs_dim` that includes role.

### Critic Network

**No changes needed.** Critic remains centralized, taking both observations and actions.

Keep **separate critics** for pursuer and evader (not shared) to:
- Reduce gradient conflict
- Maintain stability
- Allow independent Q-estimation

### EMA Networks

Single `ema_actor` (shared, like the actor). Updated once per step after both agent updates.

## Training Loop Changes

### Current (separate networks)

```python
# Sample batch
batch = replay_buffer.sample(batch_size)

# Pursuer update
c_loss_p, a_loss_p = pursuer.update(batch, ema_evader, agent_id="pursuer")

# Evader update
for _ in range(evader_train_steps):
    c_loss_e, a_loss_e = evader.update(batch, ema_pursuer, agent_id="evader")
```

### New (shared actor)

```python
# Sample batch
batch = replay_buffer.sample(batch_size)

# Shared actor: compute BOTH losses from SAME batch
# 1. Pursuer actor loss (role=0)
# 2. Evader actor loss (role=1)
# 3. Sum and backprop ONCE through shared actor

# Critic updates remain separate (pursuer critic + evader critic)
c_loss_p = pursuer.update_critic(batch)  # only critic, not actor
c_loss_e = evader.update_critic(batch)

# Shared actor update: sum of both role losses
a_loss_p, a_loss_e = shared_actor.update_both(batch, ema_actor)

# Update EMA (single network)
ema_actor.soft_update(shared_actor)
```

### Key Detail: Actor Update

```python
def update_shared_actor(self, batch, ema_actor):
    obs_p = batch["obs_p"]
    obs_e = batch["obs_e"]
    actions_p = batch["actions_p"]  # from replay buffer (old actions)
    actions_e = batch["actions_e"]
    
    # --- Pursuer role loss ---
    # Current pursuer action (with role=0)
    role_p = torch.zeros(batch_size, 1)
    new_actions_p = shared_actor(torch.cat([obs_p, role_p], dim=-1))
    
    # Q(s_p, s_e, a_p_new, a_e_old)  — evader action from batch
    q1_p = critic_p(obs_all, torch.cat([new_actions_p, actions_e.detach()], dim=-1))
    q2_p = critic_p2(obs_all, torch.cat([new_actions_p, actions_e.detach()], dim=-1))
    loss_p = -torch.min(q1_p, q2_p).mean()
    
    # --- Evader role loss ---
    # Current evader action (with role=1)
    role_e = torch.ones(batch_size, 1)
    new_actions_e = shared_actor(torch.cat([obs_e, role_e], dim=-1))
    
    # Q(s_p, s_e, a_p_old, a_e_new)  — pursuer action from batch
    q1_e = critic_e(obs_all, torch.cat([actions_p.detach(), new_actions_e], dim=-1))
    q2_e = critic_e2(obs_all, torch.cat([actions_p.detach(), new_actions_e], dim=-1))
    loss_e = -torch.min(q1_e, q2_e).mean()
    
    # Combined loss
    total_loss = loss_p + loss_e
    
    # Single backprop through shared actor
    actor_optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(shared_actor.parameters(), 1.0)
    actor_optimizer.step()
    
    return loss_p.item(), loss_e.item()
```

## MADDPGAgent Changes

### Constructor

```python
class SharedMADDPGAgent:
    def __init__(self, ...):
        # Shared actor (both agents use this!)
        self.shared_actor = ActorNetwork(obs_dim + 1, action_dim)
        self.shared_actor_target = ActorNetwork(obs_dim + 1, action_dim)
        
        # Separate critics (one per role)
        self.critic_p = CriticNetwork(...)
        self.critic_p2 = CriticNetwork(...)
        self.critic_e = CriticNetwork(...)
        self.critic_e2 = CriticNetwork(...)
        
        # Separate optimizers for critics
        # Shared optimizer for actor (both roles update same network)
```

### Action Selection

```python
def select_action(self, obs, role, noise_sigma=0.1):
    # role: 0 = pursuer, 1 = evader
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    role_tensor = torch.FloatTensor([[role]])
    input_tensor = torch.cat([obs_tensor, role_tensor], dim=-1)
    with torch.no_grad():
        action = self.shared_actor(input_tensor).cpu().numpy()[0]
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)
    return action
```

## Config Design

```yaml
# Shared Policy Config
symmetric: true
shared_policy: true

maddpg:
  episodes: 10000        # More episodes (shared policy learns slower)
  actor_lr: 0.0005
  critic_lr: 0.0005
  ema_tau: 0.001
  exploration_sigma: 0.15  # Higher exploration (need diverse strategies)
  
  # No pretrain, no mixing — both roles learn simultaneously from scratch
```

## Experimental Protocol

### Phase 1: Shared Policy + Symmetric Dynamics (Primary, 3 seeds)
- Both agents: point-mass, equal speeds, symmetric FOV
- Shared actor with role conditioning
- EMA τ=0.001
- 10,000 episodes (shared policy converges slower)
- Expected: CR→0.5 with oscillation amplitude < 0.3

### Phase 2: Shared Policy + Asymmetric Dynamics (Secondary, 3 seeds)
- Test whether shared policy fixes asymmetric dynamics (GI setup)
- If this works: the architecture matters more than dynamics
- If this fails: dynamics asymmetry is the harder barrier

### Control: Separate Policy + Symmetric Dynamics (3 seeds)
- Same symmetric setup but separate actors
- Test whether shared policy is the key difference
- Expected: same as before (seed lottery)

## Theoretical Predictions

| Metric | Separate Policy (Previous) | Shared Policy (Predicted) |
|--------|---------------------------|---------------------------|
| Seed variance | HIGH (1/7 balanced) | LOW (all seeds similar) |
| Oscillation amplitude | ~0.30 | ~0.15 |
| CR at convergence | 0.0 or 1.0 | **~0.5** |
| Reproducibility | 1/7 | **≥2/3** |

## Implementation Checklist

### Step 1: Modify ActorNetwork (1 hour)
- [ ] Add `role_dim=1` to constructor
- [ ] Concatenate role to observation in forward()
- [ ] Test with dummy inputs

### Step 2: Create SharedMADDPGAgent (3 hours)
- [ ] Constructor with shared actor, separate critics
- [ ] `select_action(obs, role)` method
- [ ] `update_critics(batch)` — separate critic updates
- [ ] `update_shared_actor(batch, ema_actor)` — combined actor update
- [ ] `update_targets()` — soft update for both critics + shared actor target

### Step 3: Modify train.py (2 hours)
- [ ] Create single `shared_actor` and `ema_actor`
- [ ] Create two agents (pursuer and evader) using same shared actor
- [ ] Action selection: use `shared_actor` with role=0 or role=1
- [ ] Update loop: critic updates separate → combined actor update → EMA update
- [ ] Logging: track CR, both actor losses, combined loss

### Step 4: Config and Launch (1 hour)
- [ ] Create `config_shared_policy.yaml`
- [ ] Test smoke run locally
- [ ] Launch 3 seeds on niro-2

### Step 5: Monitor and Evaluate (ongoing)
- [ ] Check CR convergence
- [ ] Compute robustness at checkpoints
- [ ] Compare to separate policy baseline

**Total implementation time: ~1 day**
**Training time: ~10 days (10000 episodes × 3 seeds)**

## Why This Could Fail (And What We'd Learn)

1. **Gradient conflict:** L_p and L_e gradients might cancel, causing slow/no learning.
   - *Learning:* Shared policy needs careful gradient management.

2. **Mode collapse:** Network learns mediocre strategy for both roles.
   - *Learning:* Higher exploration or entropy bonus needed.

3. **Non-stationarity still wins:** Even with shared policy, EMA lag might still cause collapse.
   - *Learning:* Problem is deeper than architecture — possibly fundamental.

If it fails, we'd have tested:
- Separate policy + asymmetric dynamics ❌
- Separate policy + symmetric dynamics ❌
- Shared policy + symmetric dynamics ❌

This would make the paper's conclusion **even stronger**: "No architectural fix (symmetry, sharing, EMA, Pop-FSP) achieves balance. The instability is fundamental to capture-based PE with discontinuous payoffs."

## Go/No-Go Decision

**GO if:** We accept the risk of another 10-day run for a potentially transformative result.

**NO-GO if:** We write the paper now with current negative results.

**My recommendation: GO.** The shared policy is the only approach with a genuine theoretical mechanism for balance. It fundamentally changes the problem structure. If it works, the paper becomes a method paper. If it fails, the negative result is bulletproof.
