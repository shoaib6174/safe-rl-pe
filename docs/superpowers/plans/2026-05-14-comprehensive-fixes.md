# Comprehensive Fix Plan: Paper-vs-Implementation Discrepancies

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all identified discrepancies between our implementation and Gonultas & Isler (IROS 2025) to achieve correct oscillatory training dynamics and target capture rates (CR=0.98 vs random, CR=0.54 self-play).

**Architecture:** Five independent fixes across MADDPG agent, BiMDN architecture, training loop, and evaluation protocol. Each fix is self-contained but must all be applied together for correct behavior.

**Tech Stack:** PyTorch, NumPy, YAML

---

## Severity Classification

| # | Fix | Severity | File(s) | Root Cause |
|---|-----|----------|---------|-----------|
| 1 | Actor observation bug | **CRITICAL** | `agents/maddpg.py` | Evader's actor always sees pursuer's observation |
| 2 | BiMDN 10K warm-up | **MAJOR** | `training/vectorized_env.py` | BiMDN trains too early with noisy data |
| 3 | BiMDN architecture | **MAJOR** | `agents/bimdn.py` | Missing FC layers in each output head |
| 4 | Evaluation protocol | **CRITICAL** | `evaluation/evaluate.py` | Only evaluates final checkpoint, not tournament |
| 5 | Checkpoint overwriting | **MODERATE** | `training/train.py` | All seeds write to same directory |

---

## Task 1: Fix Actor Observation Bug (CRITICAL)

**Files:**
- Modify: `repro_gonultas_isler_2024/agents/maddpg.py`
- Modify: `repro_gonultas_isler_2024/training/train.py`

**Root Cause:** In `MADDPGAgent.update()`, line 173/185 uses `self.actor(obs_p)` unconditionally. When `self` is the evader agent, it should see `obs_e` instead. This means the evader's actor network has never actually seen its own observation space — it has been trained on the pursuer's observations. The evader cannot learn ego-motion control, explaining the permanent CR=1.00 (pursuer dominance).

- [ ] **Step 1: Add `agent_id` parameter to `update()` signature**

  In `agents/maddpg.py`, change the `update` method signature:

  ```python
  def update(
      self,
      batch: dict,
      other_agent: "MADDPGAgent",
      agent_id: str = "pursuer",  # "pursuer" or "evader"
  ) -> tuple:
  ```

- [ ] **Step 2: Select correct observation based on agent_id**

  Replace the unconditional `obs_p` usage in the actor update block. At line ~170 (inside the actor update section):

  ```python
  # Before (BUG):
  new_actions_p = self.actor(obs_p)

  # After (FIX):
  own_obs = obs_p if agent_id == "pursuer" else obs_e
  new_actions = self.actor(own_obs)
  ```

  Also update the `new_actions_all` concatenation to use the correct variable name.

- [ ] **Step 3: Update `new_actions_all` to use `new_actions`**

  Replace:
  ```python
  new_actions_all = torch.cat([new_actions_p, actions_e.detach()], dim=-1)
  ```
  With:
  ```python
  if agent_id == "pursuer":
      new_actions_all = torch.cat([new_actions, actions_e.detach()], dim=-1)
  else:
      new_actions_all = torch.cat([actions_p.detach(), new_actions], dim=-1)
  ```

- [ ] **Step 4: Update train.py to pass agent_id**

  In `training/train.py`, line ~172:

  ```python
  # Before:
  c_loss_p, a_loss_p = pursuer.update(batch, evader)
  c_loss_e, a_loss_e = evader.update(batch, pursuer)

  # After:
  c_loss_p, a_loss_p = pursuer.update(batch, evader, agent_id="pursuer")
  c_loss_e, a_loss_e = evader.update(batch, pursuer, agent_id="evader")
  ```

- [ ] **Step 5: Verify the fix**

  Run a short training test (100 episodes, 1 env) and check that:
  1. No errors occur
  2. Both pursuer and evader actor losses are non-zero and changing
  3. Evader reward starts negative (it should be losing initially, then improving)

---

## Task 2: Implement BiMDN 10K Warm-up (MAJOR)

**Files:**
- Modify: `repro_gonultas_isler_2024/training/vectorized_env.py`
- Modify: `repro_gonultas_isler_2024/config_belief_noamp.yaml`

**Root Cause:** Paper Section IV-C states "a warm-up period of 10,000 samples is used before training the BiMDN." Our code starts training at 256 samples. Early training with insufficient data produces noisy beliefs that corrupt the MADDPG observation space.

- [ ] **Step 1: Add warm-up counter to VectorizedEnvRunner**

  In `training/vectorized_env.py`, add to `__init__`:

  ```python
  self.bimdn_warmup = config["bimdn"].get("belief_warmup", 10000)
  self.bimdn_samples_collected = 0
  ```

- [ ] **Step 2: Count samples during collection**

  In the `step()` method, in the frame skip loop where `store_training_sample` is called, increment the counter:

  ```python
  self.belief_pursuers[i].store_training_sample(evader_pos)
  self.belief_evaders[i].store_training_sample(pursuer_pos)
  self.bimdn_samples_collected += 1  # Add this line
  ```

- [ ] **Step 3: Gate BiMDN updates behind warm-up check**

  In the BiMDN update block (line ~236), add the warm-up check:

  ```python
  # Before:
  if self.use_belief and global_step > 0 and global_step % bimdn_update_freq == 0:

  # After:
  if (self.use_belief
      and global_step > 0
      and global_step % bimdn_update_freq == 0
      and self.bimdn_samples_collected >= self.bimdn_warmup):
  ```

- [ ] **Step 4: Verify config has warm-up parameter**

  Check `config_belief_noamp.yaml` has:

  ```yaml
  bimdn:
    belief_warmup: 10000
  ```

  Add it if missing.

- [ ] **Step 5: Verify warm-up behavior**

  Run a short test and verify:
  1. BiMDN losses remain 0.0 for the first ~10K samples
  2. After 10K samples, BiMDN losses become non-zero
  3. Console output shows `BP=0.000 BE=0.000` during warm-up

---

## Task 3: Fix BiMDN Architecture (MAJOR)

**Files:**
- Modify: `repro_gonultas_isler_2024/agents/bimdn.py`

**Root Cause:** Paper Figure 4 shows "fully connected layer in each head before the output layers." Current code has `BiLSTM → FC(32) → [softmax(π), μ, ELU+1(σ)]` directly. Missing: `FC(32) → FC(32) → output` for each head.

- [ ] **Step 1: Add intermediate FC layers to each head**

  In `agents/bimdn.py`, modify `__init__`:

  ```python
  # Before:
  self.fc = nn.Linear(2 * hidden_dim, feature_dim)
  self.relu = nn.ReLU()
  self.pi_head = nn.Linear(feature_dim, num_components)
  self.mu_head = nn.Linear(feature_dim, num_components * 2)
  self.sigma_head = nn.Linear(feature_dim, num_components * 2)

  # After:
  self.fc = nn.Linear(2 * hidden_dim, feature_dim)
  self.relu = nn.ReLU()

  # Intermediate FC layers for each head (per paper Fig 4)
  self.pi_fc = nn.Linear(feature_dim, feature_dim)
  self.mu_fc = nn.Linear(feature_dim, feature_dim)
  self.sigma_fc = nn.Linear(feature_dim, feature_dim)

  self.pi_head = nn.Linear(feature_dim, num_components)
  self.mu_head = nn.Linear(feature_dim, num_components * 2)
  self.sigma_head = nn.Linear(feature_dim, num_components * 2)
  ```

- [ ] **Step 2: Update forward() to use intermediate layers**

  ```python
  # Before:
  features = self.relu(self.fc(last_hidden))
  pi = torch.softmax(self.pi_head(features), dim=-1)
  mu = self.mu_head(features).view(-1, self.num_components, 2)
  sigma = nn.functional.elu(self.sigma_head(features)) + 1.0

  # After:
  features = self.relu(self.fc(last_hidden))
  pi = torch.softmax(self.pi_head(self.relu(self.pi_fc(features))), dim=-1)
  mu = self.mu_head(self.relu(self.mu_fc(features))).view(-1, self.num_components, 2)
  sigma = nn.functional.elu(self.sigma_head(self.relu(self.sigma_fc(features)))) + 1.0
  ```

- [ ] **Step 3: Verify architecture matches paper**

  Print model summary and verify layer count:

  ```python
  model = BiMDN()
  print(model)
  ```

  Expected layers: `LSTM → FC → [FC→pi_head, FC→mu_head, FC→sigma_head]`

---

## Task 4: Implement Checkpoint Tournament Evaluation (CRITICAL)

**Files:**
- Modify: `repro_gonultas_isler_2024/evaluation/evaluate.py`
- Create: `repro_gonultas_isler_2024/scripts/evaluate_tournament.py`

**Root Cause:** Paper Section V-A: "For each checkpoint, we evaluate the agent by having it play against 32 randomly sampled opponents from previous checkpoints and compute the win rate." We only evaluate the final checkpoint. The CR=0.54 in the paper comes from the best checkpoint selected via this tournament, not the final model.

- [ ] **Step 1: Add `load_policy_from_state_dict()` helper**

  In `evaluation/evaluate.py`, add:

  ```python
  def load_policy_from_state_dict(agent: MADDPGAgent, state_dict: dict):
      agent.actor.load_state_dict(state_dict)
      agent.actor.eval()
  ```

- [ ] **Step 2: Implement `evaluate_checkpoint_tournament()`**

  Add a new function that evaluates one checkpoint against N random opponents:

  ```python
  def evaluate_checkpoint_tournament(
      env: PursuitEvasionEnv,
      checkpoint_state: dict,  # {"pursuer": state_dict, "evader": state_dict, "bimdn_p": ..., "bimdn_e": ...}
      opponent_pool: list,  # List of opponent state dicts
      num_opponents: int = 32,
      num_episodes_per_match: int = 10,
      frame_stack: int = 2,
      use_belief: bool = False,
      belief_pursuer: BeliefStateManager = None,
      belief_evader: BeliefStateManager = None,
  ) -> dict:
      """Evaluate one checkpoint vs N random opponents from pool."""
      # Sample opponents
      sampled_opponents = np.random.choice(opponent_pool, size=min(num_opponents, len(opponent_pool)), replace=False)

      # Load main checkpoint
      main_pursuer = MADDPGAgent(...)  # same construction as run_evaluation
      main_evader = MADDPGAgent(...)
      load_policy_from_state_dict(main_pursuer, checkpoint_state["pursuer"])
      load_policy_from_state_dict(main_evader, checkpoint_state["evader"])

      wins = 0
      total_matches = 0

      for opp_state in sampled_opponents:
          opp_pursuer = MADDPGAgent(...)
          opp_evader = MADDPGAgent(...)
          load_policy_from_state_dict(opp_pursuer, opp_state["pursuer"])
          load_policy_from_state_dict(opp_evader, opp_state["evader"])

          # Main pursuer vs opponent evader
          results = evaluate_agent(
              env,
              lambda obs: main_pursuer.select_action(obs, noise_sigma=0.0),
              lambda obs: opp_evader.select_action(obs, noise_sigma=0.0),
              num_episodes=num_episodes_per_match,
              frame_stack=frame_stack,
              use_belief=use_belief,
              belief_pursuer=belief_pursuer,
              belief_evader=belief_evader,
          )
          wins += results["capture_rate"] * num_episodes_per_match
          total_matches += num_episodes_per_match

      return {"win_rate": wins / total_matches if total_matches > 0 else 0.0}
  ```

- [ ] **Step 3: Implement `run_full_tournament()`**

  Add a function that loads all checkpoints, runs tournament for each, and returns best:

  ```python
  def run_full_tournament(
      checkpoint_dir: str,
      config: dict,
      num_opponents: int = 32,
      episodes_per_match: int = 10,
  ):
      checkpoint_path = Path(checkpoint_dir)
      # Find all checkpoint subdirectories
      checkpoints = sorted(checkpoint_path.glob("episode_*"))

      # Load all checkpoint state dicts into memory
      pool = []
      for cp in checkpoints:
          state = {
              "pursuer": torch.load(cp / "pursuer_actor.pt", map_location="cpu"),
              "evader": torch.load(cp / "evader_actor.pt", map_location="cpu"),
          }
          if use_belief:
              state["bimdn_p"] = cp / "bimdn_pursuer.pt"
              state["bimdn_e"] = cp / "bimdn_evader.pt"
          pool.append(state)

      env = PursuitEvasionEnv(config)
      # ... construct belief managers if needed ...

      results = []
      for i, cp_state in enumerate(pool):
          result = evaluate_checkpoint_tournament(
              env, cp_state, pool[:i] + pool[i+1:],  # exclude self
              num_opponents, episodes_per_match,
              frame_stack, use_belief, belief_pursuer, belief_evader,
          )
          results.append((checkpoints[i], result["win_rate"]))
          print(f"Checkpoint {checkpoints[i].name}: win_rate={result['win_rate']:.3f}")

      best_cp, best_wr = max(results, key=lambda x: x[1])
      print(f"\nBest checkpoint: {best_cp.name} with win_rate={best_wr:.3f}")
      return best_cp, best_wr, results
  ```

- [ ] **Step 4: Create CLI script for tournament**

  Create `scripts/evaluate_tournament.py`:

  ```python
  import yaml
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent / "repro_gonultas_isler_2024"))

  from evaluation.evaluate import run_full_tournament

  if __name__ == "__main__":
      config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
      checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "results/checkpoints"

      with open(config_path, "r") as f:
          config = yaml.safe_load(f)

      best_cp, best_wr, all_results = run_full_tournament(checkpoint_dir, config)
      print(f"\n=== FINAL RESULT ===")
      print(f"Best checkpoint: {best_cp}")
      print(f"Win rate: {best_wr:.4f}")
  ```

- [ ] **Step 5: Verify tournament script runs**

  Run on an existing checkpoint directory with at least 2 checkpoints:

  ```bash
  ./venv/bin/python scripts/evaluate_tournament.py config_belief_noamp.yaml results_belief_noamp/checkpoints/
  ```

  Expected: prints win rates for each checkpoint, identifies best.

---

## Task 5: Fix Checkpoint Overwriting (MODERATE)

**Files:**
- Modify: `repro_gonultas_isler_2024/training/train.py`
- Modify: `repro_gonultas_isler_2024/config_belief_noamp.yaml`

**Root Cause:** All seeds save checkpoints to `results_belief_noamp/checkpoints/`. Later seeds overwrite earlier seeds' checkpoints, making multi-seed training invalid.

- [ ] **Step 1: Add seed to output directory in train.py**

  In `training/train.py`, after loading config:

  ```python
  output_dir = Path(config.get("output_dir", "."))
  seed = config.get("seed", 0)
  output_dir = output_dir / f"seed_{seed}"
  output_dir.mkdir(parents=True, exist_ok=True)
  ```

- [ ] **Step 2: Ensure config has seed parameter**

  Check `config_belief_noamp.yaml` has:

  ```yaml
  seed: 0
  output_dir: "results_belief_noamp"
  ```

  This will produce `results_belief_noamp/seed_0/checkpoints/`, `results_belief_noamp/seed_1/checkpoints/`, etc.

- [ ] **Step 3: Update launch script to pass seeds**

  Modify the launch script (e.g., `launch_gi_v3_belief_noamp.sh`) to generate configs with different seeds:

  ```bash
  for seed in 0 1 2; do
      python -c "
import yaml
with open('config_belief_noamp.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['seed'] = $seed
with open(f'config_seed_{seed}.yaml', 'w') as f:
    yaml.dump(config, f)
"
      nohup ./venv/bin/python repro_gonultas_isler_2024/training/train.py config_seed_${seed}.yaml > seed_${seed}.log 2>&1 &
  done
  ```

- [ ] **Step 4: Verify directory structure**

  After launching a test run with seed=42:

  ```bash
  ls results_belief_noamp/seed_42/checkpoints/
  ```

  Should show `episode_1000/`, `episode_2000/`, etc. without collision.

---

## Integration & Verification

### Integration Order

1. **Task 1** (actor bug) — must be first; fixes fundamental learning
2. **Task 2** (BiMDN warm-up) — independent, but order matters for training stability
3. **Task 3** (BiMDN architecture) — independent of Tasks 1-2
4. **Task 5** (checkpoint paths) — independent, but needed before launching
5. **Task 4** (evaluation) — can be done last, needed after training completes

### Combined Verification Test

- [ ] **Step 1: Run a 500-episode smoke test with all fixes applied**

  ```bash
  ./venv/bin/python repro_gonultas_isler_2024/training/train.py config_test.yaml
  ```

  Config should have: `episodes: 500`, `parallel_envs: 2`, `seed: 999`

- [ ] **Step 2: Verify no crashes and meaningful loss curves**

  Check:
  1. Pursuer critic loss: starts high, decreases
  2. Evader critic loss: starts high, decreases (THIS IS NEW — was impossible before)
  3. Pursuer actor loss: non-zero, changing sign
  4. Evader actor loss: non-zero, changing sign (THIS IS NEW)
  5. BiMDN losses: 0.0 for first ~10K samples, then non-zero
  6. Capture rate: starts at ~1.0, may drop as evader learns (oscillation beginning)

- [ ] **Step 3: Verify checkpoint directory has seed subfolder**

  ```bash
  ls results_belief_noamp/seed_999/checkpoints/
  ```

- [ ] **Step 4: Run tournament evaluation on smoke test checkpoints**

  ```bash
  ./venv/bin/python scripts/evaluate_tournament.py config_test.yaml results_belief_noamp/seed_999/checkpoints/
  ```

  Should complete without errors.

### Full Training Launch

- [ ] **Step 5: Launch 3-seed full training on niro-2**

  ```bash
  # On niro-2
  for seed in 0 1 2; do
      # generate config with seed
      python -c "
import yaml
with open('config_belief_noamp.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['seed'] = $seed
with open(f'config_seed_{seed}.yaml', 'w') as f:
    yaml.dump(config, f)
"
      nohup ./venv/bin/python repro_gonultas_isler_2024/training/train.py config_seed_${seed}.yaml > seed_${seed}.log 2>&1 &
  done
  ```

- [ ] **Step 6: Monitor for 24 hours and verify oscillatory behavior**

  Check logs for:
  1. Capture rate oscillating (not stuck at 1.0)
  2. Both pursuer and evader losses are active
  3. No NaN values
  4. BiMDN losses become non-zero after warm-up and stabilize

---

## Expected Outcomes

| Metric | Before Fixes | After Fixes (Expected) | Paper |
|--------|-------------|----------------------|-------|
| CR vs random | ~0.80 | **~0.98** | ~0.98 |
| CR self-play (best ckpt) | 1.00 (stuck) | **~0.54** (oscillatory) | ~0.54 |
| Training dynamics | Pursuer-dominant, flat | Oscillatory, both agents improve | Oscillatory |
| BiMDN loss | Noisy from start | Stable after 10K warm-up | Stable |

---

## Rollback Plan

If any fix introduces instability:
1. Revert that specific fix via git
2. Re-run smoke test
3. If still unstable, revert to last known good commit (`dd2f2af`)

---

## References

- Gonultas & Isler, "Game-Theoretic Planning with Belief State Encoding...", IROS 2025
- Paper Section IV-C: BiMDN architecture and warm-up
- Paper Section V-A: Training protocol and checkpoint evaluation
- Ackert et al. 2019: Twin critics in MADDPG
