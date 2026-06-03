# Belief-State Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an explicit belief-state particle filter for pursuit-evasion self-play under pure partial observability, eliminating the masking curriculum that causes distributional shift and exploitability.

**Architecture:** Add a `BeliefTracker` (particle filter) to `PartialObsWrapper` that outputs a 6-D belief vector (mean, std, entropy, time-since-seen). Extend the policy network with a belief MLP branch. Update `PartialObsOpponentAdapter` to also provide belief observations so frozen belief-trained policies work in BR tests and evaluation.

**Tech Stack:** Python 3.11, NumPy, PyTorch, Stable-Baselines3, Gymnasium, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `envs/belief_tracker.py` | **Create** | Particle filter belief tracker: maintains distribution over opponent position, outputs 6-D statistics |
| `tests/test_belief_tracker.py` | **Create** | Unit tests for BeliefTracker: visible clustering, hidden diffusion, reappearance convergence, arena clamping |
| `envs/partial_obs_wrapper.py` | **Modify** | Add BeliefTracker instance, update observation space with `"belief"` key, wire into `step()`/`reset()` |
| `agents/partial_obs_policy.py` | **Modify** | Add `belief_mlp` branch to `PartialObsPolicyNetwork`, update `PartialObsFeaturesExtractor` to infer `belief_dim` from obs space |
| `envs/opponent_adapter.py` | **Modify** | Add BeliefTracker instance, append `"belief"` to obs dict in `_build_obs_dict()`, wire `reset()` |
| `tests/test_training_smoke.py` | **Modify** (add test) | Smoke test: launch 100 steps of training with belief-enabled wrapper, verify obs dict contains `"belief"` |

---

## Task 1: BeliefTracker Core Implementation

**Files:**
- Create: `envs/belief_tracker.py`
- Test: `tests/test_belief_tracker.py`

### Step 1.1: Write the failing test

Create `tests/test_belief_tracker.py`:

```python
"""Unit tests for BeliefTracker particle filter.

Test Index:
  A: test_visible_target_clusters
  B: test_hidden_target_diffuses
  C: test_reappearance_reconverges
  D: test_particles_clamped_to_arena
  E: test_safe_default_before_first_update
"""

import numpy as np
import pytest

from envs.belief_tracker import BeliefTracker


# --- Test A: Visible target clusters around true position ---

def test_visible_target_clusters():
    """When opponent is visible, particles should cluster near true position."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    tracker.update(own_state, opp_state, is_visible=True)
    stats = tracker.get_statistics()

    mean_x, mean_y = stats[0], stats[1]
    assert abs(mean_x - 8.0) < 0.1, f"mean_x should be ~8.0, got {mean_x}"
    assert abs(mean_y - 8.0) < 0.1, f"mean_y should be ~8.0, got {mean_y}"


# --- Test B: Hidden target diffuses over time ---

def test_hidden_target_diffuses():
    """When opponent is hidden, particle spread should grow with sqrt(t)."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=500,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    # First observe
    tracker.update(own_state, opp_state, is_visible=True)
    stats_visible = tracker.get_statistics()
    std_visible = (stats_visible[2] + stats_visible[3]) / 2

    # Hide for 200 steps (10 seconds at dt=0.05)
    for _ in range(200):
        tracker.update(own_state, opp_state, is_visible=False)

    stats_hidden = tracker.get_statistics()
    std_hidden = (stats_hidden[2] + stats_hidden[3]) / 2

    # Diffusion: std should grow from ~0.2 (resample noise) to ~0.7m
    assert std_visible < 0.3, f"Visible std too large: {std_visible}"
    assert std_hidden > std_visible, f"Hidden std should exceed visible std"
    assert 0.5 < std_hidden < 1.5, f"Hidden std should be ~0.7m, got {std_hidden}"

    # time_since_seen should be 10.0 seconds
    assert abs(stats_hidden[5] - 10.0) < 0.01, f"time_since_seen should be 10.0, got {stats_hidden[5]}"


# --- Test C: Reappearance reconverges quickly ---

def test_reappearance_reconverges():
    """After diffusion, one visible update should reconverge to true position."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    own_state = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    opp_state = np.array([8.0, 8.0, 0.0], dtype=np.float32)

    # Observe, hide, then reappear at a different location
    tracker.update(own_state, opp_state, is_visible=True)
    for _ in range(100):
        tracker.update(own_state, opp_state, is_visible=False)

    # Opponent reappears at new position
    new_opp = np.array([12.0, 12.0, 0.0], dtype=np.float32)
    tracker.update(own_state, new_opp, is_visible=True)
    stats = tracker.get_statistics()

    assert abs(stats[0] - 12.0) < 0.1, f"mean_x should reconverge to 12.0, got {stats[0]}"
    assert abs(stats[1] - 12.0) < 0.1, f"mean_y should reconverge to 12.0, got {stats[1]}"


# --- Test D: Particles clamped to arena ---

def test_particles_clamped_to_arena():
    """Particles should never leave [0, arena_width] x [0, arena_height]."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=10.0,
        arena_height=10.0,
        opponent_v_max=5.0,  # High speed to accelerate diffusion
        n_particles=1000,
        seed=42,
    )
    own_state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    opp_state = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    tracker.update(own_state, opp_state, is_visible=True)
    for _ in range(500):
        tracker.update(own_state, opp_state, is_visible=False)

    # All particles should be inside arena
    assert np.all(tracker.particles[:, 0] >= 0.0)
    assert np.all(tracker.particles[:, 0] <= 10.0)
    assert np.all(tracker.particles[:, 1] >= 0.0)
    assert np.all(tracker.particles[:, 1] <= 10.0)


# --- Test E: Safe default before first update ---

def test_safe_default_before_first_update():
    """Before any update, get_statistics should return safe default."""
    tracker = BeliefTracker(
        role="pursuer",
        arena_width=20.0,
        arena_height=20.0,
        opponent_v_max=1.0,
        n_particles=100,
        seed=42,
    )
    stats = tracker.get_statistics()

    # mean should be arena center
    assert abs(stats[0] - 10.0) < 0.01, f"default mean_x should be 10.0, got {stats[0]}"
    assert abs(stats[1] - 10.0) < 0.01, f"default mean_y should be 10.0, got {stats[1]}"
    # std should be arena diagonal / 2 ≈ 14.14
    expected_std = np.sqrt(20.0**2 + 20.0**2) / 2
    assert abs(stats[2] - expected_std) < 0.1
    assert abs(stats[3] - expected_std) < 0.1
    # time_since_seen should be 0
    assert stats[5] == 0.0
```

Run to verify it fails:

```bash
pytest tests/test_belief_tracker.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'envs.belief_tracker'`

### Step 1.2: Implement BeliefTracker

Create `envs/belief_tracker.py`:

```python
"""Particle filter belief tracker for opponent position in pursuit-evasion.

Maintains a distribution over the opponent's (x, y) position using a
simplified particle filter. Outputs geometric statistics (mean, std, entropy,
time-since-seen) as a 6-D belief vector for the policy network.
"""

from __future__ import annotations

import numpy as np


class BeliefTracker:
    """Track opponent position belief via particle filter.

    Args:
        role: 'pursuer' or 'evader' — which agent this tracker serves.
        arena_width: Arena width in meters.
        arena_height: Arena height in meters.
        opponent_v_max: Max speed of the opponent being tracked (m/s).
        n_particles: Number of particles (default 100).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        role: str,
        arena_width: float,
        arena_height: float,
        opponent_v_max: float,
        n_particles: int = 100,
        seed: int = 0,
    ):
        self.role = role
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.opponent_v_max = opponent_v_max
        self.n_particles = n_particles
        self.dt = 0.05  # Simulation timestep (matches PursuitEvasionEnv)

        self.rng = np.random.default_rng(seed)

        # Particle state
        self.particles = None
        self.weights = None
        self.steps_since_seen = 0
        self._ever_updated = False

        self.reset()

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        # Initialize particles uniformly over arena
        self.particles = np.stack([
            self.rng.uniform(0.0, self.arena_width, size=self.n_particles),
            self.rng.uniform(0.0, self.arena_height, size=self.n_particles),
        ], axis=1).astype(np.float32)
        self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles
        self.steps_since_seen = 0
        self._ever_updated = False

    def update(
        self,
        own_state: np.ndarray,
        opponent_state: np.ndarray,
        is_visible: bool,
    ) -> None:
        """Update belief given current visibility.

        Args:
            own_state: [x, y, theta] of the tracking agent.
            opponent_state: [x, y, theta] of the opponent.
            is_visible: Whether opponent is currently visible.
        """
        self._ever_updated = True

        if is_visible:
            # Resample all particles around observed position
            true_pos = opponent_state[:2]
            self.particles = self.rng.normal(
                loc=true_pos,
                scale=0.2,
                size=(self.n_particles, 2),
            ).astype(np.float32)
            self._clamp_particles()
            self.steps_since_seen = 0
        else:
            # Propagate particles with diffusion
            self.steps_since_seen += 1
            sigma = self.opponent_v_max * self.dt * np.sqrt(self.steps_since_seen)
            noise = self.rng.normal(loc=0.0, scale=sigma, size=(self.n_particles, 2))
            self.particles += noise.astype(np.float32)
            self._clamp_particles()

    def get_statistics(self) -> np.ndarray:
        """Return 6-D belief statistics vector.

        Returns:
            [mean_x, mean_y, std_x, std_y, entropy, time_since_seen]
        """
        if not self._ever_updated:
            # Safe default before first update
            mean_x = self.arena_width / 2.0
            mean_y = self.arena_height / 2.0
            std_x = np.sqrt(self.arena_width**2 + self.arena_height**2) / 2.0
            std_y = std_x
            entropy = np.log(std_x * std_y + 1e-6)
            time_since_seen = 0.0
            return np.array([mean_x, mean_y, std_x, std_y, entropy, time_since_seen], dtype=np.float32)

        mean_x = float(np.mean(self.particles[:, 0]))
        mean_y = float(np.mean(self.particles[:, 1]))
        std_x = float(np.std(self.particles[:, 0]))
        std_y = float(np.std(self.particles[:, 1]))

        # Entropy: log(det(cov) + eps)
        # For independent axes: det(cov) = std_x^2 * std_y^2
        det_cov = (std_x ** 2) * (std_y ** 2)
        entropy = float(np.log(det_cov + 1e-6))

        time_since_seen = float(self.steps_since_seen * self.dt)

        return np.array([mean_x, mean_y, std_x, std_y, entropy, time_since_seen], dtype=np.float32)

    def _clamp_particles(self) -> None:
        """Clamp all particles to arena bounds."""
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, self.arena_width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, self.arena_height)
```

### Step 1.3: Run tests to verify they pass

```bash
pytest tests/test_belief_tracker.py -v
```

Expected: All 5 tests PASS.

### Step 1.4: Commit

```bash
git add envs/belief_tracker.py tests/test_belief_tracker.py
git commit -m "feat: add BeliefTracker particle filter with unit tests"
```

---

## Task 2: PartialObsWrapper Belief Integration

**Files:**
- Modify: `envs/partial_obs_wrapper.py`

### Step 2.1: Modify PartialObsWrapper to include BeliefTracker

In `envs/partial_obs_wrapper.py`, make these changes:

**Add import at top:**

```python
from envs.belief_tracker import BeliefTracker
```

**Add BeliefTracker to `__init__` (after sensor init):**

```python
# Add after line ~76 (after lidar_max_range assignment):

# Belief tracker for opponent position
if self.role == "pursuer":
    opp_v_max = self._base_env.evader_v_max
else:
    opp_v_max = self._base_env.pursuer_v_max

self.belief_tracker = BeliefTracker(
    role=self.role,
    arena_width=self._base_env.arena_width,
    arena_height=self._base_env.arena_height,
    opponent_v_max=opp_v_max,
    n_particles=100,
    seed=0,  # Deterministic per instance
)
```

**Add `"belief"` to observation space (after `"state"` key):**

```python
# In the spaces.Dict dict, after "state" entry (around line ~97):
            "belief": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
```

**Update `reset()` to reset belief tracker (after buffer.reset):**

```python
# After line ~117 (self.steps_since_seen = 0):
        self.belief_tracker.reset()
```

**Update `_build_obs_dict()` to include belief (before return statement):**

```python
# After line ~197 (self.buffer.add(raw_obs)), before the return statement:

        # Update belief tracker
        is_visible = d_to_opp >= 0
        self.belief_tracker.update(own_state, target_state, is_visible)
        belief = self.belief_tracker.get_statistics()
```

**Update the return dict to include `"belief"`:**

```python
        return {
            "obs_history": self.buffer.get_history(),
            "lidar": lidar.reshape(1, -1).astype(np.float32),
            "state": state_features,
            "belief": belief,
        }
```

### Step 2.2: Run existing PartialObsWrapper tests

```bash
pytest tests/test_env.py -v -k partial
```

Expected: Any existing partial-obs tests should still pass (they may not reference `"belief"`).

### Step 2.3: Commit

```bash
git add envs/partial_obs_wrapper.py
git commit -m "feat: integrate BeliefTracker into PartialObsWrapper"
```

---

## Task 3: Policy Network Belief Branch

**Files:**
- Modify: `agents/partial_obs_policy.py`

### Step 3.1: Modify PartialObsPolicyNetwork

In `agents/partial_obs_policy.py`, update `PartialObsPolicyNetwork.__init__`:

**Add `belief_dim` parameter (default `None` for backward compatibility):**

```python
    def __init__(
        self,
        raw_obs_dim: int = 43,
        lidar_dim: int = 36,
        state_dim: int = 7,
        belief_dim: int | None = None,  # NEW
        encoder_type: str = "bimdn",
        encoder_kwargs: dict | None = None,
        features_dim: int = 256,
    ):
```

**Add belief MLP branch after state_mlp (around line ~78):**

```python
        # State branch: [x, y, theta, v, omega, d_to_opp, bearing]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Belief branch (optional for backward compatibility)
        self.belief_dim = belief_dim
        if belief_dim is not None:
            self.belief_mlp = nn.Sequential(
                nn.Linear(belief_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
            )
            combined_dim = 256 + 64 + encoder_latent_dim + 64
        else:
            self.belief_mlp = None
            combined_dim = 256 + 64 + encoder_latent_dim
```

**Update `forward()` to handle `"belief"` key:**

```python
    def forward(self, obs_dict: dict) -> torch.Tensor:
        # Belief encoding from observation history
        latent = self.encoder.encode(obs_dict["obs_history"])

        # Lidar spatial features
        lidar_features = self.lidar_conv(obs_dict["lidar"])

        # State features
        state_features = self.state_mlp(obs_dict["state"])

        # Combine all branches
        branches = [lidar_features, state_features, latent]

        # Belief branch (if present in obs_dict and network has belief_mlp)
        if self.belief_mlp is not None and "belief" in obs_dict:
            belief_features = self.belief_mlp(obs_dict["belief"])
            branches.append(belief_features)

        combined = torch.cat(branches, dim=-1)
        return self.combined_mlp(combined)
```

### Step 3.2: Modify PartialObsFeaturesExtractor

In `PartialObsFeaturesExtractor.__init__`, infer `belief_dim` from observation space:

```python
        # Infer dimensions from observation space
        obs_hist_shape = observation_space["obs_history"].shape
        lidar_shape = observation_space["lidar"].shape
        state_shape = observation_space["state"].shape
        raw_obs_dim = obs_hist_shape[-1]
        lidar_dim = lidar_shape[-1]
        state_dim = state_shape[0]

        # NEW: infer belief_dim if present in observation space
        belief_dim = None
        if "belief" in observation_space.spaces:
            belief_dim = observation_space["belief"].shape[0]

        self.net = PartialObsPolicyNetwork(
            raw_obs_dim=raw_obs_dim,
            lidar_dim=lidar_dim,
            state_dim=state_dim,
            belief_dim=belief_dim,  # NEW
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            features_dim=features_dim,
        )
```

### Step 3.3: Run policy network tests

```bash
pytest tests/test_bimdn.py -v  # Or any test that uses PartialObsFeaturesExtractor
```

If no such tests exist, run a quick Python check:

```bash
python -c "
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents.partial_obs_policy import PartialObsFeaturesExtractor

obs_space = spaces.Dict({
    'obs_history': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 43), dtype=np.float32),
    'lidar': spaces.Box(low=0.0, high=5.0, shape=(1, 36), dtype=np.float32),
    'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
    'belief': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
})
extractor = PartialObsFeaturesExtractor(obs_space, features_dim=256, encoder_type='mlp')
print('belief_dim:', extractor.net.belief_dim)
print('combined_mlp input dim:', extractor.net.combined_mlp[0].in_features)
assert extractor.net.belief_dim == 6
assert extractor.net.combined_mlp[0].in_features == 256 + 64 + 32 + 64  # mlp latent=32
print('OK')
"
```

Expected: Prints `belief_dim: 6`, `combined_mlp input dim: 416`, then `OK`.

### Step 3.4: Commit

```bash
git add agents/partial_obs_policy.py
git commit -m "feat: add belief branch to PartialObsPolicyNetwork"
```

---

## Task 4: PartialObsOpponentAdapter Belief Support

**Files:**
- Modify: `envs/opponent_adapter.py`

### Step 4.1: Add BeliefTracker to PartialObsOpponentAdapter

In `envs/opponent_adapter.py`, make these changes:

**Add import at top:**

```python
from envs.belief_tracker import BeliefTracker
```

**Add BeliefTracker to `__init__` (after sensor init, around line ~60):**

```python
        # Belief tracker for opponent position
        if self.role == "pursuer":
            opp_v_max = base_env.evader_v_max
        else:
            opp_v_max = base_env.pursuer_v_max

        self.belief_tracker = BeliefTracker(
            role=self.role,
            arena_width=base_env.arena_width,
            arena_height=base_env.arena_height,
            opponent_v_max=opp_v_max,
            n_particles=100,
            seed=0,
        )
```

**Update `_build_obs_dict()` to append belief (after the existing return dict creation):**

In `_build_obs_dict()`, after building the `raw_obs` and adding to buffer, but before the `return` statement, add:

```python
        # Update belief tracker
        is_visible = d_to_opp >= 0
        self.belief_tracker.update(own_state, target_state, is_visible)
        belief = self.belief_tracker.get_statistics()
```

**Update the return dict:**

```python
        return {
            "obs_history": self.buffer.get_history(),
            "lidar": lidar.reshape(1, -1).astype(np.float32),
            "state": state_features,
            "belief": belief,
        }
```

**Update `reset()` to reset belief tracker (after buffer.reset):**

```python
    def reset(self):
        """Reset the adapter's internal state for a new episode."""
        self.buffer.reset()
        self.last_known_opp_pos = None
        self.steps_since_seen = 0
        self.belief_tracker.reset()
```

### Step 4.2: Verify adapter works

Run a quick Python check:

```bash
python -c "
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.opponent_adapter import PartialObsOpponentAdapter
import numpy as np

env = PursuitEvasionEnv(arena_width=20, arena_height=20)
env.reset(seed=42)

# Create a dummy model that just checks obs dict keys
class DummyModel:
    def predict(self, obs, deterministic=False):
        assert 'belief' in obs, f'obs missing belief key. Got: {obs.keys()}'
        return np.array([0.5, 0.1]), None

adapter = PartialObsOpponentAdapter(
    model=DummyModel(), role='pursuer', base_env=env, deterministic=True,
)
obs_dict = adapter._build_obs_dict()
print('obs keys:', obs_dict.keys())
print('belief shape:', obs_dict['belief'].shape)
assert obs_dict['belief'].shape == (6,)
print('OK')
"
```

Expected: Prints `obs keys: dict_keys([...])`, `belief shape: (6,)`, then `OK`.

### Step 4.3: Commit

```bash
git add envs/opponent_adapter.py
git commit -m "feat: add BeliefTracker support to PartialObsOpponentAdapter"
```

---

## Task 5: Integration Smoke Test

**Files:**
- Modify: `tests/test_training_smoke.py` (or create if not exists)

### Step 5.1: Add smoke test for belief-enabled training

At the end of `tests/test_training_smoke.py` (or create the file):

```python
# --- Smoke test: Belief-enabled partial obs wrapper ---

def test_belief_observation_dict():
    """PartialObsWrapper should include 'belief' key in obs dict."""
    from envs.pursuit_evasion_env import PursuitEvasionEnv
    from envs.partial_obs_wrapper import PartialObsWrapper

    base_env = PursuitEvasionEnv(arena_width=20, arena_height=20)
    wrapper = PartialObsWrapper(base_env, role="pursuer")

    obs, info = wrapper.reset(seed=42)

    assert "belief" in obs, f"obs dict missing 'belief' key. Got: {obs.keys()}"
    assert obs["belief"].shape == (6,), f"belief shape should be (6,), got {obs['belief'].shape}"
    assert obs["belief"].dtype == np.float32


def test_belief_after_step():
    """Belief should update after each step."""
    from envs.pursuit_evasion_env import PursuitEvasionEnv
    from envs.partial_obs_wrapper import PartialObsWrapper

    base_env = PursuitEvasionEnv(arena_width=20, arena_height=20)
    wrapper = PartialObsWrapper(base_env, role="pursuer")

    obs, _ = wrapper.reset(seed=42)
    initial_belief = obs["belief"].copy()

    # Take one step (need to pass both actions through base env)
    action = np.array([0.5, 0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapper.step(action)

    assert "belief" in obs
    # Belief should have changed after a step
    assert not np.array_equal(obs["belief"], initial_belief), \
        "belief should change after stepping"
```

Run:

```bash
pytest tests/test_training_smoke.py -v -k belief
```

Expected: Both tests PASS.

### Step 5.2: Commit

```bash
git add tests/test_training_smoke.py
git commit -m "test: add smoke tests for belief-enabled PartialObsWrapper"
```

---

## Task 6: Full Test Suite Verification

### Step 6.1: Run all related tests

```bash
pytest tests/test_belief_tracker.py tests/test_training_smoke.py -v
```

Expected: All tests PASS.

### Step 6.2: Quick integration sanity check

```bash
python -c "
import numpy as np
from envs.pursuit_evasion_env import PursuitEvasionEnv
from envs.partial_obs_wrapper import PartialObsWrapper
from envs.opponent_adapter import PartialObsOpponentAdapter
from agents.partial_obs_policy import PartialObsFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

# 1. Wrapper produces correct obs dict
base = PursuitEvasionEnv(arena_width=20, arena_height=20)
wrapper = PartialObsWrapper(base, role='pursuer')
obs, _ = wrapper.reset(seed=42)
assert 'belief' in obs
print('Wrapper OK')

# 2. Policy network accepts belief obs
obs_space = wrapper.observation_space
extractor = PartialObsFeaturesExtractor(obs_space, features_dim=256, encoder_type='mlp')
import torch
batch_obs = {
    'obs_history': torch.from_numpy(obs['obs_history']).unsqueeze(0),
    'lidar': torch.from_numpy(obs['lidar']).unsqueeze(0),
    'state': torch.from_numpy(obs['state']).unsqueeze(0),
    'belief': torch.from_numpy(obs['belief']).unsqueeze(0),
}
features = extractor(batch_obs)
assert features.shape == (1, 256)
print('Policy network OK')

# 3. Opponent adapter produces belief obs
class DummyModel:
    def predict(self, obs, deterministic=False):
        return np.array([0.5, 0.1]), None

adapter = PartialObsOpponentAdapter(model=DummyModel(), role='pursuer', base_env=base)
action, _ = adapter.predict(np.zeros(6))
print('Opponent adapter OK')

print('All integration checks passed!')
"
```

Expected: Prints `Wrapper OK`, `Policy network OK`, `Opponent adapter OK`, `All integration checks passed!`.

### Step 6.3: Final commit

```bash
git commit -m "chore: verify belief-state integration end-to-end"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Plan Task | Status |
|-------------|-----------|--------|
| 3.1 BeliefTracker (particle filter, 6-D output) | Task 1 | Covered |
| 3.1 Process noise (σ ∝ sqrt(t), dt=0.05) | Task 1 | Covered |
| 3.1 Arena clamping | Task 1 (test D) | Covered |
| 3.1 Safe default before update | Task 1 (test E) | Covered |
| 3.2 PartialObsWrapper (add belief key) | Task 2 | Covered |
| 3.3 PartialObsPolicyNetwork (belief_mlp branch) | Task 3 | Covered |
| 3.3 Backward compatibility (belief optional) | Task 3 | Covered (belief_dim=None default) |
| 3.4 PartialObsOpponentAdapter (belief support) | Task 4 | Covered |
| 5 Error handling (clamping, safe default) | Task 1 | Covered |
| 6.1 Unit tests (visible, hidden, reappearance) | Task 1 | Covered |
| 6.2 Smoke test | Task 5 | Covered |

### Placeholder Scan

- [x] No "TBD", "TODO", "implement later"
- [x] No vague "add error handling" steps
- [x] All test code is complete with assertions
- [x] All implementation code is complete
- [x] No "similar to Task N" references

### Type Consistency

- [x] `BeliefTracker.get_statistics()` returns `np.ndarray` shape `(6,)` dtype `float32` — used consistently
- [x] `belief_dim` parameter is `int | None` throughout
- [x] `obs_dict["belief"]` shape `(6,)` matches `belief_mlp` input `Linear(6, 32)`
- [x] `dt = 0.05` used consistently in BeliefTracker

---

## Post-Implementation: Launch Instructions

After all tasks are complete, launch BELIEF_s200 on niro-2:

```bash
# 1. Push code to GitHub
git push origin main

# 2. SSH to niro-2 and pull
sshpass -p '123456' ssh niro-2@100.71.2.97 'cd /home/niro-2/Codes/safe-rl-pe && git pull'

# 3. Launch training
sshpass -p '123456' ssh niro-2@100.71.2.97 'cd /home/niro-2/Codes/safe-rl-pe && nohup .venv/bin/python -u scripts/train_amsdrl.py \
    --algorithm sac \
    --fov_angle 90 --sensing_radius 8.0 --combined_masking \
    --arena_width 20 --arena_height 20 --max_steps 1200 \
    --n_obstacles_min 0 --n_obstacles_max 3 \
    --pursuer_v_max 1.0 --evader_v_max 1.0 \
    --lr 3e-4 --ent_coef 0.03 \
    --micro_phase_steps 2048 --eval_interval 150 \
    --alternate_freeze --freeze_switch_threshold 0.65 \
    --force_switch_steps 2000000 \
    --w_vis_pursuer 0.2 --w_search 0.0001 --t_stale 100 \
    --collapse_threshold 0.1 --collapse_streak_limit 5 \
    --pfsp \
    --max_total_steps 30000000 \
    --seed 200 \
    --output results/BELIEF_s200 \
    > results/BELIEF_s200.log 2>&1 &'
```

Note: `--masking_curriculum` and `--p_full_anneal_steps` are **intentionally omitted** — pure PO from start.
