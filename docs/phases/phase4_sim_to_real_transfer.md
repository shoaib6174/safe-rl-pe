# Phase 4: Sim-to-Real Transfer (Months 6-8)

## Self-Contained Implementation Document
**Project**: Safe Deep RL for 1v1 Ground Robot Pursuit-Evasion
**Phase**: 4 of 5
**Timeline**: Months 6-8
**Status**: Implementation-ready
**Date**: 2026-02-21

---

## 1. Phase Overview

### 1.1 Title and Timeline
**Phase 4: Sim-to-Real Transfer** covers months 6-8 of the project. This phase takes the trained, safety-guaranteed pursuit-evasion policies from simulation and deploys them on physical robots through a staged transfer pipeline.

### 1.2 Objectives
1. Port the Gymnasium PE environment to Isaac Lab for GPU-accelerated training with domain randomization
2. Re-train policies with domain randomization to produce robust, transferable behaviors
3. Export trained policies to ONNX format for edge deployment
4. Validate in Gazebo/ROS2 as an intermediate step (catches ~80% of sim-to-real issues)
5. Implement the real-time RCBF-QP safety filter in C++ for deployment
6. Implement GP disturbance estimation with cold-start protocol for online adaptation
7. Deploy on physical robots (TurtleBot4 or F1TENTH) and achieve <10% performance gap vs simulation
8. Achieve zero safety violations on the real robot at 20Hz control

### 1.3 Prerequisites (from Phases 1-3)
The following artifacts MUST exist before starting Phase 4:

| Artifact | Source Phase | Description |
|----------|-------------|-------------|
| `pursuit_evasion_env.py` | Phase 1 | Gymnasium PE environment with unicycle dynamics |
| Trained PPO pursuer policy (`.pt`) | Phase 3 | Self-play converged, CBF-Beta constrained |
| Trained PPO evader policy (`.pt`) | Phase 3 | Self-play converged, CBF-Beta constrained |
| VCP-CBF implementation | Phase 2 | Virtual control point CBFs for nonholonomic robots |
| RCBF-QP safety filter (Python) | Phase 2 | Robust CBF-QP with 3-tier infeasibility handling |
| BiMDN belief encoder | Phase 3 | Handles partial observability |
| Safety architecture decision | Phase 2.5 | CBF-Beta -> RCBF-QP OR BarrierNet end-to-end |
| NE convergence verification | Phase 3 | Self-play converged with |SR_P - SR_E| < 0.10 |
| W&B training logs | Phases 1-3 | Full training history for comparison |

### 1.4 Relationship to Project
Phase 4 is the bridge between simulation research (Phases 1-3) and real-world demonstration (Phase 5 paper). It transforms simulation-validated algorithms into deployable robot software. The sim-to-real pipeline follows [N10] Salimpour 2025, which validates this exact approach (Isaac -> ONNX -> Gazebo -> Real) with 80-100% real-world success rates.

### 1.5 Key Deliverables
- Isaac Lab PE environment with domain randomization and PettingZoo MARL API
- ONNX-exported policy models (pursuer and evader) with quantization
- ROS2 Humble node architecture for real-time PE
- C++ RCBF-QP safety filter achieving <50ms solve time
- GPyTorch-based online disturbance estimator with cold-start protocol
- Real robot deployment with zero safety violations
- Comprehensive sim-to-real performance gap measurements
- Hydra config files for DR parameters, deployment configs, and QP solver settings (`conf/deployment/`, `conf/domain_randomization/`)
- wandb dashboards for all sim-to-real experiments (Isaac Lab, Gazebo, real robot), with eval videos and gap analysis tables

---

## 2. Background and Theoretical Foundations

### 2.1 The Sim-to-Real Gap

The sim-to-real gap refers to the performance degradation when transferring policies trained in simulation to physical robots. For safety-critical PE, this gap is especially dangerous because:

**Sources of the gap:**

| Source | Description | Impact on Safety |
|--------|-------------|-----------------|
| Dynamics mismatch | Simulated dynamics differ from real robot (mass, friction, motor response) | Robot may move faster/slower than expected, CBF margins may be violated |
| Sensor noise | Real sensors (lidar, odometry) have noise, dropouts, and latency | Noisy observations lead to incorrect state estimates, unsafe actions |
| Actuator differences | Real motors have dead zones, backlash, saturation, nonlinear response curves | Commanded velocities not achieved, tracking error |
| Control latency | Computation, communication, and actuator delays (10-100ms) | Safety filter decisions based on stale state |
| Unmodeled dynamics | Wheel slip, uneven floor, battery voltage sag, thermal effects | Unpredictable disturbances to nominal model |
| Localization error | Real localization (AMCL, visual odometry) drifts and has bounded accuracy | Position uncertainty affects CBF constraint evaluation |

**Why it matters for safety:** A policy that achieves zero safety violations in simulation may violate constraints on a real robot if the dynamics mismatch causes the robot to overshoot, undershoot, or respond with unexpected latency. The RCBF-QP safety filter with GP disturbance estimation is specifically designed to handle this -- but it must be properly initialized and tuned.

### 2.2 Domain Randomization (DR)

**Principle:** Train the policy across a distribution of environment parameters so that the real world falls within the training distribution. Instead of matching the simulator to reality (system identification), DR makes the policy robust to a range of conditions.

**What to randomize and how much:**

| Parameter | Nominal | Range | Distribution | Rationale |
|-----------|---------|-------|--------------|-----------|
| Robot mass | 1.0 kg | [0.8, 1.2] kg | Uniform | Battery, payload variation |
| Wheel friction | 0.5 | [0.3, 0.8] | Uniform | Floor surface variation |
| Motor gain | 1.0 | [0.8, 1.2] | Uniform | Motor aging, voltage variation |
| Sensor noise (lidar) | 0.0 m | N(0, 0.02m) | Gaussian per ray | Lidar measurement noise |
| Sensor noise (odom) | 0.0 | N(0, 0.01m) pos, N(0, 0.5 deg) heading | Gaussian per step | Odometry drift |
| Control delay | 0 ms | [0, 50] ms | Uniform integer | Communication + compute |
| Obstacle positions | fixed | +/- 0.5m | Uniform per axis | Placement uncertainty |
| Arena size | 20 m | [18, 22] m | Uniform | Arena calibration error |
| v_max | 1.0 m/s | [0.85, 1.15] m/s | Uniform | Motor capability variation |

**Key papers:**
- [07] Salimpour 2025: DR for TurtleBot sim-to-real with Isaac Sim
- [N10] Salimpour 2025: Validates DR in Isaac -> ONNX -> Gazebo -> Real pipeline (80-100% success)
- [N11] Mittal 2025: Isaac Lab native DR capabilities, GPU-accelerated

**Implementation strategy:** Randomize at episode reset. Each parallel environment gets an independent parameter sample. This is trivial in Isaac Lab which supports thousands of parallel environments.

### 2.3 Isaac Lab [N11]

Isaac Lab (formerly Isaac Gym / Orbit) is NVIDIA's GPU-accelerated robot learning framework. Key features for this project:

- **Massively parallel simulation**: Up to 4096+ environments on a single GPU, achieving up to 1.6M FPS for simple environments
- **PettingZoo Parallel API for MARL**: Native support for multi-agent RL, which is exactly what our 2-player PE game requires. Both agents step simultaneously in each environment.
- **Built-in domain randomization**: Randomize physics parameters (mass, friction, damping), sensor noise, and actuator properties through configuration
- **Sensor simulation**: Simulated lidar (ray-casting), cameras (RTX rendering), and IMU
- **Direct GPU tensor interface**: Observations and actions as GPU tensors -- no CPU-GPU transfer bottleneck
- **USD-based scene definition**: Articulated robots, obstacles, and arenas defined in Universal Scene Description format
- **PyTorch integration**: Direct PyTorch tensor outputs for seamless RL training

**Why Isaac Lab over PyBullet/MuJoCo:** Our Phase 1-3 Gymnasium environment runs on CPU. Isaac Lab provides 100-1000x speedup through GPU parallelism. This is critical for domain randomization, which requires training across many parameter configurations. A policy that takes 10 hours to train in Gymnasium can train in minutes in Isaac Lab.

### 2.4 ONNX Export Pipeline [N10]

The Open Neural Network Exchange (ONNX) format enables cross-platform deployment of trained PyTorch models:

```
PyTorch model (.pt)
    |-- torch.onnx.export() --> ONNX model (.onnx)
        |-- onnxruntime.quantization --> Quantized ONNX (.onnx, INT8/FP16)
            |-- ONNX Runtime inference on target hardware
```

**Key steps:**
1. **Export**: `torch.onnx.export(model, dummy_input, "policy.onnx", opset_version=17)`
2. **Verify**: Compare PyTorch output vs ONNX output on test inputs (rtol=1e-5)
3. **Quantize**: Dynamic quantization (INT8) for RPi4, FP16 for Jetson
4. **Benchmark**: Measure inference latency on target hardware (must be <5ms per step)

**[N10] validates this pipeline:** Salimpour et al. 2025 used Isaac Sim -> ONNX -> Gazebo -> Real robot with 80-100% task success. Their ONNX models ran at ~5ms inference on embedded hardware.

### 2.5 Gazebo/ROS2 Intermediate Validation

Gazebo Fortress (with ROS2 Humble) serves as an intermediate validation step between Isaac Lab training and real robot deployment. According to [N10], this step catches approximately 80% of sim-to-real issues.

**What Gazebo catches that Isaac Lab does not:**
- ROS2 communication latency and topic synchronization issues
- Realistic sensor plugins (lidar noise models, camera distortion)
- Control loop timing under real OS scheduling
- Nav2 stack integration issues (localization, costmaps)
- Message serialization/deserialization overhead

**What only the real robot reveals (~20% of issues):**
- Actual motor response curves and dead zones
- Real floor surface friction
- WiFi latency for multi-robot coordination
- Battery voltage effects on motor performance
- Physical sensor mounting inaccuracies

### 2.6 GP Disturbance Estimation [06]

Gaussian Process (GP) regression provides a principled way to learn the dynamics residual (difference between nominal model and real dynamics) online.

**Model:**
```
x_{t+1} = f(x_t) + g(x_t) * u_t + d(x_t)    [true dynamics]

d(x_t) ~ GP(m(x), k(x, x'))                    [disturbance model]

where:
  f(x) + g(x)*u  = nominal unicycle dynamics
  d(x)            = unmodeled disturbance (wheel slip, motor nonlinearity, etc.)
  m(x)            = mean function (zero or pre-filled from simulation)
  k(x, x')        = kernel function (Squared Exponential with ARD)
```

**Online learning loop:**
1. At each timestep, observe actual next state x_{t+1}
2. Compute residual: d_t = x_{t+1} - f(x_t) - g(x_t) * u_t
3. Add (x_t, d_t) to GP dataset
4. Update GP posterior (every N=10 steps to manage compute)
5. GP provides d_hat(x) (mean prediction) and sigma_d(x) (uncertainty)

**Kernel selection:** Squared Exponential (SE) with Automatic Relevance Determination (ARD):
```
k(x, x') = sigma_f^2 * exp(-0.5 * sum_i (x_i - x'_i)^2 / l_i^2)
```
ARD learns separate lengthscales l_i per input dimension, automatically determining which state dimensions affect the disturbance most.

**Update frequency:** Every 10 control steps (0.5s at 20Hz). GP posterior update is O(n^3) in dataset size; use a sliding window of the most recent 200 data points to keep this tractable.

### 2.7 GP Cold-Start Protocol [06]

When deploying on a new real robot, the GP has no data. This creates a chicken-and-egg problem: the RCBF-QP needs GP predictions for robust safety, but the GP needs real data to make predictions.

**Three-step protocol:**

```
Step 1: Pre-fill GP with simulation residual data (~500-1000 points)
  - Run the trained policy in simulation WITH domain randomization
  - Collect (x, u, d_residual) tuples where d comes from the randomized dynamics
  - Initialize the GP dataset with these points
  - This gives a reasonable prior: the GP "knows" roughly what disturbances look like
  - Use the MEDIAN randomization setting, not extremes

Step 2: Conservative margin for first 100 real-world steps
  - Set kappa_init = 2.0 * kappa_nominal (double the robust margin)
  - This forces the RCBF-QP to be extra conservative
  - The robot will move more slowly and cautiously
  - Accept performance degradation for safety during initialization

Step 3: Exponential transition to normal margin
  - kappa(t) = kappa_nominal + (kappa_init - kappa_nominal) * exp(-t / tau)
  - tau = 50 steps (2.5 seconds at 20Hz) -- tune based on GP convergence
  - Monitor: mean GP posterior variance sigma_bar_d
  - Early termination: if sigma_bar_d < 0.01, immediately set kappa = kappa_nominal
```

### 2.8 RCBF-QP for Deployment

The Robust Control Barrier Function QP combines the trained RL policy with real-time safety enforcement:

```
u* = argmin_{u} ||u - u_RL||^2
subject to:
  L_f h_i + L_g h_i * u + L_d h_i * d_hat >= -alpha * h_i(x) + kappa * sigma_d(x)
  u_min <= u <= u_max

where:
  u_RL     = ONNX policy output (desired action)
  h_i(x)  = VCP-CBF constraint i (arena, obstacles, inter-robot)
  d_hat    = GP mean disturbance prediction
  sigma_d  = GP uncertainty (standard deviation)
  kappa    = robust margin coefficient (from cold-start protocol)
  alpha    = CBF class-K function parameter (1.0)
```

The QP minimally modifies the RL action to satisfy all safety constraints. The GP uncertainty sigma_d provides a probabilistic safety margin -- larger uncertainty means more conservative behavior.

### 2.9 QP Solver Benchmarking

The RCBF-QP must solve within the 50ms control budget (20Hz). Two candidate solvers:

| Solver | Language | License | Warm-start | Typical Solve Time |
|--------|----------|---------|------------|-------------------|
| OSQP | C (Python/C++ bindings) | Apache 2.0 | Yes | 0.1-5ms for small QPs |
| qpOASES | C++ | LGPL 2.1 | Yes (active set) | 0.05-2ms for small QPs |

**Our QP size:** 2 decision variables (v, omega), 3-6 inequality constraints (arena + obstacles + collision), 4 bound constraints. This is a tiny QP. Both solvers should handle it in <1ms.

**Benchmarking protocol:**
1. Generate 10,000 random state-action pairs from simulation rollouts
2. Solve the RCBF-QP for each pair
3. Record solve times, report: mean, median, 95th percentile, 99th percentile, max
4. Target: 95th percentile < 50ms (should be achievable with large margin)
5. Test on target hardware (RPi4 for TurtleBot, Jetson for F1TENTH)

### 2.10 Hardware Platforms

| Platform | Dynamics | Max Speed | Compute | Cost | Pros | Cons |
|----------|----------|-----------|---------|------|------|------|
| TurtleBot3 Burger | Diff-drive | 0.22 m/s | RPi 4 (4GB) | ~$600 | Cheap, well-documented, ROS2 native | Very slow, limited compute |
| TurtleBot4 | Diff-drive | 0.3 m/s | RPi 4 + iRobot Create3 | ~$1,200 | Better sensors, Create3 base, ROS2 native | Still slow, moderate cost |
| F1TENTH | Ackermann | 3+ m/s | Jetson Xavier NX | ~$3,000 | Fast, powerful compute, active community | Expensive, Ackermann (not diff-drive), safety risk at speed |
| Custom diff-drive | Diff-drive | 1.0 m/s | Jetson Nano (4GB) | ~$500 | Matched to training v_max, budget-friendly | Requires custom build, no warranty |

**Recommendation:** TurtleBot4 for initial deployment (safest, best documented, ROS2 native). F1TENTH for high-speed demonstration if budget allows. The v_max mismatch (training: 1.0 m/s, TurtleBot4: 0.3 m/s) requires re-scaling velocity commands or re-training with matched v_max.

**Velocity matching strategy:** Re-train in Isaac Lab with v_max = 0.3 m/s (TurtleBot4) or v_max = 3.0 m/s (F1TENTH) during domain randomization. The DR range for v_max should center on the real robot's capability.

### 2.11 ROS2 Integration

**Node architecture:**
```
/lidar_scan ─────┐
/odom ───────────┤
/camera (opt) ───┴──> /sensor_fusion ──> /observation_encoder ──> /policy_inference ──> /safety_filter ──> /cmd_vel
                                              (BiMDN)               (ONNX Runtime)       (RCBF-QP C++)
                                                                                              ^
                                                                                              |
                                                                                        /gp_estimator
                                                                                         (GPyTorch)
```

**Topic structure:**

| Topic | Type | Rate | Publisher | Subscriber |
|-------|------|------|-----------|------------|
| `/scan` | `sensor_msgs/LaserScan` | 10Hz | Lidar driver | `/sensor_fusion` |
| `/odom` | `nav_msgs/Odometry` | 20Hz | Robot base | `/sensor_fusion` |
| `/fused_state` | custom `PEState` | 20Hz | `/sensor_fusion` | `/observation_encoder` |
| `/observation` | custom `PEObservation` | 20Hz | `/observation_encoder` | `/policy_inference` |
| `/policy_action` | `geometry_msgs/Twist` | 20Hz | `/policy_inference` | `/safety_filter` |
| `/gp_disturbance` | custom `GPPrediction` | 2Hz | `/gp_estimator` | `/safety_filter` |
| `/cmd_vel` | `geometry_msgs/Twist` | 20Hz | `/safety_filter` | Robot base |
| `/pe_diagnostics` | custom `PEDiagnostics` | 1Hz | All nodes | W&B logger |

**Control loop timing (20Hz = 50ms budget):**
```
Sensor read:        ~2ms  (lidar + odom subscription callback)
Sensor fusion:      ~1ms  (EKF or simple fusion)
Observation encode: ~2ms  (BiMDN forward pass)
Policy inference:   ~5ms  (ONNX Runtime)
Safety filter:     ~10ms  (RCBF-QP solve + GP lookup)
Motor command:      ~1ms  (publish cmd_vel)
Total:            ~21ms   (29ms margin for scheduling jitter)

GP update (async):  ~50ms every 10 steps (runs in separate thread)
```

---

## 3. Relevant Literature

| Paper | Key Contribution for Phase 4 | What to Use |
|-------|------------------------------|-------------|
| [06] Emam 2022 | RCBF + GP disturbance estimation | GP kernel, online update, robust QP formulation |
| [07] Salimpour 2025 | Sim-to-real pipeline for TurtleBot | Domain randomization ranges, Isaac Sim setup, Gazebo validation |
| [N10] Salimpour 2025 | Isaac -> ONNX -> Gazebo -> Real validated | ONNX export procedure, quantization, 80-100% success rate benchmark |
| [N11] Mittal 2025 | Isaac Lab framework | PettingZoo Parallel API for MARL, GPU-accelerated DR, sensor simulation |
| [N12] Zhang & Yang 2025 | VCP-CBF for nonholonomic robots | Use VCP formulation in the C++ RCBF-QP solver |
| [02] Gonultas 2024 | Real PE on F1TENTH | Sensor setup, BiMDN on real hardware, curriculum for transfer |
| [18] Xiao 2024 | AMS-DRL, real drone deployment | ONNX export, real-time inference, sim-real gap measurement (4%) |
| [05] Yang 2025 | CBF safety filtering | Closed-form CBF as fallback if QP too slow |
| [16] Suttle 2024 | CBF-Beta policy | Verify that ONNX-exported policy preserves CBF-Beta structure |

---

## 4. Session-wise Implementation Breakdown

### Session 1: Set Up Isaac Lab Environment for PE

**Objectives:**
- Install Isaac Lab (Isaac Sim 4.x + isaaclab extension)
- Configure a basic empty arena scene with a ground plane and walls
- Verify GPU-accelerated simulation runs with at least 1000 parallel environments

**Files to create/modify:**
- `isaac_lab/install_isaac_lab.sh` -- installation script
- `isaac_lab/envs/__init__.py` -- environment registration
- `isaac_lab/envs/pe_scene.py` -- USD scene definition (arena, walls)
- `isaac_lab/config/pe_env_cfg.py` -- environment configuration

**Instructions:**

1. Install Isaac Lab following the official documentation (requires NVIDIA GPU with CUDA 11.8+):
```bash
# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
# Install in conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install -e .
# Verify installation
python -c "import isaaclab; print(isaaclab.__version__)"
```

2. Create a basic PE arena scene in USD format:
   - Ground plane (20m x 20m)
   - Four wall segments (rectangular boundary)
   - Two differential-drive robot assets (TurtleBot3 URDF or simplified cylinder)
   - Set up physics materials (friction, restitution)

3. Create the environment configuration dataclass:
   - `num_envs`: 2048 (start with this, scale up later)
   - `dt`: 0.05 (20Hz)
   - `arena_size`: 20.0m
   - `episode_length`: 1200 steps (60s)

4. Verify by running a random action loop for 100 steps across all parallel envs.

**Verification:**
- Isaac Lab loads without errors
- At least 1000 parallel environments run simultaneously
- FPS > 10,000 (should be much higher for simple unicycle)
- GPU memory usage is reasonable (< 8GB for 2048 envs)

---

### Session 2: Port Gymnasium PE Env to Isaac Lab

**Objectives:**
- Implement the full PE environment logic in Isaac Lab
- Use PettingZoo Parallel API for the two-agent structure
- Verify observations, rewards, and termination match the Gymnasium version

**Files to create/modify:**
- `isaac_lab/envs/pursuit_evasion_env.py` -- main environment class
- `isaac_lab/envs/pe_reward.py` -- reward computation (GPU-vectorized)
- `isaac_lab/envs/pe_observation.py` -- observation construction
- `isaac_lab/envs/pe_termination.py` -- termination conditions

**Instructions:**

1. Implement the PettingZoo Parallel API wrapper:
```python
# Pseudocode for Isaac Lab PE env
class IsaacLabPEEnv:
    """
    PettingZoo Parallel API compatible PE environment in Isaac Lab.
    Both agents step simultaneously.
    """
    def __init__(self, cfg):
        self.num_envs = cfg.num_envs  # e.g. 2048
        # Isaac Lab handles physics stepping internally

    def step(self, actions_dict):
        # actions_dict = {"pursuer": tensor[num_envs, 2], "evader": tensor[num_envs, 2]}
        # Apply unicycle dynamics on GPU
        # Compute observations, rewards, dones (all GPU tensors)
        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict
```

2. Implement unicycle dynamics on GPU:
   - State: [x, y, theta] per robot (6D joint state per env)
   - Control: [v, omega] per robot
   - Forward Euler integration: x += v*cos(theta)*dt, y += v*sin(theta)*dt, theta += omega*dt
   - All operations as batched PyTorch tensor ops on GPU

3. Implement observations matching Phase 3 format:
   - Relative distance and bearing to opponent (if in FOV)
   - Own velocity
   - Lidar scan (use Isaac Lab ray-casting for GPU-accelerated lidar)
   - Belief state placeholder (will be filled by BiMDN)

4. Implement reward function matching Phase 3 (distance-based zero-sum + CBF margin bonus)

5. Cross-validate: Run 100 episodes with the same initial conditions in both Gymnasium and Isaac Lab, compare trajectories.

**Verification:**
- Observations shape matches Gymnasium version
- Rewards match within rtol=1e-4 for identical states and actions
- Episode termination (capture, timeout) triggers correctly
- 2048 environments step in parallel on GPU

---

### Session 3: Implement Domain Randomization in Isaac Lab

**Objectives:**
- Add domain randomization for all parameters in the DR table
- Verify randomization produces the expected distributions
- Test that a policy trained WITHOUT DR degrades with DR (confirms DR is working)

**Files to create/modify:**
- `isaac_lab/envs/pe_domain_randomization.py` -- DR parameter sampling
- `isaac_lab/config/dr_config.py` -- DR ranges and distributions
- `tests/test_domain_randomization.py` -- verify DR parameter distributions

**Instructions:**

1. Implement per-environment parameter randomization at episode reset:
```python
class DomainRandomizer:
    def randomize(self, env_ids):
        """Called at episode reset for specified environments."""
        n = len(env_ids)
        # Mass: Uniform [0.8, 1.2] kg
        self.masses[env_ids] = torch.empty(n).uniform_(0.8, 1.2)
        # Friction: Uniform [0.3, 0.8]
        self.frictions[env_ids] = torch.empty(n).uniform_(0.3, 0.8)
        # Motor gain: Uniform [0.8, 1.2]
        self.motor_gains[env_ids] = torch.empty(n).uniform_(0.8, 1.2)
        # Sensor noise std (lidar): fixed 0.02m
        self.lidar_noise_std[env_ids] = 0.02
        # Control delay: Uniform integer [0, 50] ms -> [0, 1] steps at 20Hz
        self.control_delay_steps[env_ids] = torch.randint(0, 2, (n,))
        # Arena size: Uniform [18, 22] m
        self.arena_sizes[env_ids] = torch.empty(n).uniform_(18.0, 22.0)
        # v_max: Uniform [0.85, 1.15] m/s
        self.v_max[env_ids] = torch.empty(n).uniform_(0.85, 1.15)
```

2. Modify the dynamics step to use randomized parameters:
   - Scale applied forces by motor_gain
   - Add Gaussian noise to lidar readings
   - Add Gaussian noise to odometry readings
   - Implement control delay buffer (store action, apply delayed)
   - Randomize obstacle positions at reset by +/-0.5m

3. Create a histogram visualization of all randomized parameters across environments.

**Verification:**
- Run 10,000 environment resets, plot parameter distributions -- all should match specified ranges
- A policy trained WITHOUT DR in Gymnasium shows measurably worse performance (>10% degradation in capture rate) when evaluated in Isaac Lab WITH DR
- DR overhead is <5% FPS reduction vs non-DR simulation

---

### Session 4: Re-train Policies with Domain Randomization

**Objectives:**
- Train pursuer and evader policies from scratch in Isaac Lab with DR
- Use the same self-play protocol (AMS-DRL) from Phase 3
- Achieve NE convergence comparable to Phase 3
- **Match v_max to target robot hardware** (critical for sim-to-real)

**Files to create/modify:**
- `isaac_lab/training/train_pe_dr.py` -- training script for Isaac Lab
- `isaac_lab/training/ams_drl_isaaclab.py` -- AMS-DRL self-play for Isaac Lab
- `isaac_lab/config/training_config.py` -- hyperparameters

**Instructions:**

1. **Match dynamics to target robot** (critical change from Phase 3):
   - For TurtleBot4: set `v_max = 0.3 m/s`, `omega_max = 1.82 rad/s`
   - For F1TENTH: set `v_max = 3.0 m/s`, `omega_max = 3.14 rad/s`
   - Scale arena to maintain similar time-to-cross ratio:
     - Phase 3 used 20m x 20m at 1.0 m/s → time-to-cross = 20s
     - TurtleBot4: 6m x 6m at 0.3 m/s → time-to-cross = 20s (matched)
     - F1TENTH: 20m x 20m at 3.0 m/s → time-to-cross = 6.7s (faster game)
   - Update VCP-CBF parameters: `d_vcp` and `safety_margin` scale with v_max
     - `d_vcp = 0.05 * (v_max / 1.0)` → 0.015m for TurtleBot4
     - `safety_margin = 0.1 * (v_max / 1.0)` → 0.03m for TurtleBot4
   - Update `max_steps` to preserve similar episode duration:
     - `max_steps = int(60.0 / dt)` = 1200 at 20Hz (same as Phase 1-3)
   - **DR center the v_max range on the REAL robot's v_max, not 1.0 m/s**:
     ```python
     dr_config['v_max'] = {'mean': 0.3, 'std': 0.06, 'distribution': 'normal'}  # TurtleBot4
     ```

2. Set up the training pipeline:
   - Use the same PPO hyperparameters from Phase 3 (lr=3e-4, gamma=0.99, etc.)
   - Use the same CBF-Beta constrained policy architecture
   - Use 2048-4096 parallel environments with DR enabled
   - Target: 10-50M total timesteps (DR requires more data than non-DR)

3. Run AMS-DRL self-play:
   - S0: Cold-start evader (500K steps)
   - S1: Train pursuer vs frozen evader (1M steps)
   - S2: Train evader vs frozen pursuer (1M steps)
   - Continue alternating until |SR_P - SR_E| < 0.10

4. Log all metrics to W&B:
   - Capture rate, escape rate per self-play phase
   - CBF violation rate (should be zero with CBF-Beta)
   - Episode length distribution
   - DR parameter distributions
   - **Log v_max and arena size in run config for traceability**

5. Compare DR-trained policies vs Phase 3 non-DR policies:
   - In nominal simulation: DR policies may be slightly worse (robustness vs optimality trade-off)
   - Under perturbation: DR policies should be significantly better
   - **Compare matched-v_max DR policy vs Phase 3 (1.0 m/s) policy deployed with velocity scaling**

**Verification:**
- NE convergence: |SR_P - SR_E| < 0.10
- Zero CBF violations during training
- DR-trained policies degrade <5% in nominal simulation vs non-DR policies
- **Matched-v_max policy performs >= velocity-scaled policy on Gazebo evaluation**
- DR-trained policies maintain >80% of nominal performance under worst-case DR parameters

---

### Session 5: ONNX Model Export

**Objectives:**
- Export pursuer and evader PyTorch policies to ONNX format
- Verify numerical equivalence between PyTorch and ONNX outputs
- Export both the policy network and the BiMDN belief encoder

**Files to create/modify:**
- `deployment/onnx_export.py` -- export script
- `deployment/onnx_verify.py` -- verification script
- `deployment/models/` -- directory for exported ONNX files

**Instructions:**

1. Export the policy network:
```python
import torch
import onnx

# Load trained PyTorch model
model = load_trained_policy("pursuer_dr.pt")
model.eval()

# Create dummy input matching observation shape
dummy_obs = torch.randn(1, obs_dim)  # single observation
dummy_hidden = torch.zeros(1, 1, hidden_dim)  # LSTM hidden state for BiMDN

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_obs, dummy_hidden),
    "deployment/models/pursuer_policy.onnx",
    opset_version=17,
    input_names=["observation", "hidden_state"],
    output_names=["action_mean", "action_std", "value", "hidden_state_out"],
    dynamic_axes={
        "observation": {0: "batch"},
        "hidden_state": {1: "batch"},
    }
)
```

2. Handle the BiMDN belief encoder:
   - If BiMDN is part of the policy network, it exports together
   - If separate, export as a second ONNX model
   - The LSTM/GRU hidden state must be managed externally in the ROS2 node

3. Verify numerical equivalence:
```python
import onnxruntime as ort
import numpy as np

# PyTorch output
with torch.no_grad():
    pt_output = model(test_obs, test_hidden)

# ONNX output
session = ort.InferenceSession("pursuer_policy.onnx")
onnx_output = session.run(None, {
    "observation": test_obs.numpy(),
    "hidden_state": test_hidden.numpy()
})

# Compare
for pt, ox in zip(pt_output, onnx_output):
    assert np.allclose(pt.numpy(), ox, rtol=1e-5, atol=1e-6), \
        f"Mismatch: max diff = {np.max(np.abs(pt.numpy() - ox))}"
```

4. Run verification on 1000 random inputs. Report max absolute difference.

**Verification:**
- ONNX model loads without errors in onnxruntime
- Max absolute difference between PyTorch and ONNX outputs < 1e-5 across 1000 test inputs
- ONNX model file size is reasonable (<50MB)
- Both pursuer and evader models exported successfully

---

### Session 6: ONNX Quantization and Inference Benchmarking

**Objectives:**
- Apply quantization (INT8 for RPi4, FP16 for Jetson) to reduce model size and inference time
- Benchmark inference latency on target hardware
- Verify quantized model output is acceptable (some degradation is okay)

**Files to create/modify:**
- `deployment/onnx_quantize.py` -- quantization script
- `deployment/onnx_benchmark.py` -- inference benchmarking
- `deployment/benchmark_results/` -- results directory

**Instructions:**

1. Apply dynamic quantization:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# INT8 quantization for RPi4
quantize_dynamic(
    "deployment/models/pursuer_policy.onnx",
    "deployment/models/pursuer_policy_int8.onnx",
    weight_type=QuantType.QInt8
)

# FP16 quantization for Jetson
# Use onnxconverter-common or ONNX Runtime's built-in FP16 converter
from onnxconverter_common import float16
model_fp16 = float16.convert_float_to_float16(
    onnx.load("deployment/models/pursuer_policy.onnx")
)
onnx.save(model_fp16, "deployment/models/pursuer_policy_fp16.onnx")
```

2. Benchmark on target hardware:
```python
import time
import onnxruntime as ort

session = ort.InferenceSession("pursuer_policy_int8.onnx")

# Warm-up
for _ in range(100):
    session.run(None, {"observation": dummy_obs, "hidden_state": dummy_hidden})

# Benchmark
latencies = []
for _ in range(1000):
    start = time.perf_counter_ns()
    session.run(None, {"observation": dummy_obs, "hidden_state": dummy_hidden})
    end = time.perf_counter_ns()
    latencies.append((end - start) / 1e6)  # ms

print(f"Mean: {np.mean(latencies):.2f}ms")
print(f"P95:  {np.percentile(latencies, 95):.2f}ms")
print(f"P99:  {np.percentile(latencies, 99):.2f}ms")
print(f"Max:  {np.max(latencies):.2f}ms")
```

3. Compare quantized vs full precision:
   - Run 1000 test observations through both models
   - Measure output difference (should be small)
   - Measure performance difference in simulation (capture rate, safety)

4. Run benchmarks on ALL target platforms (development machine, RPi4 via SSH, Jetson if available).

**Verification:**
- INT8 model inference < 5ms on RPi4 (mean)
- FP16 model inference < 2ms on Jetson (mean)
- Quantized model capture rate within 3% of full-precision model in simulation
- Zero additional safety violations from quantization
- Model size reduction > 2x (INT8) or > 1.5x (FP16)

---

### Session 7: Set Up Gazebo Fortress + ROS2 Humble

**Objectives:**
- Install Gazebo Fortress and ROS2 Humble
- Create a PE arena world file matching the training environment
- Spawn TurtleBot3/4 models in the arena
- Verify basic teleop control works

**Files to create/modify:**
- `ros2_ws/src/pe_gazebo/worlds/pe_arena.sdf` -- Gazebo world file
- `ros2_ws/src/pe_gazebo/launch/pe_gazebo.launch.py` -- launch file
- `ros2_ws/src/pe_gazebo/models/` -- custom obstacle models
- `ros2_ws/src/pe_gazebo/config/turtlebot_params.yaml` -- robot parameters

**Instructions:**

1. Set up ROS2 workspace:
```bash
mkdir -p ros2_ws/src
cd ros2_ws/src
# Create packages
ros2 pkg create pe_gazebo --build-type ament_cmake
ros2 pkg create pe_bringup --build-type ament_python
ros2 pkg create pe_msgs --build-type ament_cmake  # custom messages
ros2 pkg create pe_safety --build-type ament_cmake  # C++ safety filter
ros2 pkg create pe_inference --build-type ament_python  # ONNX inference
```

2. Create the PE arena world:
   - 20m x 20m ground plane
   - Four walls (rectangular boundary)
   - 2-4 cylindrical obstacles (matching training configuration)
   - Appropriate lighting and physics settings
   - Ground friction matching DR nominal (0.5)

3. Configure TurtleBot3/4 model:
   - Use the official turtlebot3_gazebo or turtlebot4_simulator package
   - Verify lidar publishes to `/scan`
   - Verify odometry publishes to `/odom`
   - Verify cmd_vel controls the robot

4. Launch and verify:
```bash
ros2 launch pe_gazebo pe_gazebo.launch.py
# In another terminal:
ros2 topic list  # verify all expected topics
ros2 run teleop_twist_keyboard teleop_twist_keyboard  # test control
```

**Verification:**
- Gazebo world loads with arena and obstacles
- Two TurtleBot models spawn at configurable positions
- Lidar scan visible in RViz2
- Teleoperation moves both robots correctly
- `/scan`, `/odom`, `/cmd_vel` topics active for both robots

---

### Session 8: Create ROS2 Node Architecture

**Objectives:**
- Implement all ROS2 nodes: sensor_fusion, observation_encoder, policy_inference, safety_filter, gp_estimator
- Define custom message types
- Verify node communication via topic pub/sub

**Files to create/modify:**
- `ros2_ws/src/pe_msgs/msg/PEState.msg` -- fused state message
- `ros2_ws/src/pe_msgs/msg/PEObservation.msg` -- observation message
- `ros2_ws/src/pe_msgs/msg/GPPrediction.msg` -- GP prediction message
- `ros2_ws/src/pe_msgs/msg/PEDiagnostics.msg` -- diagnostics message
- `ros2_ws/src/pe_inference/pe_inference/sensor_fusion_node.py`
- `ros2_ws/src/pe_inference/pe_inference/observation_encoder_node.py`
- `ros2_ws/src/pe_inference/pe_inference/policy_inference_node.py`
- `ros2_ws/src/pe_inference/pe_inference/gp_estimator_node.py`
- `ros2_ws/src/pe_safety/src/safety_filter_node.cpp` -- C++ for real-time
- `ros2_ws/src/pe_bringup/launch/pe_full.launch.py` -- launch all nodes

**Instructions:**

1. Define custom messages:
```
# PEState.msg
std_msgs/Header header
float64[3] pursuer_state    # [x, y, theta]
float64[3] evader_state     # [x, y, theta]
float64[2] pursuer_velocity # [v, omega]
float64[2] evader_velocity  # [v, omega]
float64[36] lidar_ranges    # 36-ray lidar
bool evader_in_fov

# GPPrediction.msg
std_msgs/Header header
float64[3] disturbance_mean  # [dx, dy, dtheta]
float64[3] disturbance_std   # [sigma_dx, sigma_dy, sigma_dtheta]
float64 mean_posterior_variance
```

2. Implement sensor fusion node:
   - Subscribe to `/scan` (LaserScan), `/odom` (Odometry)
   - For multi-robot: subscribe to opponent's `/odom` (or use detection)
   - Apply simple EKF or direct passthrough for now
   - Publish fused `PEState` at 20Hz

3. Implement observation encoder node:
   - Subscribe to `PEState`
   - Run BiMDN forward pass to encode belief state
   - Construct full observation vector matching training format
   - Publish `PEObservation`

4. Implement policy inference node:
   - Subscribe to `PEObservation`
   - Load ONNX model, run inference with onnxruntime
   - Manage LSTM hidden state across timesteps
   - Publish raw action as `Twist` message

5. Implement GP estimator node (Python):
   - Subscribe to `PEState` and `/cmd_vel`
   - Compute dynamics residual every step
   - Update GP every 10 steps
   - Publish `GPPrediction` at 2Hz

6. Implement safety filter node (C++ -- placeholder with passthrough for now):
   - Subscribe to policy action (`Twist`) and `GPPrediction`
   - For now, pass through the action unchanged
   - Publish to `/cmd_vel`
   - Full RCBF-QP implementation in Session 10

**Verification:**
- All nodes launch without errors
- `ros2 topic echo` shows data flowing through the pipeline
- End-to-end latency from sensor to cmd_vel < 30ms (measured with ROS2 tracing)
- Node graph (via `rqt_graph`) matches the architecture diagram

---

### Session 9: Deploy ONNX Policy in Gazebo with ROS2

**Objectives:**
- Run the full PE game in Gazebo using ONNX-exported policies
- Verify qualitative behavior matches Isaac Lab training
- Measure performance gap between Isaac Lab and Gazebo

**Files to create/modify:**
- `ros2_ws/src/pe_bringup/launch/pe_gazebo_eval.launch.py` -- evaluation launch
- `ros2_ws/src/pe_inference/pe_inference/eval_logger_node.py` -- logs metrics
- `evaluation/gazebo_eval.py` -- evaluation script (runs N episodes)

**Instructions:**

1. Launch the full system in Gazebo:
```bash
ros2 launch pe_bringup pe_gazebo_eval.launch.py \
    pursuer_model:=deployment/models/pursuer_policy_int8.onnx \
    evader_model:=deployment/models/evader_policy_int8.onnx \
    num_episodes:=100
```

2. Implement the evaluation logger:
   - Record per-episode: capture/escape, episode length, min distance, safety violations
   - Record per-timestep: positions, velocities, CBF values, actions
   - Save to CSV and log to W&B

3. Run 100 evaluation episodes with random initial positions.

4. Compare Gazebo results with Isaac Lab simulation results:
   - Capture rate: should be within 10%
   - Mean capture time: should be within 15%
   - Safety violations: should be zero (even without RCBF-QP, the policy should be inherently safe from CBF-Beta training)

5. Identify failure modes:
   - If performance gap > 10%, analyze which episodes fail and why
   - Common issues: timing, sensor noise response, dynamics mismatch
   - Feed findings back to DR configuration

**Verification:**
- Full PE game runs in Gazebo without crashes
- Pursuer captures evader in >60% of episodes (or matches training rate)
- Performance gap vs Isaac Lab < 10% (capture rate, capture time)
- Zero safety violations (boundary, obstacle, collision)
- Qualitative behavior: smooth trajectories, intentional pursuit/evasion strategies

---

### Session 10: Implement RCBF-QP Safety Filter in C++

**Objectives:**
- Implement the full RCBF-QP safety filter in C++ for real-time performance
- Integrate OSQP or qpOASES solver
- Implement 3-tier infeasibility handling
- Verify correctness against the Python RCBF-QP from Phase 2

**Files to create/modify:**
- `ros2_ws/src/pe_safety/src/rcbf_qp_solver.cpp` -- QP solver wrapper
- `ros2_ws/src/pe_safety/src/rcbf_qp_solver.hpp` -- header
- `ros2_ws/src/pe_safety/src/vcp_cbf.cpp` -- VCP-CBF constraint computation
- `ros2_ws/src/pe_safety/src/vcp_cbf.hpp` -- header
- `ros2_ws/src/pe_safety/src/safety_filter_node.cpp` -- ROS2 node (replace placeholder)
- `ros2_ws/src/pe_safety/CMakeLists.txt` -- build configuration

**Instructions:**

1. Install QP solver libraries:
```bash
# OSQP
sudo apt install libosqp-dev  # or build from source
# qpOASES
sudo apt install libqpoases-dev  # or build from source
```

2. Implement VCP-CBF constraint computation:
```cpp
// vcp_cbf.hpp
struct CBFConstraint {
    double a_v;    // coefficient for v in Lgh * u
    double a_omega; // coefficient for omega in Lgh * u
    double b;       // right-hand side: -Lfh - alpha*h + kappa*sigma_d
};

class VCPCBF {
public:
    // Compute CBF constraint for arena boundary
    CBFConstraint arena_constraint(
        double x, double y, double theta,
        double d_vcp,    // VCP offset distance
        double R_arena,  // arena radius
        double alpha,    // class-K parameter
        double kappa,    // robust margin
        double sigma_d   // GP uncertainty
    );

    // Compute CBF constraint for circular obstacle
    CBFConstraint obstacle_constraint(
        double x, double y, double theta,
        double d_vcp,
        double obs_x, double obs_y, double obs_r,
        double alpha, double kappa, double sigma_d
    );

    // Compute CBF constraint for inter-robot collision
    CBFConstraint collision_constraint(
        double x_p, double y_p, double theta_p,
        double x_e, double y_e, double theta_e,
        double d_vcp,
        double r_min,     // minimum separation
        double alpha, double kappa, double sigma_d
    );
};
```

3. Implement the QP solver:
```cpp
// rcbf_qp_solver.hpp
class RCBFQPSolver {
public:
    struct Solution {
        double v_safe;
        double omega_safe;
        bool feasible;
        int tier_used;      // 1=exact, 2=relaxed, 3=backup
        double solve_time_ms;
    };

    // Tier 1: Exact CBF-QP
    Solution solve_exact(
        double v_desired, double omega_desired,
        const std::vector<CBFConstraint>& constraints,
        double v_min, double v_max,
        double omega_min, double omega_max
    );

    // Tier 2: Relaxed CBF-QP (relax least-important constraint)
    Solution solve_relaxed(
        double v_desired, double omega_desired,
        const std::vector<CBFConstraint>& constraints,
        const std::vector<int>& priorities  // 0=highest priority
    );

    // Tier 3: Backup controller
    Solution backup_controller(
        double x, double y, double theta,
        const std::vector<Obstacle>& obstacles
    );

    // Full 3-tier solve
    Solution solve(
        double v_desired, double omega_desired,
        double x, double y, double theta,
        const std::vector<CBFConstraint>& constraints,
        const std::vector<Obstacle>& obstacles
    );
};
```

4. Implement the safety filter ROS2 node:
   - Subscribe to policy action, GP prediction, and fused state
   - Compute VCP-CBF constraints for current state
   - Solve RCBF-QP with 3-tier handling
   - Publish safe action to `/cmd_vel`
   - Log: QP solve time, tier used, constraint margins, CBF values

5. Cross-validate with Python implementation:
   - Generate 1000 (state, action) pairs from simulation
   - Solve RCBF-QP in both Python (Phase 2) and C++
   - Verify outputs match within rtol=1e-4

**Verification:**
- C++ RCBF-QP produces same output as Python version (rtol=1e-4) on 1000 test cases
- QP solve time < 1ms for typical cases (2 variables, 3-6 constraints)
- 3-tier infeasibility handling works: test with deliberately infeasible inputs
- Safety filter node integrates into ROS2 pipeline without latency issues

---

### Session 11: QP Solver Benchmarking

**Objectives:**
- Comprehensive benchmarking of OSQP and qpOASES on target hardware
- Measure solve time distributions, worst-case performance
- Select the solver for deployment

**Files to create/modify:**
- `deployment/benchmarks/qp_benchmark.cpp` -- benchmarking harness
- `deployment/benchmarks/qp_benchmark_results.json` -- results
- `deployment/benchmarks/plot_qp_benchmark.py` -- visualization

**Instructions:**

1. Generate test cases:
   - 10,000 random (state, action) pairs from simulation rollouts
   - Include easy cases (far from constraints) and hard cases (near constraint boundaries)
   - Include infeasible cases (10% of test set)

2. Benchmark both solvers:
   - For each test case, solve with OSQP and qpOASES
   - Record: solve time, solution quality (||u - u_rl||), feasibility status
   - Benchmark on: development machine (x86), RPi4 (ARM), Jetson Nano (ARM)

3. Report metrics:
   - Mean solve time
   - Median solve time
   - 95th percentile solve time (MUST be < 50ms)
   - 99th percentile solve time
   - Maximum solve time
   - Infeasibility detection rate (should be 100%)

4. Warm-starting test:
   - Solve sequences of QPs from consecutive timesteps
   - Measure speedup from warm-starting (using previous solution as initial guess)

5. Select solver: Choose the one with better worst-case performance on target hardware.

**Verification:**
- Both solvers produce correct solutions (verified against Python reference)
- 95th percentile solve time < 50ms on target hardware
- Warm-starting provides measurable speedup (>20%)
- Clear winner selected with documented rationale

---

### Session 12: Implement GP Disturbance Estimation

**Objectives:**
- Implement GP disturbance estimation using GPyTorch
- Verify GP predictions improve over time with more data
- Test GP update does not cause latency spikes in the control loop

**Files to create/modify:**
- `ros2_ws/src/pe_inference/pe_inference/gp_estimator.py` -- GP model class
- `ros2_ws/src/pe_inference/pe_inference/gp_estimator_node.py` -- ROS2 node (update from Session 8)
- `tests/test_gp_estimator.py` -- unit tests

**Instructions:**

1. Implement the GP model:
```python
import gpytorch
import torch

class DisturbanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=6  # [x, y, theta, v, omega, t]
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

2. Implement the online update loop:
   - Maintain a sliding window buffer of 200 most recent (state, residual) pairs
   - Every 10 control steps, refit GP hyperparameters (1-2 gradient steps on marginal likelihood)
   - Provide predict() method that returns (mean, std) for current state
   - Use GPyTorch's `fast_pred_var` for efficient uncertainty estimation

3. Implement the data collection:
   - At each step: observe x_t, u_t, x_{t+1}
   - Compute residual: d_t = x_{t+1} - f_nominal(x_t, u_t)
   - Add to sliding window buffer

4. Test in simulation:
   - Add known disturbance to the Gymnasium environment (e.g., constant wind force)
   - Run GP estimator, verify it converges to the true disturbance
   - Plot predicted vs actual disturbance over time

**Verification:**
- GP prediction RMSE decreases over time (< 0.1 after 100 data points)
- GP update time < 50ms (runs asynchronously, does not block control loop)
- GP posterior variance sigma_d decreases as data accumulates
- Sliding window prevents memory growth

---

### Session 13: GP Cold-Start Protocol Implementation

**Objectives:**
- Implement the 3-step cold-start protocol
- Generate simulation residual data for pre-filling
- Implement the conservative-to-normal kappa transition
- Verify protocol in Gazebo simulation

**Files to create/modify:**
- `deployment/gp_cold_start.py` -- cold-start data generation
- `deployment/gp_prefill_data/` -- pre-generated GP data
- `ros2_ws/src/pe_inference/pe_inference/gp_cold_start_manager.py` -- manages the transition

**Instructions:**

1. Generate pre-fill data from simulation:
```python
def generate_gp_prefill_data(env, policy, n_points=1000):
    """Run policy in domain-randomized simulation, collect residuals."""
    data_x = []  # states
    data_y = []  # residuals

    obs = env.reset()
    for _ in range(n_points):
        action = policy.predict(obs)
        next_obs, _, done, _, info = env.step(action)

        # Compute residual
        state = extract_state(obs)
        next_state = extract_state(next_obs)
        nominal_next = unicycle_step(state, action, dt=0.05)
        residual = next_state - nominal_next

        data_x.append(np.concatenate([state, action]))
        data_y.append(residual)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

    return np.array(data_x), np.array(data_y)
```

2. Implement the cold-start manager:
```python
class GPColdStartManager:
    def __init__(self, prefill_data_path, kappa_nominal=1.0, kappa_init=2.0, tau=50):
        self.kappa_nominal = kappa_nominal
        self.kappa_init = kappa_init
        self.tau = tau
        self.step_count = 0
        self.transition_complete = False

        # Load pre-fill data
        data = np.load(prefill_data_path)
        self.prefill_x = torch.tensor(data["x"], dtype=torch.float32)
        self.prefill_y = torch.tensor(data["y"], dtype=torch.float32)

    def get_kappa(self):
        """Return current robust margin coefficient."""
        if self.transition_complete:
            return self.kappa_nominal

        kappa = self.kappa_nominal + \
                (self.kappa_init - self.kappa_nominal) * \
                np.exp(-self.step_count / self.tau)
        return kappa

    def step(self, mean_posterior_variance):
        """Called every control step."""
        self.step_count += 1

        # Early termination if GP is confident
        if mean_posterior_variance < 0.01:
            self.transition_complete = True

    def get_prefill_data(self):
        return self.prefill_x, self.prefill_y
```

3. Integrate into the safety filter node:
   - On startup, load pre-fill data into GP
   - Use kappa from cold-start manager in RCBF-QP
   - Log kappa value and sigma_bar_d over time

4. Test in Gazebo with intentional dynamics perturbation:
   - Add a constant force disturbance to one robot in Gazebo
   - Verify cold-start protocol keeps the robot safe during initialization
   - Verify kappa transitions from 2.0 to 1.0 as GP learns

**Verification:**
- Pre-fill data generates 1000 points without errors
- Cold-start kappa starts at 2.0 * kappa_nominal and decays exponentially
- GP initialized with pre-fill data has lower initial variance than empty GP
- Zero safety violations during the cold-start phase in Gazebo
- Kappa reaches nominal value within 200 steps (10 seconds at 20Hz)

---

### Session 14: Real Robot Setup

**Objectives:**
- Hardware bringup of the selected robot platform (TurtleBot4 recommended)
- Sensor calibration (lidar, odometry)
- Verify all ROS2 topics publish correctly
- Set up the arena in the physical lab space

**Files to create/modify:**
- `hardware/robot_setup.md` -- setup documentation
- `hardware/calibration/lidar_calibration.py` -- lidar calibration script
- `hardware/calibration/odom_calibration.py` -- odometry calibration
- `hardware/arena_setup.md` -- physical arena specifications

**Instructions:**

1. Robot bringup:
   - Power on TurtleBot4, connect to WiFi network
   - SSH into robot: `ssh ubuntu@<robot_ip>`
   - Verify ROS2 topics: `ros2 topic list`
   - Expected: `/scan`, `/odom`, `/cmd_vel`, `/tf`, `/joint_states`
   - Test basic teleop: `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

2. Lidar calibration:
   - Place robot at known positions relative to walls
   - Compare lidar readings to tape-measured distances
   - Compute and record systematic bias (offset, scaling)
   - Verify lidar noise is within DR training range (sigma < 0.02m)

3. Odometry calibration:
   - Drive robot in a known square (1m x 1m)
   - Compare odometry-reported position to measured position
   - Compute drift rate (m/m traveled, deg/m traveled)
   - Verify odometry noise is within DR training range

4. Arena setup:
   - Mark a rectangular arena on the floor matching the training arena from Session 4
   - For TurtleBot4: 6m x 6m arena (matched to v_max=0.3 m/s re-training in Session 4)
   - Arena size preserves the same time-to-cross ratio as simulation (20s at robot v_max)
   - Place physical obstacles (cardboard cylinders, boxes) matching training layout
   - Set up external tracking if available (OptiTrack, AprilTag ceiling) for ground truth

5. Set up two robots (pursuer and evader):
   - Each robot runs its own policy inference node
   - Shared arena, independent compute
   - Configure ROS2 namespaces: `/pursuer/...` and `/evader/...`

**Verification:**
- Both robots respond to teleop commands
- Lidar data matches physical measurements within 2cm
- Odometry drift < 5% over a 2m path
- Arena is set up with obstacles matching training configuration
- ROS2 communication between robots works (can see each other's topics)

---

### Session 15: Deploy Single Agent Obstacle Avoidance

**Objectives:**
- Deploy ONE robot with the ONNX policy + RCBF-QP safety filter
- Test obstacle avoidance only (no pursuit-evasion)
- Verify safety filter activates correctly near walls and obstacles
- This is a conservative first test before full PE

**Files to create/modify:**
- `ros2_ws/src/pe_bringup/launch/pe_single_robot.launch.py` -- single robot launch
- `evaluation/real_robot_single_eval.py` -- evaluation script

**Instructions:**

1. Deploy single robot with safety filter:
   - Launch policy inference node with the pursuer ONNX model
   - Launch RCBF-QP safety filter with GP cold-start
   - Set a dummy "evader" target at a fixed position across the arena

2. Test scenarios (in order of increasing difficulty):
   a. Robot drives in open space (no obstacles nearby) -- verify smooth motion
   b. Robot approaches a wall -- verify safety filter steers away (NOT brakes -- VCP-CBF should prefer steering)
   c. Robot navigates around a single obstacle -- verify smooth avoidance
   d. Robot in a corner -- verify it can navigate out without getting stuck

3. Monitor and record:
   - QP solve times (histogram)
   - CBF constraint values over time
   - Safety filter intervention rate (% of steps where action is modified)
   - GP disturbance predictions vs actual (plot over time)
   - Cold-start kappa transition
   - Robot trajectory (from odometry + external tracking if available)

4. Tune parameters if needed:
   - CBF alpha: increase if too conservative, decrease if too aggressive
   - VCP offset d: adjust if steering behavior is wrong
   - GP update rate: increase if disturbance learning is too slow

**Verification:**
- Zero collisions with walls or obstacles across 10 test runs (5 minutes each)
- QP solve time 95th percentile < 50ms on real hardware
- Control loop maintains 20Hz (no drops below 15Hz)
- Safety filter intervention rate < 20% (policy should be mostly safe from training)
- GP cold-start protocol works: kappa decreases, sigma_bar_d decreases
- VCP-CBF prefers steering over braking when approaching walls [N12]

---

### Session 16: Deploy Full PE Game on Real Robots

**Objectives:**
- Run the full 1v1 pursuit-evasion game on two physical robots
- Both robots run their respective ONNX policies with RCBF-QP safety filters
- Achieve zero safety violations

**Files to create/modify:**
- `ros2_ws/src/pe_bringup/launch/pe_two_robots.launch.py` -- two-robot launch
- `evaluation/real_robot_pe_eval.py` -- PE evaluation script
- `evaluation/real_robot_pe_metrics.py` -- metric computation

**Instructions:**

1. Launch both robots:
```bash
# On pursuer robot (or from control PC):
ros2 launch pe_bringup pe_two_robots.launch.py \
    pursuer_model:=pursuer_policy_int8.onnx \
    evader_model:=evader_policy_int8.onnx \
    pursuer_robot_ip:=<ip1> \
    evader_robot_ip:=<ip2>
```

2. Run evaluation episodes:
   - Random initial positions (within arena)
   - 30 episodes minimum (60-second timeout each)
   - Record all data for analysis

3. Monitor in real-time:
   - RViz2 visualization: robot positions, lidar scans, CBF constraint boundaries
   - Terminal output: capture/escape events, safety filter activations, QP solve times
   - W&B real-time logging

4. Safety test scenarios (run 5 times each):
   a. Head-on approach: start robots facing each other, 2m apart
   b. Corner encounter: both robots near the same corner
   c. Obstacle squeeze: narrow gap between obstacles
   d. High-speed approach: pursuer at max velocity toward evader

5. Record demonstration videos:
   - Overhead camera (full arena view)
   - Robot-mounted camera (if available)
   - Screen recording of RViz2

**Verification:**
- ZERO safety violations across all episodes (wall collision, obstacle collision, inter-robot collision)
- Pursuer captures evader in >50% of episodes (or matches simulation rate within 10%)
- Both robots maintain 20Hz control loop
- Inter-robot collision avoidance activates correctly (robots maintain r_min separation)
- Qualitative behavior matches simulation (intentional pursuit/evasion strategies)

---

### Session 17: Train-Deploy Gap Measurement

**Objectives:**
- Quantitatively measure the performance gap at each pipeline stage
- Identify the biggest sources of degradation
- Tune parameters to minimize the gap

**Files to create/modify:**
- `evaluation/sim_to_real_gap_analysis.py` -- gap analysis script
- `evaluation/gap_analysis_results/` -- results directory

**Instructions:**

1. Run identical evaluation protocol (same initial conditions, 100 episodes each) at each pipeline stage:
   - Stage A: Isaac Lab (DR enabled, GPU simulation)
   - Stage B: Gazebo (ONNX policy, no safety filter)
   - Stage C: Gazebo (ONNX policy + RCBF-QP safety filter)
   - Stage D: Real robot (ONNX policy + RCBF-QP + GP)

2. Compute metrics at each stage:

| Metric | Stage A | Stage B | Stage C | Stage D | Gap A-D |
|--------|---------|---------|---------|---------|---------|
| Capture rate (%) | | | | | |
| Mean capture time (s) | | | | | |
| Mean min distance (m) | | | | | |
| Safety violation rate (%) | | | | | |
| Mean episode length (s) | | | | | |
| CBF filter intervention rate (%) | | | | | |
| QP solve time P95 (ms) | N/A | N/A | | | |

3. Analyze where degradation occurs:
   - A -> B gap: Simulation engine differences, sensor model differences
   - B -> C gap: Safety filter conservatism (RCBF-QP modifying actions)
   - C -> D gap: Real-world dynamics, sensor noise, latency

4. If gap > 10%:
   - Increase DR aggressiveness (widen parameter ranges)
   - Tune CBF alpha for less conservatism
   - Improve GP model (more pre-fill data, better kernel)
   - Check for systematic errors (e.g., odometry bias)

**Verification:**
- Complete data for all four pipeline stages
- Total gap (Stage A vs Stage D) < 10% in capture rate
- Each pipeline transition contributes <5% degradation
- Root causes of degradation identified and documented

---

### Session 18: Comprehensive Real-World Evaluation

**Objectives:**
- Run the final comprehensive evaluation on real robots
- Collect all data needed for Phase 5 (paper writing)
- Record demonstration videos
- Document all findings

**Files to create/modify:**
- `evaluation/final_real_world_eval.py` -- final evaluation script
- `evaluation/final_results/` -- all results, videos, plots
- `evaluation/final_results/summary.md` -- evaluation summary

**Instructions:**

1. Run 50+ evaluation episodes with random initial conditions:
   - 25 with obstacles (training configuration)
   - 25 without obstacles (open arena)
   - Record everything

2. Run safety stress tests:
   - 10 episodes with deliberately challenging initial conditions (close to walls, tight spaces)
   - 5 episodes with external perturbation (push robot gently during play)
   - 5 episodes starting from GP cold-start (reset GP, verify protocol)

3. Record demonstration videos:
   - 3 "best" episodes (exciting pursuits, clever evasions)
   - 3 "safety" episodes (close calls where RCBF-QP saves the robot)
   - 1 GP cold-start episode (show conservative-to-normal transition)
   - All from overhead camera + RViz2 screen recording

4. Generate publication-quality plots:
   - Trajectory plots (pursuer blue, evader red, obstacles gray)
   - QP solve time histogram
   - GP disturbance convergence plot
   - Cold-start kappa transition plot
   - Sim-to-real comparison bar charts
   - CBF value time series (showing safety margin maintained)

5. Compute final metrics (Table 1 for the paper):
   - Capture rate, mean capture time, safety violation rate
   - 95th percentile QP solve time
   - Control loop frequency (mean, min)
   - Sim-to-real gap percentages
   - GP convergence time (steps to sigma_bar_d < 0.01)

6. Document lessons learned:
   - What worked well
   - What required significant tuning
   - What would you do differently
   - Recommendations for future work

**Verification:**
- 50+ episodes completed without system failures
- Zero safety violations across all episodes
- All demonstration videos recorded and clearly show PE behavior
- Publication-quality plots generated
- Complete metrics table filled
- Lessons learned documented

---

### Sessions 19-21: Iteration Buffer (Debugging & Refinement)

**Purpose**: Real sim-to-real transfer is inherently iterative. These buffer sessions provide dedicated time for debugging, parameter tuning, and re-running failed experiments. In practice, the first attempt rarely works end-to-end.

**Session 19: Sim-to-Gazebo Iteration** (3-5h)
- Debug issues discovered during Sessions 7-9 (Gazebo/ROS2 integration)
- Re-tune RCBF-QP parameters if safety margin is too conservative or too aggressive in Gazebo
- Fix ONNX inference issues (BiMDN LSTM state management, quantization artifacts)
- Re-run Session 9 evaluation if sim-to-Gazebo gap > 10%

**Session 20: Gazebo-to-Real Iteration** (4-6h)
- Debug real-robot issues discovered during Sessions 14-16
- Tune GP cold-start parameters based on actual sensor noise characteristics
- Adjust arena scaling if robots behave differently at real-world scale
- Re-calibrate sensor offsets and timing delays
- Re-run safety stress tests after parameter changes

**Session 21: End-to-End Polish** (3-4h)
- Re-run comprehensive evaluation (Session 18) with final tuned parameters
- Record additional demonstration videos if earlier recordings had issues
- Fill gaps in data collection for Phase 5 tables/figures
- Document all parameter changes made during iteration with rationale

**Note**: If all prior sessions succeed on first attempt, these buffer sessions can be used for additional experiments (e.g., testing with F1TENTH, exploring larger arenas, additional challenge scenarios). In practice, at least 1-2 of these sessions will be needed for debugging.

---

## 5. Testing Plan (Automated)

### 5.1 Unit Tests

| Test Name | Description | Inputs | Expected Output | Pass Criteria |
|-----------|-------------|--------|-----------------|---------------|
| `test_onnx_output_matches_pytorch` | Verify ONNX model produces same output as PyTorch model | 1000 random observations | Matching action distributions | Max absolute diff < 1e-5 for all outputs |
| `test_onnx_quantized_output_acceptable` | Verify quantized ONNX output is close to full precision | 1000 random observations | Similar actions | Max absolute diff < 0.01, capture rate diff < 3% |
| `test_dr_parameter_ranges` | Verify domain randomization produces correct distributions | 10,000 env resets | Parameter histograms | All parameters within specified ranges, uniform for uniform params |
| `test_dr_mass_range` | Mass randomization is within [0.8, 1.2] | 5000 env resets | Mass values | min >= 0.8, max <= 1.2, mean approx 1.0 |
| `test_dr_friction_range` | Friction randomization is within [0.3, 0.8] | 5000 env resets | Friction values | min >= 0.3, max <= 0.8 |
| `test_gp_prediction_accuracy` | GP converges to known disturbance | 200 data points from known disturbance | GP mean close to true | RMSE < 0.05 after 200 points |
| `test_gp_uncertainty_decreases` | GP uncertainty decreases with data | Sequential data points | Decreasing sigma | sigma at point 200 < 0.5 * sigma at point 10 |
| `test_qp_feasible_solution` | QP solver returns feasible solution for valid inputs | 1000 feasible (state, action) pairs | Feasible solutions | 100% feasibility rate, all constraints satisfied |
| `test_qp_infeasible_detection` | QP solver correctly detects infeasible inputs | 100 infeasible inputs | Infeasibility flag | 100% detection rate |
| `test_qp_solve_within_time` | QP solver runs within time budget | 1000 random inputs | Solve times | 95th percentile < 50ms |
| `test_vcp_cbf_arena_constraint` | VCP-CBF arena constraint is correct | States near arena boundary | Correct constraint coefficients | Matches analytical computation |
| `test_vcp_cbf_obstacle_constraint` | VCP-CBF obstacle constraint is correct | States near obstacles | Correct constraint coefficients | Matches analytical computation |
| `test_vcp_cbf_collision_constraint` | VCP-CBF inter-robot constraint is correct | States with two robots nearby | Correct constraint coefficients | Matches analytical computation |
| `test_cold_start_kappa_decay` | Kappa decays from kappa_init to kappa_nominal | 200 steps | Exponential decay | kappa at step 0 = 2*kappa_nominal, at step 200 approx kappa_nominal |
| `test_cold_start_early_termination` | Cold-start terminates early when GP is confident | Low sigma_bar_d | kappa = kappa_nominal | Transition complete flag set |
| `test_cpp_qp_matches_python` | C++ QP solver matches Python implementation | 1000 (state, action) pairs | Same solutions | Max absolute diff < 1e-4 |

### 5.2 Integration Tests

| Test Name | Description | Inputs | Expected Output | Pass Criteria |
|-----------|-------------|--------|-----------------|---------------|
| `test_isaac_lab_valid_observations` | Isaac Lab env produces valid observation tensors | 100 random steps | Valid obs tensors | No NaN/Inf, correct shape, values in expected ranges |
| `test_isaac_lab_parallel_consistency` | Parallel envs produce consistent results | Same initial state, same actions across envs | Identical trajectories (when DR is off) | Max state diff < 1e-6 |
| `test_onnx_inference_latency` | ONNX inference within latency budget | 1000 inference calls | Latency measurements | Mean < 5ms, P95 < 10ms |
| `test_ros2_topic_communication` | All ROS2 nodes communicate correctly | Launch full pipeline | Messages on all topics | All topics active, correct message types, correct rates |
| `test_ros2_end_to_end_latency` | End-to-end latency from sensor to cmd_vel | Timestamped messages | Latency distribution | P95 < 30ms |
| `test_rcbf_qp_ros2_integration` | Safety filter integrates with ROS2 control loop | Policy actions + states | Safe actions published | cmd_vel satisfies all CBF constraints |
| `test_gp_no_latency_spikes` | GP updates don't cause latency spikes in control loop | 1000 steps with GP updates | Control loop timing | No control loop iteration > 50ms |
| `test_gazebo_matches_isaac_lab` | Gazebo PE behavior matches Isaac Lab | 50 episodes, same initial conditions | Similar metrics | Capture rate diff < 10%, mean capture time diff < 15% |
| `test_full_pipeline_gazebo` | Full pipeline runs in Gazebo without crashes | 10 full episodes | Completed episodes | Zero crashes, all episodes complete |
| `test_multi_robot_ros2` | Two-robot ROS2 setup works | Launch two robots | Both controlled independently | Both robots respond to separate cmd_vel, no topic conflicts |

### 5.3 System Tests

| Test Name | Description | Inputs | Expected Output | Pass Criteria |
|-----------|-------------|--------|-----------------|---------------|
| `test_sim_gazebo_gap` | Performance gap between Isaac Lab and Gazebo | 100 episodes each | Metrics comparison | Gap < 10% in capture rate |
| `test_gazebo_real_gap` | Performance gap between Gazebo and real robot | 50+ episodes each | Metrics comparison | Gap < 10% in capture rate |
| `test_zero_safety_violations_real` | Zero safety violations on real robot | 50+ episodes | Safety metrics | Zero wall/obstacle/robot collisions |
| `test_20hz_control_loop` | Control loop maintains 20Hz on real hardware | 1000 consecutive steps | Loop timing | Mean > 19Hz, no iteration > 100ms |
| `test_gp_cold_start_protocol` | Full cold-start protocol works on real robot | Fresh GP, 200 steps | Kappa and sigma_d traces | Conservative start, smooth transition, sigma_d < 0.01 within 200 steps |
| `test_real_robot_capture` | Pursuer actually captures evader on real robot | 30 episodes | Capture events | Capture rate > 50% (or within 10% of simulation) |
| `test_long_running_stability` | System runs for 30+ minutes without degradation | Continuous operation | System metrics | No memory leaks, no timing degradation, no crashes |

### 5.4 Hardware-in-Loop Tests

| Test Name | Description | Inputs | Expected Output | Pass Criteria |
|-----------|-------------|--------|-----------------|---------------|
| `test_motor_response` | Motor response matches commanded velocity | Step input commands | Velocity measurements | Achieved velocity within 10% of command within 0.5s |
| `test_lidar_accuracy` | Lidar returns match expected obstacle distances | Known obstacle placement | Lidar readings | Error < 2cm for obstacles within 3m |
| `test_odometry_accuracy` | Odometry matches known trajectory | Drive 1m square | Position after loop | Drift < 5% of distance traveled |
| `test_localization_accuracy` | Localization accuracy sufficient for CBF | Known positions | Position estimates | Error < 5cm (95th percentile) |
| `test_wifi_latency` | WiFi communication latency acceptable | Ping tests, topic latency | Latency measurements | P95 < 10ms for local network |

---

## 6. Manual Validation Checklist

### 6.1 Gazebo Visual Validation
- [ ] Trajectories in Gazebo look qualitatively similar to Isaac Lab training visualizations
- [ ] Pursuer actively chases the evader (not wandering randomly)
- [ ] Evader actively flees from the pursuer (not standing still)
- [ ] Both robots avoid obstacles with smooth trajectories (not jerky stop-and-go)
- [ ] No clipping through walls or obstacles
- [ ] Lidar rays visible in RViz2, correctly detecting obstacles

### 6.2 Real Robot Visual Validation
- [ ] Robot motion is smooth (no jerky or oscillating movements)
- [ ] Robot maintains safe distance from walls (> 0.15m)
- [ ] Robot maintains safe distance from obstacles (> 0.15m + obstacle radius)
- [ ] Two robots maintain minimum separation distance during play
- [ ] VCP-CBF causes steering behavior near walls (NOT braking to a stop)
- [ ] Pursuit behavior is visible: pursuer approaches evader intentionally
- [ ] Evasion behavior is visible: evader moves away from approaching pursuer

### 6.3 Quantitative Measurements
- [ ] Measure actual capture times on real robot, compare to simulation capture times (should be within 15%)
- [ ] Check QP solve time histogram on real hardware: 95th percentile MUST be < 50ms
- [ ] Plot GP predicted disturbance vs actual dynamics residual over time -- lines should converge within 100 steps
- [ ] Measure control loop frequency over a 5-minute run: mean should be > 19Hz, no drops below 10Hz

### 6.4 Safety Scenario Tests (run each 3 times)
- [ ] **Wall approach**: Manually drive robot toward wall at 0.5 * v_max -- safety filter should steer away before contact
- [ ] **Obstacle approach**: Drive toward obstacle -- safety filter should divert around it
- [ ] **Head-on collision**: Start both robots facing each other, 1m apart -- inter-robot CBF should prevent collision
- [ ] **Corner approach**: Navigate robot into corner -- should be able to escape without getting stuck
- [ ] **Rapid approach**: Pursuer at max speed toward evader near a wall -- both should remain safe
- [ ] **GP cold-start**: Reset GP, run from scratch -- robot should be extra conservative initially, then relax

### 6.5 Video Recording
- [ ] Record 3 best PE episodes from overhead camera (full arena view)
- [ ] Record 3 safety demonstration episodes showing RCBF-QP intervention
- [ ] Record 1 GP cold-start demonstration (conservative-to-normal transition)
- [ ] Record RViz2 screen for each video (showing lidar, CBF margins, paths)
- [ ] All videos include timestamp overlay for synchronization with data logs

### 6.6 Data Logging
- [ ] W&B logging active during all real robot experiments
- [ ] Per-episode metrics: capture/escape, episode length, min distance, safety violations, CBF intervention rate
- [ ] Per-timestep data: positions (x, y, theta), velocities (v, omega), actions (commanded and executed), CBF values, QP solve time, GP predictions
- [ ] GP cold-start data: kappa over time, sigma_bar_d over time, GP data count
- [ ] System health: CPU usage, memory usage, control loop frequency, WiFi latency

---

## 7. Success Criteria and Phase Gates

### 7.1 Hard Requirements (MUST pass to proceed to Phase 5)

| Criterion | Metric | Target | Measurement Method | Protocol |
|-----------|--------|--------|-------------------|----------|
| Sim-to-real gap | Capture rate difference (sim vs real) | < 10% | 100 sim episodes vs 50 real episodes | Same initial conditions (seeded); 5 seeds; Welch's t-test, p < 0.05 |
| Safety | Safety violation count on real robot | ZERO | All real robot episodes (50+) | Any wall/obstacle/robot collision = FAIL; log min CBF value per step |
| Real-time inference | Control loop frequency | >= 20Hz (mean), no drop below 10Hz | ROS2 timing logs per-callback | Record 10,000 consecutive steps; plot histogram |
| QP solve time | 95th percentile solve time | < 50ms | QP solver C++ benchmark on target hardware | 10,000 random (state, action) pairs including 10% near-infeasible |
| GP cold-start | Protocol completes without safety violation | Validated | 5 cold-start runs on real robot | Fresh GP each run; record kappa(t) and sigma_bar_d(t); zero violations |
| ONNX equivalence | Max output diff PyTorch vs ONNX | < 1e-5 | 1000 random inputs | Both pursuer and evader models; report max abs diff |
| Gazebo validation | Gazebo capture rate vs Isaac Lab | < 10% gap | 100 episodes each, same seeds | Catches 80% of sim-to-real issues per [N10] |

### 7.2 Soft Targets (desirable but not blocking)

| Criterion | Metric | Target | How to Measure | Notes |
|-----------|--------|--------|---------------|-------|
| Capture rate (real) | Pursuer captures evader | > 50% | 50+ real episodes, report mean ± 95% CI | May be lower with slow robots |
| CBF intervention rate | % steps where safety filter modifies action | < 15% | `count(||u_safe - u_RL|| > 0.01) / total_steps` | Lower = policy learned safety |
| GP convergence time | Steps to sigma_bar_d < 0.01 | < 200 steps | Monitor `sigma_bar_d` per GP update; log step when threshold first crossed | Faster = better |
| Backup controller activation | % steps using Tier 3 backup | < 0.1% | Count tier_used == 3 in QP solve logs | Should be extremely rare |
| ONNX model size | Compressed model file size | < 10MB | `ls -la deployment/models/*.onnx` | For edge deployment |
| DR robustness | Performance under worst-case DR | > 80% of nominal | Eval with extreme DR parameters (mass=1.2, friction=0.3, delay=50ms) | Test policy generalization |

### 7.3 Definition of Done

> **Phase 4 is COMPLETE when:**
> 1. ALL hard requirements in Section 7.1 are met (with documented evidence)
> 2. At least 50 real-robot PE episodes completed and logged to W&B
> 3. ZERO safety violations across all real-robot episodes
> 4. Demonstration videos recorded (minimum 7 videos per Section 6.5)
> 5. Sim-to-real gap analysis complete (all 4 pipeline stages: Isaac Lab → Gazebo (no filter) → Gazebo (with filter) → Real)
> 6. GP cold-start protocol validated: 5 runs, kappa transition logged, zero violations
> 7. QP solver benchmarked on target hardware: OSQP vs qpOASES, winner selected with rationale
> 8. Minimum test suite (Section 7.5) passes: 18+ tests, all green
> 9. All code committed, documented, and reproducible
> 10. Phase 4 summary with key findings and Phase 5 data requirements

### 7.4 Phase Gate Checklist

Before declaring Phase 4 complete, verify:
- [ ] All hard requirements met (Section 7.1, documented with numbers)
- [ ] Minimum test suite passes (Section 7.5, 18+ tests)
- [ ] At least 50 real-robot PE episodes completed and logged
- [ ] Demonstration videos recorded (minimum 7 videos per Section 6.5)
- [ ] Sim-to-real gap analysis complete (all 4 pipeline stages measured)
- [ ] GP cold-start protocol validated on real robot (5 runs)
- [ ] QP solver benchmarked on target hardware with documented results
- [ ] All code committed, documented, and reproducible
- [ ] W&B dashboard with all real-robot experiment data
- [ ] Publication-quality plots generated for Phase 5

### 7.5 Minimum Test Suite (18+ Tests)

**File: `tests/test_onnx_export.py`** (4 tests)

```python
# Test A: ONNX model loads and runs
def test_onnx_model_loads():
    """Exported ONNX model loads in onnxruntime without errors."""
    session = ort.InferenceSession("deployment/models/pursuer_policy.onnx")
    dummy_obs = np.random.randn(1, obs_dim).astype(np.float32)
    dummy_hidden = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    outputs = session.run(None, {"observation": dummy_obs, "hidden_state": dummy_hidden})
    assert len(outputs) == 4  # action_mean, action_std, value, hidden_out
    assert not any(np.isnan(o).any() for o in outputs)

# Test B: ONNX matches PyTorch output
def test_onnx_matches_pytorch():
    """Max absolute difference < 1e-5 across 1000 random inputs."""
    model = load_trained_policy("pursuer_dr.pt")
    session = ort.InferenceSession("deployment/models/pursuer_policy.onnx")
    max_diff = 0.0
    for _ in range(1000):
        obs = torch.randn(1, obs_dim)
        hidden = torch.zeros(1, 1, hidden_dim)
        with torch.no_grad():
            pt_out = model(obs, hidden)
        onnx_out = session.run(None, {"observation": obs.numpy(), "hidden_state": hidden.numpy()})
        for pt, ox in zip(pt_out, onnx_out):
            max_diff = max(max_diff, np.max(np.abs(pt.numpy() - ox)))
    assert max_diff < 1e-5, f"Max diff = {max_diff}"

# Test C: Quantized model output acceptable
def test_quantized_output_acceptable():
    """INT8 quantized model output diff < 0.01 from full precision."""
    session_fp = ort.InferenceSession("deployment/models/pursuer_policy.onnx")
    session_q = ort.InferenceSession("deployment/models/pursuer_policy_int8.onnx")
    for _ in range(100):
        obs = np.random.randn(1, obs_dim).astype(np.float32)
        hidden = np.zeros((1, 1, hidden_dim), dtype=np.float32)
        out_fp = session_fp.run(None, {"observation": obs, "hidden_state": hidden})
        out_q = session_q.run(None, {"observation": obs, "hidden_state": hidden})
        assert np.max(np.abs(out_fp[0] - out_q[0])) < 0.01  # action mean

# Test D: ONNX inference latency on host machine
def test_onnx_inference_latency():
    """Mean inference < 10ms, P95 < 20ms on development machine."""
    session = ort.InferenceSession("deployment/models/pursuer_policy_int8.onnx")
    obs = np.random.randn(1, obs_dim).astype(np.float32)
    hidden = np.zeros((1, 1, hidden_dim), dtype=np.float32)
    # Warm-up
    for _ in range(100):
        session.run(None, {"observation": obs, "hidden_state": hidden})
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        session.run(None, {"observation": obs, "hidden_state": hidden})
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
    assert np.mean(latencies) < 10.0
    assert np.percentile(latencies, 95) < 20.0
```

**File: `tests/test_domain_randomization.py`** (4 tests)

```python
# Test E: DR mass within specified range
def test_dr_mass_range():
    """Mass randomization stays in [0.8, 1.2] kg across 5000 resets."""
    dr = DomainRandomizer()
    masses = []
    for _ in range(5000):
        dr.randomize(torch.arange(1))
        masses.append(dr.masses[0].item())
    assert min(masses) >= 0.8
    assert max(masses) <= 1.2
    assert abs(np.mean(masses) - 1.0) < 0.05  # Mean ≈ 1.0

# Test F: DR friction within specified range
def test_dr_friction_range():
    """Friction randomization stays in [0.3, 0.8] across 5000 resets."""
    dr = DomainRandomizer()
    frictions = []
    for _ in range(5000):
        dr.randomize(torch.arange(1))
        frictions.append(dr.frictions[0].item())
    assert min(frictions) >= 0.3
    assert max(frictions) <= 0.8

# Test G: Policy trained WITHOUT DR degrades with DR
def test_no_dr_policy_degrades_with_dr():
    """Non-DR policy loses >10% capture rate under DR perturbation."""
    policy = load_trained_policy("pursuer_no_dr.pt")
    rate_nominal = evaluate_capture_rate(policy, env_no_dr, n_episodes=50)
    rate_dr = evaluate_capture_rate(policy, env_with_dr, n_episodes=50)
    assert rate_nominal - rate_dr > 0.10  # DR hurts non-robust policy

# Test H: DR overhead < 5% FPS reduction
def test_dr_fps_overhead():
    """Enabling DR reduces FPS by < 5%."""
    fps_no_dr = benchmark_fps(env_no_dr, n_steps=10000)
    fps_dr = benchmark_fps(env_with_dr, n_steps=10000)
    assert (fps_no_dr - fps_dr) / fps_no_dr < 0.05
```

**File: `tests/test_qp_solver.py`** (5 tests)

```python
# Test I: C++ QP matches Python implementation
def test_cpp_matches_python():
    """C++ and Python QP give same output on 1000 test cases."""
    for state, action in generate_test_cases(1000):
        sol_py = python_rcbf_qp(state, action)
        sol_cpp = cpp_rcbf_qp(state, action)
        assert abs(sol_py.v - sol_cpp.v) < 1e-4
        assert abs(sol_py.omega - sol_cpp.omega) < 1e-4

# Test J: QP solve time within budget
def test_qp_solve_time():
    """95th percentile QP solve time < 50ms."""
    times = []
    for state, action in generate_test_cases(10000):
        t0 = time.perf_counter_ns()
        cpp_rcbf_qp(state, action)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    assert np.percentile(times, 95) < 50.0

# Test K: QP returns feasible solution for valid inputs
def test_qp_feasible():
    """100% feasibility rate for 1000 feasible inputs."""
    for state, action in generate_feasible_cases(1000):
        sol = cpp_rcbf_qp(state, action)
        assert sol.feasible
        assert sol.tier_used == 1  # Exact solution

# Test L: QP detects infeasible inputs and escalates
def test_qp_infeasible_escalation():
    """Infeasible inputs trigger Tier 2 or Tier 3."""
    for state, action in generate_infeasible_cases(100):
        sol = cpp_rcbf_qp(state, action)
        assert sol.tier_used in [2, 3]  # Must escalate

# Test M: VCP-CBF arena constraint matches analytical
def test_vcp_cbf_arena_constraint():
    """VCP-CBF arena constraint coefficients match hand-computed values."""
    # Robot at (9.5, 5.0, 0.0), wall at x=10, d_vcp=0.1
    constraint = vcp_cbf_arena(x=9.5, y=5.0, theta=0.0, d_vcp=0.1,
                                arena_bounds={'x_max': 10.0})
    # h = R_arena^2 - ||q - center||^2 where q = (9.6, 5.0)
    # Expected: a_v < 0 (moving toward wall reduces h)
    assert constraint.a_v < 0
    assert abs(constraint.b) > 0  # Non-trivial constraint
```

**File: `tests/test_gp_cold_start.py`** (5 tests)

```python
# Test N: Cold-start kappa starts at 2x nominal
def test_cold_start_initial_kappa():
    """Kappa starts at 2.0 * kappa_nominal."""
    mgr = GPColdStartManager("prefill.npz", kappa_nominal=1.0, kappa_init=2.0)
    assert abs(mgr.get_kappa() - 2.0) < 1e-6

# Test O: Kappa decays exponentially
def test_kappa_exponential_decay():
    """Kappa decays toward nominal over 200 steps."""
    mgr = GPColdStartManager("prefill.npz", kappa_nominal=1.0, kappa_init=2.0, tau=50)
    kappas = []
    for _ in range(200):
        kappas.append(mgr.get_kappa())
        mgr.step(mean_posterior_variance=0.1)  # GP still uncertain
    assert kappas[0] > kappas[-1]
    assert abs(kappas[-1] - 1.0) < 0.05  # Near nominal after 200 steps

# Test P: Early termination when GP is confident
def test_cold_start_early_termination():
    """Kappa jumps to nominal when sigma_bar_d < 0.01."""
    mgr = GPColdStartManager("prefill.npz", kappa_nominal=1.0, kappa_init=2.0, tau=50)
    for _ in range(10):
        mgr.step(mean_posterior_variance=0.1)
    assert not mgr.transition_complete
    mgr.step(mean_posterior_variance=0.005)  # GP confident
    assert mgr.transition_complete
    assert abs(mgr.get_kappa() - 1.0) < 1e-6

# Test Q: GP prediction accuracy improves with data
def test_gp_prediction_improves():
    """GP RMSE < 0.05 after 200 data points from known disturbance."""
    gp = DisturbanceGP()
    true_disturbance = lambda x: 0.1 * np.sin(x[0])  # Known
    rmse_at_10 = train_and_eval_gp(gp, true_disturbance, n_points=10)
    rmse_at_200 = train_and_eval_gp(gp, true_disturbance, n_points=200)
    assert rmse_at_200 < rmse_at_10
    assert rmse_at_200 < 0.05

# Test R: GP uncertainty decreases with data
def test_gp_uncertainty_decreases():
    """GP sigma at 200 points < 0.5 * sigma at 10 points."""
    gp = DisturbanceGP()
    sigma_10 = train_gp_and_get_sigma(gp, n_points=10)
    sigma_200 = train_gp_and_get_sigma(gp, n_points=200)
    assert sigma_200 < 0.5 * sigma_10
```

### 7.6 Worked Examples

#### Example 1: ONNX Export and Verification

```
Setup:
  Trained pursuer policy: obs_dim=40, hidden_dim=64, action_dim=2
  Export to ONNX opset 17

Step 1: Export
  dummy_obs = torch.randn(1, 40)       # single observation
  dummy_hidden = torch.zeros(1, 1, 64)  # LSTM hidden state
  torch.onnx.export(model, (dummy_obs, dummy_hidden), "pursuer.onnx")
  File size: 2.1 MB (FP32)

Step 2: Verify on test input
  obs = [0.5, -0.3, 0.1, ...]  (40 values)
  PyTorch output:  action_mean = [0.4523, -0.1287]
  ONNX output:     action_mean = [0.4523, -0.1287]
  Max diff: 2.3e-7 < 1e-5 ✓

Step 3: Quantize to INT8
  Input:  pursuer.onnx (2.1 MB, FP32)
  Output: pursuer_int8.onnx (0.7 MB, INT8)
  Size reduction: 3.0x

Step 4: Verify quantized
  INT8 output:     action_mean = [0.4519, -0.1290]
  Diff from FP32:  [0.0004, 0.0003] < 0.01 ✓

Step 5: Benchmark on RPi4
  Warm-up: 100 inferences
  Mean latency:  3.2ms (INT8 on RPi4)
  P95 latency:   4.8ms < 5ms ✓
  P99 latency:   6.1ms

Key insight: INT8 quantization gives 3x size reduction and ~2x speed
with negligible accuracy loss for our small policy network.
```

#### Example 2: GP Cold-Start Protocol on Real Robot

```
Setup:
  kappa_nominal = 1.0, kappa_init = 2.0, tau = 50
  Pre-fill data: 1000 points from DR simulation
  Robot: TurtleBot4, arena: 6m x 6m

Step 0: Initialize GP with pre-fill data
  GP initialized with 1000 (state, residual) pairs
  Initial sigma_bar_d = 0.15 (moderate uncertainty)
  kappa(0) = 2.0 (double margin → robot is extra cautious)

Step 10 (t = 0.5s):
  kappa(10) = 1.0 + 1.0 * exp(-10/50) = 1.0 + 0.819 = 1.819
  sigma_bar_d = 0.12 (learning real dynamics)
  Robot moves slowly, large safety margins

Step 50 (t = 2.5s):
  kappa(50) = 1.0 + 1.0 * exp(-1.0) = 1.368
  sigma_bar_d = 0.05 (GP improving)
  Safety margins relaxing, robot moves more freely

Step 100 (t = 5.0s):
  kappa(100) = 1.0 + 1.0 * exp(-2.0) = 1.135
  sigma_bar_d = 0.008 < 0.01 → EARLY TERMINATION
  kappa = kappa_nominal = 1.0 immediately
  Robot now operates at full performance

Total cold-start time: 5.0s (100 steps at 20Hz)
Safety violations during cold-start: ZERO ✓
Performance during cold-start: ~60% of nominal (conservative but safe)
```

#### Example 3: C++ RCBF-QP Solve Near Obstacle

```
Setup:
  Pursuer at state: x=8.5, y=5.0, theta=0.0 (facing right wall)
  Wall at x_max = 10.0m, d_vcp = 0.1m
  Obstacle at (9.0, 5.5, r=0.3m)
  Policy desires: v_RL = 0.8 m/s, omega_RL = 0.0 (straight ahead)

Step 1: Compute VCP position
  q = (8.5 + 0.1*cos(0), 5.0 + 0.1*sin(0)) = (8.6, 5.0)

Step 2: Build CBF constraints
  Arena CBF (right wall):
    h_arena = 10.0 - 8.6 = 1.4 (VCP is 1.4m from wall)
    Lf_h = 0, Lg_h = [-cos(0), 0.1*sin(0)] = [-1.0, 0.0]
    Constraint: -1.0*v + 0.0*omega >= -1.0*1.4 + 1.0*0.03
    → v <= 1.43 (not binding, margin is large)

  Obstacle CBF:
    dist_to_obs = sqrt((8.6-9.0)^2 + (5.0-5.5)^2) = 0.64m
    h_obs = 0.64^2 - (0.3+0.15)^2 = 0.41 - 0.20 = 0.21
    Lg_h = [-0.72, -0.05]  (moving toward obstacle reduces h)
    Constraint: -0.72*v - 0.05*omega >= -1.0*0.21 + 1.0*0.02
    → 0.72*v + 0.05*omega <= 0.19
    → v <= 0.26 (at omega=0) — THIS IS BINDING!

Step 3: Solve QP (Tier 1)
  min ||[v, omega] - [0.8, 0.0]||^2
  s.t. v <= 1.43 (arena), 0.72*v + 0.05*omega <= 0.19 (obstacle)
       0 <= v <= 1.0, -2.0 <= omega <= 2.0

  Solution: v* = 0.15, omega* = 1.14
  Tier used: 1 (exact feasible solution found)
  Solve time: 0.3ms on RPi4

Step 4: Interpretation
  Policy wanted v=0.8 straight ahead → toward obstacle
  Safety filter: slow down to v=0.15 AND steer left (omega=1.14)
  VCP-CBF prefers steering over braking — this is the key advantage!
  Action modification: ||u_safe - u_RL|| = 1.28 (significant intervention)
```

---

## 8. Troubleshooting Guide

### 8.1 Sim-to-Real Gap Too Large (> 10%)

**Symptoms:** Real robot capture rate is much lower than simulation, or robot behaves erratically.

**Diagnosis steps:**
1. Check which pipeline stage introduces the most degradation (Session 17 analysis)
2. If A -> B (Isaac Lab -> Gazebo) gap is large: simulation engine differences
3. If C -> D (Gazebo -> Real) gap is large: dynamics mismatch or sensor issues

**Solutions:**
- **Increase DR aggressiveness**: Widen all DR parameter ranges by 50%. Retrain.
- **System identification**: Measure real robot parameters (mass, friction, motor gain) and center DR around them instead of nominal.
- **Add more DR parameters**: Randomize wheel diameter, center of mass offset, motor deadzone.
- **Improve GP model**: Use more pre-fill data, try different kernels (Matern 3/2 instead of SE).
- **Reduce control delay**: Optimize ROS2 node pipeline to reduce end-to-end latency.
- **Velocity matching**: Ensure v_max in training matches real robot capability.

### 8.2 QP Solver Too Slow

**Symptoms:** QP solve time exceeds 50ms, control loop drops below 15Hz.

**Diagnosis steps:**
1. Check which QP configurations are slow (many constraints? near-infeasible?)
2. Profile the C++ code for bottlenecks
3. Check if GP prediction is the bottleneck (not the QP itself)

**Solutions:**
- **Warm-starting**: Use previous solution as initial guess (reduces iterations by 50-80%).
- **Reduce constraints**: If many obstacles, only include nearest 3 in QP.
- **Switch solver**: Try qpOASES if OSQP is slow (or vice versa).
- **Closed-form fallback**: For the simple 2D case, the CBF-QP may have a closed-form solution [05]. Implement as fallback.
- **Reduce GP overhead**: Cache GP predictions, update less frequently (every 20 steps instead of 10).
- **Hardware upgrade**: If on RPi4, offload QP to a connected Jetson or laptop.

### 8.3 ONNX Conversion Errors

**Symptoms:** `torch.onnx.export()` fails or produces incorrect model.

**Diagnosis steps:**
1. Check error message -- usually unsupported operations
2. Identify which layer/operation is not supported in ONNX opset 17

**Solutions:**
- **Unsupported ops**: Replace with ONNX-compatible alternatives (e.g., use `torch.nn.functional` instead of custom ops).
- **Dynamic shapes**: Use `dynamic_axes` parameter for variable batch sizes.
- **LSTM/GRU export**: These require special handling. Export the recurrent part separately if needed.
- **BiMDN mixture density**: The mixture sampling is not needed at inference (just use the mean). Simplify the model before export.
- **Beta distribution**: Export only the alpha/beta parameters, compute the distribution externally.
- **Opset version**: Try different opset versions (13-17). Some ops are better supported in different versions.

### 8.4 ROS2 Timing Issues

**Symptoms:** Control loop is inconsistent, topics arrive late, actions are delayed.

**Diagnosis steps:**
1. Use `ros2 topic hz` to check actual publication rates
2. Use ROS2 tracing tools to measure node latencies
3. Check CPU usage on the robot

**Solutions:**
- **Timer-based control**: Use ROS2 timer callbacks (not subscription-driven) for the control loop. Set timer to 20Hz.
- **Executor configuration**: Use MultiThreadedExecutor for parallel node execution.
- **Priority scheduling**: Set real-time priority for the safety filter node.
- **Reduce subscription queue sizes**: Set queue_size=1 with `best_effort` QoS for sensor topics (always use latest data).
- **Profile and optimize**: Use `perf` or `callgrind` to find bottlenecks in node callbacks.
- **Reduce node count**: Merge sensor_fusion + observation_encoder into a single node to reduce IPC overhead.

### 8.5 GP Divergence

**Symptoms:** GP predictions become large or oscillatory, sigma_d grows instead of shrinking.

**Diagnosis steps:**
1. Plot GP training data (input states vs residuals)
2. Check for outliers in residual computation
3. Verify nominal dynamics model is correct

**Solutions:**
- **Outlier rejection**: Reject residuals > 3 * median absolute deviation.
- **Kernel hyperparameters**: Fix lengthscales if they diverge; use informative priors.
- **Sliding window size**: Reduce from 200 to 100 if data is non-stationary.
- **Multiple GPs**: Use separate GPs for dx, dy, dtheta residuals (instead of multi-output).
- **Nominal model fix**: If residuals are systematically large, the nominal model is wrong. Re-derive or use system identification.
- **Learning rate**: Reduce GP hyperparameter learning rate to prevent oscillation.

### 8.6 Hardware Communication Failures

**Symptoms:** Robot stops responding, topics drop out, connection lost.

**Diagnosis steps:**
1. Check WiFi signal strength and stability
2. Check robot battery level
3. Check ROS2 DDS discovery

**Solutions:**
- **Wired connection**: Use ethernet instead of WiFi for development (USB ethernet adapter).
- **DDS configuration**: Set `ROS_DOMAIN_ID` explicitly, use Cyclone DDS with tuned settings.
- **Watchdog timer**: Implement a safety watchdog that stops the robot if no cmd_vel received for 200ms.
- **Battery monitoring**: Add battery level check; stop experiments below 20%.
- **Automatic reconnection**: ROS2 nodes should handle temporary disconnections gracefully.

### 8.7 Localization Errors

**Symptoms:** Robot position estimates are wrong, CBF constraints evaluated at incorrect states, safety filter makes wrong decisions.

**Diagnosis steps:**
1. Compare localization output to external tracking (OptiTrack/AprilTag)
2. Check lidar scan quality (reflective surfaces, glass walls)
3. Check odometry drift over time

**Solutions:**
- **External tracking**: Use OptiTrack or ceiling-mounted cameras with AprilTags for ground truth. Fuse with onboard sensors.
- **AMCL tuning**: Tune particle filter parameters (num_particles, laser_model_type, recovery behaviors).
- **Map quality**: Create a high-quality map of the arena using SLAM Toolbox before experiments.
- **EKF fusion**: Fuse wheel odometry + lidar AMCL + IMU in an Extended Kalman Filter.
- **Conservative CBF margins**: Increase safety margins to account for localization uncertainty (add position_uncertainty to obstacle radii in CBF).

---

## 9. Guide to Next Phase

### 9.1 Artifacts for Phase 5 (Analysis and Publication)

Phase 4 produces the following artifacts needed for Phase 5:

| Artifact | Location | Used For |
|----------|----------|----------|
| Real-robot evaluation data (50+ episodes) | `evaluation/final_results/` | Paper results tables, statistical tests |
| Sim-to-real gap analysis (4-stage comparison) | `evaluation/gap_analysis_results/` | Paper sim-to-real section |
| QP solver benchmarks | `deployment/benchmarks/` | Paper real-time performance claims |
| GP convergence data | W&B logs | Paper disturbance estimation section |
| Cold-start protocol traces | W&B logs | Paper safety during deployment section |
| Demonstration videos (7+ videos) | `evaluation/final_results/videos/` | Paper supplementary, conference demos |
| Trajectory plots (publication-quality) | `evaluation/final_results/plots/` | Paper figures |
| ONNX model files | `deployment/models/` | Reproducibility, open-source release |
| ROS2 package source | `ros2_ws/src/` | Open-source release |
| Isaac Lab environment | `isaac_lab/` | Open-source release |
| Hardware setup documentation | `hardware/` | Reproducibility |

### 9.2 Data to Collect for Paper

**Table 1: Main Results** (collect during Session 18):
- Capture rate (pursuer perspective): sim, Gazebo, real
- Mean capture time: sim, Gazebo, real
- Safety violation rate: sim, Gazebo, real (must be zero for real)
- Mean episode length: sim, Gazebo, real
- Inter-robot minimum distance: sim, Gazebo, real

**Table 2: Real-Time Performance** (collect during Sessions 11, 15-16):
- ONNX inference time: mean, P95, P99
- QP solve time: mean, P95, P99
- GP update time: mean, P95
- Total control loop time: mean, P95
- Control loop frequency: mean, min

**Table 3: Ablation -- Domain Randomization** (collect during Sessions 4, 17):
- With DR vs without DR: capture rate on real robot
- With GP vs without GP: safety violation rate on real robot
- With cold-start vs without cold-start: safety during first 100 steps

**Table 4: Sim-to-Real Pipeline** (collect during Session 17):
- Performance at each stage: Isaac Lab, Gazebo (no filter), Gazebo (with filter), Real
- Degradation at each transition

**Figure 1: Trajectory Plots** (collect during Session 18):
- 3 example trajectories showing pursuit-evasion behavior on real robot
- Color-coded: pursuer (blue), evader (red), obstacles (gray), arena (black)
- Include CBF constraint boundaries (dashed lines)

**Figure 2: QP Solve Time Histogram** (collect during Session 11):
- Histogram of 10,000 QP solve times on target hardware
- Mark 50ms threshold line

**Figure 3: GP Convergence** (collect during Sessions 12-13):
- Plot GP predicted disturbance vs actual residual over 500 steps
- Show uncertainty bounds (2*sigma) shrinking over time

**Figure 4: Cold-Start Protocol** (collect during Session 13):
- Plot kappa(t) transition from conservative to nominal
- Plot sigma_bar_d(t) decreasing
- Mark when transition completes

### 9.3 Statistical Tests for Paper
- **Sim-to-real gap significance**: Paired t-test or Wilcoxon signed-rank test on per-episode metrics
- **DR ablation significance**: Independent samples t-test, DR-trained vs non-DR-trained on real robot
- **Confidence intervals**: Report 95% CI for all metrics (capture rate, capture time, etc.)
- **Minimum sample size**: 50 real episodes for 95% CI width < 0.1 on capture rate (binomial)

### 9.4 Phase 5 Entry Criteria
Phase 5 can begin once ALL of the following are satisfied:
1. All Phase 4 hard requirements met (Section 7.1)
2. All demonstration videos recorded
3. Complete metrics tables filled for all 4 pipeline stages
4. GP cold-start protocol validated
5. At least 50 real-robot episodes logged with full data
6. All code committed and documented

---

## 10. Software Versions & Reproducibility

### 10.1 Pinned Package Versions

```bash
# Core (inherited from Phases 1-3)
stable-baselines3==2.6.0
gymnasium==1.1.1
torch==2.6.0
numpy==2.2.2

# Isaac Lab (GPU simulation)
# Isaac Lab pinned to specific release tag:
# git checkout v1.4.0  # or latest stable at time of implementation
# Requires: CUDA >= 11.8, Isaac Sim >= 4.2

# ONNX export and deployment
onnx==1.17.0
onnxruntime==1.20.1
onnxconverter-common==1.14.0    # for FP16 conversion

# GP disturbance estimation
gpytorch==1.14
linear_operator==0.5.3

# QP solvers (C++ libraries, version at apt/build time)
# osqp: >= 0.6.7 (C library)
# qpOASES: >= 3.2.1 (C++ library)
# Python OSQP for cross-validation:
osqp==0.6.7.post3

# ROS2 Humble (Ubuntu 22.04)
# ros-humble-desktop (system packages, pinned by distro)
# Gazebo Fortress (system packages)

# Trajectory clustering and analysis
scikit-learn==1.6.1

# Visualization & Tracking (inherited from Phase 1, listed for completeness)
# pygame==2.6.1            # Available for offline trajectory replay / video generation
# wandb==0.19.1            # Already in Phase 1 — sim-to-real dashboards, real-robot logging
# hydra-core==1.3.2        # Already in Phase 1 — DR configs, deployment configs
# omegaconf==2.3.0         # Already in Phase 1
# Note: Phase 4 primarily uses Isaac Lab, Gazebo, and RViz2 for real-time visualization
```

**Compatibility notes**:
- Isaac Lab requires NVIDIA GPU with CUDA; incompatible with Apple Silicon / AMD GPUs
- ONNX opset version 17 used for export; requires onnx >= 1.14
- ROS2 Humble EOL: May 2027; compatible with Ubuntu 22.04
- GPyTorch 1.14 requires torch >= 2.0; our torch 2.6.0 is compatible
- onnxruntime ARM build (for RPi4) may need to be compiled from source

### 10.2 Reproducibility Protocol

1. **Random seeds**: All experiments use seeds `[0, 1, 2, 3, 4]` as in prior phases
2. **Isaac Lab determinism**: Set `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True`
3. **ONNX determinism**: ONNX models are deterministic by design; verify by comparing 1000 inference calls
4. **Domain randomization seed**: DR parameters seeded per-environment at reset; record `env_seed = base_seed + env_idx`
5. **GP reproducibility**: Fix GP hyperparameter init; set `torch.manual_seed(seed)` before GP training
6. **Gazebo reproducibility**: Gazebo physics is NOT perfectly deterministic; run 5 seeds and report mean ± std
7. **Real robot**: Real experiments are inherently non-repeatable; report over 50+ episodes for statistical power
8. **Hardware documentation**: Record exact hardware specs (RPi4 model, Jetson model, WiFi router, robot firmware version)
9. **Git state**: Tag each experiment with the exact git commit hash; store in W&B run metadata
10. **Environment checksum**: Hash the Isaac Lab environment config to detect inadvertent changes

---

## Appendix A: File Structure Overview

```
claude_pursuit_evasion/
├── isaac_lab/
│   ├── install_isaac_lab.sh
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── pe_scene.py
│   │   ├── pursuit_evasion_env.py
│   │   ├── pe_reward.py
│   │   ├── pe_observation.py
│   │   ├── pe_termination.py
│   │   └── pe_domain_randomization.py
│   ├── config/
│   │   ├── pe_env_cfg.py
│   │   ├── dr_config.py
│   │   └── training_config.py
│   └── training/
│       ├── train_pe_dr.py
│       └── ams_drl_isaaclab.py
├── deployment/
│   ├── onnx_export.py
│   ├── onnx_verify.py
│   ├── onnx_quantize.py
│   ├── onnx_benchmark.py
│   ├── gp_cold_start.py
│   ├── models/
│   │   ├── pursuer_policy.onnx
│   │   ├── pursuer_policy_int8.onnx
│   │   ├── pursuer_policy_fp16.onnx
│   │   ├── evader_policy.onnx
│   │   ├── evader_policy_int8.onnx
│   │   └── evader_policy_fp16.onnx
│   ├── gp_prefill_data/
│   │   └── prefill_1000.npz
│   ├── benchmarks/
│   │   ├── qp_benchmark.cpp
│   │   ├── qp_benchmark_results.json
│   │   └── plot_qp_benchmark.py
│   └── benchmark_results/
├── ros2_ws/
│   └── src/
│       ├── pe_msgs/
│       │   └── msg/
│       │       ├── PEState.msg
│       │       ├── PEObservation.msg
│       │       ├── GPPrediction.msg
│       │       └── PEDiagnostics.msg
│       ├── pe_gazebo/
│       │   ├── worlds/
│       │   │   └── pe_arena.sdf
│       │   ├── launch/
│       │   │   └── pe_gazebo.launch.py
│       │   ├── models/
│       │   └── config/
│       │       └── turtlebot_params.yaml
│       ├── pe_inference/
│       │   └── pe_inference/
│       │       ├── sensor_fusion_node.py
│       │       ├── observation_encoder_node.py
│       │       ├── policy_inference_node.py
│       │       ├── gp_estimator_node.py
│       │       ├── gp_estimator.py
│       │       ├── gp_cold_start_manager.py
│       │       └── eval_logger_node.py
│       ├── pe_safety/
│       │   ├── src/
│       │   │   ├── safety_filter_node.cpp
│       │   │   ├── rcbf_qp_solver.cpp
│       │   │   ├── rcbf_qp_solver.hpp
│       │   │   ├── vcp_cbf.cpp
│       │   │   └── vcp_cbf.hpp
│       │   └── CMakeLists.txt
│       └── pe_bringup/
│           └── launch/
│               ├── pe_full.launch.py
│               ├── pe_gazebo_eval.launch.py
│               ├── pe_single_robot.launch.py
│               └── pe_two_robots.launch.py
├── hardware/
│   ├── robot_setup.md
│   ├── arena_setup.md
│   └── calibration/
│       ├── lidar_calibration.py
│       └── odom_calibration.py
├── evaluation/
│   ├── gazebo_eval.py
│   ├── real_robot_single_eval.py
│   ├── real_robot_pe_eval.py
│   ├── real_robot_pe_metrics.py
│   ├── sim_to_real_gap_analysis.py
│   ├── final_real_world_eval.py
│   ├── gap_analysis_results/
│   └── final_results/
│       ├── summary.md
│       ├── plots/
│       └── videos/
└── tests/
    ├── test_domain_randomization.py
    ├── test_gp_estimator.py
    ├── test_onnx_export.py
    ├── test_qp_solver.py
    ├── test_vcp_cbf.py
    ├── test_cold_start.py
    ├── test_ros2_integration.py
    └── test_full_pipeline.py
```

## Appendix B: Key Equations Quick Reference

### B.1 Unicycle Dynamics
```
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = omega
u = [v, omega],  v in [0, v_max], omega in [-omega_max, omega_max]
```

### B.2 VCP-CBF [N12]
```
Virtual control point: q = [x + d*cos(theta), y + d*sin(theta)]
VCP derivative: q_dot = [v*cos(theta) - d*omega*sin(theta),
                          v*sin(theta) + d*omega*cos(theta)]
CBF condition: dh/dq * q_dot + alpha * h(x) >= 0
This is affine in u = [v, omega] (uniform relative degree 1)
```

### B.3 RCBF-QP [06]
```
u* = argmin_{u} ||u - u_RL||^2
subject to: L_f h + L_g h * u >= -alpha * h(x) + kappa * sigma_d(x)
            u_min <= u <= u_max
```

### B.4 GP Cold-Start Transition
```
kappa(t) = kappa_nominal + (kappa_init - kappa_nominal) * exp(-t / tau)
where: kappa_init = 2.0 * kappa_nominal, tau = 50 steps
Early termination: if sigma_bar_d < 0.01, set kappa = kappa_nominal
```

### B.5 GP Disturbance Model
```
d(x) ~ GP(0, k(x, x'))
k(x, x') = sigma_f^2 * exp(-0.5 * sum_i (x_i - x'_i)^2 / l_i^2)
Online update: d_t = x_{t+1} - f(x_t) - g(x_t) * u_t
Sliding window: keep most recent 200 data points
Update frequency: every 10 control steps
```
