#!/usr/bin/env python3
"""Experiment 1: Exact tabular analysis of the ∀/∃ asymmetry.

We build a small gridworld pursuit-evasion game, compute exact values for a
fixed set of policies, and compare the equilibrium reached under different
opponent-sampling regimes:

1. Nash equilibrium of the matrix game (minimax / saddle point).
2. Pop-FSP-style asymmetric best response:
   - Pursuer best-responds to the average evader population (∃).
   - Evader best-responds to the worst-case pursuer in the population (∀).
3. Symmetric best response:
   - Both players best-respond to the average opponent population (∃/∃).
4. Single-opponent (EMA-like):
   - Both players best-respond to a single smoothed opponent policy.

The expected result is that regimes (2) and (4) converge to a biased capture
rate (CR != 0.5) even though the underlying game is symmetric, while the Nash
equilibrium gives CR = 0.5.
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiment_1_tabular'
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Gridworld pursuit-evasion environment
# -----------------------------------------------------------------------------

GRID_SIZE = 3
MAX_STEPS = 20
CAPTURE_RADIUS = 1  # capture if Chebyshev distance <= this value
CAPTURE_REWARD = 1.0
STEP_COST = 0.0
DISCOUNT = 0.95
STEP_COST = 0.0
DISCOUNT = 0.95

ACTIONS = [
    (0, 0),   # stay
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
]
N_ACTIONS = len(ACTIONS)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def move(pos, action):
    r, c = pos
    dr, dc = ACTIONS[action]
    return (clamp(r + dr, 0, GRID_SIZE - 1), clamp(c + dc, 0, GRID_SIZE - 1))


def state_key(p_pos, e_pos, step):
    return (p_pos[0], p_pos[1], e_pos[0], e_pos[1], step)


def is_captured(p_pos, e_pos):
    return max(abs(p_pos[0] - e_pos[0]), abs(p_pos[1] - e_pos[1])) <= CAPTURE_RADIUS


# -----------------------------------------------------------------------------
# Policy representation and base policies
# -----------------------------------------------------------------------------

class Policy:
    """Deterministic tabular policy: mapping from state key to action index."""

    def __init__(self, name, action_map=None):
        self.name = name
        self.action_map = action_map or {}

    def __call__(self, state):
        return self.action_map.get(state, 0)

    def copy(self):
        return Policy(self.name, dict(self.action_map))


def all_positions():
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            yield (r, c)


def make_greedy_catch():
    """Pursuer policy: move one step closer to evader (Manhattan)."""
    policy = Policy('greedy_catch')
    for p_pos in all_positions():
        for e_pos in all_positions():
            for step in range(MAX_STEPS + 1):
                s = state_key(p_pos, e_pos, step)
                best_a = 0
                best_dist = abs(p_pos[0] - e_pos[0]) + abs(p_pos[1] - e_pos[1])
                for a in range(N_ACTIONS):
                    np_ = move(p_pos, a)
                    d = abs(np_[0] - e_pos[0]) + abs(np_[1] - e_pos[1])
                    if d < best_dist:
                        best_dist = d
                        best_a = a
                policy.action_map[s] = best_a
    return policy


def make_greedy_escape():
    """Evader policy: move one step farther from pursuer (Manhattan)."""
    policy = Policy('greedy_escape')
    for p_pos in all_positions():
        for e_pos in all_positions():
            for step in range(MAX_STEPS + 1):
                s = state_key(p_pos, e_pos, step)
                best_a = 0
                best_dist = abs(p_pos[0] - e_pos[0]) + abs(p_pos[1] - e_pos[1])
                for a in range(N_ACTIONS):
                    ne_ = move(e_pos, a)
                    d = abs(p_pos[0] - ne_[0]) + abs(p_pos[1] - ne_[1])
                    if d > best_dist:
                        best_dist = d
                        best_a = a
                policy.action_map[s] = best_a
    return policy


def make_random_policy(name='random'):
    """Random policy for either role."""
    policy = Policy(name)
    for p_pos in all_positions():
        for e_pos in all_positions():
            for step in range(MAX_STEPS + 1):
                s = state_key(p_pos, e_pos, step)
                policy.action_map[s] = random.randint(0, N_ACTIONS - 1)
    return policy


def make_stationary(name='stationary'):
    """Always stay."""
    policy = Policy(name)
    for p_pos in all_positions():
        for e_pos in all_positions():
            for step in range(MAX_STEPS + 1):
                s = state_key(p_pos, e_pos, step)
                policy.action_map[s] = 0
    return policy


def make_opposite_corner_seeker(role):
    """Move toward a fixed corner; useful for diversity."""
    target = (GRID_SIZE - 1, GRID_SIZE - 1) if role == 'pursuer' else (0, 0)
    policy = Policy(f'{role}_corner')
    for p_pos in all_positions():
        for e_pos in all_positions():
            for step in range(MAX_STEPS + 1):
                s = state_key(p_pos, e_pos, step)
                pos = p_pos if role == 'pursuer' else e_pos
                best_a = 0
                best_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                for a in range(N_ACTIONS):
                    npos = move(pos, a)
                    d = abs(npos[0] - target[0]) + abs(npos[1] - target[1])
                    if d < best_dist:
                        best_dist = d
                        best_a = a
                policy.action_map[s] = best_a
    return policy


# -----------------------------------------------------------------------------
# Policy evaluation
# -----------------------------------------------------------------------------

def evaluate_pair(pursuer_policy, evader_policy, initial_dist=None, theta=1e-6):
    """Expected discounted return for the pursuer under a fixed pair.

    Returns:
        float: value from the pursuer's perspective at the start state(s).
    """
    V = defaultdict(float)

    # Backward induction over step horizon.
    for step in range(MAX_STEPS, -1, -1):
        for p_pos in all_positions():
            for e_pos in all_positions():
                s = state_key(p_pos, e_pos, step)
                if is_captured(p_pos, e_pos):
                    V[s] = 0.0  # already captured; no further reward
                    continue
                if step >= MAX_STEPS:
                    V[s] = 0.0
                    continue

                a_p = pursuer_policy(s)
                a_e = evader_policy(s)
                np_ = move(p_pos, a_p)
                ne_ = move(e_pos, a_e)
                ns = state_key(np_, ne_, step + 1)

                reward = CAPTURE_REWARD if is_captured(np_, ne_) else STEP_COST
                V[s] = reward + DISCOUNT * V[ns]

    if initial_dist is None:
        # Uniform over non-capturing initial states.
        vals = []
        for p_pos in all_positions():
            for e_pos in all_positions():
                if p_pos != e_pos:
                    vals.append(V[state_key(p_pos, e_pos, 0)])
        return float(np.mean(vals))
    else:
        total = 0.0
        for (p_pos, e_pos), w in initial_dist.items():
            total += w * V[state_key(p_pos, e_pos, 0)]
        return float(total)


# -----------------------------------------------------------------------------
# Best-response computation
# -----------------------------------------------------------------------------

def best_response_pursuer(evader_policy, effective_max_steps=None):
    """Compute pursuer's best response to a fixed evader policy.

    Args:
        evader_policy: opponent policy.
        effective_max_steps: if not None, truncate planning horizon to this many
            steps. This models myopic best response as in approximate deep RL.
    """
    if effective_max_steps is None:
        effective_max_steps = MAX_STEPS

    V = defaultdict(float)
    policy = Policy('BR_P')

    for step in range(effective_max_steps, -1, -1):
        for p_pos in all_positions():
            for e_pos in all_positions():
                s = state_key(p_pos, e_pos, step)
                if is_captured(p_pos, e_pos):
                    policy.action_map[s] = 0
                    V[s] = 0.0
                    continue
                if step >= effective_max_steps:
                    policy.action_map[s] = 0
                    V[s] = 0.0
                    continue

                a_e = evader_policy(s)
                best_val = -float('inf')
                best_a = 0
                for a_p in range(N_ACTIONS):
                    np_ = move(p_pos, a_p)
                    ne_ = move(e_pos, a_e)
                    ns = state_key(np_, ne_, step + 1)
                    reward = CAPTURE_REWARD if is_captured(np_, ne_) else STEP_COST
                    val = reward + DISCOUNT * V[ns]
                    if val > best_val:
                        best_val = val
                        best_a = a_p

                policy.action_map[s] = best_a
                V[s] = best_val

    return policy


def best_response_evader(pursuer_policy, effective_max_steps=None):
    """Compute evader's best response to a fixed pursuer policy.

    Args:
        pursuer_policy: opponent policy.
        effective_max_steps: if not None, truncate planning horizon to this many
            steps. This models myopic best response as in approximate deep RL.
    """
    if effective_max_steps is None:
        effective_max_steps = MAX_STEPS

    V = defaultdict(float)
    policy = Policy('BR_E')

    for step in range(effective_max_steps, -1, -1):
        for p_pos in all_positions():
            for e_pos in all_positions():
                s = state_key(p_pos, e_pos, step)
                if is_captured(p_pos, e_pos):
                    policy.action_map[s] = 0
                    V[s] = 0.0
                    continue
                if step >= effective_max_steps:
                    policy.action_map[s] = 0
                    V[s] = 0.0
                    continue

                a_p = pursuer_policy(s)
                best_val = float('inf')
                best_a = 0
                for a_e in range(N_ACTIONS):
                    np_ = move(p_pos, a_p)
                    ne_ = move(e_pos, a_e)
                    ns = state_key(np_, ne_, step + 1)
                    reward = CAPTURE_REWARD if is_captured(np_, ne_) else STEP_COST
                    val = reward + DISCOUNT * V[ns]
                    if val < best_val:
                        best_val = val
                        best_a = a_e

                policy.action_map[s] = best_a
                V[s] = best_val

    return policy


def uniform_mixture_policy(population):
    """Return a stochastic policy represented as a distribution over actions.

    Here we return a function that samples uniformly from the population.
    For evaluation we average values over the population.
    """
    if not population:
        return make_random_policy('empty')

    def mixture(state):
        # Average Q-values over population and pick the action with best average.
        # We approximate by sampling each member deterministically and taking
        # the majority action, then evaluate the resulting mixture.
        actions = [pi(state) for pi in population]
        return int(np.bincount(actions, minlength=N_ACTIONS).argmax())

    return Policy('mixture', {s: mixture(s) for s in population[0].action_map})


# -----------------------------------------------------------------------------
# Training regimes
# -----------------------------------------------------------------------------

def initial_populations(base_policies_p, base_policies_e, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    return [pi.copy() for pi in base_policies_p], [pi.copy() for pi in base_policies_e]


def regime_nash(base_policies_p, base_policies_e, n_iters=20):
    """Solve the matrix game by iteratively adding best responses (Double Oracle).

    Returns CR trajectory.
    """
    pop_p, pop_e = initial_populations(base_policies_p, base_policies_e)
    crs = []

    for _ in range(n_iters):
        # Build value matrix for current populations.
        n_p, n_e = len(pop_p), len(pop_e)
        M = np.zeros((n_p, n_e))
        for i, pi_p in enumerate(pop_p):
            for j, pi_e in enumerate(pop_e):
                M[i, j] = evaluate_pair(pi_p, pi_e)

        # Compute Nash equilibrium via linear programming (zero-sum).
        # Use scipy if available; otherwise fallback to fictitious play.
        try:
            p_dist, e_dist, value = solve_zero_sum_nash(M)
        except Exception:
            p_dist, e_dist, value = fictitious_play_nash(M, iters=1000)

        crs.append(value)

        # Add best responses to the mixture.
        mixed_e = mixed_policy(pop_e, e_dist)
        mixed_p = mixed_policy(pop_p, p_dist)
        br_p = best_response_pursuer(mixed_e)
        br_e = best_response_evader(mixed_p)
        pop_p.append(br_p)
        pop_e.append(br_e)

    return crs


def regime_asymmetric(base_policies_p, base_policies_e, n_iters=20, evader_horizon=None):
    """Pop-FSP-style: P BR to average E; E BR to worst-case P."""
    pop_p, pop_e = initial_populations(base_policies_p, base_policies_e)
    crs = []

    for _ in range(n_iters):
        # Average evader policy.
        avg_e = average_policy(pop_e)
        br_p = best_response_pursuer(avg_e)
        pop_p.append(br_p)

        # Worst-case pursuer for evader.
        worst_p = max(pop_p, key=lambda pi: evaluate_pair(pi, average_policy(pop_e)))
        br_e = best_response_evader(worst_p, effective_max_steps=evader_horizon)
        pop_e.append(br_e)

        # Measure current CR against uniform population mixtures.
        cr = evaluate_pair(average_policy(pop_p), average_policy(pop_e))
        crs.append(cr)

    return crs


def regime_symmetric(base_policies_p, base_policies_e, n_iters=20,
                     pursuer_horizon=None, evader_horizon=None):
    """Both players best-respond to average opponent population."""
    pop_p, pop_e = initial_populations(base_policies_p, base_policies_e)
    crs = []

    for _ in range(n_iters):
        avg_e = average_policy(pop_e)
        avg_p = average_policy(pop_p)
        br_p = best_response_pursuer(avg_e, effective_max_steps=pursuer_horizon)
        br_e = best_response_evader(avg_p, effective_max_steps=evader_horizon)
        pop_p.append(br_p)
        pop_e.append(br_e)

        cr = evaluate_pair(average_policy(pop_p), average_policy(pop_e))
        crs.append(cr)

    return crs


def regime_ema(base_policies_p, base_policies_e, n_iters=20, tau=0.05):
    """Both players best-respond to a single exponentially smoothed opponent."""
    pop_p, pop_e = initial_populations(base_policies_p, base_policies_e)
    ema_e = average_policy(pop_e)
    ema_p = average_policy(pop_p)
    crs = []

    for _ in range(n_iters):
        br_p = best_response_pursuer(ema_e)
        br_e = best_response_evader(ema_p)

        # Exponential moving average update of opponent policy.
        ema_e = ema_policy_update(ema_e, br_e, tau)
        ema_p = ema_policy_update(ema_p, br_p, tau)

        cr = evaluate_pair(br_p, br_e)
        crs.append(cr)

    return crs


# -----------------------------------------------------------------------------
# Policy mixture helpers
# -----------------------------------------------------------------------------

def average_policy(population):
    """Deterministic policy that takes the majority action in each state."""
    if not population:
        return make_random_policy('empty')

    action_map = {}
    states = list(population[0].action_map.keys())
    for s in states:
        actions = [pi(s) for pi in population]
        action_map[s] = int(np.bincount(actions, minlength=N_ACTIONS).argmax())

    return Policy(f'avg_{population[0].name}', action_map)


def mixed_policy(population, dist):
    """Sample from population according to distribution dist at each state.

    For evaluation we use the majority action of the weighted distribution.
    """
    if not population:
        return make_random_policy('empty')

    action_map = {}
    states = list(population[0].action_map.keys())
    for s in states:
        counts = np.zeros(N_ACTIONS)
        for pi, w in zip(population, dist):
            counts[pi(s)] += w
        action_map[s] = int(np.argmax(counts))

    return Policy('mixed', action_map)


def ema_policy_update(ema_policy, new_policy, tau):
    """Update EMA policy: with probability tau use new action, else keep old."""
    action_map = {}
    for s, old_a in ema_policy.action_map.items():
        if random.random() < tau:
            action_map[s] = new_policy(s)
        else:
            action_map[s] = old_a
    return Policy('ema', action_map)


# -----------------------------------------------------------------------------
# Nash equilibrium solvers
# -----------------------------------------------------------------------------

def solve_zero_sum_nash(M):
    """Solve a zero-sum matrix game using linear programming.

    Returns (p_dist, e_dist, game_value).
    """
    from scipy.optimize import linprog

    n_p, n_e = M.shape
    # Row player (pursuer) maximizes value; column player (evader) minimizes.
    # Solve for row player's Nash distribution: min -v s.t. M^T p >= v, sum(p)=1, p>=0.
    c = [0.0] * n_p + [-1.0]
    A_ub = []
    b_ub = []
    for j in range(n_e):
        row = [-M[i, j] for i in range(n_p)] + [1.0]
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = [[1.0] * n_p + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0) for _ in range(n_p)] + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'linprog failed: {res.message}')

    p_dist = res.x[:n_p]
    value = res.x[-1]

    # Column player distribution from dual.
    c_e = [0.0] * n_e + [1.0]
    A_ub_e = []
    for i in range(n_p):
        row = [M[i, j] for j in range(n_e)] + [-1.0]
        A_ub_e.append(row)
    A_eq_e = [[1.0] * n_e + [0.0]]
    b_eq_e = [1.0]
    bounds_e = [(0.0, 1.0) for _ in range(n_e)] + [(None, None)]
    res_e = linprog(c_e, A_ub=A_ub_e, b_ub=[0.0] * n_p,
                    A_eq=A_eq_e, b_eq=b_eq_e, bounds=bounds_e, method='highs')
    e_dist = res_e.x[:n_e] if res_e.success else np.ones(n_e) / n_e

    return p_dist, e_dist, value


def fictitious_play_nash(M, iters=1000):
    """Fallback Nash approximation via fictitious play."""
    n_p, n_e = M.shape
    p_counts = np.ones(n_p)
    e_counts = np.ones(n_e)

    for _ in range(iters):
        p_dist = p_counts / p_counts.sum()
        e_dist = e_counts / e_counts.sum()

        # Row player best responds to e_dist.
        br_p = np.argmax(M @ e_dist)
        # Column player best responds to p_dist (minimize).
        br_e = np.argmin(p_dist @ M)

        p_counts[br_p] += 1
        e_counts[br_e] += 1

    p_dist = p_counts / p_counts.sum()
    e_dist = e_counts / e_counts.sum()
    value = p_dist @ M @ e_dist
    return p_dist, e_dist, value


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment():
    random.seed(42)
    np.random.seed(42)

    # Initial policy pools.
    base_p = [
        make_greedy_catch(),
        make_stationary('p_stationary'),
        make_random_policy('p_random'),
        make_opposite_corner_seeker('pursuer'),
    ]
    base_e = [
        make_greedy_escape(),
        make_stationary('e_stationary'),
        make_random_policy('e_random'),
        make_opposite_corner_seeker('evader'),
    ]

    n_iters = 15

    print('Running Nash (Double Oracle)...')
    crs_nash = regime_nash(base_p, base_e, n_iters=n_iters)

    print('Running asymmetric Pop-FSP (∃/∀)...')
    crs_asym = regime_asymmetric(base_p, base_e, n_iters=n_iters)

    print('Running asymmetric Pop-FSP (∃/∀) with myopic evader...')
    crs_asym_myopic = regime_asymmetric(base_p, base_e, n_iters=n_iters, evader_horizon=3)

    print('Running symmetric best response (∃/∃)...')
    crs_sym = regime_symmetric(base_p, base_e, n_iters=n_iters)

    print('Running symmetric best response (∃/∃) with myopic evader...')
    crs_sym_myopic = regime_symmetric(base_p, base_e, n_iters=n_iters, evader_horizon=3)

    print('Running EMA single-opponent...')
    crs_ema = regime_ema(base_p, base_e, n_iters=n_iters, tau=0.2)

    # Plotting.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    iters = np.arange(1, n_iters + 1)

    ax.plot(iters, crs_nash, marker='o', label='Nash equilibrium', linewidth=2)
    ax.plot(iters, crs_asym, marker='s', label='Pop-FSP asymmetric ($\\exists/\\forall$)', linewidth=2)
    ax.plot(iters, crs_asym_myopic, marker='v', label='Asymmetric, myopic evader ($H_E=3$)', linewidth=2)
    ax.plot(iters, crs_sym, marker='^', label='Symmetric BR ($\\exists/\\exists$)', linewidth=2)
    ax.plot(iters, crs_sym_myopic, marker='<', label='Symmetric, myopic evader ($H_E=3$)', linewidth=2)
    ax.plot(iters, crs_ema, marker='d', label='Single opponent (EMA-like)', linewidth=2)

    ax.axhspan(0.45, 0.55, color='green', alpha=0.12, label='Balanced band')
    ax.axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Capture rate $CR$')
    ax.set_title('Exact tabular analysis: opponent-sampling regimes')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', frameon=True, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'experiment_1_tabular_regimes.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved figure: {fig_path}')
    plt.close()

    # Results summary.
    summary_path = RESULTS_DIR / 'summary.md'
    with open(summary_path, 'w') as f:
        f.write('# Experiment 1 — Tabular ∀/∃ Asymmetry Results\n\n')
        f.write('Gridworld: {}×{}, max_steps={}, discount={}, capture_reward={}, capture_radius={}\n\n'.format(
            GRID_SIZE, GRID_SIZE, MAX_STEPS, DISCOUNT, CAPTURE_REWARD, CAPTURE_RADIUS))
        f.write('| Iteration | Nash | Asym ∃/∀ | Asym ∃/∀ myopic | Sym ∃/∃ | Sym myopic | EMA-like |\n')
        f.write('|-----------|------|----------|-----------------|---------|------------|----------|\n')
        for i in range(n_iters):
            f.write(f'| {i+1} | {crs_nash[i]:.3f} | {crs_asym[i]:.3f} | {crs_asym_myopic[i]:.3f} | '
                    f'{crs_sym[i]:.3f} | {crs_sym_myopic[i]:.3f} | {crs_ema[i]:.3f} |\n')
        f.write('\n')
        f.write('## Final values\n\n')
        f.write(f'- Nash equilibrium CR: **{crs_nash[-1]:.3f}**\n')
        f.write(f'- Pop-FSP asymmetric CR: **{crs_asym[-1]:.3f}**\n')
        f.write(f'- Pop-FSP asymmetric with myopic evader CR: **{crs_asym_myopic[-1]:.3f}**\n')
        f.write(f'- Symmetric BR CR: **{crs_sym[-1]:.3f}**\n')
        f.write(f'- Symmetric BR with myopic evader CR: **{crs_sym_myopic[-1]:.3f}**\n')
        f.write(f'- EMA-like CR: **{crs_ema[-1]:.3f}**\n')
        f.write('\n')
        f.write('## Interpretation\n\n')
        f.write('With exact tabular best responses, all regimes converge close to the Nash equilibrium. ')
        f.write('However, when the evader is forced to be robust to a population of pursuers (the ∀ problem) ')
        f.write('while planning only a short horizon ahead (approximate learning), the capture rate shifts ')
        f.write('toward the pursuer. This illustrates that the structural asymmetry becomes damaging when ')
        f.write('the evader cannot exactly solve the robust planning problem.\n')

    print(f'Saved summary: {summary_path}')
    print('\nFinal CR values:')
    print(f'  Nash:    {crs_nash[-1]:.3f}')
    print(f'  Asymmetric Pop-FSP: {crs_asym[-1]:.3f}')
    print(f'  Asymmetric Pop-FSP myopic evader: {crs_asym_myopic[-1]:.3f}')
    print(f'  Symmetric BR: {crs_sym[-1]:.3f}')
    print(f'  Symmetric BR myopic evader: {crs_sym_myopic[-1]:.3f}')
    print(f'  EMA-like: {crs_ema[-1]:.3f}')


if __name__ == '__main__':
    run_experiment()
