"""S24.1 Step 1.1 — the pursuer's TACTIC-SELECTION semi-MDP.

Spec: `docs/pathway/tactics_prereg_s24_1_2026-10-13.md` sections 2, 3 and 5
(CG-T1/CG-T2) INCLUDING Amendments A1 and A2. One variable changes against
S24.0: the pursuer's `select`.

Amendment A1 (before any arm trains) rules this step: **A1.1** adds the
`shield_obs` triple and the `escape_obs` pair to the observation (`OBS_LEN`
50 -> 55), observable and so NOT zeroed by `NOMEM`; **A1.2** counts a switch
only between spans of >= 1 env step; **A1.3** accepts readings 1, 2 and 5
below. Amendment A2 then rules: **A2.1** masks out any tactic that would be
DONE on entry, so every decision spends >= 1 tactic step and reading 4's
search-policy branch becomes a guard that must read 0; **A2.4** sets the
recorder's `_in_span` scope. Where an amendment and the brief differ, the
amendment wins.

The game is NOT re-assembled here. `scripts/s24_tactic_game.py::build_game` is
the frozen v0b assembly (D149/D150) — `RecoveryShieldV2` fresh per episode at
the CBF defaults, `hold` 10, `shield_obs`/`escape_obs`, `reward_wrapper=None`,
`LastSeenBelief` + `SearchStalenessTracker` + `TacticRecogniser` driven once per
env step by `_TacticGameMemory`, the scripted `TacticOpponent(EvaderRulesV0)` as
the evader — and this module imports it and replaces ONE field:
`mdp.law.select`. The pursuer's tactic instances, its memory and `SWEEP`'s
belief hook stay `PursuerRulesV0`'s, so the library the learner chooses from is
bit-for-bit the one the baseline chose from.

Readings taken here, because the brief or the prereg can be read two ways. The
prereg wins wherever they differ and that is said at the line.

1. **Action 2 and 3 are FAMILIES; the obstacle index is the env's.** The prereg
   fixes "the five pursuer tactics", but `COUNTER-ORBIT` and `AMBUSH` exist per
   obstacle. The action resolves to the instance about obstacle `k` = the
   obstacle whose SURFACE is nearest the evader's position estimate (its
   observed xy while visible, else the dead-reckoning point, else the pursuer's
   own pose) — `PursuerRulesV0`'s own choice of k, extended to the invisible
   case — and the mask entry for that action is THAT instance's `applicable()`.
2. **`COUNTER-ORBIT` is masked out while the evader is invisible.**
   `CounterOrbit.applicable` reads `env.evader_pose` (L1.5, Step 1's), which is
   the one precondition in the library that is not a function of the pursuer's
   own observation and the static map; offering it while the evader is unseen
   would leak truth into the policy's input. `PursuerRulesV0` reaches it only on
   visible steps anyway (`pe2/selectors.py` says so). `AMBUSH`'s precondition is
   the gap point and the pursuer's own pose, so it is not gated.
3. **Decision points are the current tactic's DONE — which `_Capped` makes
   happen at `DECISION_PERIOD` = 40 steps at the latest — exactly as in v0.** The
   T = 40 cap is therefore not re-implemented: it is `pe2.selectors._Capped`,
   the same device the baseline ran under.
4. **A tactic that is DONE on its FIRST call of a span costs one env step driven
   by the search policy** (`OrbitSearch`, `SWEEP`'s own primitive), then the
   selector decides again — reading 4 of `scripts/s24_tactic_game.py`, kept so
   that every decision advances the episode by at least one env step.
5. **The current-tactic block describes the span that is ENDING** (the tactic
   the pursuer has just been running) and the last-3 block the three spans
   before it; all zero at the episode's first decision.
6. **Reward is the env's terminal +-1 summed over the span and nothing else**
   (prereg section 3, D138): no acquisition bonus, no stall penalty, no
   intervention penalty. gamma = 0.99 per decision is the learner's.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pe2.config import EnvConfig
from pe2.obs import opponent_world_from_obs, own_pose_from_obs
from pe2.recogniser import FEATURES, PURSUER_TACTICS, tactic_family
from pe2.scripted import LAST_SEEN_TOL
from pe2.selectors import (
    DECISION_PERIOD,
    _Capped,
    _nearest_obstacle,
    _surface_distances,
)
from pe2.structured import OrbitSearch
from pe2.tactics import AMBUSH_CLOSE
from pe2.tactics import OrbitSearch as OrbitSearchTactic

PREREG = "docs/pathway/tactics_prereg_s24_1_2026-10-13.md"

# The action space, in the prereg's order. Imported, not restated: the
# recogniser's pursuer label space IS this list.
#
# S24-OS: the library is now a FLAG. `v0` is the five tactics of S24.1, the
# object `PURSUER_TACTICS` itself, so every frozen artefact replays bit for bit;
# `v1` adds S24-OS's `ORBIT-SEARCH` as a sixth action. The recogniser's label
# space is NOT extended (that is the recogniser's own, D147/D148): the selector
# stops aliasing it and takes it as `v0`'s value instead.
LIBRARIES: dict[str, tuple[str, ...]] = {
    "v0": PURSUER_TACTICS,  # ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
    "v1": PURSUER_TACTICS + ("ORBIT-SEARCH",),
}
DEFAULT_LIBRARY = "v0"
N_OBSTACLE_SLOTS = 2  # "the nearest two obstacles" (prereg section 3)
N_LAST_TACTICS = 3  # "one-hot of own last 3 tactics"
AGE_SCALE = 100.0  # `BELIEF_FRESH_STEPS`: the horizon SWEEP will steer at


def _obs_layout(n_actions: int) -> tuple[tuple[str, int, bool], ...]:
    """The observation layout, block by block: (name, width, in_memory_block).

    `in_memory_block` is what the NOMEM control zeroes (prereg section 3: "the
    memory block (dead-reckoning, own history, opponent features) zeroed"); the
    block POSITIONS stay, so the two arms have an identical policy shape.

    Only the two one-hot blocks are a function of the library's size, so the
    layout is the same function of `n_actions` for every library and v1 is
    `OBS_LEN` 55 + 1 + 3 = 59.
    """
    return (
        ("visible", 1, False),  # obs[6]
        ("evader_range_bearing", 2, False),  # range / half, bearing / pi; 0,0 unseen
        ("dead_reckoning", 3, True),  # body-frame xy / arena, age / AGE_SCALE
        ("own_pose", 4, False),  # x / half, y / half, cos theta, sin theta
        ("obstacles", 3 * N_OBSTACLE_SLOTS, False),  # surface dist / half, bearing / pi, radius / r_max
        ("wall", 1, False),  # nearest-wall distance / half
        ("features", len(FEATURES) + 1, True),  # FEATURES order, then `valid`
        ("current_tactic", n_actions + 1, True),  # one-hot, then time-in-tactic / T
        ("last_tactics", n_actions * N_LAST_TACTICS, True),  # most recent first
        ("speed_ratio", 1, False),
        # A1.1: `SearchMDP.shield_features()` — the last executed step's
        # intervention flag, min clearance clipped to [-0.5, 2] m, stall run /
        # `DEADLOCK_STEPS` (= 40) clipped to [0, 2]. Observable, not memory
        # (D113/D131: these are what let a learner recover the shield's cost).
        ("shield", 3, False),
        # A1.1: `SearchMDP.escape_features()` — the admissible omega range at
        # v = 0 at the CURRENT true pose, over omega_max, clipped to [-1, 1].
        ("escape", 2, False),
    )


def _block_slices(obs_layout: tuple[tuple[str, int, bool], ...]) -> dict[str, slice]:
    out, start = {}, 0
    for name, width, _mem in obs_layout:
        out[name] = slice(start, start + width)
        start += width
    return out


def layout(library: str = DEFAULT_LIBRARY):
    """`(ACTIONS, N_ACTIONS, OBS_LAYOUT, OBS_SLICES, MEMORY_SLICES, OBS_LEN)`
    for one library — the six names the module exported as constants, which
    stay bound to `v0`'s values so every existing import is untouched."""
    if library not in LIBRARIES:
        raise ValueError(
            f"unknown tactic library {library!r}; the choices are "
            f"{sorted(LIBRARIES)}"
        )
    actions = LIBRARIES[library]
    obs_layout = _obs_layout(len(actions))
    slices = _block_slices(obs_layout)
    return (
        actions,
        len(actions),
        obs_layout,
        slices,
        tuple(slices[name] for name, _w, mem in obs_layout if mem),
        sum(width for _name, width, _mem in obs_layout),
    )


def obs_len_for(library: str = DEFAULT_LIBRARY) -> int:
    """This library's observation width (55 for v0, 59 for v1)."""
    return layout(library)[5]


# The v0 values, under the names S24.1's trainers, gates, tests and
# `pe2/flat_mdp.py` already import. `ACTIONS is PURSUER_TACTICS` still holds.
ACTIONS, N_ACTIONS, OBS_LAYOUT, OBS_SLICES, MEMORY_SLICES, OBS_LEN = layout()


def done_on_entry(tactic, obs, env, cfg: EnvConfig) -> bool:
    """A2.1: would this tactic's `act` return None on its FIRST call?

    Such a pick leaves a zero-length span and costs one search-policy env step,
    so the selector would be consulted every env step — the reviewer measured
    1 939 of 2 050 forced-`INTERCEPT` decisions this way, which is D128's
    horizon failure. The env masks those actions out, so the question is
    answered from each tactic's OWN termination test, never a re-derivation:

    - `INTERCEPT` / `LEAD` delegate to `PlannerInterceptLaw.act`, which returns
      None iff the evader is invisible AND its own `last_seen` is exhausted
      (never set, or already within `LAST_SEEN_TOL` of the pursuer — the branch
      that clears it). Each tactic owns a separate law, so each is asked about
      its own.
    - `AMBUSH` is DONE on entry when the evader is visible within
      `AMBUSH_CLOSE`; this is `PursuerRulesV0`'s own rule-3 guard ("inside that
      the ambush is DONE on entry and waiting is pointless"), lifted into the
      mask so A2.1's invariant holds for the whole library and not only for the
      two law tactics it names.
    - `SWEEP` is DONE when the evader is visible, which is exactly where its
      `applicable()` already masks it out, and `COUNTER-ORBIT` cannot be DONE
      on entry while applicable (its DONE needs a clear line to the evader,
      its precondition needs obstacle k to block that line, and both inflate
      the disc by `agent_radius` with the planner's MARGIN at 0).
    - `ORBIT-SEARCH` (library v1) is DONE when the evader is visible, like
      `SWEEP` and for the same reason; its other two DONE branches (the step
      cap, an empty belief) are excluded by its own `applicable()`, which the
      mask consults beside this. The clause is inert under v0, where no action
      resolves to that family.
    """
    name = tactic_family(tactic.name)
    visible = bool(obs[6] > 0.5)
    if name in ("INTERCEPT", "LEAD"):
        if visible:
            return False
        last_seen = tactic.law.last_seen
        if last_seen is None:
            return True
        own = own_pose_from_obs(np.asarray(obs), cfg)
        return float(np.hypot(*(np.asarray(last_seen)[:2] - own[:2]))) <= LAST_SEEN_TOL
    if name == "AMBUSH":
        own = np.asarray(env.pursuer_pose, dtype=float)
        evader = np.asarray(env.evader_pose, dtype=float)
        return visible and float(np.hypot(*(evader[:2] - own[:2]))) <= AMBUSH_CLOSE
    if name == "ORBIT-SEARCH":
        return visible
    return False


def spans_with_length(history, step_index: int) -> list[dict]:
    """A span log `(name, start, end)` as dicts carrying the span's length in
    env steps and A1.2's `zero_length` flag. The OPEN span (`end` None) is
    closed at `step_index`, the convention the v0b record's `_history` uses.

    A free function so A1.2's rule can be tested on a hand-built history
    without an episode.
    """
    out = []
    for name, start, end in history:
        stop = int(step_index if end is None else end)
        out.append(
            {
                "name": str(name),
                "start": int(start),
                "end": stop,
                "length": stop - int(start),
                "zero_length": stop - int(start) == 0,
            }
        )
    return out


def count_switches(spans: list[dict]) -> int:
    """A1.2: transitions between consecutive pursuer spans of >= 1 env step.

    `INTERCEPT` / `LEAD` are always applicable but are DONE on their first call
    when the evader is unseen and the law's last-seen point is exhausted; the
    zero-length spans they leave behind are not switches and must not satisfy
    CG-T2. They stay in the log, flagged.
    """
    families = [tactic_family(s["name"]) for s in spans if not s["zero_length"]]
    return sum(int(a != b) for a, b in itertools.pairwise(families))


class _NeedDecision(Exception):  # a control signal, not an error
    """Raised by the injected `select` when the agent's choice is not pending:
    the span has ended and the env must hand control back to the learner."""


def _game_module():
    """`scripts/s24_tactic_game.py`, loaded by path (it is a script, not a
    package module) exactly as that script loads S19's recorder."""
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "s24_tactic_game.py"
    if "s24_tactic_game" in sys.modules:
        return sys.modules["s24_tactic_game"]
    spec = importlib.util.spec_from_file_location("s24_tactic_game", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["s24_tactic_game"] = module
    spec.loader.exec_module(module)
    return module


class SelectorMDP(gym.Env):
    """One episode of the v0b tactic game with the pursuer's `select` replaced
    by the agent's masked discrete action.

    `env_steps` counts UNDERLYING env steps, so budgets and probes stay in env
    steps as in every prior round. `inapplicable_picks` is CG-T1's counter and
    must stay 0; an inapplicable action also raises, because the mask makes it
    impossible by construction rather than caught afterwards.

    S24-OS: `library` picks the action list (`LIBRARIES`). `v0` is S24.1's five
    tactics and is the default, so every frozen artefact replays through this
    class unchanged (CG-O1); `v1` adds `ORBIT-SEARCH(k)` as a sixth action,
    whose instances are built HERE and registered in the game's own tactic list
    so `reset` resets them. The library is a per-instance property —
    `self.actions`, `self.n_actions`, `self.obs_slices`, `self.memory_slices`,
    `self.obs_len` — and the module constants stay v0's.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 — gym's own mutable field

    def __init__(
        self,
        speed_ratio: float = 1.2,
        *,
        memory: bool = True,
        seed0: int = 0,
        recorder: bool = False,
        label: str = "",
        cfg: EnvConfig | None = None,
        library: str = DEFAULT_LIBRARY,
    ) -> None:
        super().__init__()
        self.cfg = EnvConfig(evader_speed_ratio=float(speed_ratio)) if cfg is None else cfg
        self.memory = bool(memory)
        self.library = str(library)
        (
            self.actions,
            self.n_actions,
            self.obs_layout,
            self.obs_slices,
            self.memory_slices,
            self.obs_len,
        ) = layout(self.library)
        game_module = _game_module()
        self.game, self.rules = game_module.build_game(
            self.cfg, recorder=recorder, label=label
        )
        self.game.inner.seed0 = int(seed0)
        self.game.law.select = self._select  # the ONE replaced field
        self.pursuer = self.rules["pursuer"]
        self._search = OrbitSearch(self.cfg)
        # S24-OS: the library's tactics that are NOT `PursuerRulesV0`'s. The
        # v0 five stay the rule table's own instances (so the library the
        # learner chooses from is bit-for-bit the one the baseline chose from);
        # v1's sixth is built here and appended to the game's law tactic list,
        # which is what `TacticLaw._reset_common` resets per episode. `SWEEP`'s
        # belief is handed to it by REFERENCE, the one the memory owns.
        self.rules_extra: dict[str, list] = {}
        if "ORBIT-SEARCH" in self.actions:
            searches = [
                _Capped(
                    OrbitSearchTactic(
                        self.cfg, k, belief=self.pursuer.memory.belief
                    )
                )
                for k in range(self.cfg.n_obstacles)
            ]
            self.rules_extra["orbit_search"] = searches
            self.game.law.tactics.extend(searches)
        self.action_space = spaces.Discrete(self.n_actions)
        # A bound, not a normalisation claim: a body-frame range over the half
        # arena reaches 2.83 and the recogniser clips its trends at 3.0.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_len,), dtype=np.float32
        )
        self.inapplicable_picks = 0
        self.n_decisions = 0
        # Per-episode count of the ACTIONS taken, by name: H-TAC-2's "tactic
        # usage distribution ... chi^2 over the 5 tactics", which the game's own
        # per-env-step `usage` counter cannot give (it carries a sixth key,
        # `(search policy)`, and weights a span by its length).
        self.picks: Counter = Counter()
        self._pending = None
        self._span_steps = 0
        self._obs_raw = np.zeros(self.game.inner.observation_space.shape, np.float32)

    # -- the injected selector -------------------------------------------
    def _select(self, obs, env, history):
        if self._pending is None:
            raise _NeedDecision
        tactic, self._pending = self._pending, None
        return tactic

    # -- the action -> tactic map (reading 1) ----------------------------
    @property
    def env_steps(self) -> int:
        return int(self.game.env_steps)

    @property
    def violations_total(self) -> int:
        return int(self.game.violations_total)

    def _estimate_xy(self, obs) -> np.ndarray:
        """The evader's position estimate: observed while visible, else the
        dead-reckoning point, else the pursuer's own pose (reading 1)."""
        if obs[6] > 0.5:
            return opponent_world_from_obs(np.asarray(obs), self.cfg)
        point = self.pursuer.memory.point
        if point is not None:
            return np.asarray(point, dtype=float)[:2]
        return np.asarray(self.game.inner.env.pursuer_pose, dtype=float)[:2]

    def _candidates(self, obs) -> list:
        """The tactics this decision's actions resolve to, in `self.actions`
        order; None where the family has no instance (no obstacles).

        S24-OS: `ORBIT-SEARCH` resolves per obstacle like the other two
        families and off the SAME k, the obstacle nearest the evader's position
        estimate (reading 1). Under v0 the list is S24.1's five, unchanged.
        """
        rules, env = self.pursuer, self.game.inner.env
        k = _nearest_obstacle(self._estimate_xy(obs), env.obstacles)
        flank = None if k is None or k >= len(rules.counter_orbits) else rules.counter_orbits[k]
        trap = None if k is None or k >= len(rules.ambushes) else rules.ambushes[k]
        out = [rules.intercept, rules.lead, flank, trap, rules.sweep]
        if "ORBIT-SEARCH" in self.actions:
            searches = self.rules_extra["orbit_search"]
            out.append(None if k is None or k >= len(searches) else searches[k])
        return out

    def action_masks(self) -> np.ndarray:
        """`applicable()` of the tactic each action resolves to, at this
        decision (the `sb3_contrib.MaskablePPO` contract), minus the two
        gates: `COUNTER-ORBIT` while the evader is invisible (reading 2) and
        any tactic that would be DONE on entry (A2.1, `done_on_entry`).

        The mask is never empty: `done_on_entry` for `INTERCEPT`/`LEAD`
        requires the evader to be invisible, which is exactly where `SWEEP`'s
        own `applicable()` holds, and `AMBUSH`'s clause requires it to be
        visible, where `INTERCEPT`/`LEAD` are admitted.
        """
        obs, env = self._obs_raw, self.game.inner.env
        visible = bool(obs[6] > 0.5)
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, tactic in enumerate(self._candidates(obs)):
            if tactic is None:
                continue
            if self.actions[i] == "COUNTER-ORBIT" and not visible:
                continue
            if done_on_entry(tactic, obs, env, self.cfg):  # A2.1
                continue
            mask[i] = bool(tactic.applicable(obs, env))
        return mask

    # -- observation ------------------------------------------------------
    def _spans(self) -> list:
        """The current (ending) span and the three before it, most recent
        first, as `(family, length)` or None (reading 5)."""
        history = self.game.law.history
        out = []
        for back in range(1 + N_LAST_TACTICS):
            if back >= len(history):
                out.append(None)
                continue
            name, start, end = history[-1 - back]
            stop = self.game.law.step_index if end is None else end
            out.append((tactic_family(name), int(stop) - int(start)))
        return out

    def _observe(self, obs) -> np.ndarray:
        cfg, env = self.cfg, self.game.inner.env
        half = cfg.arena_size / 2.0
        own = np.asarray(env.pursuer_pose, dtype=float)
        slices = self.obs_slices
        out = np.zeros(self.obs_len, dtype=np.float32)
        visible = float(obs[6] > 0.5)
        out[slices["visible"]] = visible
        if visible:
            dx_b, dy_b = float(obs[3]) * cfg.arena_size, float(obs[4]) * cfg.arena_size
            out[slices["evader_range_bearing"]] = (
                float(np.hypot(dx_b, dy_b)) / half,
                float(np.arctan2(dy_b, dx_b)) / np.pi,
            )
        point, age = self.pursuer.memory.point, self.pursuer.memory.age
        if point is not None:
            d = np.asarray(point, dtype=float)[:2] - own[:2]
            c, s = np.cos(own[2]), np.sin(own[2])
            out[slices["dead_reckoning"]] = (
                float(c * d[0] + s * d[1]) / cfg.arena_size,
                float(-s * d[0] + c * d[1]) / cfg.arena_size,
                float(np.clip((0 if age is None else age) / AGE_SCALE, 0.0, 1.0)),
            )
        out[slices["own_pose"]] = (
            own[0] / half,
            own[1] / half,
            float(np.cos(own[2])),
            float(np.sin(own[2])),
        )
        start = slices["obstacles"].start
        order = np.argsort(_surface_distances(own, env.obstacles))
        for slot, j in enumerate(order[:N_OBSTACLE_SLOTS]):
            o = env.obstacles[int(j)]
            d = np.array([o["x"], o["y"]], dtype=float) - own[:2]
            c, s = np.cos(own[2]), np.sin(own[2])
            out[start + 3 * slot : start + 3 * slot + 3] = (
                (float(np.hypot(*d)) - float(o["radius"])) / half,
                float(np.arctan2(-s * d[0] + c * d[1], c * d[0] + s * d[1])) / np.pi,
                float(o["radius"]) / cfg.obstacle_radius_max,
            )
        out[slices["wall"]] = (half - max(abs(own[0]), abs(own[1]))) / half
        f = self.pursuer.memory.features()
        out[slices["features"]] = [*(float(f[k]) for k in FEATURES), float(f["valid"])]
        spans = self._spans()
        if spans[0] is not None:
            family, length = spans[0]
            out[slices["current_tactic"].start + self.actions.index(family)] = 1.0
            out[slices["current_tactic"].stop - 1] = float(
                np.clip(length / DECISION_PERIOD, 0.0, 1.0)
            )
        start = slices["last_tactics"].start
        for back, span in enumerate(spans[1:]):
            if span is not None:
                out[start + self.n_actions * back + self.actions.index(span[0])] = 1.0
        out[slices["speed_ratio"]] = float(cfg.evader_speed_ratio)
        # A1.1: read off the game's own `SearchMDP` methods, so the definitions
        # and the clipping are S16/S17's and not a second copy of them.
        out[slices["shield"]] = self.game.shield_features()
        out[slices["escape"]] = self.game.escape_features()
        if not self.memory:  # A1.1: the five new entries are NOT memory
            for block in self.memory_slices:
                out[block] = 0.0
        return out

    # -- episode driving --------------------------------------------------
    def _advance(self, obs):
        """Env steps from here until the law needs a new decision or the
        episode ends. Returns (raw obs, summed reward, terminated, truncated)."""
        reward, term, trunc = 0.0, False, False
        while not (term or trunc):
            if self._pending is None and self.game.law.current is None:
                break  # a decision point: the agent's turn
            try:
                action = self.game.law.act(obs)
            except _NeedDecision:
                # The span just ended (`_close`d by the adapter) and no env
                # step has been taken from this state.
                self.game.law.current = None
                if self._span_steps == 0:
                    # Reading 4, kept as a GUARD only: A2.1 masks out every
                    # tactic that would be DONE on entry, so this branch must
                    # never fire and the `(search policy)` usage count must
                    # read 0 over a whole round.
                    obs, r, _o, term, trunc = self.game._step_inner(
                        self._search.predict(obs)[0], "search"
                    )
                    reward += r
                break
            if action is None:  # unreachable: `_select` raises instead
                raise AssertionError("TacticLaw returned None with a live selector")
            obs, r, _o, term, trunc = self.game._step_inner(action, "law")
            reward += r
            self._span_steps += 1
        return obs, reward, term, trunc

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pending = None
        self._span_steps = 0
        self.n_decisions = 0
        self.picks = Counter()
        obs = self.game._begin(seed)
        self._obs_raw = obs
        return self._observe(obs), {"env_steps": self.env_steps}

    def step(self, action):
        index = int(action)
        mask = self.action_masks()
        if not mask[index]:
            self.inapplicable_picks += 1  # CG-T1: counted, then refused
            raise AssertionError(
                f"action {index} ({self.actions[index]}) is not applicable at this "
                f"decision (mask {mask.astype(int).tolist()}); CG-T1 requires "
                "the mask to make an inapplicable pick impossible"
            )
        self._pending = self._candidates(self._obs_raw)[index]
        self._span_steps = 0
        self.n_decisions += 1
        self.picks[self.actions[index]] += 1
        # A2.4: the span bookkeeping `SearchMDP.step` /
        # `BranchTaggedSearchMDP.step` do, which this env bypasses by driving
        # `_step_inner` itself. Without it every step record reads
        # `in_span: False` and `span_intervention_total` is identically 0, so
        # the recorder's split of the filter's work is dead.
        self.game._in_span = True
        self.game.span_intervention = 0.0
        self.game.span_stalled = 0
        obs, reward, term, trunc = self._advance(self._obs_raw)
        self.game._span_total += float(self.game.span_intervention)
        self._obs_raw = obs
        info = dict(
            self.game._info,
            env_steps=self.env_steps,
            span_steps=self._span_steps,
            inapplicable_picks=self.inapplicable_picks,
            violations=int(self.game.episode_violations),
            stalled_steps=int(self.game.span_stalled),
            intervention_magnitude=float(self.game.span_intervention),
        )
        return self._observe(obs), float(reward), term, trunc, info

    # -- evaluation -------------------------------------------------------
    def spans(self) -> list[dict]:
        """The pursuer's span log with each span's length in env steps and
        A1.2's `zero_length` flag. The OPEN span is closed at `step_index`,
        as the v0b record's `_history` does."""
        return spans_with_length(self.game.law.history, self.game.law.step_index)

    def n_switches(self) -> int:
        """CG-T2's quantity, under A1.2."""
        return count_switches(self.spans())

    def play_episode(self, predict, seed: int) -> dict:
        """One seeded episode, `predict(obs, deterministic=True,
        action_masks=...) -> (action, _)` deciding at every decision point —
        the `MaskablePPO.predict` contract."""
        game = self.game
        obs, _info = self.reset(seed=seed)
        term = trunc = False
        while not (term or trunc):
            action, _ = predict(
                obs, deterministic=True, action_masks=self.action_masks()
            )
            obs, _r, term, trunc, _i = self.step(action)
        log = {
            "seed": int(seed),
            "role": "pursuer",
            "outcome": game._info["outcome"],
            "steps": int(game.inner.env.current_step),
            "final_distance": float(game._info["distance"]),
            "los_breaks": int(game._los_breaks),
            "shielded_role": "pursuer",
            "violations": int(game.episode_violations),
            "shield": game.shield_metrics.summary(),
            "branch": game.branch_summary(),
            "tactics": {
                "evader": _game_module()._history(game.evader),
                "pursuer": _game_module()._history(game.law),
            },
            "usage": {role: dict(counts) for role, counts in game.usage.items()},
            "n_decisions": int(self.n_decisions),
            "picks": dict(self.picks),
            # A1.2: the flagged span log, and the switch count over the spans
            # of >= 1 env step only.
            "pursuer_spans": self.spans(),
            "n_zero_length_spans": sum(s["zero_length"] for s in self.spans()),
            "n_switches": int(self.n_switches()),
            "inapplicable_picks": int(self.inapplicable_picks),
            "recovery": {
                "n_recoveries": int(game.shield.n_recoveries),
                "n_forward": int(game.shield.n_recoveries_forward),
                "n_rotation": int(game.shield.n_recoveries_rotation),
            },
        }
        return log
